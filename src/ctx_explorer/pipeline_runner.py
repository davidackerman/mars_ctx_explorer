"""End-to-end CTX similarity pipeline: download → ISIS3 → tile → embed → index.

Unifies the three legacy scripts (download_ctx_images, build_ctx_retrieval_index,
build_ctx_patch_index) behind a single ``PipelineConfig`` + ``run()`` driver so
callers can turn every cost-driving knob from one place. Defaults are tuned for
a cheap overnight run; cranking any knob up is straightforward.

This module intentionally reuses the existing pieces rather than duplicating
logic:

  - CTXDownloader      (download + ISIS3)
  - generate_chunk_tiles (PNG tile generation)
  - DINOv3HFExtractor   (CLS / patch tokens)
  - CTXSimilarityIndex.build_from_embeddings (FAISS build, IVF-PQ aware)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image

from ctx_explorer.embeddings import DINOv3HFExtractor, EmbeddingPipeline
from ctx_explorer.downloader import CTXDownloader
from ctx_explorer.retrieval import (
    CTXSimilarityIndex,
    discover_images,
    generate_chunk_tiles,
)

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class PipelineConfig:
    # corpus
    limit: int = 1000
    download_dir: Path = Path("data/raw/ctx")
    output_dir: Path = Path("outputs/ctx_similarity")
    overwrite: bool = False

    # ISIS3
    use_isis3: bool = True
    apply_calibration: bool = False
    map_resolution: float = 48.0  # m/pixel; 24 is native
    workers: int = 6  # download / ISIS3 threads (PDS rate-limits around 8)

    # tiling
    tile_scales: List[int] = field(default_factory=lambda: [512])
    tile_stride_fraction: float = 0.5
    tile_min_std: float = 5.0
    tile_max_nodata_frac: float = 0.1
    tile_workers: int = 8  # CPU processes tiling in parallel

    # embedding
    model_name: str = "facebook/dinov3-vitl16-pretrain-sat493m"
    image_size: int = 224
    batch_size: int = 128
    num_workers: int = 8
    use_half_precision: bool = True
    embed_mode: str = "cls"  # 'cls' | 'both'
    device: str = "cuda"

    # indexing
    index_type: str = "ivfpq"  # 'flat' | 'ivfpq'
    pq_bytes: int = 64
    nlist: Optional[int] = None
    normalize: bool = True

    # housekeeping
    delete_intermediates: bool = True  # delete tile PNGs after index is built
    delete_source_tifs: bool = False  # also delete THIS run's source .tifs (opt-in)
    dry_run: bool = False
    yes: bool = False

    # overrides / helpers
    search_kwargs: dict = field(default_factory=dict)  # passed to ODE search

    def index_dim(self) -> int:
        """Embedding dim for the configured backbone (hardcoded short table)."""
        known = {
            "facebook/dinov3-vitl16-pretrain-sat493m": 1024,
            "facebook/dinov3-vitl16-pretrain-lvd1689m": 1024,
            "facebook/dinov3-vitb16-pretrain-lvd1689m": 768,
            "facebook/dinov3-vit7b16-pretrain-sat493m": 4096,
        }
        return known.get(self.model_name, 1024)


# --------------------------------------------------------------------------- #
# Budget estimate
# --------------------------------------------------------------------------- #


# Heuristics calibrated against the current 91-image/67k-tile run.
# Units: bytes per image, per tile; seconds per tile.
_MB = 1024 * 1024
AVG_BYTES_PER_IMAGE_24MPP = 55 * _MB  # measured median .tif @ 24 mpp
AVG_TILES_PER_IMAGE_MULTISCALE = 740  # 256 + 512 + 1024 at 0.5 stride
AVG_TILES_PER_IMAGE_512_ONLY = 130  # rough fraction of the multiscale count
GPU_TILES_PER_SECOND_CLS_224 = 200  # DINOv3-L, half precision, 224 input
GPU_TILES_PER_SECOND_CLS_512 = 40
PER_IMAGE_ISIS3_SECONDS = 60  # wall-clock at 4 workers, per-image serial cost


def estimate_budget(cfg: PipelineConfig) -> dict:
    """Compute an order-of-magnitude resource estimate for a run."""
    # Fewer pixels at lower resolution → smaller tifs + tiles. Crude 1/(r^2).
    pixel_ratio = (24.0 / cfg.map_resolution) ** 2
    per_image_bytes = max(1, int(AVG_BYTES_PER_IMAGE_24MPP * pixel_ratio))

    scales = tuple(sorted(int(s) for s in cfg.tile_scales))
    if scales == (256, 512, 1024):
        base_tpi = AVG_TILES_PER_IMAGE_MULTISCALE
    elif scales == (512,):
        base_tpi = AVG_TILES_PER_IMAGE_512_ONLY
    else:
        # Rough: tiles per scale scale as 1/scale² and stride²
        base_tpi = int(
            sum(
                AVG_TILES_PER_IMAGE_512_ONLY
                * (512 / s) ** 2
                * (0.5 / cfg.tile_stride_fraction) ** 2
                for s in scales
            )
        )

    # Tile density also scales with ~pixel_ratio
    tiles_per_image = max(1, int(base_tpi * pixel_ratio))
    total_tiles = cfg.limit * tiles_per_image

    # Tile PNGs: 50 kB average at 8-bit, scales roughly linear in side length
    bytes_per_tile = sum(int(0.05 * _MB * (s / 256) ** 1.3) for s in scales) // max(
        1, len(scales)
    )
    tile_bytes = total_tiles * bytes_per_tile

    source_bytes = cfg.limit * per_image_bytes

    dim = cfg.index_dim()
    if cfg.index_type == "ivfpq":
        # IVF: coarse centroid codes (tiny) + nlist × quantizer + PQ codes
        index_bytes = total_tiles * cfg.pq_bytes + 4 * dim * max(32, cfg.nlist or 256)
    else:
        index_bytes = total_tiles * dim * 4

    # Embedding parquet (always fp32 on disk); small for CLS, big for patches
    emb_per_tile = dim * 4
    if cfg.embed_mode == "both":
        emb_per_tile += 196 * dim * 4  # patch tokens too
    embeddings_bytes = total_tiles * emb_per_tile

    gpu_rate = (
        GPU_TILES_PER_SECOND_CLS_224
        if cfg.image_size <= 256
        else GPU_TILES_PER_SECOND_CLS_512
    )
    gpu_seconds = total_tiles / max(1, gpu_rate)
    download_seconds = cfg.limit * PER_IMAGE_ISIS3_SECONDS / max(1, cfg.workers)

    total_seconds = download_seconds + gpu_seconds

    persistent_bytes = index_bytes + embeddings_bytes // 10  # parquet compresses ~10x
    if not cfg.delete_intermediates:
        persistent_bytes += source_bytes + tile_bytes
    peak_bytes = source_bytes + tile_bytes + index_bytes + embeddings_bytes

    return {
        "limit": cfg.limit,
        "tiles_per_image": tiles_per_image,
        "total_tiles": total_tiles,
        "source_bytes": source_bytes,
        "tile_bytes": tile_bytes,
        "embeddings_bytes": embeddings_bytes,
        "index_bytes": index_bytes,
        "persistent_bytes": persistent_bytes,
        "peak_bytes": peak_bytes,
        "download_seconds": download_seconds,
        "gpu_seconds": gpu_seconds,
        "total_seconds": total_seconds,
    }


def pretty_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def pretty_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    if s < 86400:
        return f"{s / 3600:.1f}h"
    return f"{s / 86400:.1f}d"


def print_budget(cfg: PipelineConfig, est: dict, disk_avail_bytes: int) -> None:
    print("=" * 60)
    print(f"CTX pipeline pre-flight — limit={cfg.limit}")
    print("-" * 60)
    print(f"  resolution          {cfg.map_resolution:.0f} m/px")
    print(f"  tile scales         {cfg.tile_scales} (stride {cfg.tile_stride_fraction})")
    print(f"  model / image_size  {cfg.model_name} @ {cfg.image_size}")
    print(f"  embed_mode          {cfg.embed_mode}")
    print(f"  index_type          {cfg.index_type} (pq_bytes={cfg.pq_bytes})")
    print(
        f"  delete_intermediates  "
        f"{cfg.delete_intermediates}  →  intermediates purged as they complete"
    )
    print("-" * 60)
    print(f"  est tiles/image      {est['tiles_per_image']:,}")
    print(f"  est total tiles      {est['total_tiles']:,}")
    print(f"  est source .tif disk {pretty_bytes(est['source_bytes'])}")
    print(f"  est tile PNG disk    {pretty_bytes(est['tile_bytes'])}")
    print(f"  est embeddings disk  {pretty_bytes(est['embeddings_bytes'])}")
    print(f"  est FAISS index disk {pretty_bytes(est['index_bytes'])}")
    print(
        f"  est PERSISTENT disk  "
        f"{pretty_bytes(est['persistent_bytes'])}  (what's left after run)"
    )
    print(
        f"  est PEAK disk        "
        f"{pretty_bytes(est['peak_bytes'])}  (during run, before cleanup)"
    )
    print(f"  available disk       {pretty_bytes(disk_avail_bytes)}")
    print("-" * 60)
    print(f"  est download+ISIS3   {pretty_seconds(est['download_seconds'])}")
    print(f"  est GPU embed        {pretty_seconds(est['gpu_seconds'])}")
    print(f"  est TOTAL wall-clock {pretty_seconds(est['total_seconds'])}")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


_ISIS3_CANDIDATE_ROOTS = [
    "/groups/scicompsoft/home/ackermand/miniconda3/envs/isis3",
    os.path.expanduser("~/miniconda3/envs/isis3"),
    os.path.expanduser("~/anaconda3/envs/isis3"),
]
_ISISDATA_CANDIDATE = str(Path(__file__).resolve().parents[2] / "data" / "isis3data")


def _ensure_isis3_env(cfg: PipelineConfig) -> None:
    """Auto-set ISISROOT/ISISDATA/PATH if ISIS3 is enabled and they're missing.

    Without this the ISIS3 subprocess (`mroctx2isis`, `cam2map`, ...) fails to
    find its binaries when the pipeline runs via `pixi run` without the user
    having sourced a shell helper first. Silent fallback to the GDAL converter
    produces unprojected .tifs that corrupt the downstream index, so we fail
    loud instead.
    """
    if not cfg.use_isis3:
        return

    if not os.environ.get("ISISROOT"):
        for candidate in _ISIS3_CANDIDATE_ROOTS:
            if Path(candidate, "bin", "mroctx2isis").exists():
                os.environ["ISISROOT"] = candidate
                logger.info("Auto-set ISISROOT=%s", candidate)
                break

    isisroot = os.environ.get("ISISROOT")
    if isisroot:
        bin_dir = f"{isisroot}/bin"
        current_path = os.environ.get("PATH", "")
        if bin_dir not in current_path.split(":"):
            os.environ["PATH"] = f"{bin_dir}:{current_path}"

    if not os.environ.get("ISISDATA") and Path(_ISISDATA_CANDIDATE).exists():
        os.environ["ISISDATA"] = _ISISDATA_CANDIDATE
        logger.info("Auto-set ISISDATA=%s", _ISISDATA_CANDIDATE)

    # Hard check that mroctx2isis is actually on PATH now.
    if shutil.which("mroctx2isis") is None:
        raise RuntimeError(
            "ISIS3 enabled but `mroctx2isis` is not on PATH even after auto-probe. "
            "Set ISISROOT / ISISDATA explicitly, pass --no-isis3 (will produce "
            "unprojected .tifs — unusable for global retrieval), or install ISIS3 "
            "at one of the known paths."
        )


def run(cfg: PipelineConfig) -> dict:
    """Run the pipeline end-to-end and return a summary dict."""
    cfg.download_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_isis3_env(cfg)

    disk_avail = shutil.disk_usage(cfg.output_dir).free
    est = estimate_budget(cfg)
    print_budget(cfg, est, disk_avail)

    if est["peak_bytes"] > disk_avail:
        logger.warning(
            "Estimated peak disk (%s) exceeds available (%s). "
            "Use --delete-intermediates, lower --limit, or increase --map-resolution.",
            pretty_bytes(est["peak_bytes"]),
            pretty_bytes(disk_avail),
        )
        if not cfg.yes:
            raise SystemExit("Aborting: not enough disk. Re-run with --yes to override.")

    if cfg.dry_run:
        logger.info("Dry run — exiting after estimate.")
        return {"dry_run": True, "estimate": est}

    t_start = time.time()

    # -- Stage 1: download + ISIS3 -----------------------------------------
    downloader = CTXDownloader(
        output_dir=cfg.download_dir,
        use_isis3=cfg.use_isis3,
        apply_calibration=cfg.apply_calibration,
        map_resolution=cfg.map_resolution,
    )
    search_kwargs = dict(cfg.search_kwargs)
    search_kwargs.setdefault("limit", cfg.limit)
    logger.info("Searching ODE for up to %d CTX images", cfg.limit)
    product_list = downloader.search_images(**search_kwargs)
    if not product_list:
        raise RuntimeError("ODE returned no products for the requested query.")

    logger.info("Downloading %d products with %d workers", len(product_list), cfg.workers)
    t_download = time.time()
    downloaded_paths = downloader.download_images(
        product_list, overwrite=cfg.overwrite, max_workers=cfg.workers
    )
    logger.info("Download + ISIS3 done in %.1fs", time.time() - t_download)

    # -- Stage 2: tile -----------------------------------------------------
    image_paths = discover_images(cfg.download_dir, recursive=False)
    if not image_paths:
        raise RuntimeError(f"No .tif files found under {cfg.download_dir}")

    tiles_dir = cfg.output_dir / "tiles"
    t_tile = time.time()
    tile_paths = generate_chunk_tiles(
        image_paths=image_paths,
        tile_output_dir=tiles_dir,
        tile_scales=cfg.tile_scales,
        stride_fraction=cfg.tile_stride_fraction,
        min_std=cfg.tile_min_std,
        max_nodata_frac=cfg.tile_max_nodata_frac,
        workers=cfg.tile_workers,
    )
    logger.info(
        "Tiled %d images → %d tiles in %.1fs",
        len(image_paths),
        len(tile_paths),
        time.time() - t_tile,
    )
    if not tile_paths:
        raise RuntimeError("Tile generation produced zero tiles; adjust filters.")

    # -- Stage 3: embed ----------------------------------------------------
    os.environ.setdefault("HF_HOME", "/mnt/bigdisk/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/mnt/bigdisk/hf_cache/hub")

    extractor = DINOv3HFExtractor(
        model_name=cfg.model_name,
        device=cfg.device,
        use_half_precision=cfg.use_half_precision,
    )
    transform = DINOv3HFExtractor.get_default_transforms(image_size=cfg.image_size)

    pipeline = EmbeddingPipeline(
        extractor=extractor,
        output_format="parquet",
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        transform=transform,
    )
    embeddings_path = cfg.output_dir / "embeddings.parquet"
    # Replace any stale parquet from a previous (possibly duplicated) run
    if embeddings_path.exists():
        embeddings_path.unlink()
    t_embed = time.time()
    pipeline.extract_dataset(
        image_paths=tile_paths,
        output_path=embeddings_path,
        resume=False,
    )
    logger.info("Embed done in %.1fs", time.time() - t_embed)

    # -- Stage 4: FAISS index ---------------------------------------------
    manifest_path = cfg.download_dir / "manifest.json"
    t_index = time.time()
    CTXSimilarityIndex.build_from_embeddings(
        embeddings_path=embeddings_path,
        index_path=cfg.output_dir / "faiss.index",
        metadata_path=cfg.output_dir / "metadata.parquet",
        manifest_path=manifest_path if manifest_path.exists() else None,
        normalize=cfg.normalize,
        model_name=cfg.model_name,
        image_size=cfg.image_size,
        index_type=cfg.index_type,
        pq_bytes=cfg.pq_bytes,
        nlist=cfg.nlist,
    )
    logger.info("FAISS build done in %.1fs", time.time() - t_index)

    # -- Stage 5: cleanup (if requested) ----------------------------------
    # Only nuke what THIS run produced, never pre-existing .tifs that the user
    # may still want. Tiles are cheap to regenerate; source .tifs are not.
    if cfg.delete_intermediates:
        logger.info("Deleting tile PNGs (--delete-intermediates; source .tifs kept)")
        if tiles_dir.exists():
            shutil.rmtree(tiles_dir, ignore_errors=True)
    if cfg.delete_source_tifs:
        newly_downloaded = [Path(p) for p in (downloaded_paths or []) if p is not None]
        logger.info(
            "Deleting %d source .tifs from this run (--delete-source-tifs)",
            len(newly_downloaded),
        )
        for tif in newly_downloaded:
            tif.unlink(missing_ok=True)

    total = time.time() - t_start
    logger.info("Pipeline complete in %s", pretty_seconds(total))

    return {
        "dry_run": False,
        "estimate": est,
        "actual_seconds": total,
        "tile_count": len(tile_paths),
        "embeddings_path": str(embeddings_path),
        "index_path": str(cfg.output_dir / "faiss.index"),
    }
