#!/usr/bin/env python3
"""Unified CTX similarity pipeline driver.

Runs: download → ISIS3 → tile → embed → FAISS index, with every cost-driving
knob exposed as a flag. See docs/PIPELINE.md for defaults + how to dial up.

Examples:
    # Cheap overnight run, 1000 images, CLS-only IVF-PQ index
    pixi run ctx-pipeline --limit 1000

    # Higher fidelity: 24 mpp, multi-scale tiles, 512-input DINO, flat index
    pixi run ctx-pipeline --limit 500 --map-resolution 24 \
        --tile-scales 256,512,1024 --image-size 512 --index-type flat

    # Dry-run: show estimated disk + time, don't actually download
    pixi run ctx-pipeline --limit 5000 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from scientific_pipelines.planetary.mars.ctx.pipeline_runner import (
    PipelineConfig,
    run,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_scales(raw: str) -> list[int]:
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end CTX similarity pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Corpus
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--download-dir", type=Path, default=Path("data/raw/ctx"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ctx_similarity"))
    parser.add_argument("--overwrite", action="store_true")

    # ODE / ISIS3
    parser.add_argument("--no-isis3", action="store_true")
    parser.add_argument("--apply-calibration", action="store_true")
    parser.add_argument("--map-resolution", type=float, default=48.0)
    parser.add_argument("--workers", type=int, default=6,
                        help="Download/ISIS3 threads (PDS rate-limits around 8)")

    # Tiling
    parser.add_argument(
        "--tile-scales",
        type=_parse_scales,
        default=[512],
        help="Comma-separated tile sizes in px (e.g. 256,512,1024)",
    )
    parser.add_argument("--tile-stride-fraction", type=float, default=0.5)
    parser.add_argument("--tile-min-std", type=float, default=5.0)
    parser.add_argument("--tile-max-nodata-frac", type=float, default=0.1)
    parser.add_argument("--tile-workers", type=int, default=8,
                        help="CPU processes for parallel tile generation")

    # Embedding
    parser.add_argument("--model-name", default="facebook/dinov3-vitl16-pretrain-sat493m")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--embed-mode", choices=["cls", "both"], default="cls")
    parser.add_argument("--device", default="cuda")

    # Index
    parser.add_argument("--index-type", choices=["flat", "ivfpq"], default="ivfpq")
    parser.add_argument("--pq-bytes", type=int, default=64)
    parser.add_argument("--nlist", type=int, default=None)
    parser.add_argument("--no-normalize", action="store_true")

    # Housekeeping
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="Keep tile PNGs after index build (default: delete)")
    parser.add_argument("--delete-source-tifs", action="store_true",
                        help="Also delete THIS run's source .tifs (opt-in). "
                             "Pre-existing .tifs in the download dir are never touched.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", "/mnt/bigdisk/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/mnt/bigdisk/hf_cache/hub")

    cfg = PipelineConfig(
        limit=args.limit,
        download_dir=args.download_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        use_isis3=not args.no_isis3,
        apply_calibration=args.apply_calibration,
        map_resolution=args.map_resolution,
        workers=args.workers,
        tile_scales=args.tile_scales,
        tile_stride_fraction=args.tile_stride_fraction,
        tile_min_std=args.tile_min_std,
        tile_max_nodata_frac=args.tile_max_nodata_frac,
        tile_workers=args.tile_workers,
        model_name=args.model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_half_precision=not args.no_half,
        embed_mode=args.embed_mode,
        device=args.device,
        index_type=args.index_type,
        pq_bytes=args.pq_bytes,
        nlist=args.nlist,
        normalize=not args.no_normalize,
        delete_intermediates=not args.keep_intermediates,
        delete_source_tifs=args.delete_source_tifs,
        dry_run=args.dry_run,
        yes=args.yes,
    )

    summary = run(cfg)
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
