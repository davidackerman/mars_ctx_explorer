#!/usr/bin/env python3
"""Stream Murray Lab CTX tiles → DINOv3 patch tokens → IVF-PQ index.

Companion to stream_murray_index.py (which stores one CLS vector per tile).
This variant stores every patch token (196 per tile at 224 input) so queries
can localise "where within a tile" the match is — no post-hoc lookup needed.

Size @ zoom 10 (1.77 M valid tiles × 196 patches × 64 B PQ code) ≈ 22 GB.
Wall-clock: ~3.5 h on one GPU (network-bound; same as the CLS run).

Pixel aspect is corrected the same way as the CLS indexer
(scripts/stream_murray_index.fetch_tile) so embeddings at all latitudes live
in the same feature space.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import queue
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from tqdm import tqdm

# Share the tile-fetching + geometry utilities with the CLS indexer.
from stream_murray_index import (
    LEVEL0_RES_DEG,
    TILE_PX,
    TILE_URL,
    TileItem,
    enumerate_tiles,
    fetch_tile,
    pixel_size_deg,
    tile_center_deg,
)

Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run(
    zoom: int,
    output_dir: Path,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    max_tiles: Optional[int] = None,
    concurrency: int = 48,
    model_name: str = "facebook/dinov3-vitl16-pretrain-sat493m",
    image_size: int = 224,
    batch_size: int = 64,
    train_sample: int = 200_000,
    pq_bytes: int = 64,
    nlist: Optional[int] = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_tiles = list(enumerate_tiles(zoom, bbox))
    if max_tiles is not None:
        all_tiles = all_tiles[:max_tiles]
    logger.info(
        "Zoom %d: %d tiles (~%.1f km/side at equator)",
        zoom,
        len(all_tiles),
        TILE_PX * pixel_size_deg(zoom) * 59.3,
    )

    os.environ.setdefault("HF_HOME", "/mnt/bigdisk/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/mnt/bigdisk/hf_cache/hub")
    from scientific_pipelines.core.embeddings import DINOv3HFExtractor

    extractor = DINOv3HFExtractor(
        model_name=model_name, device="cuda", use_half_precision=True
    )
    transform = DINOv3HFExtractor.get_default_transforms(image_size=image_size)
    dim = extractor.get_embedding_dim()
    # Probe num_patches_per_tile with a dummy forward
    probe = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    probe_patches = extractor.extract_patches(transform(probe).unsqueeze(0))
    num_patches_per_tile = int(probe_patches.shape[1])
    logger.info(
        "Extractor: %s @ %d, dim=%d, %d patches/tile",
        model_name,
        image_size,
        dim,
        num_patches_per_tile,
    )

    # --- Fetcher thread-pool feeding a queue ------------------------------ #
    tile_queue: queue.Queue[Optional[TileItem]] = queue.Queue(maxsize=concurrency * 4)
    stop_flag = threading.Event()

    def fetch_worker(subset: List[Tuple[int, int, int]]) -> None:
        sess = requests.Session()
        sess.headers["User-Agent"] = (
            "mars-astrobio-research/0.1 (contact: ackermand@janelia.hhmi.org)"
        )
        for z, x, y in subset:
            if stop_flag.is_set():
                return
            item = fetch_tile(sess, z, x, y, correct_aspect=True)
            tile_queue.put(item)
        tile_queue.put(None)

    subsets = [all_tiles[i::concurrency] for i in range(concurrency)]
    pool = ThreadPoolExecutor(max_workers=concurrency)
    futures = [pool.submit(fetch_worker, s) for s in subsets]

    def _shutdown(*_):
        stop_flag.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # --- Embed loop ------------------------------------------------------- #
    import faiss

    training_vectors: List[np.ndarray] = []
    training_done = False
    index: Optional[faiss.IndexIVFPQ] = None
    effective_nlist = nlist or max(
        256, int(math.sqrt(len(all_tiles) * num_patches_per_tile))
    )

    index_path = output_dir / "patches.faiss"
    tile_rows: List[dict] = []
    pending_images: List[np.ndarray] = []
    pending_coords: List[Tuple[int, int, int]] = []
    alive_fetchers = concurrency
    errors = 0
    total_tiles_added = 0
    start_time = time.time()

    def flush_batch() -> None:
        nonlocal training_done, index, total_tiles_added
        if not pending_images:
            return
        # Varying widths (cos-lat correction) — apply transform per-image so
        # each ends up 224×224 before stacking.
        batch_tensors = []
        for arr in pending_images:
            batch_tensors.append(transform(Image.fromarray(arr, mode="RGB")))
        batch = torch.stack(batch_tensors, dim=0)
        patches = extractor.extract_patches(batch).astype("float32")
        # patches: (B, P, D) → flatten to (B*P, D)
        B, P, D = patches.shape
        patches_flat = patches.reshape(B * P, D)
        faiss.normalize_L2(patches_flat)

        if not training_done:
            training_vectors.append(patches_flat.copy())
            collected = sum(v.shape[0] for v in training_vectors)
            if collected >= train_sample:
                train_mat = np.vstack(training_vectors)[:train_sample]
                logger.info(
                    "Training IndexIVFPQ(nlist=%d, m=%d, nbits=8) on %d patch vectors",
                    effective_nlist,
                    pq_bytes,
                    train_mat.shape[0],
                )
                quantizer = faiss.IndexFlatIP(D)
                index = faiss.IndexIVFPQ(
                    quantizer, D, effective_nlist, pq_bytes, 8, faiss.METRIC_INNER_PRODUCT
                )
                index.train(train_mat)
                for v in training_vectors:
                    index.add(v)
                training_vectors.clear()
                training_done = True
                logger.info("IVF-PQ trained; switching to stream-add mode")
        else:
            index.add(patches_flat)

        for coord in pending_coords:
            z_i, x_i, y_i = coord
            lat, lon = tile_center_deg(z_i, x_i, y_i)
            tile_rows.append(
                {
                    "tile_row_id": total_tiles_added,
                    "z": z_i,
                    "x": x_i,
                    "y": y_i,
                    "lat": lat,
                    "lon": lon,
                }
            )
            total_tiles_added += 1

        pending_images.clear()
        pending_coords.clear()

    with tqdm(total=len(all_tiles), desc="Stream") as pbar:
        while alive_fetchers > 0 or not tile_queue.empty():
            item = tile_queue.get()
            if item is None:
                alive_fetchers -= 1
                continue
            if item.err is not None or item.image is None:
                errors += 1
                pbar.update(1)
                continue
            pending_images.append(item.image)
            pending_coords.append((item.z, item.x, item.y))
            pbar.update(1)
            if len(pending_images) >= batch_size:
                flush_batch()
        if pending_images:
            flush_batch()

    if not training_done and training_vectors:
        logger.warning(
            "Only %d patch vectors — below train_sample=%d; building flat index",
            sum(v.shape[0] for v in training_vectors),
            train_sample,
        )
        train_mat = np.vstack(training_vectors)
        index = faiss.IndexFlatIP(dim)
        index.add(train_mat)
        # In this path, pending_coords already consumed — tile_rows is empty.
        # Redo the per-tile metadata using implicit ordering (training_vectors
        # was appended in flush order, but we flushed per-batch so the tile
        # rows we already pushed are correct; this only matters for the
        # fallback case which shouldn't fire at scale).

    faiss.write_index(index, str(index_path))
    tile_meta_path = output_dir / "tiles.parquet"
    pd.DataFrame(tile_rows).to_parquet(tile_meta_path, index=False)

    sidecar = {
        "model_name": model_name,
        "image_size": image_size,
        "embedding_dim": dim,
        "tile_source": "arcgis_online_ctx1",
        "zoom": zoom,
        "tile_url_template": TILE_URL,
        "normalize": True,
        "index_type": "ivfpq" if training_done else "flat",
        "pq_bytes": pq_bytes,
        "nlist": effective_nlist if training_done else None,
        "tile_px": TILE_PX,
        "aspect_corrected": True,
        "num_patches_per_tile": num_patches_per_tile,
        "patch_grid_n": int(round(math.sqrt(num_patches_per_tile))),
        "kind": "patch",
    }
    with open(output_dir / "patches.model.json", "w") as fp:
        json.dump(sidecar, fp, indent=2)

    elapsed = time.time() - start_time
    logger.info(
        "Done in %.1fs: %d tiles indexed (%d patches total), %d fetch errors, index=%s",
        elapsed,
        total_tiles_added,
        total_tiles_added * num_patches_per_tile,
        errors,
        index_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream Murray Lab CTX mosaic to a patch-level FAISS index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--zoom", type=int, default=10)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/murray_z10_patch")
    )
    parser.add_argument("--bbox", type=str, default=None)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-sample", type=int, default=200_000)
    parser.add_argument("--pq-bytes", type=int, default=64)
    parser.add_argument("--nlist", type=int, default=None)
    parser.add_argument(
        "--model-name", default="facebook/dinov3-vitl16-pretrain-sat493m"
    )
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bbox = None
    if args.bbox:
        parts = [float(s) for s in args.bbox.split(",")]
        if len(parts) != 4:
            raise ValueError("--bbox must have 4 comma-separated floats")
        bbox = (parts[0], parts[1], parts[2], parts[3])
    run(
        zoom=args.zoom,
        output_dir=args.output_dir,
        bbox=bbox,
        max_tiles=args.max_tiles,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        train_sample=args.train_sample,
        pq_bytes=args.pq_bytes,
        nlist=args.nlist,
        model_name=args.model_name,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
