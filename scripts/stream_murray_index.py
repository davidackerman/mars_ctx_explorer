#!/usr/bin/env python3
"""Stream Murray Lab CTX mosaic tiles → embed → IVF-PQ index, no persistent intermediates.

Tile source: Esri ArcGIS Online MapServer (public, CORS-enabled, no auth).
    https://astro.arcgis.com/arcgis/rest/services/OnMars/CTX1/MapServer/tile/{z}/{y}/{x}

Coordinate system is Mars GCS 2000 equirectangular (wkid 104971):
  - Origin: (lon=-180°, lat=+90°) at tile (0, 0).
  - Level 0 resolution: 0.3515625 deg/pixel.
  - Level z resolution: 0.3515625 / 2^z deg/pixel.
  - Tile size: 512×512 px.
  - Row (y) increases southward; column (x) increases eastward.

Per-tile disk is zero: fetch → decode → embed → PQ-encode → append codes to a
memmapped file. A tiny metadata parquet records (z, x, y, lat, lon, row_id).

Usage:
    pixi run python scripts/stream_murray_index.py \
        --zoom 8 --concurrency 32 --train-sample 20000 --output-dir outputs/murray_z8

Validation:
    --bbox "lon_min,lat_min,lon_max,lat_max"  (default: whole planet)
    --max-tiles N                             (stop after N tiles; prototype)
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import os
import queue
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TILE_URL = (
    "https://astro.arcgis.com/arcgis/rest/services/OnMars/CTX1/MapServer/tile/{z}/{y}/{x}"
)
TILE_PX = 512
LEVEL0_RES_DEG = 0.3515625


# --------------------------------------------------------------------------- #
# Tile math
# --------------------------------------------------------------------------- #


def pixel_size_deg(z: int) -> float:
    return LEVEL0_RES_DEG / (2**z)


def tile_bbox_deg(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) in degrees for a tile."""
    px = pixel_size_deg(z)
    size = TILE_PX * px
    lon_min = -180.0 + x * size
    lat_max = 90.0 - y * size  # north edge
    lon_max = lon_min + size
    lat_min = lat_max - size
    return lon_min, lat_min, lon_max, lat_max


def tile_center_deg(z: int, x: int, y: int) -> Tuple[float, float]:
    lon_min, lat_min, lon_max, lat_max = tile_bbox_deg(z, x, y)
    return (lat_min + lat_max) / 2, (lon_min + lon_max) / 2


def enumerate_tiles(
    z: int, bbox: Optional[Tuple[float, float, float, float]] = None
) -> Iterable[Tuple[int, int, int]]:
    """Yield (z, x, y) tile coords covering the given bbox (lon/lat in deg)."""
    size = TILE_PX * pixel_size_deg(z)
    total_x = int(round(360.0 / size))
    total_y = int(round(180.0 / size))
    if bbox is None:
        x_range = range(total_x)
        y_range = range(total_y)
    else:
        lon_min, lat_min, lon_max, lat_max = bbox
        x_min = max(0, int(math.floor((lon_min + 180.0) / size)))
        x_max = min(total_x - 1, int(math.ceil((lon_max + 180.0) / size)) - 1)
        y_min = max(0, int(math.floor((90.0 - lat_max) / size)))  # y grows southward
        y_max = min(total_y - 1, int(math.ceil((90.0 - lat_min) / size)) - 1)
        x_max = max(x_min, x_max)
        y_max = max(y_min, y_max)
        x_range = range(x_min, x_max + 1)
        y_range = range(y_min, y_max + 1)
    for y in y_range:
        for x in x_range:
            yield z, x, y


# --------------------------------------------------------------------------- #
# Tile fetcher (thread pool)
# --------------------------------------------------------------------------- #


@dataclass
class TileItem:
    z: int
    x: int
    y: int
    image: Optional[np.ndarray]
    err: Optional[str] = None


def fetch_tile(
    session: requests.Session,
    z: int,
    x: int,
    y: int,
    timeout: float = 15.0,
    correct_aspect: bool = True,
) -> TileItem:
    """Fetch a tile and (optionally) undo plate-carrée horizontal stretch.

    Murray Lab's tiles are plate-carrée: a 512×512 raster at latitude φ
    represents cos(φ)·W × W km on the ground. Features get "fatter" by
    1/cos(φ) horizontally toward the poles — a circular crater at 60° looks
    like a 2:1 ellipse to DINO. Left uncorrected, the embedding at high
    latitude is dominated by this artefact.

    With ``correct_aspect=True`` we resample the tile to
    (int(512 · cos(φ)), 512) pixels before handing it downstream. Now both
    axes represent the same km/pixel; DINO's standard 224×224 resize (which
    happens in the transform) stretches that corrected raster back up
    proportionally, so features have the right shape regardless of latitude.
    Pixels below lat ~85° survive; above that we clamp cos(lat) ≥ 0.05 to
    avoid degenerate 1-column tiles.
    """
    url = TILE_URL.format(z=z, x=x, y=y)
    try:
        resp = session.get(url, timeout=timeout)
        if resp.status_code != 200:
            return TileItem(z, x, y, None, f"HTTP {resp.status_code}")
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        if correct_aspect:
            import math

            lat_center = tile_center_deg(z, x, y)[0]
            cos_lat = max(0.05, math.cos(math.radians(abs(lat_center))))
            if cos_lat < 0.999:
                new_w = max(32, int(round(TILE_PX * cos_lat)))
                img = img.resize((new_w, TILE_PX), Image.LANCZOS)
        arr = np.asarray(img)
        return TileItem(z, x, y, arr)
    except Exception as e:
        return TileItem(z, x, y, None, str(e))


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def run(
    zoom: int,
    output_dir: Path,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    max_tiles: Optional[int] = None,
    concurrency: int = 32,
    model_name: str = "facebook/dinov3-vitl16-pretrain-sat493m",
    image_size: int = 224,
    batch_size: int = 128,
    train_sample: int = 20_000,
    pq_bytes: int = 64,
    nlist: Optional[int] = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Tile enumeration ------------------------------------------------ #
    all_tiles = list(enumerate_tiles(zoom, bbox))
    if max_tiles is not None:
        all_tiles = all_tiles[:max_tiles]
    logger.info(
        "Zoom %d: %d tiles (~%.1f km/side at equator)",
        zoom,
        len(all_tiles),
        TILE_PX * pixel_size_deg(zoom) * 59.3,
    )
    if not all_tiles:
        logger.error("No tiles to process; check --zoom / --bbox")
        return

    # --- Extractor ------------------------------------------------------- #
    os.environ.setdefault("HF_HOME", "/mnt/bigdisk/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/mnt/bigdisk/hf_cache/hub")
    from scientific_pipelines.core.embeddings import DINOv3HFExtractor

    extractor = DINOv3HFExtractor(model_name=model_name, device="cuda", use_half_precision=True)
    transform = DINOv3HFExtractor.get_default_transforms(
        image_size=image_size,
        preserve_aspect=True,
    )
    dim = extractor.get_embedding_dim()
    logger.info("Extractor ready: %s @ %d, dim=%d", model_name, image_size, dim)

    # --- Fetcher thread-pool feeding a queue ---------------------------- #
    tile_queue: queue.Queue[TileItem] = queue.Queue(maxsize=concurrency * 4)
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
        tile_queue.put(None)  # poison pill per worker

    # Partition tiles across fetcher threads
    subsets = [all_tiles[i::concurrency] for i in range(concurrency)]
    pool = ThreadPoolExecutor(max_workers=concurrency)
    _futures = [pool.submit(fetch_worker, s) for s in subsets]

    def shutdown_handler(_sig, _frm):
        stop_flag.set()
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # --- Embed loop + IVF-PQ build ------------------------------------- #
    import faiss

    training_vectors: List[np.ndarray] = []
    training_done = False
    index: Optional[faiss.IndexIVFPQ] = None
    effective_nlist = nlist or max(256, int(math.sqrt(len(all_tiles))))

    index_path = output_dir / "faiss.index"
    metadata_rows: List[dict] = []

    alive_fetchers = concurrency
    total_ingested = 0
    pending_images: List[np.ndarray] = []
    pending_coords: List[Tuple[int, int, int]] = []
    errors = 0
    start_time = time.time()

    def flush_batch() -> None:
        nonlocal training_done, index, total_ingested
        if not pending_images:
            return
        # With aspect correction enabled, tiles come in with varying widths
        # (cos(lat) * 512), so we can't np.stack them. Apply transform
        # per-image — transform resizes to 224×224 so the resulting tensors
        # are all the same shape and stack cleanly.
        batch_tensors = []
        for arr in pending_images:
            pil = Image.fromarray(arr, mode="RGB")
            batch_tensors.append(transform(pil))
        batch = torch.stack(batch_tensors, dim=0)
        vectors = extractor.extract(batch).astype("float32")
        faiss.normalize_L2(vectors)

        if not training_done:
            training_vectors.append(vectors.copy())
            collected = sum(v.shape[0] for v in training_vectors)
            if collected >= train_sample:
                train_mat = np.vstack(training_vectors)[:train_sample]
                logger.info(
                    "Training IndexIVFPQ(nlist=%d, m=%d, nbits=8) on %d vectors",
                    effective_nlist,
                    pq_bytes,
                    train_mat.shape[0],
                )
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFPQ(
                    quantizer, dim, effective_nlist, pq_bytes, 8, faiss.METRIC_INNER_PRODUCT
                )
                index.train(train_mat)
                # Flush all training-phase vectors into the index so we don't lose them
                for v in training_vectors:
                    index.add(v)
                training_vectors.clear()
                training_done = True
                logger.info("IVF-PQ trained; switching to stream-add mode")
        else:
            index.add(vectors)

        # Metadata (lat/lon via tile_center_deg)
        for z, x, y in pending_coords:
            lat, lon = tile_center_deg(z, x, y)
            metadata_rows.append(
                {"row_id": total_ingested, "z": z, "x": x, "y": y, "lat": lat, "lon": lon}
            )
            total_ingested += 1

        pending_images.clear()
        pending_coords.clear()

    with tqdm(total=len(all_tiles), desc="Stream") as pbar:
        while alive_fetchers > 0 or not tile_queue.empty():
            item = tile_queue.get()
            if item is None:
                alive_fetchers -= 1
                continue
            if item.err is not None:
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
        # Not enough tiles to train IVF-PQ; fall back to a flat IP index
        logger.warning(
            "Only %d vectors — below train_sample=%d; building Flat index instead.",
            sum(v.shape[0] for v in training_vectors),
            train_sample,
        )
        train_mat = np.vstack(training_vectors)
        index = faiss.IndexFlatIP(dim)
        index.add(train_mat)
        for row_id, (z, x, y) in enumerate(pending_coords):
            lat, lon = tile_center_deg(z, x, y)
            metadata_rows.append(
                {"row_id": row_id, "z": z, "x": x, "y": y, "lat": lat, "lon": lon}
            )

    faiss.write_index(index, str(index_path))
    md = pd.DataFrame(metadata_rows)
    md.to_parquet(output_dir / "metadata.parquet", index=False)

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
        "preprocess": "aspect_preserve_pad_v1",
    }
    import json

    with open(output_dir / "faiss.model.json", "w") as fp:
        json.dump(sidecar, fp, indent=2)

    elapsed = time.time() - start_time
    logger.info(
        "Done in %.1fs: %d indexed, %d errors, index=%s",
        elapsed,
        total_ingested,
        errors,
        index_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream Murray Lab CTX mosaic to a compressed FAISS index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--zoom", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/murray_stream"))
    parser.add_argument("--bbox", type=str, default=None,
                        help="lon_min,lat_min,lon_max,lat_max (default: whole planet)")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-sample", type=int, default=20000)
    parser.add_argument("--pq-bytes", type=int, default=64)
    parser.add_argument("--nlist", type=int, default=None)
    parser.add_argument("--model-name", default="facebook/dinov3-vitl16-pretrain-sat493m")
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
