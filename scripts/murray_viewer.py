#!/usr/bin/env python3
"""FastAPI viewer for the Murray Lab global CTX index.

Serves a Leaflet map whose base layer IS the Murray Lab mosaic (via Esri's
public ArcGIS Online tile endpoint). Click anywhere on Mars → the backend
figures out the underlying tile at the current zoom, fetches it, embeds,
queries the pre-built FAISS index, and returns top-k similar tiles.

Unlike ctx_viewer_server.py (which assumes local tile PNGs), everything here
is streamed — no persistent tiles, no source-image cache. Thumbnails in the
results panel are rendered directly from the Murray Lab tile URL.

Run:
    pixi run python scripts/murray_viewer.py --index-dir outputs/murray_z8_global
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi import Response
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = None

TILE_URL = (
    "https://astro.arcgis.com/arcgis/rest/services/OnMars/CTX1/MapServer/tile/{z}/{y}/{x}"
)
TILE_PX = 512
LEVEL0_RES_DEG = 0.3515625


def pixel_size_deg(z: int) -> float:
    return LEVEL0_RES_DEG / (2**z)


def latlon_to_tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    size = TILE_PX * pixel_size_deg(z)
    x = int((lon + 180.0) / size)
    y = int((90.0 - lat) / size)
    x = max(0, min(2 * (2**z) - 1, x))
    y = max(0, min(1 * (2**z) - 1, y))
    return x, y


def tile_center_deg(z: int, x: int, y: int) -> tuple[float, float]:
    size = TILE_PX * pixel_size_deg(z)
    lon = -180.0 + (x + 0.5) * size
    lat = 90.0 - (y + 0.5) * size
    return lat, lon


def tile_bounds_deg(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Return (lat_min, lat_max, lon_min, lon_max) in degrees."""
    size = TILE_PX * pixel_size_deg(z)
    lon_min = -180.0 + x * size
    lon_max = lon_min + size
    lat_max = 90.0 - y * size
    lat_min = lat_max - size
    return lat_min, lat_max, lon_min, lon_max


APP_STATE: dict = {}


def _load_single_index(index_dir: Path) -> dict:
    import faiss

    with open(index_dir / "faiss.model.json") as fp:
        sidecar = json.load(fp)
    index = faiss.read_index(str(index_dir / "faiss.index"))
    # Bump IVF probe count — default is 1 (scan only the single nearest cell),
    # which gives ~80% recall@10. ~sqrt(nlist) is the usual sweet spot;
    # nprobe=32 on nlist~1331 gets us well above 95% recall@10 at ~20-30 ms
    # per query (still interactive).
    if isinstance(index, faiss.IndexIVF):
        index.nprobe = 32
        logger.info("Set nprobe=32 on IVF index at %s (nlist=%d)", index_dir, index.nlist)
    return {
        "index_dir": index_dir,
        "index": index,
        "metadata": pd.read_parquet(index_dir / "metadata.parquet"),
        "sidecar": sidecar,
        "zoom": int(sidecar["zoom"]),
    }


def _load_patch_index(patch_dir: Path) -> dict:
    import faiss

    with open(patch_dir / "patches.model.json") as fp:
        sidecar = json.load(fp)
    index = faiss.read_index(str(patch_dir / "patches.faiss"))
    if isinstance(index, faiss.IndexIVF):
        index.nprobe = 64
        logger.info(
            "Set nprobe=64 on patch IVF index at %s (nlist=%d)",
            patch_dir, index.nlist,
        )
    return {
        "index_dir": patch_dir,
        "index": index,
        "tiles_df": pd.read_parquet(patch_dir / "tiles.parquet"),
        "sidecar": sidecar,
        "zoom": int(sidecar["zoom"]),
        "num_patches_per_tile": int(sidecar["num_patches_per_tile"]),
        "patch_grid_n": int(sidecar["patch_grid_n"]),
    }


def _load(index_root: Path, patch_root: Optional[Path] = None) -> None:
    """Load one or many indices from ``index_root``.

    - If ``index_root/faiss.index`` exists, treat it as a single-zoom index.
    - Otherwise scan subdirectories for faiss.index files and register each as
      a zoom level, enabling multi-scale queries.
    - If ``patch_root`` is given, also scan its subdirectories for patch
      indices (``patches.faiss`` + ``tiles.parquet`` + ``patches.model.json``).
    """
    indices: dict[int, dict] = {}
    if (index_root / "faiss.index").exists():
        state = _load_single_index(index_root)
        indices[state["zoom"]] = state
    else:
        for child in sorted(index_root.iterdir()):
            if (child / "faiss.index").exists():
                try:
                    state = _load_single_index(child)
                    indices[state["zoom"]] = state
                    logger.info(
                        "Registered zoom %d from %s (%d vectors)",
                        state["zoom"],
                        child,
                        state["index"].ntotal,
                    )
                except Exception as e:
                    logger.warning("Skipping %s: %s", child, e)
    if not indices:
        raise RuntimeError(f"No FAISS indices found in {index_root}")

    patch_indices: dict[int, dict] = {}
    if patch_root is not None and patch_root.exists():
        candidates: list[Path] = []
        if (patch_root / "patches.faiss").exists():
            candidates.append(patch_root)
        else:
            for child in sorted(patch_root.iterdir()):
                if (child / "patches.faiss").exists():
                    candidates.append(child)
        for child in candidates:
            try:
                state = _load_patch_index(child)
                patch_indices[state["zoom"]] = state
                logger.info(
                    "Registered patch index zoom %d from %s (%d patches, %d tiles)",
                    state["zoom"], child, state["index"].ntotal, len(state["tiles_df"]),
                )
            except Exception as e:
                logger.warning("Skipping patch index %s: %s", child, e)
    APP_STATE["patch_indices"] = patch_indices

    APP_STATE["index_root"] = index_root
    APP_STATE["indices"] = indices
    APP_STATE["available_zooms"] = sorted(indices)
    # Back-compat: point the "current default" at the finest available zoom
    default_zoom = APP_STATE["available_zooms"][-1]
    APP_STATE["default_zoom"] = default_zoom
    APP_STATE["index_dir"] = indices[default_zoom]["index_dir"]
    APP_STATE["index"] = indices[default_zoom]["index"]
    APP_STATE["metadata"] = indices[default_zoom]["metadata"]
    APP_STATE["sidecar"] = indices[default_zoom]["sidecar"]
    logger.info(
        "Multi-scale available zooms: %s (default=%d)",
        APP_STATE["available_zooms"],
        default_zoom,
    )

    os.environ.setdefault("HF_HOME", "/mnt/bigdisk/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/mnt/bigdisk/hf_cache/hub")
    from scientific_pipelines.core.embeddings import DINOv3HFExtractor

    APP_STATE["extractor"] = DINOv3HFExtractor(
        model_name=APP_STATE["sidecar"]["model_name"],
        device="cuda",
        use_half_precision=True,
    )
    APP_STATE["transform"] = DINOv3HFExtractor.get_default_transforms(
        image_size=APP_STATE["sidecar"].get("image_size", 224)
    )
    APP_STATE["session"] = requests.Session()
    APP_STATE["session"].headers["User-Agent"] = (
        "mars-astrobio-viewer/0.1 (contact: ackermand@janelia.hhmi.org)"
    )


def _select_index_for_bbox(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> int:
    """Pick the available zoom whose tile side length best matches the bbox's
    long edge, in log-space. Favours genuine scale-to-scale matching over the
    old "always use index_zoom" behaviour.
    """
    long_edge_deg = max(lat_max - lat_min, lon_max - lon_min)
    if long_edge_deg <= 0:
        return APP_STATE["default_zoom"]
    target_log = math.log(long_edge_deg)
    best = APP_STATE["default_zoom"]
    best_delta = float("inf")
    for z in APP_STATE["available_zooms"]:
        tile_deg = TILE_PX * pixel_size_deg(z)
        delta = abs(math.log(tile_deg) - target_log)
        if delta < best_delta:
            best_delta = delta
            best = z
    return best


def _select_index_for_tile_at_zoom(z: int) -> int:
    """Pick the best available zoom for a single-tile query at zoom ``z``."""
    if z in APP_STATE["indices"]:
        return z
    return min(APP_STATE["available_zooms"], key=lambda zz: abs(zz - z))


def _index_is_aspect_corrected(zoom: int) -> bool:
    """Did the index at this zoom level get built with cos(lat) correction?

    Critical: applying correction at query time when the index wasn't built
    that way (or vice versa) puts query and index in different feature
    spaces and degrades results. We read the flag from each per-zoom
    sidecar.
    """
    state = APP_STATE["indices"].get(zoom)
    if state is None:
        return False
    return bool(state["sidecar"].get("aspect_corrected", False))


def _apply_aspect_correction(img: Image.Image, lat_center: float) -> Image.Image:
    cos_lat = max(0.05, math.cos(math.radians(abs(lat_center))))
    if cos_lat >= 0.999:
        return img
    w, h = img.size
    new_w = max(32, int(round(w * cos_lat)))
    return img.resize((new_w, h), Image.LANCZOS)


TILE_CACHE: dict[tuple[int, int, int], bytes] = {}
TILE_CACHE_MAX = 4096


def _fetch_tile_bytes(z: int, x: int, y: int, max_attempts: int = 6) -> bytes:
    """Fetch raw tile bytes, cached in-memory, with retries+backoff for the
    transient 429/500/503 storm we see while the patch indexer is running.

    max_attempts=6 (default) is the long-retry path for query-side embedding
    where we really need the tile. Cesium imagery hits use max_attempts=2
    so one missing tile doesn't stall LOD loading; Cesium will reschedule
    the failed tile on its own later anyway."""
    key = (z, x, y)
    cached = TILE_CACHE.get(key)
    if cached is not None:
        return cached
    url = TILE_URL.format(z=z, x=x, y=y)
    last_status = "no response"
    for attempt in range(max_attempts):
        try:
            r = APP_STATE["session"].get(url, timeout=20)
            if r.status_code == 200:
                if len(TILE_CACHE) >= TILE_CACHE_MAX:
                    for k in list(TILE_CACHE.keys())[: TILE_CACHE_MAX // 10]:
                        TILE_CACHE.pop(k, None)
                TILE_CACHE[key] = r.content
                return r.content
            last_status = r.status_code
        except requests.RequestException as e:
            last_status = f"exception:{type(e).__name__}"
        time.sleep(min(4.0, 0.4 * (2 ** attempt)))
    raise HTTPException(status_code=502, detail=f"Tile fetch failed after retries: {last_status}")


def _fetch_and_embed(z: int, x: int, y: int, aspect_correct: bool = False) -> np.ndarray:
    import faiss

    content = _fetch_tile_bytes(z, x, y)
    img = Image.open(io.BytesIO(content)).convert("RGB")
    if aspect_correct:
        img = _apply_aspect_correction(img, tile_center_deg(z, x, y)[0])
    tensor = APP_STATE["transform"](img).unsqueeze(0)
    vec = APP_STATE["extractor"].extract(tensor).astype("float32")
    faiss.normalize_L2(vec)
    return vec


def _tile_range_for_bbox(
    z: int, lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> tuple[int, int, int, int]:
    """Return (x_min, x_max, y_min, y_max) tile indices covering the bbox."""
    size = TILE_PX * pixel_size_deg(z)
    x_min = max(0, int((lon_min + 180.0) / size))
    x_max = min(2 * (2**z) - 1, int((lon_max + 180.0) / size))
    y_min = max(0, int((90.0 - lat_max) / size))  # lat_max is north; y small
    y_max = min(1 * (2**z) - 1, int((90.0 - lat_min) / size))
    return x_min, x_max, y_min, y_max


def _fetch_bbox_composite(
    z: int, lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> Image.Image:
    """Fetch all tiles intersecting the bbox, composite into one image, then
    crop to the exact bbox extent so DINO sees just the region the user drew.
    """
    from concurrent.futures import ThreadPoolExecutor

    x_min, x_max, y_min, y_max = _tile_range_for_bbox(z, lat_min, lat_max, lon_min, lon_max)
    n_x = x_max - x_min + 1
    n_y = y_max - y_min + 1
    if n_x <= 0 or n_y <= 0:
        raise HTTPException(status_code=400, detail="Empty bbox")

    session: requests.Session = APP_STATE["session"]
    canvas = Image.new("RGB", (n_x * TILE_PX, n_y * TILE_PX), (0, 0, 0))

    def fetch_one(x: int, y: int) -> tuple[int, int, Optional[Image.Image]]:
        try:
            content = _fetch_tile_bytes(z, x, y)
            return x, y, Image.open(io.BytesIO(content)).convert("RGB")
        except Exception:
            return x, y, None

    with ThreadPoolExecutor(max_workers=min(16, n_x * n_y)) as pool:
        futures = [
            pool.submit(fetch_one, x, y)
            for y in range(y_min, y_max + 1)
            for x in range(x_min, x_max + 1)
        ]
        for fut in futures:
            x, y, tile_img = fut.result()
            if tile_img is None:
                continue
            canvas.paste(tile_img, ((x - x_min) * TILE_PX, (y - y_min) * TILE_PX))

    # Convert bbox (deg) → pixel coords inside the composite.
    px = pixel_size_deg(z)
    composite_lon0 = -180.0 + x_min * TILE_PX * px  # left edge
    composite_lat0 = 90.0 - y_min * TILE_PX * px    # top edge
    crop_x0 = int(round((lon_min - composite_lon0) / px))
    crop_y0 = int(round((composite_lat0 - lat_max) / px))
    crop_x1 = int(round((lon_max - composite_lon0) / px))
    crop_y1 = int(round((composite_lat0 - lat_min) / px))
    crop_x0 = max(0, min(canvas.width, crop_x0))
    crop_y0 = max(0, min(canvas.height, crop_y0))
    crop_x1 = max(crop_x0 + 1, min(canvas.width, crop_x1))
    crop_y1 = max(crop_y0 + 1, min(canvas.height, crop_y1))
    return canvas.crop((crop_x0, crop_y0, crop_x1, crop_y1))


def _embed_pil(img: Image.Image) -> np.ndarray:
    import faiss

    if img.mode != "RGB":
        img = img.convert("RGB")
    tensor = APP_STATE["transform"](img).unsqueeze(0)
    vec = APP_STATE["extractor"].extract(tensor).astype("float32")
    faiss.normalize_L2(vec)
    return vec


def _patch_bounds_deg(
    tile_z: int, tile_x: int, tile_y: int, patch_row: int, patch_col: int, grid_n: int
) -> tuple[float, float, float, float]:
    """lat/lon bounds of the (patch_row, patch_col) cell of the `grid_n`x`grid_n`
    patch grid inside tile (z, x, y). Row 0 = top (highest lat)."""
    lat_min, lat_max, lon_min, lon_max = tile_bounds_deg(tile_z, tile_x, tile_y)
    dlat = (lat_max - lat_min) / grid_n
    dlon = (lon_max - lon_min) / grid_n
    p_lat_max = lat_max - patch_row * dlat
    p_lat_min = p_lat_max - dlat
    p_lon_min = lon_min + patch_col * dlon
    p_lon_max = p_lon_min + dlon
    return p_lat_min, p_lat_max, p_lon_min, p_lon_max


def _search_patches(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    fetch_z: int, search_zoom: int, top_k: int,
    spatial_diversity: bool = False, diversity_tiles: int = 4,
) -> list[dict]:
    """Sub-tile retrieval. Crops the user's bbox, runs DINO patch tokens,
    searches each query patch against the global patch index, merges hits
    by (tile_row_id, patch_idx) keeping the max score across query patches.

    Returns a list of patch-level results with tile_url, tile_bounds, and
    patch_bounds for drawing at patch resolution on the client."""
    import faiss

    patch_indices = APP_STATE.get("patch_indices", {})
    state = patch_indices.get(search_zoom)
    if state is None:
        # Fall back to the finest available patch zoom.
        if not patch_indices:
            raise HTTPException(status_code=503, detail="Patch index not available.")
        state = patch_indices[max(patch_indices)]
    tiles_df: pd.DataFrame = state["tiles_df"]
    patch_index = state["index"]
    grid_n = state["patch_grid_n"]
    P = state["num_patches_per_tile"]
    z_i = state["zoom"]
    aspect_correct = bool(state["sidecar"].get("aspect_corrected", False))

    region_img = _fetch_bbox_composite(fetch_z, lat_min, lat_max, lon_min, lon_max)
    if aspect_correct:
        center_lat = 0.5 * (lat_min + lat_max)
        cos_lat = max(0.05, math.cos(math.radians(abs(center_lat))))
        if cos_lat < 0.999:
            new_w = max(32, int(round(region_img.width * cos_lat)))
            region_img = region_img.resize((new_w, region_img.height), Image.LANCZOS)

    transform = APP_STATE["transform"]
    extractor = APP_STATE["extractor"]
    tensor = transform(region_img).unsqueeze(0)
    qpatches = extractor.extract_patches(tensor).astype("float32")  # (1, Q, D)
    Q = int(qpatches.shape[1])
    D = int(qpatches.shape[2])
    qvec = qpatches.reshape(Q, D)
    faiss.normalize_L2(qvec)

    # Search all query patches at once. Faiss returns (Q, per_query_k).
    per_query_k = max(8, top_k // max(1, Q) * 4)
    fetch_k = per_query_k * (8 if spatial_diversity else 1)
    distances, indices = patch_index.search(qvec, fetch_k)

    # Merge: key by (tile_row_id, patch_idx) → best similarity across query patches
    best: dict[tuple[int, int], float] = {}
    for qi in range(Q):
        for score, idx in zip(distances[qi], indices[qi]):
            if idx < 0:
                continue
            tile_row_id = int(idx // P)
            patch_idx = int(idx % P)
            key = (tile_row_id, patch_idx)
            prev = best.get(key)
            if prev is None or score > prev:
                best[key] = float(score)

    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    out: list[dict] = []
    seen_tile_and_patch: list[tuple[int, int, int, int, int]] = []  # (z, x, y, pr, pc)

    for (tile_row_id, patch_idx), score in ranked:
        if tile_row_id >= len(tiles_df):
            continue
        row = tiles_df.iloc[tile_row_id]
        t_z, t_x, t_y = int(row.z), int(row.x), int(row.y)
        pr, pc = patch_idx // grid_n, patch_idx % grid_n
        if spatial_diversity:
            too_close = False
            for (sz, sx, sy, spr, spc) in seen_tile_and_patch:
                if sz != t_z:
                    continue
                dx = (sx - t_x) * grid_n + (spc - pc)
                dy = (sy - t_y) * grid_n + (spr - pr)
                if max(abs(dx), abs(dy)) < diversity_tiles * grid_n:
                    too_close = True
                    break
            if too_close:
                continue
            seen_tile_and_patch.append((t_z, t_x, t_y, pr, pc))
        p_lat_min, p_lat_max, p_lon_min, p_lon_max = _patch_bounds_deg(
            t_z, t_x, t_y, pr, pc, grid_n
        )
        center_lat = 0.5 * (p_lat_min + p_lat_max)
        center_lon = 0.5 * (p_lon_min + p_lon_max)
        out.append({
            "z": t_z, "x": t_x, "y": t_y,
            "patch_row": pr, "patch_col": pc,
            "lat": center_lat, "lon": center_lon,
            "similarity": score,
            "tile_url": f"/api/tile_img?z={t_z}&x={t_x}&y={t_y}",
            "tile_bounds": list(tile_bounds_deg(t_z, t_x, t_y)),
            "patch_bounds": [p_lat_min, p_lat_max, p_lon_min, p_lon_max],
        })
        if len(out) >= top_k:
            break
    return out


def _nms_spatial(results: list[dict], top_k: int, min_separation_tiles: int) -> list[dict]:
    """Greedy non-maximum suppression by tile-index distance. Results are
    expected to be sorted by similarity desc. `min_separation_tiles` is the
    minimum Chebyshev distance (in tile units at each result's zoom) between
    two kept hits — raising it spreads results geographically."""
    kept: list[dict] = []
    for r in results:
        too_close = False
        for k in kept:
            if k["z"] != r["z"]:
                continue
            if max(abs(k["x"] - r["x"]), abs(k["y"] - r["y"])) < min_separation_tiles:
                too_close = True
                break
        if not too_close:
            kept.append(r)
            if len(kept) >= top_k:
                break
    return kept


def _search(
    vec: np.ndarray,
    top_k: int,
    zoom: Optional[int] = None,
    spatial_diversity: bool = False,
    diversity_tiles: int = 4,
) -> list[dict]:
    state = APP_STATE["indices"][zoom] if zoom in APP_STATE["indices"] else None
    if state is None:
        z = APP_STATE["default_zoom"]
        state = APP_STATE["indices"][z]
    # Oversample when diversity is on so NMS has room to suppress clusters.
    fetch_k = top_k * 8 if spatial_diversity else top_k
    distances, indices = state["index"].search(vec, fetch_k)
    md: pd.DataFrame = state["metadata"]
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        row = md.iloc[idx]
        z_i, x_i, y_i = int(row.z), int(row.x), int(row.y)
        results.append(
            {
                "z": z_i,
                "x": x_i,
                "y": y_i,
                "lat": float(row.lat),
                "lon": float(row.lon),
                "similarity": float(dist),
                "tile_url": f"/api/tile_img?z={z_i}&x={x_i}&y={y_i}",
                "tile_bounds": list(tile_bounds_deg(z_i, x_i, y_i)),
            }
        )
    if spatial_diversity:
        results = _nms_spatial(results, top_k, diversity_tiles)
    else:
        results = results[:top_k]
    return results


app = FastAPI(title="Murray Lab CTX Similarity Viewer")


class LatLonQuery(BaseModel):
    lat: float
    lon: float
    zoom: Optional[int] = None
    top_k: int = 20
    spatial_diversity: bool = False
    diversity_tiles: int = 4


class BboxQuery(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    zoom: Optional[int] = None  # at which zoom to fetch source pixels
    top_k: int = 20
    mode: str = "composite"  # "composite" | "aggregate" | "multi"
    localize: bool = False  # find best sub-region within each result tile
    spatial_diversity: bool = False
    diversity_tiles: int = 4


@app.post("/api/query_bbox")
def api_query_bbox(q: BboxQuery) -> JSONResponse:
    """Query by an arbitrary geographic region (smaller, equal, or larger than
    one tile). Fetches all tiles intersecting the bbox at the requested fetch
    zoom, composites into one image in memory, crops to the exact bbox, runs
    DINO, and searches the pre-built index.

    If the caller doesn't specify `zoom`, we pick the finest fetch zoom that
    uses at most 9 tiles to keep network cost bounded.
    """
    # Scale-aware: pick the indexed zoom whose tile size best matches the
    # user's bbox long edge. That way we search tiles at a physical scale
    # that's comparable to what the user drew, not always at the coarsest
    # index.
    search_zoom = _select_index_for_bbox(q.lat_min, q.lat_max, q.lon_min, q.lon_max)
    # Fetch source pixels at the same zoom (so the query embedding matches
    # the index's feature scale), but cap by bbox spanning ≤9 tiles.
    if q.zoom is not None:
        fetch_z = q.zoom
    else:
        fetch_z = search_zoom
        for candidate in range(search_zoom + 1, 13):
            x0, x1, y0, y1 = _tile_range_for_bbox(
                candidate, q.lat_min, q.lat_max, q.lon_min, q.lon_max
            )
            n_tiles = (x1 - x0 + 1) * (y1 - y0 + 1)
            if n_tiles > 9:
                break
            fetch_z = candidate

    x0, x1, y0, y1 = _tile_range_for_bbox(
        fetch_z, q.lat_min, q.lat_max, q.lon_min, q.lon_max
    )
    n_tiles = (x1 - x0 + 1) * (y1 - y0 + 1)
    if n_tiles > 64:
        raise HTTPException(
            status_code=413,
            detail=f"Region spans {n_tiles} tiles at zoom {fetch_z}; pick a smaller box or lower zoom.",
        )

    mode = (q.mode or "composite").lower()
    aspect_correct_query = _index_is_aspect_corrected(search_zoom)
    import base64, faiss

    if mode == "composite":
        # One forward pass over the entire composited region → one query
        # vector. Matches "this whole region's overall signature."
        region_img = _fetch_bbox_composite(
            fetch_z, q.lat_min, q.lat_max, q.lon_min, q.lon_max
        )
        if aspect_correct_query:
            center_lat = 0.5 * (q.lat_min + q.lat_max)
            cos_lat = max(0.05, math.cos(math.radians(abs(center_lat))))
            if cos_lat < 0.999:
                new_w = max(32, int(round(region_img.width * cos_lat)))
                region_img = region_img.resize(
                    (new_w, region_img.height), Image.LANCZOS
                )
        vec = _embed_pil(region_img)
        results = _search(
            vec, q.top_k, zoom=search_zoom,
            spatial_diversity=q.spatial_diversity,
            diversity_tiles=q.diversity_tiles,
        )
        preview_img = region_img.copy()
    elif mode == "patch":
        # True sub-tile retrieval against the pre-built global patch index.
        # Every patch of every tile is a candidate.
        results = _search_patches(
            q.lat_min, q.lat_max, q.lon_min, q.lon_max,
            fetch_z, search_zoom, q.top_k,
            spatial_diversity=q.spatial_diversity,
            diversity_tiles=q.diversity_tiles,
        )
        preview_img = _fetch_bbox_composite(
            fetch_z, q.lat_min, q.lat_max, q.lon_min, q.lon_max
        )
    else:
        # Fetch each intersecting tile at the SEARCH zoom (not the fetch
        # zoom). Each constituent tile is embedded on its own, so we see
        # multiple sub-region vectors instead of one averaged signature.
        sx0, sx1, sy0, sy1 = _tile_range_for_bbox(
            search_zoom, q.lat_min, q.lat_max, q.lon_min, q.lon_max
        )
        tiles_xy = [
            (x, y) for y in range(sy0, sy1 + 1) for x in range(sx0, sx1 + 1)
        ]
        if len(tiles_xy) > 64:
            raise HTTPException(
                status_code=413,
                detail=f"Region spans {len(tiles_xy)} tiles at search zoom "
                f"{search_zoom}; pick a smaller bbox.",
            )
        vecs = []
        for tx, ty in tiles_xy:
            try:
                v = _fetch_and_embed(
                    search_zoom, tx, ty, aspect_correct=aspect_correct_query
                )
                vecs.append(v[0])
            except HTTPException:
                continue
        if not vecs:
            raise HTTPException(
                status_code=502, detail="No constituent tiles fetched."
            )
        vecs_np = np.stack(vecs, axis=0).astype("float32")

        if mode == "aggregate":
            # Mean-pool the constituent tile vectors, re-normalize, one
            # search. "Characteristic average of what's in this region."
            mean_vec = vecs_np.mean(axis=0, keepdims=True)
            faiss.normalize_L2(mean_vec)
            results = _search(
                mean_vec, q.top_k, zoom=search_zoom,
                spatial_diversity=q.spatial_diversity,
                diversity_tiles=q.diversity_tiles,
            )
        elif mode == "multi":
            # Run one search per constituent, merge by best similarity per
            # result row_id. "Anywhere that looks like ANY piece of this
            # region." More recall-y, less precise.
            all_hits: dict[tuple, dict] = {}
            for v in vecs_np:
                hits = _search(
                    v.reshape(1, -1), q.top_k, zoom=search_zoom,
                )
                for h in hits:
                    key = (h["z"], h["x"], h["y"])
                    if key not in all_hits or h["similarity"] > all_hits[key]["similarity"]:
                        all_hits[key] = h
            merged = sorted(
                all_hits.values(), key=lambda h: h["similarity"], reverse=True
            )
            if q.spatial_diversity:
                results = _nms_spatial(merged, q.top_k, q.diversity_tiles)
            else:
                results = merged[: q.top_k]
        else:
            raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}")

        # Build a preview image showing the constituent tiles stitched.
        preview_img = _fetch_bbox_composite(
            fetch_z, q.lat_min, q.lat_max, q.lon_min, q.lon_max
        )

    # Optional: localize where within each result tile the query best matches.
    # Uses DINO patch tokens: compute the mean query-patch vector, score
    # every patch position in each result tile, pick the max. Adds ~3-10 s
    # to the query (one forward pass per result tile).
    if q.localize and results and mode != "patch":
        try:
            # Build query's per-patch embeddings
            if mode == "composite":
                q_img_for_patches = region_img
            else:
                # Re-composite for patch extraction when we didn't keep one
                q_img_for_patches = _fetch_bbox_composite(
                    fetch_z, q.lat_min, q.lat_max, q.lon_min, q.lon_max
                )
                if aspect_correct_query:
                    cl = 0.5 * (q.lat_min + q.lat_max)
                    cos_ = max(0.05, math.cos(math.radians(abs(cl))))
                    if cos_ < 0.999:
                        q_img_for_patches = q_img_for_patches.resize(
                            (max(32, int(round(q_img_for_patches.width * cos_))),
                             q_img_for_patches.height),
                            Image.LANCZOS,
                        )
            q_tensor = APP_STATE["transform"](q_img_for_patches).unsqueeze(0)
            q_patches = APP_STATE["extractor"].extract_patches(q_tensor)[0]  # (P, D)
            q_patches /= np.clip(np.linalg.norm(q_patches, axis=1, keepdims=True), 1e-9, None)
            q_mean = q_patches.mean(axis=0)
            q_mean /= max(1e-9, float(np.linalg.norm(q_mean)))

            for r in results:
                try:
                    url = TILE_URL.format(z=r["z"], x=r["x"], y=r["y"])
                    resp = APP_STATE["session"].get(url, timeout=15)
                    if resp.status_code != 200:
                        continue
                    tile_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    if aspect_correct_query:
                        tile_img = _apply_aspect_correction(
                            tile_img, tile_center_deg(r["z"], r["x"], r["y"])[0]
                        )
                    t_tensor = APP_STATE["transform"](tile_img).unsqueeze(0)
                    t_patches = APP_STATE["extractor"].extract_patches(t_tensor)[0]  # (P, D)
                    t_patches /= np.clip(
                        np.linalg.norm(t_patches, axis=1, keepdims=True), 1e-9, None
                    )
                    sims = t_patches @ q_mean  # (P,)
                    grid_n = int(round(math.sqrt(t_patches.shape[0])))
                    best_idx = int(np.argmax(sims))
                    row = best_idx // grid_n
                    col = best_idx % grid_n
                    # In the displayed thumbnail coord system (512 px ref),
                    # each patch is 512/grid_n px. Client scales these to
                    # however large it renders the thumb.
                    cell = 512.0 / grid_n
                    r["best_patch_bbox"] = [
                        int(col * cell),
                        int(row * cell),
                        int((col + 1) * cell),
                        int((row + 1) * cell),
                    ]
                    r["best_patch_sim"] = float(sims[best_idx])
                except Exception as exc:
                    logger.warning("localize failed for %s: %s", r, exc)
        except Exception as exc:
            logger.warning("localize path failed: %s", exc)

    preview_img.thumbnail((512, 512), Image.LANCZOS)
    buf = io.BytesIO()
    preview_img.save(buf, format="JPEG", quality=85)
    preview_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return JSONResponse(
        {
            "query": {
                "bbox": {
                    "lat_min": q.lat_min, "lat_max": q.lat_max,
                    "lon_min": q.lon_min, "lon_max": q.lon_max,
                },
                "fetch_z": fetch_z,
                "search_zoom": search_zoom,
                "available_zooms": APP_STATE["available_zooms"],
                "n_tiles": n_tiles,
                "mode": mode,
                "composite_w": preview_img.width,
                "composite_h": preview_img.height,
                "preview_b64": preview_b64,
                "aspect_corrected": aspect_correct_query,
            },
            "results": results,
        }
    )


@app.post("/api/query_latlon")
def api_query_latlon(q: LatLonQuery) -> JSONResponse:
    # Pick the best-matching zoom we have indexed. If the user specifies one,
    # snap to the nearest available; otherwise default to the finest zoom.
    if q.zoom is not None:
        index_zoom = _select_index_for_tile_at_zoom(q.zoom)
    else:
        index_zoom = APP_STATE["default_zoom"]
    # Fetch source pixels at the indexed zoom so the query vector lives in
    # the same feature space as the searched tiles.
    x, y = latlon_to_tile(q.lat, q.lon, index_zoom)
    aspect_correct = _index_is_aspect_corrected(index_zoom)
    vec = _fetch_and_embed(index_zoom, x, y, aspect_correct=aspect_correct)
    results = _search(
        vec, q.top_k, zoom=index_zoom,
        spatial_diversity=q.spatial_diversity,
        diversity_tiles=q.diversity_tiles,
    )
    return JSONResponse(
        {
            "query": {
                "lat": q.lat, "lon": q.lon, "z": index_zoom, "x": x, "y": y,
                "tile_url": f"/api/tile_img?z={index_zoom}&x={x}&y={y}",
                "tile_bounds": list(tile_bounds_deg(index_zoom, x, y)),
                "index_zoom_used": index_zoom,
                "available_zooms": APP_STATE["available_zooms"],
            },
            "results": results,
        }
    )


@app.get("/api/anomalies")
def api_anomalies(k: int = 20) -> JSONResponse:
    """Lazy-compute (and cache) anomaly scores for the current default zoom.

    Uses FAISS IVF-PQ nearest-neighbor distance as the outlier signal: for
    each indexed vector, search the index and take the distance to its k-th
    nearest neighbour. Vectors far from their neighbours are outliers. This
    scales linearly in n queries and each query is O(nprobe * list_size),
    dramatically faster than sklearn LOF on 1M+ vectors (LOF on 1.77M × 1024d
    takes 30-60 min; FAISS-NN-distance takes 3-5 min).
    """
    index_dir: Path = APP_STATE["index_dir"]
    scores_path = index_dir / "anomaly_scores.parquet"
    if not scores_path.exists():
        import faiss, time as _time

        faiss_index = APP_STATE["index"]
        ntotal = faiss_index.ntotal
        dim = faiss_index.d
        logger.info(
            "Computing FAISS-NN anomaly scores over %d vectors (k=40 neighbours)...",
            ntotal,
        )
        # Read vectors back from the index (IVF-PQ reconstruction) in chunks to
        # avoid blowing RAM on very large indices.
        K = 40  # neighbours
        chunk = 20_000
        all_scores = np.empty(ntotal, dtype=np.float32)
        t0 = _time.time()
        for start in range(0, ntotal, chunk):
            end = min(start + chunk, ntotal)
            vecs = np.empty((end - start, dim), dtype=np.float32)
            faiss_index.reconstruct_n(start, end - start, vecs)
            faiss.normalize_L2(vecs)
            # K+1 because the closest neighbour to a vector is itself
            dists, _ = faiss_index.search(vecs, K + 1)
            # Inner-product similarity → convert to distance (1 - sim), then
            # take the median of the K real neighbours (exclude self at idx 0).
            sims = dists[:, 1:]
            neighbour_dist = 1.0 - sims.mean(axis=1)
            all_scores[start:end] = neighbour_dist.astype(np.float32)
            if (start // chunk) % 5 == 0:
                logger.info("  %d/%d (%.0fs)", end, ntotal, _time.time() - t0)
        md = APP_STATE["metadata"].copy()
        md["anomaly_score"] = all_scores
        md.to_parquet(scores_path, index=False)
        logger.info("Wrote %s in %.0fs", scores_path, _time.time() - t0)
    df = pd.read_parquet(scores_path)
    s = df["anomaly_score"]
    # Clip to sane cosine-distance range only — IVF-PQ reconstruction of
    # near-zero-norm tiles can produce inf/huge numeric garbage. We do NOT
    # filter by latitude: polar results stay in. The correct fix for the
    # plate-carrée squash at high latitudes is the cos(lat) aspect
    # correction applied at embed time (see stream_murray_index.fetch_tile).
    # Indexes built without aspect correction (sidecar
    # aspect_corrected=False) will surface polar tiles prominently in
    # anomaly lists for that reason; rebuilding them with correction is the
    # proper remedy.
    df = df[np.isfinite(s) & (s >= 0.0) & (s <= 2.0)]
    df = df.sort_values("anomaly_score", ascending=False).head(k)
    payload = []
    for _, row in df.iterrows():
        z_i, x_i, y_i = int(row.z), int(row.x), int(row.y)
        payload.append(
            {
                "z": z_i, "x": x_i, "y": y_i,
                "lat": float(row.lat), "lon": float(row.lon),
                "anomaly_score": float(row.anomaly_score),
                "tile_url": f"/api/tile_img?z={z_i}&x={x_i}&y={y_i}",
                "tile_bounds": list(tile_bounds_deg(z_i, x_i, y_i)),
            }
        )
    return JSONResponse({"anomalies": payload})


@app.get("/api/tile_img")
def api_tile_img(z: int, x: int, y: int) -> Response:
    """Proxy Murray Lab tiles through the server so clients inherit our
    retry/backoff policy and don't hit ArcGIS directly while the patch
    indexer is saturating the upstream rate limit. Fast path: 2 attempts
    only, so one bad tile doesn't stall Cesium LOD loading (Cesium will
    reschedule a failed imagery tile on its own)."""
    content = _fetch_tile_bytes(z, x, y, max_attempts=2)
    return Response(
        content=content,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return HTML


@app.get("/globe", response_class=HTMLResponse)
def globe() -> HTMLResponse:
    """3D Mars globe viewer (CesiumJS + Murray Lab imagery on Mars ellipsoid)."""
    return HTMLResponse(
        content=GLOBE_HTML,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


HTML = r"""<!doctype html>
<html><head>
<meta charset="utf-8"/>
<title>Murray Lab CTX Similarity</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{height:100%;margin:0;font-family:system-ui,sans-serif}
  #layout{display:grid;grid-template-columns:2fr 1fr;height:100vh}
  #map{height:100%}
  #side{overflow-y:auto;padding:12px;background:#111;color:#eee;border-left:1px solid #333}
  h2{margin:6px 0;font-size:1.05em}
  .muted{color:#888;font-size:.85em}
  button{background:#244;color:#eee;border:1px solid #466;padding:6px 10px;cursor:pointer;margin-right:6px}
  .result{display:flex;gap:8px;padding:6px 4px;border-bottom:1px solid #222;cursor:pointer}
  .result:hover{background:#1c1c1c}
  .result img{width:80px;height:80px;object-fit:cover;border:1px solid #333}
  .result .meta{font-size:.85em;line-height:1.3}
</style>
</head><body>
<div id="layout">
  <div id="map"></div>
  <div id="side">
    <h2>Murray Lab CTX similarity</h2>
    <p class="muted">Click anywhere on Mars → the tile there is embedded and the top-k most similar tiles globally are pinned. Background = full Murray Lab 5 m mosaic.</p>
    <div style="margin-bottom:10px">
      <button id="surprise">🔭 Surprise me (top anomalies)</button>
      <button id="mode-toggle">🔲 Region select: OFF</button>
    </div>
    <div id="region-mode-panel" style="margin-bottom:10px;display:none">
      <label class="muted">Region-query mode:
        <select id="region-query-mode">
          <option value="composite" selected>composite — one vector from stitched region (match overall signature)</option>
          <option value="aggregate">aggregate — average constituent tile vectors (characteristic average)</option>
          <option value="multi">multi — search each constituent, merge (matches ANY piece of region)</option>
        </select>
      </label>
      <label class="muted" style="display:block;margin-top:4px">
        <input type="checkbox" id="region-localize"/>
        Localize WITHIN each result tile (highlights best-matching sub-region; +3-10 s)
      </label>
    </div>
    <p class="muted" id="mode-hint">Click a spot on Mars = query that tile. Toggle region select, then drag a rectangle to query an arbitrary region (smaller or larger than a tile).</p>
    <div id="query"></div>
    <h2>Results</h2>
    <div id="results"><p class="muted">Click anywhere on Mars.</p></div>
  </div>
</div>

<script>
const TILE_URL = "https://astro.arcgis.com/arcgis/rest/services/OnMars/CTX1/MapServer/tile/{z}/{y}/{x}";
const AVAILABLE_ZOOMS = %(AVAILABLE_ZOOMS)s;

const map = L.map("map", {
  crs: L.CRS.EPSG4326,
  minZoom: 1,
  maxZoom: 12,
  worldCopyJump: false,
}).setView([0, 0], 2);

L.tileLayer(TILE_URL, {
  tileSize: 512,
  zoomOffset: -1,  // Murray Lab level 0 has 2×1 tiles; Leaflet EPSG:4326 level 0 has 2×1 as well, so offset -1 aligns.
  attribution: "NASA/JPL/MSSS/The Murray Lab (Dickson et al. 2024)",
  noWrap: true,
}).addTo(map);

let resultMarkers = [];
let queryMarker = null;
let queryRect = null;
let regionMode = false;
let dragging = null;

const modeBtn = document.getElementById("mode-toggle");
const regionModePanel = document.getElementById("region-mode-panel");
modeBtn.onclick = () => {
  regionMode = !regionMode;
  modeBtn.innerText = regionMode ? "🔲 Region select: ON" : "🔲 Region select: OFF";
  regionModePanel.style.display = regionMode ? "block" : "none";
  if (regionMode) map.dragging.disable();
  else map.dragging.enable();
};

map.on("click", async (e) => {
  if (regionMode) return;  // region mode uses mousedown/mouseup, not click
  const { lat, lng } = e.latlng;
  document.getElementById("query").innerHTML =
    `<p>Querying lat ${lat.toFixed(2)}, lon ${lng.toFixed(2)} (available zooms: ${AVAILABLE_ZOOMS.join(", ")})...</p>`;
  if (queryMarker) map.removeLayer(queryMarker);
  if (queryRect) { map.removeLayer(queryRect); queryRect = null; }
  queryMarker = L.circleMarker([lat, lng], {radius:10,color:"#7df",weight:2,fillColor:"#7df",fillOpacity:0.5}).addTo(map);

  // Match client's current zoom to one of our index zooms.
  const leafletZoom = map.getZoom();
  const requestedZoom = AVAILABLE_ZOOMS.reduce(
    (best, z) => (Math.abs(z - leafletZoom) < Math.abs(best - leafletZoom) ? z : best),
    AVAILABLE_ZOOMS[0],
  );

  const resp = await fetch("/api/query_latlon", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({lat, lon: lng, zoom: requestedZoom, top_k: 20}),
  });
  const data = await resp.json();
  renderQuery(data.query);
  renderResults(data.results || [], "similarity");
});

map.on("mousedown", (e) => {
  if (!regionMode) return;
  dragging = { start: e.latlng, end: e.latlng };
  if (queryRect) { map.removeLayer(queryRect); queryRect = null; }
  queryRect = L.rectangle(L.latLngBounds(e.latlng, e.latlng), {
    color: "#ffe600", weight: 2, fillOpacity: 0.15,
  }).addTo(map);
});
map.on("mousemove", (e) => {
  if (!regionMode || !dragging || !queryRect) return;
  dragging.end = e.latlng;
  queryRect.setBounds(L.latLngBounds(dragging.start, dragging.end));
});
map.on("mouseup", async (e) => {
  if (!regionMode || !dragging) return;
  const start = dragging.start, end = dragging.end;
  dragging = null;
  const lat_min = Math.min(start.lat, end.lat);
  const lat_max = Math.max(start.lat, end.lat);
  const lon_min = Math.min(start.lng, end.lng);
  const lon_max = Math.max(start.lng, end.lng);
  if (lat_max - lat_min < 1e-4 || lon_max - lon_min < 1e-4) return;  // ignore tiny drags
  document.getElementById("query").innerHTML =
    `<p>Querying region (${lat_min.toFixed(2)},${lon_min.toFixed(2)})–(${lat_max.toFixed(2)},${lon_max.toFixed(2)})...</p>`;
  const queryMode = document.getElementById("region-query-mode").value;
  const localize = document.getElementById("region-localize").checked;
  const resp = await fetch("/api/query_bbox", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({lat_min, lat_max, lon_min, lon_max, top_k: 20, mode: queryMode, localize}),
  });
  const data = await resp.json();
  renderBboxQuery(data.query);
  renderResults(data.results || [], "similarity");
});

function renderBboxQuery(q) {
  document.getElementById("query").innerHTML = `
    <h3>Query region (${q.mode})</h3>
    <img src="data:image/jpeg;base64,${q.preview_b64}" style="width:100%;max-width:400px;border:1px solid #333"/>
    <p class="muted">${q.composite_w}×${q.composite_h} px previewed from ${q.n_tiles} tile(s) fetched at z=${q.fetch_z}</p>
    <p class="muted">Searched at z=${q.search_zoom} (available: ${q.available_zooms.join(", ")}) • aspect-corrected: ${q.aspect_corrected}</p>
    <p class="muted">bbox: lat ${q.bbox.lat_min.toFixed(2)}..${q.bbox.lat_max.toFixed(2)}, lon ${q.bbox.lon_min.toFixed(2)}..${q.bbox.lon_max.toFixed(2)}</p>
  `;
}

function renderQuery(q) {
  document.getElementById("query").innerHTML = `
    <h3>Query tile</h3>
    <img src="${q.tile_url}" style="width:100%;max-width:300px;border:1px solid #333"/>
    <p class="muted">z=${q.z} • tile (${q.x},${q.y}) • lat ${q.lat.toFixed(2)} lon ${q.lon.toFixed(2)}</p>
  `;
}

function renderResults(results, scoreCol) {
  resultMarkers.forEach(m => map.removeLayer(m));
  resultMarkers = [];
  const el = document.getElementById("results");
  if (!results || !results.length) { el.innerHTML = "<p class='muted'>No hits.</p>"; return; }
  el.innerHTML = "";
  results.forEach((r, idx) => {
    const row = document.createElement("div");
    row.className = "result";
    const score = r[scoreCol] ?? r.similarity ?? r.anomaly_score;
    const bbox = r.best_patch_bbox;
    const hl = bbox
      ? `<div style="position:absolute;left:${bbox[0]/512*100}%;top:${bbox[1]/512*100}%;width:${(bbox[2]-bbox[0])/512*100}%;height:${(bbox[3]-bbox[1])/512*100}%;border:2px solid #ffe600;box-shadow:0 0 6px rgba(255,230,0,.6);pointer-events:none"></div>`
      : "";
    const matchNote = r.best_patch_sim
      ? `<div class="muted">sub-region sim: ${r.best_patch_sim.toFixed(3)}</div>`
      : "";
    row.innerHTML = `
      <div style="position:relative;width:80px;height:80px;flex:none">
        <img src="${r.tile_url}" loading="lazy" style="width:100%;height:100%;object-fit:cover;border:1px solid #333"/>
        ${hl}
      </div>
      <div class="meta">
        <div><b>#${idx + 1}</b></div>
        <div>${scoreCol}: ${score?.toFixed(3)}</div>
        ${matchNote}
        <div>lat ${r.lat.toFixed(2)} • lon ${r.lon.toFixed(2)}</div>
        <div class="muted">z=${r.z} • (${r.x},${r.y})</div>
      </div>`;
    row.onclick = () => map.setView([r.lat, r.lon], Math.min(map.getMaxZoom(), 6));
    el.appendChild(row);
    const marker = L.circleMarker([r.lat, r.lon], {
      radius: 7,
      color: scoreCol === "anomaly_score" ? "#ffd700" : "#ff6b9a",
      weight: 2,
      fillOpacity: 0.5,
    }).addTo(map);
    marker.bindTooltip(`#${idx + 1} ${scoreCol}: ${score?.toFixed(2)}`);
    marker.on("click", () => map.setView([r.lat, r.lon], Math.min(map.getMaxZoom(), 6)));
    resultMarkers.push(marker);
  });
}

document.getElementById("surprise").onclick = async () => {
  const btn = document.getElementById("surprise");
  btn.disabled = true; btn.innerText = "Scoring (first time: ~1-2 min)...";
  try {
    const resp = await fetch("/api/anomalies?k=20");
    const data = await resp.json();
    renderResults(data.anomalies || [], "anomaly_score");
  } finally {
    btn.disabled = false; btn.innerText = "🔭 Surprise me (top anomalies)";
  }
};
</script>
</body></html>
"""


GLOBE_HTML = r"""<!doctype html>
<html><head>
<meta charset="utf-8"/>
<title>Mars CTX 3D globe</title>
<link href="https://cesium.com/downloads/cesiumjs/releases/1.119/Build/Cesium/Widgets/widgets.css" rel="stylesheet"/>
<script src="https://cesium.com/downloads/cesiumjs/releases/1.119/Build/Cesium/Cesium.js"></script>
<style>
  html, body { height: 100%; margin: 0; padding: 0; overflow: hidden; }
  body { font-family: system-ui, sans-serif; background: #000; color: #eee; }
  #layout {
    display: grid;
    grid-template-columns: 2fr 1fr;
    grid-template-rows: 100vh;
    height: 100vh;
    width: 100vw;
    overflow: hidden;
  }
  #cesiumContainer { width: 100%; height: 100vh; min-height: 0; overflow: hidden; position: relative; }
  #side { overflow-y: auto; min-height: 0; padding: 12px; background: #111; border-left: 1px solid #333; }
  h2 { margin: 6px 0; font-size: 1.05em; }
  .muted { color: #888; font-size: .85em; }
  .result { display: flex; gap: 8px; padding: 6px 4px; border-bottom: 1px solid #222; cursor: pointer; }
  .result:hover { background: #1c1c1c; }
  .result img { width: 80px; height: 80px; object-fit: cover; border: 1px solid #333; }
  .result .meta { font-size: .85em; line-height: 1.3; }
  button { background: #244; color: #eee; border: 1px solid #466; padding: 6px 10px; cursor: pointer; }
  a { color: #6fb7ff; }
</style>
</head><body>
<div id="layout">
  <div id="cesiumContainer"></div>
  <div id="side">
    <h2>3D Mars globe</h2>
    <p class="muted"><b>Click</b> = query that tile. <b>Draw region</b> (button below) → one drag → rectangle staged for querying. <a href="#" id="mode-help-link">[what do the modes mean?]</a></p>
    <div style="margin-bottom:8px">
      <button id="draw-region-btn">🔲 Draw region</button>
    </div>
    <div id="mode-help" style="display:none;background:#1a1a1a;border:1px solid #333;padding:8px;margin-bottom:8px;font-size:.85em;line-height:1.35">
      <p><b>patch</b> — true sub-tile retrieval. Your region is cropped, split into DINO patches (~700 m each at z10), and every query patch is searched against a pre-built global patch index of every patch of every tile. Results are patch-level rectangles, not full tiles. This is the most accurate mode for finding specific small features.</p>
      <p><b>composite</b> — all tiles inside your rectangle are stitched into one image; DINO gives one query vector. Matches the <i>overall signature</i> of the scene at tile-resolution results.</p>
      <p><b>aggregate</b> — each constituent tile's own CLS vector is averaged; one search. Matches the <i>average look</i> of the constituents.</p>
      <p><b>multi</b> — one search per constituent tile, results merged by best score. Matches if <i>any piece</i> of your region resembles the hit.</p>
      <p><b>Spatial diversity (NMS)</b> — after ranking, suppress hits that share a neighbourhood (within N tiles of another kept hit) so you don't get 20 results all clustered in the same crater field.</p>
      <p><b>Localize within each result</b> — for every top-k hit, run patch-level matching to find the best sub-region inside that tile and draw a yellow highlight box. Adds 3-10 s.</p>
    </div>
    <fieldset style="border:1px solid #333;padding:8px;margin:8px 0">
      <legend class="muted">Query options</legend>
      <label class="muted" style="display:block;margin-bottom:4px">Region mode:
        <select id="region-query-mode" style="width:100%">
          <option value="patch" selected>patch — sub-tile matches globally (recommended)</option>
          <option value="composite">composite — one vector from stitched region</option>
          <option value="aggregate">aggregate — average of tile vectors</option>
          <option value="multi">multi — search each tile, merge</option>
        </select>
      </label>
      <label class="muted" style="display:block;margin-top:4px">
        <input type="checkbox" id="spatial-diversity" checked/>
        Spatial diversity (spread hits apart;
        <input type="number" id="diversity-tiles" value="4" min="1" max="32" style="width:3em"/>
        tile buffer)
      </label>
      <label class="muted" style="display:block">
        <input type="checkbox" id="region-localize"/>
        Localize within each result tile (+3-10 s)
      </label>
    </fieldset>
    <p class="muted"><a href="/">← back to 2D viewer</a></p>
    <div id="query"></div>
    <div id="result-preview" style="margin-top:6px"></div>
    <h2>Results <span class="muted" style="font-weight:normal">(hover = highlight on globe · click = fly + preview)</span></h2>
    <div id="results"><p class="muted">Click anywhere on Mars.</p></div>
  </div>
</div>
<script>
// Mars ellipsoid (MOLA IAU2000): equatorial 3396190 m, polar 3376200 m.
const MARS_EQ = 3396190.0;
const MARS_POLAR = 3376200.0;
const marsEllipsoid = new Cesium.Ellipsoid(MARS_EQ, MARS_EQ, MARS_POLAR);
// Override Cesium's default ellipsoid BEFORE constructing the viewer, so
// Cartesian3.fromDegrees, the globe tessellator, and the camera controller
// all treat lat/lon on Mars instead of Earth. Cesium.Ellipsoid.WGS84 is a
// frozen constant; Cesium.Ellipsoid.default (1.117+) is the writable knob.
Cesium.Ellipsoid.default = marsEllipsoid;
Cesium.Ion.defaultAccessToken = "";
// Route Cesium imagery through our own proxy so requests inherit the
// server-side retry/backoff policy and — crucially — come from the same
// origin as the page, avoiding the intermittent missing-CORS-header
// failures we see hitting astro.arcgis.com directly under load.
const TILE_URL = "/api/tile_img?z={z}&x={x}&y={y}";
const AVAILABLE_ZOOMS = %(AVAILABLE_ZOOMS)s;

const tilingScheme = new Cesium.GeographicTilingScheme({
  ellipsoid: marsEllipsoid,
  rectangle: Cesium.Rectangle.fromDegrees(-180, -90, 180, 90),
  numberOfLevelZeroTilesX: 2,
  numberOfLevelZeroTilesY: 1,
});
const imagery = new Cesium.UrlTemplateImageryProvider({
  url: TILE_URL,
  tileWidth: 512,
  tileHeight: 512,
  maximumLevel: 14,
  minimumLevel: 0,
  tilingScheme: tilingScheme,
  credit: "NASA/JPL/MSSS/The Murray Lab (Dickson et al. 2024)",
});

Cesium.Camera.DEFAULT_VIEW_RECTANGLE = Cesium.Rectangle.fromDegrees(-180, -90, 180, 90);
Cesium.Camera.DEFAULT_VIEW_FACTOR = 0;

const viewer = new Cesium.Viewer("cesiumContainer", {
  baseLayerPicker: false,
  geocoder: false,
  timeline: false,
  animation: false,
  infoBox: false,
  selectionIndicator: false,
  shouldAnimate: false,
  homeButton: false,
  sceneModePicker: false,
  navigationHelpButton: false,
  fullscreenButton: false,
  baseLayer: new Cesium.ImageryLayer(imagery),
  terrainProvider: new Cesium.EllipsoidTerrainProvider({ ellipsoid: marsEllipsoid }),
  skyBox: false,
  skyAtmosphere: false,
  useDefaultRenderLoop: true,
  globe: new Cesium.Globe(marsEllipsoid),
});
viewer.scene.globe.showGroundAtmosphere = false;
viewer.scene.backgroundColor = Cesium.Color.BLACK;
viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString("#2a1a12");
// Render only when the scene actually changes (not every frame). This alone
// stops the "camera keeps approaching" visual — without it, Cesium renders
// at 60 Hz and any latent tween in the camera controller keeps firing.
viewer.scene.requestRenderMode = true;
viewer.clock.shouldAnimate = false;

// Default Cesium SSE is 2 (sharpest). Keep at 2 so zoom-in actually
// requests higher-resolution tiles instead of stopping at a coarse level.
viewer.scene.globe.maximumScreenSpaceError = 2;
// Allow more concurrent imagery requests so LOD fills in faster.
Cesium.RequestScheduler.maximumRequestsPerServer = 24;

// Zero out all camera-controller inertia. Without this, any pre-Cesium
// browser wheel/scroll energy bleeds into a decaying zoom over ~2-3 s
// that looks exactly like a continuous approach.
const cc = viewer.scene.screenSpaceCameraController;
cc.inertiaSpin = 0;
cc.inertiaTranslate = 0;
cc.inertiaZoom = 0;
cc.enableCollisionDetection = false;

// Hard-freeze the camera: every frame just BEFORE rendering, reset its
// position + orientation to a fixed orbit view. preRender is the last
// scene event before the draw, so no tween/controller nudge can sneak in
// after us. We hold the freeze for 3 s then release so the user can
// rotate/zoom with the mouse.
const FIXED_POS = new Cesium.Cartesian3(12_000_000, 0, 0);
const FIXED_DIR = new Cesium.Cartesian3(-1, 0, 0);
const FIXED_UP = new Cesium.Cartesian3(0, 0, 1);
const FIXED_RIGHT = new Cesium.Cartesian3();
Cesium.Cartesian3.cross(FIXED_DIR, FIXED_UP, FIXED_RIGHT);
Cesium.Cartesian3.normalize(FIXED_RIGHT, FIXED_RIGHT);
const FIXED_FOV = Cesium.Math.toRadians(60);
function clampCamera() {
  Cesium.Cartesian3.clone(FIXED_POS, viewer.camera.position);
  Cesium.Cartesian3.clone(FIXED_DIR, viewer.camera.direction);
  Cesium.Cartesian3.clone(FIXED_UP, viewer.camera.up);
  Cesium.Cartesian3.clone(FIXED_RIGHT, viewer.camera.right);
  if (viewer.camera.frustum && "fov" in viewer.camera.frustum) {
    viewer.camera.frustum.fov = FIXED_FOV;
  }
}
viewer.camera.setView({
  destination: FIXED_POS,
  orientation: { direction: FIXED_DIR, up: FIXED_UP },
});
cc.enableInputs = false;
const removePre = viewer.scene.preRender.addEventListener(clampCamera);
const removePost = viewer.scene.postUpdate.addEventListener(clampCamera);
setTimeout(() => {
  removePre();
  removePost();
  cc.enableInputs = true;
}, 3000);

// Tiny debug HUD so we can see, live, whether the camera is moving. If
// these numbers stay constant but the globe still appears to approach,
// the visual change is imagery LOD (the surface sharpening), not camera.
const hud = document.createElement("div");
hud.style.cssText = "position:absolute;top:4px;left:4px;padding:4px 8px;background:#000a;color:#0f0;font:12px monospace;z-index:9999;pointer-events:none";
document.getElementById("cesiumContainer").appendChild(hud);
viewer.scene.postRender.addEventListener(() => {
  const p = viewer.camera.position;
  const r = Math.sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
  const fov = viewer.camera.frustum.fov !== undefined
    ? Cesium.Math.toDegrees(viewer.camera.frustum.fov).toFixed(1)
    : "?";
  const cvs = viewer.canvas;
  const ge = viewer.scene.globe.ellipsoid.radii;
  hud.innerHTML =
    `cam ${(p.x/1e6).toFixed(2)}, ${(p.y/1e6).toFixed(2)}, ${(p.z/1e6).toFixed(2)} Mm · r=${(r/1e6).toFixed(2)} Mm<br>` +
    `fov=${fov}° · canvas=${cvs.width}x${cvs.height} · globe r=${(ge.x/1e6).toFixed(2)}/${(ge.y/1e6).toFixed(2)}/${(ge.z/1e6).toFixed(2)} Mm`;
});
viewer.scene.requestRender();

let resultEntities = [];
let queryEntity = null;

function clearResults() {
  resultEntities.forEach(e => viewer.entities.remove(e));
  resultEntities = [];
}

async function queryLatLon(lat, lon) {
  document.getElementById("query").innerHTML =
    `<p>Querying lat ${lat.toFixed(2)}, lon ${lon.toFixed(2)}…</p>`;
  if (queryEntity) viewer.entities.remove(queryEntity);
  queryEntity = viewer.entities.add({
    position: Cesium.Cartesian3.fromDegrees(lon, lat, 0, marsEllipsoid),
    point: { pixelSize: 12, color: Cesium.Color.CYAN.withAlpha(0.7), outlineColor: Cesium.Color.WHITE, outlineWidth: 1 },
  });
  const leafletZoomApprox = Math.max(...AVAILABLE_ZOOMS);
  const diversity = document.getElementById("spatial-diversity")?.checked ?? false;
  const diversityTiles = parseInt(document.getElementById("diversity-tiles")?.value) || 4;
  try {
    const resp = await fetch("/api/query_latlon", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        lat, lon, zoom: leafletZoomApprox, top_k: 20,
        spatial_diversity: diversity,
        diversity_tiles: diversityTiles,
      }),
    });
    if (!resp.ok) {
      const err = await resp.text();
      document.getElementById("query").innerHTML =
        `<p style="color:#f88">Query failed (HTTP ${resp.status}): ${err.slice(0,200)}</p>`;
      return;
    }
    const data = await resp.json();
    if (!data.query) {
      document.getElementById("query").innerHTML = `<p style="color:#f88">Empty response from server.</p>`;
      return;
    }
    renderQuery(data.query);
    renderResults(data.results || [], "similarity");
  } catch (e) {
    document.getElementById("query").innerHTML = `<p style="color:#f88">Network error: ${e.message}</p>`;
  }
}

function renderQuery(q) {
  document.getElementById("result-preview").innerHTML = "";  // reset on new search
  if (!q) {
    document.getElementById("query").innerHTML = `<p style="color:#f88">Empty query response.</p>`;
    return;
  }
  const onErr = `this.onerror=null;this.style.opacity='.3';this.alt='tile unavailable'`;
  const preview = q.tile_url
    ? `<img src="${q.tile_url}" style="width:100%;max-width:320px;border:1px solid #333" onerror="${onErr}"/>`
    : (q.preview_png_b64 ? `<img src="data:image/png;base64,${q.preview_png_b64}" style="width:100%;max-width:320px;border:1px solid #333"/>` : "");
  const loc = (q.z !== undefined && q.x !== undefined)
    ? `z=${q.z} • tile (${q.x},${q.y})`
    : (q.bbox ? `bbox lat ${q.bbox.lat_min.toFixed(2)}..${q.bbox.lat_max.toFixed(2)}, lon ${q.bbox.lon_min.toFixed(2)}..${q.bbox.lon_max.toFixed(2)}` : "");
  const ll = (q.lat !== undefined && q.lon !== undefined)
    ? ` • lat ${q.lat.toFixed(2)} lon ${q.lon.toFixed(2)}`
    : "";
  document.getElementById("query").innerHTML = `<h3>Query</h3>${preview}<p class="muted">${loc}${ll}</p>`;
}

function renderResults(results, scoreCol) {
  clearResults();
  const el = document.getElementById("results");
  if (!results.length) { el.innerHTML = "<p class='muted'>No hits.</p>"; return; }
  el.innerHTML = "";
  results.forEach((r, idx) => {
    const row = document.createElement("div");
    row.className = "result";
    const score = r[scoreCol] ?? r.similarity ?? r.anomaly_score;
    row.innerHTML = `
      <img src="${r.tile_url}" loading="lazy" onerror="this.onerror=null;this.style.opacity='.3';this.alt='•'"/>
      <div class="meta">
        <div><b>#${idx + 1}</b></div>
        <div>${scoreCol}: ${score?.toFixed(3)}</div>
        <div>lat ${r.lat.toFixed(2)} • lon ${r.lon.toFixed(2)}</div>
      </div>`;

    // Draw the matching rectangle on the globe. For patch-mode hits we use
    // the sub-tile patch_bounds so the highlight is ~700 m instead of ~10 km.
    const t = r.patch_bounds || r.tile_bounds;  // [latMin, latMax, lonMin, lonMax]
    const isAnom = scoreCol === "anomaly_score";
    const baseColor = isAnom
      ? Cesium.Color.GOLD
      : Cesium.Color.fromCssColorString("#ff6b9a");
    const rectEnt = t ? viewer.entities.add({
      rectangle: {
        coordinates: Cesium.Rectangle.fromDegrees(t[2], t[0], t[3], t[1]),
        material: baseColor.withAlpha(0.18),
        outline: true,
        outlineColor: baseColor.withAlpha(0.9),
        height: 0,
      },
    }) : null;
    const pinEnt = viewer.entities.add({
      position: Cesium.Cartesian3.fromDegrees(r.lon, r.lat, 0, marsEllipsoid),
      point: {
        pixelSize: 8,
        color: baseColor.withAlpha(0.9),
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 1,
      },
    });
    if (rectEnt) resultEntities.push(rectEnt);
    resultEntities.push(pinEnt);

    // Hover: brighten this result's rectangle.
    row.onmouseenter = () => {
      if (rectEnt) {
        rectEnt.rectangle.material = baseColor.withAlpha(0.45);
        rectEnt.rectangle.outlineColor = Cesium.Color.WHITE;
      }
      viewer.scene.requestRender();
    };
    row.onmouseleave = () => {
      if (rectEnt) {
        rectEnt.rectangle.material = baseColor.withAlpha(0.18);
        rectEnt.rectangle.outlineColor = baseColor.withAlpha(0.9);
      }
      viewer.scene.requestRender();
    };
    // Click: fly to it AND render the tile image in a separate preview
    // pane, so the original query image stays visible for comparison.
    row.onclick = () => {
      viewer.camera.flyTo({
        destination: Cesium.Cartesian3.fromDegrees(r.lon, r.lat, 400_000, marsEllipsoid),
      });
      document.getElementById("result-preview").innerHTML = `
        <h3>Result #${idx + 1} preview</h3>
        <img src="${r.tile_url}" style="width:100%;max-width:320px;border:1px solid #333"
             onerror="this.onerror=null;this.style.opacity='.3';this.alt='tile unavailable'"/>
        <p class="muted">z=${r.z} • tile (${r.x},${r.y}) • lat ${r.lat.toFixed(2)} lon ${r.lon.toFixed(2)} • ${scoreCol}: ${score?.toFixed(3)}</p>`;
    };
    el.appendChild(row);
  });
  viewer.scene.requestRender();
}

// ---- Region-select (Shift+drag a rectangle on the globe) --------------- //
let dragState = null;  // {startLat, startLon, endLat, endLon}
let queryRectEntity = null;
const cesiumContainer = document.getElementById("cesiumContainer");
document.getElementById("mode-help-link").onclick = (e) => {
  e.preventDefault();
  const h = document.getElementById("mode-help");
  h.style.display = h.style.display === "none" ? "block" : "none";
};

function pickLatLon(windowPos) {
  const cartesian = viewer.camera.pickEllipsoid(windowPos, marsEllipsoid);
  if (!cartesian) return null;
  const carto = marsEllipsoid.cartesianToCartographic(cartesian);
  return {
    lat: Cesium.Math.toDegrees(carto.latitude),
    lon: Cesium.Math.toDegrees(carto.longitude),
  };
}

// Live rectangle: Cesium reads lonMin/latMin/lonMax/latMax from this closure
// via a CallbackProperty, so we just mutate these numbers during the drag
// instead of removing and re-adding the entity every mouse-move frame (which
// made the rectangle look laggy under requestRenderMode).
const rectBounds = { lonMin: 0, latMin: 0, lonMax: 0, latMax: 0 };
function setQueryRect(latMin, lonMin, latMax, lonMax) {
  rectBounds.lonMin = lonMin;
  rectBounds.latMin = latMin;
  rectBounds.lonMax = lonMax;
  rectBounds.latMax = latMax;
  if (!queryRectEntity) {
    queryRectEntity = viewer.entities.add({
      rectangle: {
        coordinates: new Cesium.CallbackProperty(() =>
          Cesium.Rectangle.fromDegrees(
            rectBounds.lonMin, rectBounds.latMin,
            rectBounds.lonMax, rectBounds.latMax,
          ),
        false),
        material: Cesium.Color.YELLOW.withAlpha(0.2),
        outline: true,
        outlineColor: Cesium.Color.YELLOW,
        height: 0,
      },
    });
  }
  viewer.scene.requestRender();
}

const ssh = viewer.screenSpaceEventHandler;

// "Arm" a single region-drag via the Draw region button. Next drag becomes
// a region-select; it auto-disarms on mouseup so the very next drag goes
// back to rotating the camera.
let regionArmed = false;
const drawBtn = document.getElementById("draw-region-btn");
function setArmed(armed) {
  regionArmed = armed;
  drawBtn.innerText = armed ? "⏳ Draw one drag…" : "🔲 Draw region";
  drawBtn.style.background = armed ? "#644" : "#244";
  cesiumContainer.style.cursor = armed ? "crosshair" : "default";
}
drawBtn.onclick = () => setArmed(!regionArmed);
window.addEventListener("keydown", (e) => { if (e.key === "Escape" && regionArmed) setArmed(false); });

ssh.setInputAction((evt) => {
  if (!regionArmed) return;
  const p = pickLatLon(evt.position);
  if (!p) return;
  viewer.scene.screenSpaceCameraController.enableInputs = false;
  dragState = { startLat: p.lat, startLon: p.lon, endLat: p.lat, endLon: p.lon };
  setQueryRect(p.lat, p.lon, p.lat, p.lon);
}, Cesium.ScreenSpaceEventType.LEFT_DOWN);

// Click-to-query on LEFT_CLICK (fires on release, so plain drags to rotate
// don't fire queries). If a region is staged, the click first clears the
// stage; a second click runs the tile-level query.
ssh.setInputAction((evt) => {
  if (dragState) return;
  if (stagedBbox) {
    clearStagedRegion();
    document.getElementById("query").innerHTML = "";
    return;
  }
  const p = pickLatLon(evt.position);
  if (!p) return;
  queryLatLon(p.lat, p.lon);
}, Cesium.ScreenSpaceEventType.LEFT_CLICK);

// Mouse-move: we track the drag regardless of whether Shift is still held,
// since the user may release Shift mid-drag.
ssh.setInputAction((evt) => {
  if (!dragState) return;
  const p = pickLatLon(evt.endPosition);
  if (!p) return;
  dragState.endLat = p.lat;
  dragState.endLon = p.lon;
  setQueryRect(
    Math.min(dragState.startLat, p.lat),
    Math.min(dragState.startLon, p.lon),
    Math.max(dragState.startLat, p.lat),
    Math.max(dragState.startLon, p.lon),
  );
}, Cesium.ScreenSpaceEventType.MOUSE_MOVE);

// Staged bbox: set by finishRegionDrag, consumed by runStagedRegion() or
// cleared when the user clicks elsewhere. We DO NOT fire the query
// automatically on mouseup so accidental drags or mid-drag adjustments
// don't waste upstream tile fetches.
let stagedBbox = null;  // {latMin, latMax, lonMin, lonMax}

function clearStagedRegion() {
  stagedBbox = null;
  if (queryRectEntity) {
    viewer.entities.remove(queryRectEntity);
    queryRectEntity = null;
  }
  viewer.scene.requestRender();
}

async function runStagedRegion() {
  if (!stagedBbox) return;
  const { latMin, latMax, lonMin, lonMax } = stagedBbox;
  document.getElementById("query").innerHTML =
    `<p>Querying region (${latMin.toFixed(2)},${lonMin.toFixed(2)})–(${latMax.toFixed(2)},${lonMax.toFixed(2)})…</p>`;
  const queryMode = document.getElementById("region-query-mode").value;
  const localize = document.getElementById("region-localize").checked;
  const diversity = document.getElementById("spatial-diversity").checked;
  const diversityTiles = parseInt(document.getElementById("diversity-tiles").value) || 4;
  try {
    const resp = await fetch("/api/query_bbox", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        lat_min: latMin, lat_max: latMax,
        lon_min: lonMin, lon_max: lonMax,
        top_k: 20, mode: queryMode, localize: localize,
        spatial_diversity: diversity,
        diversity_tiles: diversityTiles,
      }),
    });
    if (!resp.ok) {
      const err = await resp.text();
      document.getElementById("query").innerHTML =
        `<p style="color:#f88">Region query failed (HTTP ${resp.status}): ${err.slice(0,200)}</p>
         <button id="retry-region">🔁 Retry</button> <button id="cancel-region">✕ Clear</button>`;
      document.getElementById("retry-region").onclick = runStagedRegion;
      document.getElementById("cancel-region").onclick = clearStagedRegion;
      return;
    }
    const data = await resp.json();
    renderQuery(data.query);
    renderResults(data.results || [], "similarity");
    stagedBbox = null;  // consumed; keep the rectangle visible as context
  } catch (e) {
    document.getElementById("query").innerHTML =
      `<p style="color:#f88">Network error: ${e.message}</p>
       <button id="retry-region">🔁 Retry</button> <button id="cancel-region">✕ Clear</button>`;
    document.getElementById("retry-region").onclick = runStagedRegion;
    document.getElementById("cancel-region").onclick = clearStagedRegion;
  }
}

function finishRegionDrag() {
  if (!dragState) return;
  const d = dragState;
  dragState = null;
  viewer.scene.screenSpaceCameraController.enableInputs = true;
  setArmed(false);
  const latMin = Math.min(d.startLat, d.endLat);
  const latMax = Math.max(d.startLat, d.endLat);
  const lonMin = Math.min(d.startLon, d.endLon);
  const lonMax = Math.max(d.startLon, d.endLon);
  if (latMax - latMin < 1e-4 || lonMax - lonMin < 1e-4) {
    clearStagedRegion();
    return;
  }
  stagedBbox = { latMin, latMax, lonMin, lonMax };
  document.getElementById("query").innerHTML = `
    <h3>Region staged</h3>
    <p class="muted">(${latMin.toFixed(2)},${lonMin.toFixed(2)}) – (${latMax.toFixed(2)},${lonMax.toFixed(2)})<br>
    Adjust options in the panel, then confirm.</p>
    <button id="run-region" style="background:#264;font-weight:bold">🔍 Search this region</button>
    <button id="cancel-region">✕ Clear</button>`;
  document.getElementById("run-region").onclick = runStagedRegion;
  document.getElementById("cancel-region").onclick = () => {
    clearStagedRegion();
    document.getElementById("query").innerHTML = "";
  };
}
ssh.setInputAction(finishRegionDrag, Cesium.ScreenSpaceEventType.LEFT_UP);
</script>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, default=Path("outputs/murray_z8_global"))
    parser.add_argument(
        "--patch-dir", type=Path, default=Path("outputs/murray_patch_indices"),
        help="Root containing per-zoom patch-index subdirs (patches.faiss + tiles.parquet). "
        "If the path doesn't exist we fall back to CLS-only region queries.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8503)
    args = parser.parse_args()

    _load(args.index_dir, patch_root=args.patch_dir)
    global HTML, GLOBE_HTML
    zooms_json = json.dumps(APP_STATE["available_zooms"])
    HTML = HTML.replace("%(AVAILABLE_ZOOMS)s", zooms_json)
    GLOBE_HTML = GLOBE_HTML.replace("%(AVAILABLE_ZOOMS)s", zooms_json)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
