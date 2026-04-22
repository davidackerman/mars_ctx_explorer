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
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
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


APP_STATE: dict = {}


def _load_single_index(index_dir: Path) -> dict:
    import faiss

    with open(index_dir / "faiss.model.json") as fp:
        sidecar = json.load(fp)
    return {
        "index_dir": index_dir,
        "index": faiss.read_index(str(index_dir / "faiss.index")),
        "metadata": pd.read_parquet(index_dir / "metadata.parquet"),
        "sidecar": sidecar,
        "zoom": int(sidecar["zoom"]),
    }


def _load(index_root: Path) -> None:
    """Load one or many indices from ``index_root``.

    - If ``index_root/faiss.index`` exists, treat it as a single-zoom index.
    - Otherwise scan subdirectories for faiss.index files and register each as
      a zoom level, enabling multi-scale queries.
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


def _fetch_and_embed(z: int, x: int, y: int) -> np.ndarray:
    import faiss

    url = TILE_URL.format(z=z, x=x, y=y)
    r = APP_STATE["session"].get(url, timeout=15)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Tile fetch failed: HTTP {r.status_code}")
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
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
        url = TILE_URL.format(z=z, x=x, y=y)
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 200:
                return x, y, Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            pass
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


def _search(vec: np.ndarray, top_k: int, zoom: Optional[int] = None) -> list[dict]:
    state = APP_STATE["indices"][zoom] if zoom in APP_STATE["indices"] else None
    if state is None:
        z = APP_STATE["default_zoom"]
        state = APP_STATE["indices"][z]
    distances, indices = state["index"].search(vec, top_k)
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
                "tile_url": TILE_URL.format(z=z_i, x=x_i, y=y_i),
            }
        )
    return results


app = FastAPI(title="Murray Lab CTX Similarity Viewer")


class LatLonQuery(BaseModel):
    lat: float
    lon: float
    zoom: Optional[int] = None
    top_k: int = 20


class BboxQuery(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    zoom: Optional[int] = None  # at which zoom to fetch source pixels
    top_k: int = 20


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

    region_img = _fetch_bbox_composite(
        fetch_z, q.lat_min, q.lat_max, q.lon_min, q.lon_max
    )
    vec = _embed_pil(region_img)
    results = _search(vec, q.top_k, zoom=search_zoom)

    # Encode the composite crop as a base64 thumbnail for UI preview.
    import base64

    preview_img = region_img.copy()
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
                "composite_w": region_img.width,
                "composite_h": region_img.height,
                "preview_b64": preview_b64,
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
    vec = _fetch_and_embed(index_zoom, x, y)
    results = _search(vec, q.top_k, zoom=index_zoom)
    return JSONResponse(
        {
            "query": {
                "lat": q.lat, "lon": q.lon, "z": index_zoom, "x": x, "y": y,
                "tile_url": TILE_URL.format(z=index_zoom, x=x, y=y),
                "index_zoom_used": index_zoom,
                "available_zooms": APP_STATE["available_zooms"],
            },
            "results": results,
        }
    )


@app.get("/api/anomalies")
def api_anomalies(k: int = 20) -> JSONResponse:
    """Lazy-compute (and cache) LOF anomaly scores over the FAISS vectors.

    For IVF-PQ, we reconstruct vectors with index.reconstruct_n which returns
    the PQ-decoded approximation — fine for LOF which only needs pairwise
    distances.
    """
    index_dir: Path = APP_STATE["index_dir"]
    scores_path = index_dir / "anomaly_scores.parquet"
    if not scores_path.exists():
        logger.info("Computing LOF anomaly scores over %d vectors...", APP_STATE["index"].ntotal)
        import faiss
        from sklearn.neighbors import LocalOutlierFactor

        ntotal = APP_STATE["index"].ntotal
        dim = APP_STATE["index"].d
        vectors = np.empty((ntotal, dim), dtype="float32")
        APP_STATE["index"].reconstruct_n(0, ntotal, vectors)
        faiss.normalize_L2(vectors)
        lof = LocalOutlierFactor(n_neighbors=40, contamination=0.02, n_jobs=-1)
        lof.fit_predict(vectors)
        scores = -lof.negative_outlier_factor_
        md = APP_STATE["metadata"].copy()
        md["anomaly_score"] = scores
        md.to_parquet(scores_path, index=False)
        logger.info("Wrote %s", scores_path)
    df = pd.read_parquet(scores_path).sort_values("anomaly_score", ascending=False).head(k)
    payload = []
    for _, row in df.iterrows():
        z_i, x_i, y_i = int(row.z), int(row.x), int(row.y)
        payload.append(
            {
                "z": z_i, "x": x_i, "y": y_i,
                "lat": float(row.lat), "lon": float(row.lon),
                "anomaly_score": float(row.anomaly_score),
                "tile_url": TILE_URL.format(z=z_i, x=x_i, y=y_i),
            }
        )
    return JSONResponse({"anomalies": payload})


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return HTML


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
modeBtn.onclick = () => {
  regionMode = !regionMode;
  modeBtn.innerText = regionMode ? "🔲 Region select: ON" : "🔲 Region select: OFF";
  // Disable map drag while in region mode so mousedown starts a bbox draw.
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
  const resp = await fetch("/api/query_bbox", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({lat_min, lat_max, lon_min, lon_max, top_k: 20}),
  });
  const data = await resp.json();
  renderBboxQuery(data.query);
  renderResults(data.results || [], "similarity");
});

function renderBboxQuery(q) {
  document.getElementById("query").innerHTML = `
    <h3>Query region</h3>
    <img src="data:image/jpeg;base64,${q.preview_b64}" style="width:100%;max-width:400px;border:1px solid #333"/>
    <p class="muted">${q.composite_w}×${q.composite_h} px composited from ${q.n_tiles} tile(s) fetched at z=${q.fetch_z}</p>
    <p class="muted">Searched at z=${q.search_zoom} (available: ${q.available_zooms.join(", ")})</p>
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
    row.innerHTML = `
      <img src="${r.tile_url}" loading="lazy"/>
      <div class="meta">
        <div><b>#${idx + 1}</b></div>
        <div>${scoreCol}: ${score?.toFixed(3)}</div>
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, default=Path("outputs/murray_z8_global"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8503)
    args = parser.parse_args()

    _load(args.index_dir)
    global HTML
    HTML = HTML.replace(
        "%(AVAILABLE_ZOOMS)s", json.dumps(APP_STATE["available_zooms"])
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
