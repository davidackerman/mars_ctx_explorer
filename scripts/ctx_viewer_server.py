#!/usr/bin/env python3
"""FastAPI backend for the CTX similarity viewer (Phase C prototype).

Serves:
  GET  /              → Leaflet-based viewer HTML
  GET  /api/images    → unique CTX product list with approx lat/lon + tile counts
  GET  /api/tile      → streams a tile PNG given its disk path (limited to index dir)
  GET  /api/source    → streams the full source .tif as a downsampled JPEG
  POST /api/query     → takes a crop (base64 PNG) + optional product filter,
                        returns top-k similar tiles with lat/lon + match box

This replaces the Streamlit app for the "click anywhere on a map → find similar"
workflow. Retrieval uses the existing CTXPatchIndex we already built.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image
from pydantic import BaseModel

from scientific_pipelines.core.embeddings import DINOv3HFExtractor
from scientific_pipelines.planetary.mars.ctx.anomaly import (
    score_and_persist_anomaly,
    top_k_anomalies,
)
from scientific_pipelines.planetary.mars.ctx.atlas import build_atlas, load_atlas
from scientific_pipelines.planetary.mars.ctx.patch_retrieval import CTXPatchIndex
from scientific_pipelines.planetary.mars.ctx.retrieval import (
    CTXSimilarityIndex,
    _load_grayscale_stretched,
)

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None


# --------------------------------------------------------------------------- #
# Globals populated on startup
# --------------------------------------------------------------------------- #

APP_STATE: dict = {
    "index_dir": None,
    "cls_index": None,
    "patch_index": None,
    "extractor": None,
    "transform": None,
    "source_dir": None,
}


def _load_resources(index_dir: Path, source_dir: Path) -> None:
    """Open the retrieval index(es), load the DINO model once."""
    cls_index = CTXSimilarityIndex(
        index_path=index_dir / "faiss.index",
        metadata_path=index_dir / "metadata.parquet",
    )
    APP_STATE["cls_index"] = cls_index

    patch_faiss = index_dir / "patches.faiss"
    if patch_faiss.exists():
        APP_STATE["patch_index"] = CTXPatchIndex(index_dir)
        logger.info("Loaded patch index (%s vectors)", APP_STATE["patch_index"].index.ntotal)
    else:
        logger.warning("No patch index found; region-crop queries will fall back to CLS")

    sidecar = cls_index.metadata  # placeholder; real sidecar is JSON on disk
    sidecar_path = index_dir / "faiss.model.json"
    model_name = "facebook/dinov3-vitl16-pretrain-sat493m"
    image_size = 512
    if sidecar_path.exists():
        import json as _json

        with open(sidecar_path) as fp:
            data = _json.load(fp)
        model_name = data.get("model_name") or model_name
        image_size = int(data.get("image_size") or image_size)

    APP_STATE["extractor"] = DINOv3HFExtractor(
        model_name=model_name, device="cuda", use_half_precision=False
    )
    APP_STATE["transform"] = DINOv3HFExtractor.get_default_transforms(image_size=image_size)
    APP_STATE["index_dir"] = index_dir
    APP_STATE["source_dir"] = source_dir
    logger.info("Extractor ready: %s @ %s", model_name, image_size)


# --------------------------------------------------------------------------- #
# App setup
# --------------------------------------------------------------------------- #

app = FastAPI(title="CTX Similarity Viewer")


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return VIEWER_HTML


@app.get("/api/images")
def api_images() -> JSONResponse:
    md: pd.DataFrame = APP_STATE["cls_index"].metadata
    grouped = (
        md.dropna(subset=["product_id"])
        .groupby("product_id")
        .agg(
            lat=("approx_lat", "first"),
            lon=("approx_lon", "first"),
            tile_count=("image_path", "size"),
            first_tile=("image_path", "first"),
        )
        .reset_index()
    )
    records: List[dict] = []
    for _, row in grouped.iterrows():
        lat = float(row["lat"]) if pd.notna(row["lat"]) else None
        lon = float(row["lon"]) if pd.notna(row["lon"]) else None
        records.append(
            {
                "product_id": str(row["product_id"]),
                "lat": lat,
                "lon": lon,
                "tile_count": int(row["tile_count"]),
                "first_tile": str(row["first_tile"]),
            }
        )
    return JSONResponse({"images": records})


@app.get("/api/tile")
def api_tile(path: str = Query(..., description="Tile PNG path")) -> Response:
    target = Path(path).resolve()
    allowed_root = APP_STATE["index_dir"].resolve()
    if not str(target).startswith(str(allowed_root)):
        raise HTTPException(status_code=403, detail="Path outside index_dir")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Tile not found")
    return Response(content=target.read_bytes(), media_type="image/png")


@app.get("/api/source")
def api_source(product_id: str, max_edge: int = 1600) -> Response:
    source_dir: Path = APP_STATE["source_dir"]
    tif = source_dir / f"{product_id}.tif"
    if not tif.exists():
        raise HTTPException(status_code=404, detail=f"{tif} not found")
    # 16-bit → 8-bit percentile stretch + CLAHE for display. The base stretch
    # matches the tiles we indexed; CLAHE is a display-only enhancement that
    # pulls out local contrast in flat-looking scenes (e.g. volcanic plains)
    # that a single global stretch can't show.
    import cv2

    arr = _load_grayscale_stretched(tif)  # uint8 2-D, 0 = NoData
    mask = arr > 0
    if mask.any():
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
        enhanced = clahe.apply(arr)
        # Keep NoData as black (CLAHE would brighten 0s too)
        arr = np.where(mask, enhanced, 0).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    w, h = img.size
    scale = min(1.0, max_edge / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={
            "X-Original-Width": str(w),
            "X-Original-Height": str(h),
            "X-Preview-Scale": f"{scale:.6f}",
        },
    )


@app.post("/api/build_patch_index")
def api_build_patch_index(product_id: str) -> JSONResponse:
    """Build (or return cached) per-product patch index.

    Intended for the 'global CLS-only index' workflow: when a user focuses on
    a specific product, they can opt in to region-level patch queries for
    just that product. Each per-product patch index is tiny (tens of MB).
    """
    index_dir: Path = APP_STATE["index_dir"]
    patch_dir = index_dir / "patch_by_product" / product_id
    if (patch_dir / "patches.faiss").exists():
        return JSONResponse({"product_id": product_id, "status": "cached", "path": str(patch_dir)})

    md: pd.DataFrame = APP_STATE["cls_index"].metadata
    tiles = md[md["product_id"] == product_id]["image_path"].astype(str).tolist()
    if not tiles:
        raise HTTPException(status_code=404, detail=f"No tiles for product_id={product_id}")

    extractor = APP_STATE["extractor"]
    transform = APP_STATE["transform"]

    patch_dir.mkdir(parents=True, exist_ok=True)
    extra_meta = md[md["product_id"] == product_id].drop_duplicates(subset="image_path")

    logger.info("Building patch index for %s (%d tiles)", product_id, len(tiles))
    CTXPatchIndex.build_from_tiles(
        tile_paths=[Path(p) for p in tiles],
        extractor=extractor,
        transform=transform,
        index_dir=patch_dir,
        batch_size=32,
        device="cuda",
        normalize=True,
        extra_tile_metadata=extra_meta,
    )
    return JSONResponse(
        {
            "product_id": product_id,
            "status": "built",
            "path": str(patch_dir),
            "tile_count": len(tiles),
        }
    )


@app.get("/api/atlas")
def api_atlas(max_points: int = 10000) -> JSONResponse:
    """Return the UMAP atlas: every tile's 2-D projection + cluster id.

    Lazily computes + caches `atlas.parquet` on first call. Subsequent calls
    just read the parquet. Clients get a downsample by the ``max_points`` arg
    (default 10k) to keep the scatter responsive.
    """
    index_dir: Path = APP_STATE["index_dir"]
    atlas_path = index_dir / "atlas.parquet"
    if not atlas_path.exists():
        logger.info("No cached atlas; building UMAP now (can take a few minutes)")
        build_atlas(
            embeddings_path=index_dir / "embeddings.parquet",
            metadata_path=index_dir / "metadata.parquet",
            out_path=atlas_path,
            max_points=50_000,
        )
    df = load_atlas(atlas_path)
    df = df.dropna(subset=["atlas_x", "atlas_y"])
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "image_path": str(row["image_path"]),
                "x": float(row["atlas_x"]),
                "y": float(row["atlas_y"]),
                "cluster_id": int(row.get("cluster_id") or -1),
                "product_id": str(row.get("product_id", "") or ""),
                "tile_scale": int(row.get("tile_scale") or 0),
                "lat": float(row["approx_lat"]) if pd.notna(row.get("approx_lat")) else None,
                "lon": float(row["approx_lon"]) if pd.notna(row.get("approx_lon")) else None,
            }
        )
    return JSONResponse({"points": records})


class FewShotRequest(BaseModel):
    positive_paths: List[str]
    top_k: int = 20
    tile_scale: Optional[int] = None


@app.post("/api/few_shot")
def api_few_shot(req: FewShotRequest) -> JSONResponse:
    """Fit a logistic regression using the user's positives vs the rest of the
    corpus and return the top-k highest-scoring tiles globally.
    """
    if not req.positive_paths:
        raise HTTPException(status_code=400, detail="Need at least one positive_paths entry")

    index: CTXSimilarityIndex = APP_STATE["cls_index"]
    embeddings_path = APP_STATE["index_dir"] / "embeddings.parquet"
    if not embeddings_path.exists():
        raise HTTPException(status_code=500, detail=f"Missing {embeddings_path}")
    emb_df = pd.read_parquet(embeddings_path).drop_duplicates(
        subset="image_path", keep="first"
    )

    # Align embeddings to the index's metadata row order.
    path_to_row = {str(p): i for i, p in enumerate(emb_df["image_path"].astype(str))}
    vectors = np.vstack(emb_df["embedding"].values).astype(np.float32)
    vectors = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)

    pos_rows = [path_to_row.get(p) for p in req.positive_paths]
    pos_rows = [r for r in pos_rows if r is not None]
    if not pos_rows:
        raise HTTPException(
            status_code=400,
            detail="None of the positive_paths are in the index embeddings",
        )

    y = np.zeros(vectors.shape[0], dtype=np.int32)
    y[pos_rows] = 1

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=1.0, max_iter=200, class_weight="balanced", n_jobs=-1)
    clf.fit(vectors, y)
    # clf.decision_function is the raw score; use that for ranking
    scores = clf.decision_function(vectors).astype(np.float32)

    md = index.metadata
    md_slim = md[
        [c for c in ("image_path", "tile_scale", "product_id", "approx_lat", "approx_lon") if c in md.columns]
    ].drop_duplicates(subset="image_path", keep="first")

    df = pd.DataFrame(
        {"image_path": emb_df["image_path"].astype(str), "score": scores}
    ).merge(md_slim, on="image_path", how="left")
    df = df[~df["image_path"].isin(req.positive_paths)]
    if req.tile_scale is not None and "tile_scale" in df.columns:
        df = df[df["tile_scale"] == int(req.tile_scale)]
    df = df.sort_values("score", ascending=False).head(req.top_k)

    payload = []
    for _, row in df.iterrows():
        payload.append(
            {
                "image_path": str(row["image_path"]),
                "product_id": str(row.get("product_id", "") or ""),
                "tile_scale": int(row.get("tile_scale") or 0),
                "score": float(row["score"]),
                "lat": float(row["approx_lat"]) if pd.notna(row.get("approx_lat")) else None,
                "lon": float(row["approx_lon"]) if pd.notna(row.get("approx_lon")) else None,
            }
        )
    return JSONResponse({"results": payload})


@app.get("/api/anomalies")
def api_anomalies(k: int = 20, tile_scale: Optional[int] = None) -> JSONResponse:
    """Return the top-k most anomalous tiles.

    Lazily computes + caches the anomaly score parquet on first call; subsequent
    calls are instant. Higher score = more outlier-ish in CLS-embedding space.
    """
    index_dir: Path = APP_STATE["index_dir"]
    scores_path = index_dir / "anomaly_scores.parquet"
    if not scores_path.exists():
        logger.info("No cached anomaly scores; computing now (LOF, k=40)...")
        score_and_persist_anomaly(
            embeddings_path=index_dir / "embeddings.parquet",
            metadata_path=index_dir / "metadata.parquet",
            out_path=scores_path,
            method="lof",
            n_neighbors=40,
        )
    df = top_k_anomalies(scores_path, k=k, tile_scale=tile_scale)
    payload = df.to_dict(orient="records")
    # numpy dtypes can't serialise natively
    clean = []
    for row in payload:
        clean.append(
            {
                "image_path": str(row.get("image_path", "")),
                "product_id": str(row.get("product_id", "") or ""),
                "tile_scale": int(row.get("tile_scale") or 0),
                "anomaly_score": float(row.get("anomaly_score") or 0.0),
                "lat": float(row["approx_lat"]) if pd.notna(row.get("approx_lat")) else None,
                "lon": float(row["approx_lon"]) if pd.notna(row.get("approx_lon")) else None,
            }
        )
    return JSONResponse({"anomalies": clean})


class QueryRequest(BaseModel):
    crop_png_base64: str
    top_k: int = 12
    tile_scale: Optional[int] = None


@app.post("/api/query")
def api_query(req: QueryRequest) -> JSONResponse:
    try:
        raw = base64.b64decode(req.crop_png_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Bad base64: {exc}")

    crop = Image.open(io.BytesIO(raw))
    if crop.mode != "RGB":
        crop = crop.convert("RGB")
    tensor = APP_STATE["transform"](crop).unsqueeze(0)

    patch_index: Optional[CTXPatchIndex] = APP_STATE["patch_index"]
    if patch_index is not None:
        patches = APP_STATE["extractor"].extract_patches(tensor)  # (1, P, D)
        results = patch_index.query_by_patches(
            patches[0], k=req.top_k, tile_scale=req.tile_scale
        )
        score_col = "aggregate_sim"
    else:
        cls_vec = APP_STATE["extractor"].extract(tensor)[0]
        results = APP_STATE["cls_index"].query_by_vector(
            cls_vec, k=req.top_k, tile_scale=req.tile_scale
        )
        score_col = "similarity"

    md = APP_STATE["cls_index"].metadata
    joined = results.merge(
        md[["image_path", "approx_lat", "approx_lon"]].drop_duplicates(subset="image_path"),
        on="image_path",
        how="left",
        suffixes=("", "_md"),
    )

    payload: List[dict] = []
    for _, row in joined.iterrows():
        entry = {
            "image_path": str(row["image_path"]),
            "product_id": str(row.get("product_id", "")),
            "tile_scale": int(row.get("tile_scale") or 0),
            "score": float(row[score_col]) if score_col in row else None,
            "coverage": int(row["coverage"]) if "coverage" in row and pd.notna(row["coverage"]) else None,
            "lat": float(row["approx_lat"]) if pd.notna(row.get("approx_lat")) else None,
            "lon": float(row["approx_lon"]) if pd.notna(row.get("approx_lon")) else None,
        }
        payload.append(entry)
    return JSONResponse({"results": payload, "score_column": score_col})


# --------------------------------------------------------------------------- #
# Static HTML (inline)
# --------------------------------------------------------------------------- #

VIEWER_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>CTX Similarity Viewer</title>
<meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no" />
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
  integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
  integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<style>
  html, body { height: 100%; margin: 0; font-family: system-ui, sans-serif; }
  #layout { display: grid; grid-template-columns: 2fr 1fr; height: 100vh; }
  #map { height: 100%; }
  #side {
    border-left: 1px solid #333;
    overflow-y: auto;
    padding: 12px;
    background: #111;
    color: #eee;
  }
  #side h2 { margin: 8px 0 4px; font-size: 1.1em; }
  #crop-container {
    background: #000;
    position: relative;
    user-select: none;
  }
  #crop-container img {
    display: block;
    max-width: 100%;
    -webkit-user-drag: none;
    user-drag: none;
    -webkit-user-select: none;
    user-select: none;
    pointer-events: none;
  }
  #crop-rect {
    position: absolute;
    border: 2px solid #00ffae;
    background: rgba(0, 255, 174, 0.15);
    pointer-events: none;
  }
  #results .result {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 4px;
    border-bottom: 1px solid #222;
    cursor: pointer;
  }
  #results .result:hover { background: #1c1c1c; }
  #results img { width: 80px; height: 80px; object-fit: cover; border: 1px solid #222; }
  #results .meta { font-size: 0.85em; line-height: 1.25; }
  button { background: #244; color: #eee; border: 1px solid #466; padding: 6px 10px; cursor: pointer; }
  button:disabled { opacity: 0.5; cursor: default; }
  .muted { color: #888; font-size: 0.85em; }
</style>
</head>
<body>
<div id="layout">
  <div id="map"></div>
  <div id="side">
    <h2>CTX similarity explorer</h2>
    <p class="muted">Click a circle on the map to open its source image. Drag to draw a crop, then hit "Find similar".</p>
    <div style="margin-bottom: 10px;">
      <button id="surprise-btn">🔭 Surprise me (top anomalies)</button>
      <button id="atlas-btn">🗺️ Terrain atlas</button>
    </div>
    <div id="positive-tray" style="display: none; margin: 8px 0; padding: 6px; background: #1a1a1a; border: 1px solid #333;">
      <div style="display: flex; align-items: center; gap: 8px;">
        <span id="tray-count">0 positive examples</span>
        <button id="tray-fewshot">Find more like these</button>
        <button id="tray-clear">Clear tray</button>
      </div>
      <div id="tray-thumbs" style="display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px;"></div>
    </div>
    <div id="selected"></div>
    <h2>Results</h2>
    <div id="results"><p class="muted">Nothing yet.</p></div>
  </div>
</div>

<script>
const MARS_BASE =
  "https://trek.nasa.gov/tiles/Mars/EQ/Mars_Viking_MDIM21_ClrMosaic_global_232m/1.0.0/default/default028mm/{z}/{y}/{x}.jpg";

const map = L.map("map", {
  crs: L.CRS.EPSG4326,
  minZoom: 1,
  maxZoom: 6,
  worldCopyJump: false,
}).setView([0, 0], 2);

L.tileLayer(MARS_BASE, {
  attribution: "Mars base: NASA/USGS (MDIM 2.1)",
  tileSize: 256,
  noWrap: true,
}).addTo(map);

let selectedProduct = null;
let selectedImg = null;
let cropBox = null;
let cropStart = null;
let sourceOriginalW = 0;
let sourceOriginalH = 0;
let previewScale = 1.0;
let resultMarkers = [];

// Few-shot positives tray --------------------------------------------------
const positiveTray = new Set();

function updateTrayUI() {
  const tray = document.getElementById("positive-tray");
  const count = document.getElementById("tray-count");
  const thumbs = document.getElementById("tray-thumbs");
  count.innerText = `${positiveTray.size} positive example${positiveTray.size === 1 ? "" : "s"}`;
  tray.style.display = positiveTray.size > 0 ? "block" : "none";
  thumbs.innerHTML = "";
  positiveTray.forEach(p => {
    const img = document.createElement("img");
    img.src = `/api/tile?path=${encodeURIComponent(p)}`;
    img.style.width = "60px";
    img.style.height = "60px";
    img.style.objectFit = "cover";
    img.style.border = "1px solid #333";
    img.title = p;
    img.style.cursor = "pointer";
    img.onclick = () => { positiveTray.delete(p); updateTrayUI(); };
    thumbs.appendChild(img);
  });
}

function addToTray(imagePath) {
  positiveTray.add(imagePath);
  updateTrayUI();
}

document.getElementById("tray-clear").onclick = () => {
  positiveTray.clear();
  updateTrayUI();
};

document.getElementById("tray-fewshot").onclick = async () => {
  if (positiveTray.size === 0) return;
  const btn = document.getElementById("tray-fewshot");
  btn.disabled = true;
  btn.innerText = "Searching...";
  const resp = await fetch("/api/few_shot", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ positive_paths: Array.from(positiveTray), top_k: 20 }),
  });
  const data = await resp.json();
  renderResults(data.results || [], "score");
  btn.disabled = false;
  btn.innerText = "Find more like these";
};

// Atlas modal --------------------------------------------------------------
document.getElementById("atlas-btn").onclick = async () => {
  const btn = document.getElementById("atlas-btn");
  btn.disabled = true;
  btn.innerText = "Building atlas...";
  try {
    const resp = await fetch("/api/atlas?max_points=5000");
    const data = await resp.json();
    renderAtlas(data.points || []);
  } finally {
    btn.disabled = false;
    btn.innerText = "🗺️ Terrain atlas";
  }
};

function renderAtlas(points) {
  const existing = document.getElementById("atlas-modal");
  if (existing) existing.remove();
  if (!points.length) { return; }

  const modal = document.createElement("div");
  modal.id = "atlas-modal";
  modal.style.cssText = "position:fixed;top:5%;left:5%;width:90%;height:90%;background:#0a0a0a;color:#eee;border:1px solid #444;z-index:9999;padding:12px;overflow:hidden;";
  modal.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <span>Terrain atlas — UMAP(cosine) + HDBSCAN on CLS embeddings. Click a point to load its tile.</span>
      <button id="atlas-close">Close</button>
    </div>
    <canvas id="atlas-canvas" width="1200" height="800" style="display:block;margin:8px auto;background:#050505;border:1px solid #222;"></canvas>
    <div id="atlas-preview" style="position:absolute;right:20px;bottom:20px;background:#111;padding:6px;border:1px solid #333;display:none;">
      <img id="atlas-preview-img" style="max-width:160px;max-height:160px;display:block;"/>
      <div id="atlas-preview-cap" style="font-size:0.85em;"></div>
    </div>
  `;
  document.body.appendChild(modal);
  document.getElementById("atlas-close").onclick = () => modal.remove();

  const canvas = document.getElementById("atlas-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1, rangeY = maxY - minY || 1;
  const pad = 24;

  function toPixel(p) {
    return {
      px: pad + (p.x - minX) / rangeX * (W - 2*pad),
      py: pad + (p.y - minY) / rangeY * (H - 2*pad),
    };
  }

  function clusterColor(cid) {
    if (cid < 0) return "rgba(120,120,120,0.35)";
    const h = (cid * 137.5) % 360;
    return `hsla(${h}, 80%, 60%, 0.8)`;
  }

  ctx.fillStyle = "#050505";
  ctx.fillRect(0, 0, W, H);
  points.forEach(p => {
    const { px, py } = toPixel(p);
    ctx.fillStyle = clusterColor(p.cluster_id);
    ctx.beginPath();
    ctx.arc(px, py, 2.3, 0, Math.PI*2);
    ctx.fill();
  });

  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (W / rect.width);
    const my = (e.clientY - rect.top) * (H / rect.height);
    // Nearest-point lookup (brute force is fine for 5k)
    let best = null, bestD = 64;
    for (const p of points) {
      const { px, py } = toPixel(p);
      const d = (px - mx) ** 2 + (py - my) ** 2;
      if (d < bestD) { bestD = d; best = p; }
    }
    const prev = document.getElementById("atlas-preview");
    if (best) {
      prev.style.display = "block";
      document.getElementById("atlas-preview-img").src =
        `/api/tile?path=${encodeURIComponent(best.image_path)}`;
      document.getElementById("atlas-preview-cap").innerText =
        `${best.product_id} • cluster ${best.cluster_id} • scale ${best.tile_scale}`;
    } else {
      prev.style.display = "none";
    }
  };
  canvas.onclick = (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (W / rect.width);
    const my = (e.clientY - rect.top) * (H / rect.height);
    let best = null, bestD = 100;
    for (const p of points) {
      const { px, py } = toPixel(p);
      const d = (px - mx) ** 2 + (py - my) ** 2;
      if (d < bestD) { bestD = d; best = p; }
    }
    if (best && best.lat !== null && best.lon !== null) {
      map.setView([best.lat, best.lon], Math.min(map.getMaxZoom(), 4));
      modal.remove();
    }
  };
}

document.getElementById("surprise-btn").onclick = async () => {
  const btn = document.getElementById("surprise-btn");
  btn.disabled = true;
  btn.innerText = "Scoring anomalies (first time can take ~1 min)...";
  try {
    const resp = await fetch("/api/anomalies?k=20");
    const data = await resp.json();
    renderAnomalies(data.anomalies);
  } finally {
    btn.disabled = false;
    btn.innerText = "🔭 Surprise me (top anomalies)";
  }
};

function renderAnomalies(anomalies) {
  resultMarkers.forEach(m => map.removeLayer(m));
  resultMarkers = [];
  const el = document.getElementById("results");
  if (!anomalies || !anomalies.length) {
    el.innerHTML = '<p class="muted">No anomaly scores yet.</p>';
    return;
  }
  el.innerHTML = "<h3>Top anomalies (weirdest tiles)</h3>";
  anomalies.forEach((a, idx) => {
    const row = document.createElement("div");
    row.className = "result";
    row.innerHTML = `
      <img src="/api/tile?path=${encodeURIComponent(a.image_path)}" />
      <div class="meta">
        <div><b>#${idx + 1}</b> ${a.product_id}</div>
        <div>anomaly: ${a.anomaly_score.toFixed(3)} • scale ${a.tile_scale}</div>
        <div>lat ${a.lat?.toFixed(1)} • lon ${a.lon?.toFixed(1)}</div>
      </div>
    `;
    el.appendChild(row);
    if (a.lat !== null && a.lon !== null) {
      const marker = L.circleMarker([a.lat, a.lon], {
        radius: 9,
        color: "#ffd700",
        weight: 2,
        fillColor: "#ffd700",
        fillOpacity: 0.5,
      }).addTo(map);
      marker.bindTooltip(`#${idx + 1} anomaly: ${a.anomaly_score.toFixed(2)}`);
      const tileBox = parseTileBox(a.image_path);
      marker.on("click", () => openByProductId(a.product_id, tileBox));
      resultMarkers.push(marker);
    }
    const tileBox = parseTileBox(a.image_path);
    row.onclick = () => openByProductId(a.product_id, tileBox);
  });
}

// Keep a product_id -> image record lookup so anomaly / few-shot / query pins
// can all navigate to the source product on click.
const imagesByProductId = new Map();

fetch("/api/images").then(r => r.json()).then(({ images }) => {
  images.forEach(img => {
    imagesByProductId.set(img.product_id, img);
    if (img.lat === null || img.lon === null) return;
    const marker = L.circleMarker([img.lat, img.lon], {
      radius: 6,
      color: "#00ffae",
      weight: 1,
      fillColor: "#00ffae",
      fillOpacity: 0.6,
    }).addTo(map);
    marker.bindTooltip(img.product_id, { permanent: false });
    marker.on("click", () => openProduct(img));
  });
});

function openByProductId(product_id, highlight) {
  const img = imagesByProductId.get(product_id);
  if (img) openProduct(img, highlight);
}

// Tile filename → source-image pixel box. Filenames look like
// "<product>_s0512_tile_004096_000512.png" where s=512 is the tile side and
// the two 6-digit ints are (y_offset, x_offset) in source-image pixels.
function parseTileBox(image_path) {
  const m = image_path.match(/_s(\d{4})_tile_(\d{6})_(\d{6})\.png$/);
  if (!m) return null;
  const scale = parseInt(m[1], 10);
  const y = parseInt(m[2], 10);
  const x = parseInt(m[3], 10);
  return { x, y, width: scale, height: scale };
}

function openProduct(img, highlight) {
  selectedProduct = img;
  const el = document.getElementById("selected");
  el.innerHTML = `
    <h2>${img.product_id}</h2>
    <p class="muted">lat ${img.lat?.toFixed(1)} • lon ${img.lon?.toFixed(1)} • ${img.tile_count} tiles</p>
    <div id="crop-container">
      <img id="src" src="" alt="" />
      <div id="crop-rect" style="display:none"></div>
      <div id="highlight-rect" style="display:none;position:absolute;border:3px solid #ffd700;background:rgba(255,215,0,0.10);pointer-events:none;"></div>
    </div>
    <p class="muted">Drag on the image to draw a query crop.</p>
    <button id="go" disabled>Find similar</button>
    <button id="clear">Clear crop</button>
  `;
  const srcImg = document.getElementById("src");
  srcImg.onload = () => {
    sourceOriginalW = parseInt(lastHeaders["x-original-width"] || srcImg.naturalWidth);
    sourceOriginalH = parseInt(lastHeaders["x-original-height"] || srcImg.naturalHeight);
    previewScale = parseFloat(lastHeaders["x-preview-scale"] || 1.0);
    selectedImg = srcImg;

    if (highlight) {
      // Convert source-image pixel coords → displayed-preview pixel coords.
      // The displayed <img> is scaled by the browser to fit the container;
      // the server-side preview scale is only partial — also account for
      // display scaling via getBoundingClientRect / naturalWidth.
      const rect = document.getElementById("highlight-rect");
      const imgRect = srcImg.getBoundingClientRect();
      const displayScaleX = imgRect.width / srcImg.naturalWidth;
      const displayScaleY = imgRect.height / srcImg.naturalHeight;
      const previewX = highlight.x * previewScale;
      const previewY = highlight.y * previewScale;
      const previewW = highlight.width * previewScale;
      const previewH = highlight.height * previewScale;
      rect.style.left = (previewX * displayScaleX) + "px";
      rect.style.top = (previewY * displayScaleY) + "px";
      rect.style.width = (previewW * displayScaleX) + "px";
      rect.style.height = (previewH * displayScaleY) + "px";
      rect.style.display = "block";
    }
  };

  lastHeaders = {};
  fetch(`/api/source?product_id=${encodeURIComponent(img.product_id)}`).then(async (resp) => {
    lastHeaders["x-original-width"] = resp.headers.get("x-original-width");
    lastHeaders["x-original-height"] = resp.headers.get("x-original-height");
    lastHeaders["x-preview-scale"] = resp.headers.get("x-preview-scale");
    const blob = await resp.blob();
    srcImg.src = URL.createObjectURL(blob);
  });

  document.getElementById("clear").onclick = clearCrop;
  document.getElementById("go").onclick = runQuery;
  setupCropping();
}

let lastHeaders = {};

function setupCropping() {
  const container = document.getElementById("crop-container");
  const rect = document.getElementById("crop-rect");

  const onDown = (e) => {
    if (!selectedImg) return;
    e.preventDefault();
    const r = selectedImg.getBoundingClientRect();
    cropStart = { x: e.clientX - r.left, y: e.clientY - r.top };
    rect.style.display = "block";
    rect.style.left = cropStart.x + "px";
    rect.style.top = cropStart.y + "px";
    rect.style.width = "0px";
    rect.style.height = "0px";
  };
  const onMove = (e) => {
    if (!cropStart || !selectedImg) return;
    e.preventDefault();
    const r = selectedImg.getBoundingClientRect();
    const x = Math.max(0, Math.min(r.width, e.clientX - r.left));
    const y = Math.max(0, Math.min(r.height, e.clientY - r.top));
    const left = Math.min(x, cropStart.x);
    const top = Math.min(y, cropStart.y);
    const width = Math.abs(x - cropStart.x);
    const height = Math.abs(y - cropStart.y);
    rect.style.left = left + "px";
    rect.style.top = top + "px";
    rect.style.width = width + "px";
    rect.style.height = height + "px";
    cropBox = { left, top, width, height };
    document.getElementById("go").disabled = width < 10 || height < 10;
  };
  const onUp = () => {
    cropStart = null;
  };

  container.addEventListener("mousedown", onDown);
  // Bind move + up to document so dragging outside the container still works
  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);

  // Explicitly block native image drag (belt + suspenders with CSS above)
  container.addEventListener("dragstart", (e) => e.preventDefault());
}

function clearCrop() {
  document.getElementById("crop-rect").style.display = "none";
  cropBox = null;
  document.getElementById("go").disabled = true;
}

async function runQuery() {
  if (!cropBox || !selectedImg) return;
  document.getElementById("go").disabled = true;
  document.getElementById("results").innerHTML = '<p class="muted">Querying...</p>';

  const displayed = selectedImg.getBoundingClientRect();
  const displayScaleX = displayed.width / selectedImg.naturalWidth;
  const displayScaleY = displayed.height / selectedImg.naturalHeight;
  const naturalBox = {
    left: Math.round(cropBox.left / displayScaleX),
    top: Math.round(cropBox.top / displayScaleY),
    width: Math.round(cropBox.width / displayScaleX),
    height: Math.round(cropBox.height / displayScaleY),
  };

  // Natural-image crop on a hidden canvas (draws from the preview, not the
  // full-res source — good enough for a first demo; can be upgraded to
  // server-side full-res cropping later).
  const canvas = document.createElement("canvas");
  canvas.width = naturalBox.width;
  canvas.height = naturalBox.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(
    selectedImg,
    naturalBox.left, naturalBox.top, naturalBox.width, naturalBox.height,
    0, 0, naturalBox.width, naturalBox.height,
  );
  const dataURL = canvas.toDataURL("image/png");
  const base64 = dataURL.split(",")[1];

  const naturalLong = Math.max(naturalBox.width, naturalBox.height);
  let snapScale = null;
  const buckets = [256, 512, 1024];
  const origLong = naturalLong / previewScale;
  snapScale = buckets.reduce(
    (prev, curr) => (Math.abs(Math.log(curr) - Math.log(origLong)) <
                     Math.abs(Math.log(prev) - Math.log(origLong)) ? curr : prev),
    buckets[0],
  );

  const resp = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      crop_png_base64: base64,
      top_k: 12,
      tile_scale: snapScale,
    }),
  });
  const data = await resp.json();
  renderResults(data.results, data.score_column);
}

function renderResults(results, scoreCol) {
  resultMarkers.forEach(m => map.removeLayer(m));
  resultMarkers = [];
  const el = document.getElementById("results");
  if (!results || !results.length) {
    el.innerHTML = '<p class="muted">No matches.</p>';
    return;
  }
  el.innerHTML = "";
  results.forEach((r, idx) => {
    const row = document.createElement("div");
    row.className = "result";
    row.innerHTML = `
      <img src="/api/tile?path=${encodeURIComponent(r.image_path)}" />
      <div class="meta">
        <div><b>#${idx + 1}</b> ${r.product_id}</div>
        <div>${scoreCol}: ${r.score?.toFixed(3) ?? "?"}${r.coverage !== null && r.coverage !== undefined ? " • cov " + r.coverage : ""}</div>
        <div>lat ${r.lat?.toFixed(1)} • lon ${r.lon?.toFixed(1)} • scale ${r.tile_scale}</div>
        <button data-path="${r.image_path}" class="add-positive-btn">+ positive</button>
      </div>
    `;
    el.appendChild(row);
    row.querySelector(".add-positive-btn").onclick = (ev) => {
      ev.stopPropagation();
      addToTray(ev.target.dataset.path);
    };

    if (r.lat !== null && r.lon !== null) {
      const marker = L.circleMarker([r.lat, r.lon], {
        radius: 8,
        color: "#ff6b9a",
        weight: 2,
        fillColor: "#ff6b9a",
        fillOpacity: 0.4,
      }).addTo(map);
      marker.bindTooltip(`#${idx + 1} ${r.product_id}`);
      const tileBox = parseTileBox(r.image_path);
      marker.on("click", () => openByProductId(r.product_id, tileBox));
      resultMarkers.push(marker);
    }
    const tileBox = parseTileBox(r.image_path);
    row.onclick = (ev) => {
      if (ev.target.classList.contains("add-positive-btn")) return;
      openByProductId(r.product_id, tileBox);
    };
  });
}
</script>
</body>
</html>
"""


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, default=Path("outputs/ctx_similarity"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/ctx"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8502)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", "/mnt/bigdisk/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/mnt/bigdisk/hf_cache/hub")

    _load_resources(args.index_dir, args.source_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
