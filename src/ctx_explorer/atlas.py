"""UMAP + HDBSCAN 'terrain atlas' over indexed CTX tiles.

Produces a 2-D projection of every tile's CLS embedding plus a cluster label,
persisted next to the FAISS index as ``atlas.parquet``. The viewer renders this
as a clickable scatter plot so users can visually browse the embedding space —
click a point, navigate the map to that tile.

Only the first run is expensive (UMAP on ~100k vectors takes a few minutes).
Subsequent loads just read the parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32, copy=False)


def build_atlas(
    embeddings_path: Path,
    metadata_path: Path,
    out_path: Path,
    n_neighbors: int = 50,
    min_dist: float = 0.05,
    min_cluster_size: int = 20,
    random_state: int = 42,
    max_points: Optional[int] = 50_000,
) -> Path:
    """Compute a UMAP + HDBSCAN atlas over the indexed tile embeddings.

    Args:
        embeddings_path: path to the CLS embeddings parquet
        metadata_path: path to the tile metadata parquet (for lat/lon join)
        out_path: path to write the resulting atlas parquet
        n_neighbors: UMAP neighbour count — larger = more global structure
        min_dist: UMAP min_dist — smaller = tighter clusters
        min_cluster_size: HDBSCAN min_cluster_size
        random_state: UMAP random_state
        max_points: subsample to this many points before UMAP to keep it
            fast. Tiles kept at random; the rest get NaN coords and cluster
            -1 so the UI can still show "atlas not computed for this tile".
    """
    embeddings_path = Path(embeddings_path)
    metadata_path = Path(metadata_path)
    out_path = Path(out_path)

    logger.info("Loading embeddings from %s", embeddings_path)
    emb_df = pd.read_parquet(embeddings_path)
    emb_df = emb_df.drop_duplicates(subset="image_path", keep="first").reset_index(drop=True)
    vectors_all = np.vstack(emb_df["embedding"].values).astype(np.float32)
    vectors_all = _normalise(vectors_all)

    n_total = vectors_all.shape[0]
    if max_points is not None and n_total > max_points:
        rng = np.random.default_rng(random_state)
        keep_idx = rng.choice(n_total, max_points, replace=False)
        keep_idx.sort()
        vectors = vectors_all[keep_idx]
        logger.info("Subsampled %d → %d rows for UMAP", n_total, max_points)
    else:
        keep_idx = np.arange(n_total, dtype=np.int64)
        vectors = vectors_all

    from umap import UMAP

    logger.info(
        "Running UMAP (n=%d dim=%d, n_neighbors=%d, min_dist=%g)",
        vectors.shape[0],
        vectors.shape[1],
        n_neighbors,
        min_dist,
    )
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
        verbose=True,
    )
    coords = reducer.fit_transform(vectors).astype(np.float32)

    import hdbscan

    logger.info("Running HDBSCAN (min_cluster_size=%d)", min_cluster_size)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=-1)
    cluster_ids = clusterer.fit_predict(coords).astype(np.int32)

    # Expand back to the full tile list so every tile has a row
    x = np.full(n_total, np.nan, dtype=np.float32)
    y = np.full(n_total, np.nan, dtype=np.float32)
    clusters = np.full(n_total, -1, dtype=np.int32)
    x[keep_idx] = coords[:, 0]
    y[keep_idx] = coords[:, 1]
    clusters[keep_idx] = cluster_ids

    out = pd.DataFrame(
        {
            "image_path": emb_df["image_path"].astype(str),
            "atlas_x": x,
            "atlas_y": y,
            "cluster_id": clusters,
        }
    )

    md = pd.read_parquet(metadata_path)
    md_cols = [
        c
        for c in ("image_path", "tile_scale", "approx_lat", "approx_lon", "product_id")
        if c in md.columns
    ]
    md_slim = (
        md[md_cols]
        .drop_duplicates(subset="image_path", keep="first")
        .reset_index(drop=True)
    )
    out = out.merge(md_slim, on="image_path", how="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    n_clusters = int(clusters.max()) + 1 if (clusters >= 0).any() else 0
    logger.info(
        "Wrote atlas to %s: %d tiles, %d clusters, %d noise",
        out_path,
        n_total,
        n_clusters,
        int((clusters == -1).sum()),
    )
    return out_path


def load_atlas(atlas_path: Path) -> pd.DataFrame:
    return pd.read_parquet(atlas_path)
