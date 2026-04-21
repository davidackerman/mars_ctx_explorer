"""Anomaly scoring over indexed CTX tiles.

Given a CLS-embedding index, score each tile by how outlier-ish it is relative
to the rest of the corpus. Reuses the sklearn implementations; cosine / inner-
product space is handled by L2-normalising before scoring so Euclidean-based
estimators behave sensibly.

Typical use:

    from scientific_pipelines.planetary.mars.ctx.anomaly import (
        score_and_persist_anomaly,
        top_k_anomalies,
    )

    # One-time at build end:
    score_and_persist_anomaly(embeddings_path, metadata_path, out_path)

    # At query time:
    df = top_k_anomalies(out_path, k=20, tile_scale=512)
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


def compute_anomaly_scores(
    embeddings: np.ndarray,
    method: str = "lof",
    n_neighbors: int = 40,
    contamination: float = 0.05,
    random_state: int = 0,
) -> np.ndarray:
    """Return a per-row anomaly score where higher = more anomalous.

    - `lof`: sklearn LocalOutlierFactor (novelty=False). Score is
      ``-negative_outlier_factor_`` (so larger = more outlier).
    - `iforest`: sklearn IsolationForest. Score is
      ``-decision_function`` (so larger = more outlier).
    """
    vectors = _normalise(np.asarray(embeddings, dtype=np.float32))
    if method == "lof":
        from sklearn.neighbors import LocalOutlierFactor

        lof = LocalOutlierFactor(
            n_neighbors=min(n_neighbors, len(vectors) - 1),
            contamination=contamination,
            n_jobs=-1,
        )
        lof.fit_predict(vectors)
        return -lof.negative_outlier_factor_.astype(np.float32)
    elif method == "iforest":
        from sklearn.ensemble import IsolationForest

        iso = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        iso.fit(vectors)
        return -iso.decision_function(vectors).astype(np.float32)
    else:
        raise ValueError(f"Unknown anomaly method: {method}")


def score_and_persist_anomaly(
    embeddings_path: Path,
    metadata_path: Path,
    out_path: Path,
    method: str = "lof",
    n_neighbors: int = 40,
) -> Path:
    """Compute anomaly scores for every row in the embeddings parquet and save.

    Reads the CLS embeddings + tile metadata (for image_path / tile_scale /
    approx_lat / approx_lon), computes a score, and writes a parquet keyed
    by image_path with columns:
        image_path, tile_scale, approx_lat, approx_lon, anomaly_score.
    """
    embeddings_path = Path(embeddings_path)
    metadata_path = Path(metadata_path)
    out_path = Path(out_path)

    logger.info("Loading embeddings from %s", embeddings_path)
    emb_df = pd.read_parquet(embeddings_path)
    if "embedding" not in emb_df.columns or "image_path" not in emb_df.columns:
        raise ValueError("Embeddings parquet missing required columns")
    emb_df = emb_df.drop_duplicates(subset="image_path", keep="first").reset_index(drop=True)

    vectors = np.vstack(emb_df["embedding"].values).astype(np.float32)
    logger.info(
        "Scoring anomalies (method=%s, n_neighbors=%d, n=%d, dim=%d)",
        method,
        n_neighbors,
        vectors.shape[0],
        vectors.shape[1],
    )
    scores = compute_anomaly_scores(vectors, method=method, n_neighbors=n_neighbors)

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

    out = pd.DataFrame(
        {
            "image_path": emb_df["image_path"].astype(str),
            "anomaly_score": scores,
        }
    )
    out = out.merge(md_slim, on="image_path", how="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    logger.info(
        "Wrote anomaly scores to %s (min=%.3f, max=%.3f)",
        out_path,
        float(scores.min()),
        float(scores.max()),
    )
    return out_path


def top_k_anomalies(
    scores_path: Path,
    k: int = 20,
    tile_scale: Optional[int] = None,
    product_exclude: Optional[str] = None,
) -> pd.DataFrame:
    """Return the top-k highest anomaly-score rows as a DataFrame.

    Optionally filter to a specific `tile_scale` so "anomalies" are compared
    like-for-like (a dune at 256 vs a dune at 1024 is not directly comparable
    in this space).
    """
    df = pd.read_parquet(scores_path)
    if tile_scale is not None and "tile_scale" in df.columns:
        df = df[df["tile_scale"] == int(tile_scale)]
    if product_exclude is not None and "product_id" in df.columns:
        df = df[df["product_id"] != product_exclude]
    df = df.sort_values("anomaly_score", ascending=False).head(k).reset_index(drop=True)
    return df
