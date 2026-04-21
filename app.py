"""HuggingFace Spaces entrypoint for the CTX similarity viewer.

Pulls the index (FAISS + metadata + optional atlas/anomaly parquets) from a
companion HF Dataset on cold start, then launches the same FastAPI app used
for local development. Exposes port 7860 by default (HF Spaces convention).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _fetch_index() -> Path:
    """Download the CTX index parquets + FAISS from a companion HF dataset."""
    repo_id = os.environ.get("INDEX_REPO_ID")
    local_dir = Path(os.environ.get("INDEX_LOCAL_DIR", "/data/index"))
    if repo_id and not (local_dir / "faiss.index").exists():
        logger.info("Downloading index from HF dataset %s → %s", repo_id, local_dir)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=[
                "faiss.index",
                "metadata.parquet",
                "embeddings.parquet",
                "faiss.model.json",
                "anomaly_scores.parquet",
                "atlas.parquet",
            ],
        )
    if not local_dir.exists():
        raise RuntimeError(
            f"No index at {local_dir} and INDEX_REPO_ID not set. "
            "Point INDEX_REPO_ID at an HF Dataset containing faiss.index + "
            "metadata.parquet."
        )
    return local_dir


def main() -> None:
    index_dir = _fetch_index()
    source_dir = Path(os.environ.get("SOURCE_DIR", "/data/source"))
    source_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import so the Dockerfile layer order stays small.
    from scripts.ctx_viewer_server import _load_resources, app

    _load_resources(index_dir=index_dir, source_dir=source_dir)

    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info("Starting viewer on %s:%d (index=%s)", host, port, index_dir)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
