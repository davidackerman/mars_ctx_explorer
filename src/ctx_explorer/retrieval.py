"""Similarity retrieval utilities for CTX image search."""

import logging
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from scientific_pipelines.core.embeddings import DINOv3Extractor, EmbeddingPipeline

logger = logging.getLogger(__name__)


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _extract_sol_from_path(path_str: str):
    """Extract sol from path/filename if present."""
    path = Path(path_str)

    # Pattern 1: parent folder like sol_1613
    for parent in path.parents:
        match = re.match(r"sol_(\d{1,5})$", parent.name.lower())
        if match:
            return int(match.group(1))

    # Pattern 2: filename token like _1613_
    match = re.search(r"_(\d{3,5})_", path.name)
    if match:
        return int(match.group(1))

    return None


def discover_images(image_dir: Path, recursive: bool = True) -> List[Path]:
    """Discover image files in a directory.

    Args:
        image_dir: Root directory containing image files
        recursive: Whether to search recursively

    Returns:
        Sorted list of image paths
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if recursive:
        candidates = image_dir.rglob("*")
    else:
        candidates = image_dir.glob("*")

    image_paths = [
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]

    image_paths = sorted(image_paths)
    logger.info(f"Discovered {len(image_paths)} images in {image_dir}")
    return image_paths


def build_ctx_embeddings(
    image_paths: List[Path],
    output_path: Path,
    model_name: str = "dinov3_vitb14",
    device: str = "cuda",
    batch_size: int = 128,
    num_workers: int = 8,
    use_half_precision: bool = False,
    resume: bool = True,
) -> Path:
    """Build DINO embeddings for CTX images and save to parquet.

    Args:
        image_paths: List of image paths to encode
        output_path: Output parquet path
        model_name: DINO model variant
        device: Inference device ('cuda' or 'cpu')
        batch_size: Embedding batch size
        num_workers: Dataloader workers
        use_half_precision: Enable fp16 inference on CUDA
        resume: Resume from existing output

    Returns:
        Path to the generated embeddings parquet
    """
    if len(image_paths) == 0:
        raise ValueError("No images provided for embedding extraction")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = DINOv3Extractor(
        model_name=model_name,
        device=device,
        use_half_precision=use_half_precision,
    )

    pipeline = EmbeddingPipeline(
        extractor=extractor,
        output_format="parquet",
        batch_size=batch_size,
        num_workers=num_workers,
        transform=DINOv3Extractor.get_default_transforms(),
    )

    embeddings, metadata = pipeline.extract_dataset(
        image_paths=image_paths,
        output_path=output_path,
        resume=resume,
    )

    logger.info(
        f"Saved embeddings to {output_path} (n={embeddings.shape[0]}, dim={embeddings.shape[1]})"
    )
    logger.info(f"Embedding metadata rows: {len(metadata)}")
    return output_path


class CTXSimilarityIndex:
    """Persistent FAISS-backed similarity index for CTX images."""

    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        normalize: bool = True,
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.normalize = normalize

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        import faiss

        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_parquet(self.metadata_path)

        if "row_id" not in self.metadata.columns:
            raise ValueError("Metadata must contain 'row_id' column")
        if "image_path" not in self.metadata.columns:
            raise ValueError("Metadata must contain 'image_path' column")

        self.metadata = self.metadata.sort_values("row_id").reset_index(drop=True)

    @classmethod
    def build_from_embeddings(
        cls,
        embeddings_path: Path,
        index_path: Path,
        metadata_path: Path,
        normalize: bool = True,
    ) -> "CTXSimilarityIndex":
        """Build and persist FAISS index artifacts from embeddings parquet."""
        embeddings_path = Path(embeddings_path)
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

        logger.info(f"Loading embeddings from {embeddings_path}")
        df = pd.read_parquet(embeddings_path)

        if "embedding" not in df.columns or "image_path" not in df.columns:
            raise ValueError("Embeddings parquet must contain columns: image_path, embedding")

        vectors = np.vstack(df["embedding"].values).astype("float32")
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("Embeddings are empty or malformed")

        import faiss

        if normalize:
            faiss.normalize_L2(vectors)
            index = faiss.IndexFlatIP(vectors.shape[1])
        else:
            index = faiss.IndexFlatL2(vectors.shape[1])

        index.add(vectors)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(index_path))

        metadata = df.drop(columns=["embedding"]).copy()
        metadata["row_id"] = np.arange(len(metadata), dtype=np.int64)
        metadata["filename"] = metadata["image_path"].map(lambda value: Path(str(value)).name)
        metadata["parent_dir"] = metadata["image_path"].map(
            lambda value: Path(str(value)).parent.name
        )
        metadata["sol"] = metadata["image_path"].map(lambda value: _extract_sol_from_path(str(value)))
        metadata.to_parquet(metadata_path, index=False)

        logger.info(
            f"Built similarity index at {index_path} with {index.ntotal} vectors "
            f"(dim={vectors.shape[1]})"
        )
        logger.info(f"Saved metadata to {metadata_path}")

        return cls(index_path=index_path, metadata_path=metadata_path, normalize=normalize)

    def query_by_row_id(self, row_id: int, k: int = 12, include_self: bool = False) -> pd.DataFrame:
        """Query top-k nearest images using an indexed row id as the anchor."""
        if row_id < 0 or row_id >= len(self.metadata):
            raise IndexError(f"row_id out of range: {row_id}")

        query_vector = self.index.reconstruct(int(row_id)).reshape(1, -1).astype("float32")
        distances, indices = self.index.search(query_vector, k + 1 if not include_self else k)

        hit_indices = indices[0]
        hit_distances = distances[0]

        results = self.metadata.iloc[hit_indices].copy()
        score_column = "similarity" if self.normalize else "distance"
        results[score_column] = hit_distances

        if not include_self:
            results = results[results["row_id"] != row_id].head(k)

        return results.reset_index(drop=True)
