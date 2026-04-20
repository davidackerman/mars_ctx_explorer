"""Similarity retrieval utilities for CTX image search."""

import logging
import re
from pathlib import Path
from typing import List, Optional

import json

import numpy as np
import pandas as pd

from scientific_pipelines.core.embeddings import (
    DINOv3Extractor,
    DINOv3HFExtractor,
    EmbeddingPipeline,
)

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


def _extract_product_id(path_str: str) -> str:
    """Extract product identifier from image filename stem."""
    stem = Path(path_str).stem
    scaled_tile_match = re.match(
        r"^(?P<product>.+?)_s\d{4}_tile_\d+_\d+$", stem
    )
    if scaled_tile_match:
        return scaled_tile_match.group("product")
    tile_match = re.match(r"^(?P<product>.+?)_tile_\d+_\d+$", stem)
    if tile_match:
        return tile_match.group("product")
    return stem


def _load_grayscale_stretched(image_path: Path) -> "np.ndarray":
    """Load an image as an 8-bit grayscale numpy array, stretching 16-bit input.

    ISIS3 isis2std writes 16-bit TIFFs (U16BIT) whose radiance values span far
    beyond 0-255. Naively calling PIL's `convert("L")` clips everything >=255
    to white, destroying the signal. This helper reads the raw pixel values and
    applies a per-image linear percentile stretch before quantizing to 8-bit.
    NoData (value 0) stays 0.
    """
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(str(image_path))
    if ds is None:
        raise IOError(f"GDAL could not open {image_path}")
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()

    if arr.ndim == 3:
        arr = arr[0]

    if arr.dtype == np.uint8:
        return arr.copy()

    valid_mask = arr > 0
    if not valid_mask.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    valid_values = arr[valid_mask]
    lo = float(np.percentile(valid_values, 1.0))
    hi = float(np.percentile(valid_values, 99.0))
    if hi <= lo:
        hi = float(valid_values.max())
        lo = float(valid_values.min())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    stretched = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    stretched[~valid_mask] = 0.0
    return (stretched * 255.0 + 0.5).astype(np.uint8)


TILE_FILENAME_RE = re.compile(
    r"^(?P<product>.+?)_s(?P<scale>\d{4})_tile_(?P<y>\d{6})_(?P<x>\d{6})$"
)


def _tile_scale_from_path(path_str: str) -> Optional[int]:
    """Extract tile scale (side-length px) from tile filename stem."""
    stem = Path(path_str).stem
    match = TILE_FILENAME_RE.match(stem)
    if match is None:
        return None
    return int(match.group("scale"))


def generate_chunk_tiles(
    image_paths: List[Path],
    tile_output_dir: Path,
    tile_scales: Optional[List[int]] = None,
    stride_fraction: float = 0.5,
    tile_size: int = 1024,
    stride: Optional[int] = None,
    min_std: float = 5.0,
    max_nodata_frac: float = 0.1,
) -> List[Path]:
    """Generate chunked PNG tiles from large source images for retrieval.

    Args:
        image_paths: Source image paths
        tile_output_dir: Directory where tile PNG files are written
        tile_scales: List of tile side lengths in pixels (enables multi-scale
            indexing). Each tile gets a filename encoding its scale so the
            retrieval index can filter queries by scale.
        stride_fraction: Stride as a fraction of tile size for each scale
            (default 0.5 = half-tile overlap)
        tile_size: Single tile side length (only used when tile_scales is None)
        stride: Single sliding-window stride (only used when tile_scales is
            None; defaults to tile_size — non-overlapping)
        min_std: Minimum per-tile grayscale stddev to keep tile
        max_nodata_frac: Drop tiles whose fraction of zero pixels exceeds this
    """
    if tile_scales is None:
        if tile_size <= 0:
            raise ValueError("tile_size must be > 0")
        effective_stride = stride if stride is not None else tile_size
        if effective_stride <= 0:
            raise ValueError("stride must be > 0")
        scale_stride_pairs = [(tile_size, effective_stride)]
    else:
        if not tile_scales:
            raise ValueError("tile_scales must be non-empty when provided")
        scale_stride_pairs = [
            (int(size), max(1, int(round(size * stride_fraction))))
            for size in tile_scales
        ]

    from PIL import Image

    tile_output_dir = Path(tile_output_dir)
    tile_output_dir.mkdir(parents=True, exist_ok=True)

    tile_paths: List[Path] = []
    total_candidates = 0

    for image_path in image_paths:
        try:
            full_array = _load_grayscale_stretched(Path(image_path))
            height, width = full_array.shape

            for scale_px, scale_stride in scale_stride_pairs:
                max_x = width - scale_px
                max_y = height - scale_px
                if max_x < 0 or max_y < 0:
                    continue

                for y_offset in range(0, max_y + 1, scale_stride):
                    for x_offset in range(0, max_x + 1, scale_stride):
                        total_candidates += 1
                        tile_array = full_array[
                            y_offset : y_offset + scale_px,
                            x_offset : x_offset + scale_px,
                        ]

                        nodata_frac = float((tile_array == 0).mean())
                        if nodata_frac > max_nodata_frac:
                            continue

                        content_mask = tile_array > 0
                        if not content_mask.any():
                            continue
                        content_std = float(tile_array[content_mask].std())
                        if content_std < min_std:
                            continue

                        tile_name = (
                            f"{image_path.stem}"
                            f"_s{scale_px:04d}"
                            f"_tile_{y_offset:06d}_{x_offset:06d}.png"
                        )
                        tile_path = tile_output_dir / tile_name
                        Image.fromarray(tile_array, mode="L").save(
                            tile_path, format="PNG"
                        )
                        tile_paths.append(tile_path)

        except Exception as exc:
            logger.warning(f"Failed to tile {image_path}: {exc}")

    logger.info(
        f"Generated {len(tile_paths)} tiles from {len(image_paths)} images "
        f"(scales={[p[0] for p in scale_stride_pairs]}, "
        f"candidates={total_candidates}, min_std={min_std}, "
        f"max_nodata_frac={max_nodata_frac})"
    )
    return tile_paths


def _infer_manifest_path_from_images(image_paths: pd.Series) -> Optional[Path]:
    """Infer CTX manifest path by checking image parent dirs for manifest.json."""
    for path_str in image_paths.astype(str).tolist():
        path = Path(path_str)
        direct_candidate = path.parent / "manifest.json"
        if direct_candidate.exists():
            return direct_candidate

        for parent in path.parents:
            candidate = parent / "manifest.json"
            if candidate.exists():
                return candidate

    return None


def _load_ctx_manifest_dataframe(manifest_path: Path) -> pd.DataFrame:
    """Load CTX downloader manifest entries as a dataframe."""
    with open(manifest_path, "r") as manifest_file:
        manifest = json.load(manifest_file)

    downloaded_images = manifest.get("downloaded_images", {})
    rows = []
    for product_id, entry in downloaded_images.items():
        rows.append(
            {
                "product_id": product_id,
                "manifest_img_path": entry.get("img_path"),
                "center_lon": entry.get("center_lon"),
                "center_lat": entry.get("center_lat"),
                "emission_angle": entry.get("emission_angle"),
                "incidence_angle": entry.get("incidence_angle"),
                "solar_longitude": entry.get("solar_longitude"),
                "download_date": entry.get("download_date"),
                "file_size_mb": entry.get("file_size_mb"),
            }
        )

    return pd.DataFrame(rows)


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
    image_size: int = 518,
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

    is_hf = "/" in model_name
    if is_hf:
        extractor = DINOv3HFExtractor(
            model_name=model_name,
            device=device,
            use_half_precision=use_half_precision,
        )
        transform = DINOv3HFExtractor.get_default_transforms(image_size=image_size)
    else:
        extractor = DINOv3Extractor(
            model_name=model_name,
            device=device,
            use_half_precision=use_half_precision,
        )
        transform = DINOv3Extractor.get_default_transforms(image_size=image_size)

    pipeline = EmbeddingPipeline(
        extractor=extractor,
        output_format="parquet",
        batch_size=batch_size,
        num_workers=num_workers,
        transform=transform,
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
        manifest_path: Optional[Path] = None,
        normalize: bool = True,
        model_name: Optional[str] = None,
        image_size: Optional[int] = None,
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

        rows_before = len(df)
        df = df.drop_duplicates(subset="image_path", keep="first").reset_index(drop=True)
        if len(df) != rows_before:
            logger.warning(
                f"Dropped {rows_before - len(df)} duplicate embeddings "
                f"(kept {len(df)}); likely from intermediate-save double-writes"
            )

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
        metadata["product_id"] = metadata["image_path"].map(
            lambda value: _extract_product_id(str(value))
        )
        metadata["sol"] = metadata["image_path"].map(
            lambda value: _extract_sol_from_path(str(value))
        )
        metadata["tile_scale"] = metadata["image_path"].map(
            lambda value: _tile_scale_from_path(str(value))
        )

        selected_manifest = Path(manifest_path) if manifest_path is not None else None
        if selected_manifest is None:
            selected_manifest = _infer_manifest_path_from_images(metadata["image_path"])

        if selected_manifest is not None and selected_manifest.exists():
            logger.info(f"Joining manifest metadata from {selected_manifest}")
            manifest_df = _load_ctx_manifest_dataframe(selected_manifest)
            if len(manifest_df) > 0:
                metadata = metadata.merge(manifest_df, on="product_id", how="left")

                for column_name in [
                    "center_lon",
                    "center_lat",
                    "emission_angle",
                    "incidence_angle",
                    "solar_longitude",
                    "file_size_mb",
                ]:
                    if column_name in metadata.columns:
                        metadata[column_name] = pd.to_numeric(
                            metadata[column_name], errors="coerce"
                        )

                matched_rows = (
                    metadata["center_lon"].notna().sum() if "center_lon" in metadata.columns else 0
                )
                logger.info(f"Manifest metadata matched for {matched_rows}/{len(metadata)} images")
            else:
                logger.warning("Manifest found but contains no downloaded_images entries")
        else:
            logger.info("No CTX manifest found for metadata join; skipping geospatial enrichment")

        metadata.to_parquet(metadata_path, index=False)

        if model_name is not None or image_size is not None:
            sidecar = {
                "model_name": model_name,
                "image_size": image_size,
                "embedding_dim": int(vectors.shape[1]),
                "normalize": bool(normalize),
            }
            sidecar_path = index_path.with_suffix(".model.json")
            with open(sidecar_path, "w") as fp:
                json.dump(sidecar, fp, indent=2)
            logger.info(f"Wrote model sidecar to {sidecar_path}")

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

    def query_by_vector(
        self,
        query_vector: np.ndarray,
        k: int = 12,
        tile_scale: Optional[int] = None,
        candidate_multiplier: int = 8,
    ) -> pd.DataFrame:
        """Query top-k nearest images using a raw query embedding vector.

        When tile_scale is given, over-fetch candidates from FAISS and post-
        filter to only that scale before returning the top-k. FAISS has no
        native metadata filter so we rely on the multiplier to keep enough
        candidates after filtering.
        """
        import faiss

        vec = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        if self.normalize:
            faiss.normalize_L2(vec)

        score_column = "similarity" if self.normalize else "distance"
        if tile_scale is None or "tile_scale" not in self.metadata.columns:
            distances, indices = self.index.search(vec, k)
            results = self.metadata.iloc[indices[0]].copy()
            results[score_column] = distances[0]
            return results.reset_index(drop=True)

        candidate_k = max(k * candidate_multiplier, k)
        candidate_k = min(candidate_k, self.index.ntotal)
        distances, indices = self.index.search(vec, candidate_k)

        candidates = self.metadata.iloc[indices[0]].copy()
        candidates[score_column] = distances[0]
        filtered = candidates[candidates["tile_scale"] == int(tile_scale)]
        if len(filtered) == 0:
            filtered = candidates
        return filtered.head(k).reset_index(drop=True)
