"""Batch embedding extraction pipeline with disk caching."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .base import EmbeddingExtractor

logger = logging.getLogger(__name__)


class ImagePathDataset(Dataset):
    """Simple dataset for loading images from file paths."""

    def __init__(self, image_paths: List[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, str(image_path)


class EmbeddingPipeline:
    """
    Batch embedding extraction with disk caching and resume capability.

    Features:
    - Parallel data loading with DataLoader
    - Progress tracking with tqdm
    - Resume capability (skip already processed images)
    - Output formats: Parquet (recommended) or NPY

    Args:
        extractor: EmbeddingExtractor instance
        output_format: Output format ('parquet' or 'npy')
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
        transform: Image preprocessing transform (if None, uses extractor default)
    """

    def __init__(
        self,
        extractor: EmbeddingExtractor,
        output_format: str = "parquet",
        batch_size: int = 256,
        num_workers: int = 8,
        transform=None,
    ):
        if output_format not in ["parquet", "npy"]:
            raise ValueError("output_format must be 'parquet' or 'npy'")

        self.extractor = extractor
        self.output_format = output_format
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

        logger.info(
            f"EmbeddingPipeline initialized: "
            f"format={output_format}, batch_size={batch_size}, "
            f"num_workers={num_workers}"
        )

    def extract_dataset(
        self,
        image_paths: List[Path],
        output_path: Path,
        resume: bool = True,
        save_frequency: int = 1000,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Extract embeddings for entire dataset.

        Args:
            image_paths: List of paths to images
            output_path: Path to save embeddings (with .parquet or .npy extension)
            resume: If True, skip images that are already in output file
            save_frequency: Save intermediate results every N images (for resume)

        Returns:
            embeddings: (N, D) array of embeddings
            metadata: DataFrame with columns [image_path, embedding_index, ...]
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check for existing results if resume=True
        processed_paths = set()
        if resume and output_path.exists():
            logger.info(f"Resume mode: loading existing embeddings from {output_path}")
            if self.output_format == "parquet":
                existing_df = pd.read_parquet(output_path)
                processed_paths = set(existing_df["image_path"].values)
            else:  # npy
                metadata_path = output_path.parent / f"{output_path.stem}_metadata.csv"
                if metadata_path.exists():
                    existing_df = pd.read_csv(metadata_path)
                    processed_paths = set(existing_df["image_path"].values)

            logger.info(f"Found {len(processed_paths)} already processed images")

        # Filter out already processed images
        remaining_paths = [p for p in image_paths if str(p) not in processed_paths]
        logger.info(
            f"Processing {len(remaining_paths)} images "
            f"({len(processed_paths)} already done)"
        )

        if len(remaining_paths) == 0:
            logger.info("All images already processed!")
            return self._load_results(output_path)

        # Create dataset and dataloader
        dataset = ImagePathDataset(remaining_paths, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        # Extract embeddings
        all_embeddings = []
        all_paths = []

        for batch_images, batch_paths in tqdm(
            dataloader, desc="Extracting embeddings", total=len(dataloader)
        ):
            # Extract batch embeddings
            embeddings = self.extractor.extract(batch_images)

            all_embeddings.append(embeddings)
            all_paths.extend(batch_paths)

            # Save intermediate results
            if len(all_paths) % save_frequency == 0:
                logger.info(f"Saving intermediate results ({len(all_paths)} processed)")
                self._save_results(
                    output_path, all_embeddings, all_paths, append=resume
                )

        # Final save
        logger.info(f"Saving final results to {output_path}")
        embeddings_array = (
            np.vstack(all_embeddings) if all_embeddings else np.array([])
        )
        self._save_results(
            output_path, [embeddings_array], all_paths, append=resume, final=True
        )

        return self._load_results(output_path)

    def _save_results(
        self,
        output_path: Path,
        embeddings_list: List[np.ndarray],
        paths: List[str],
        append: bool = False,
        final: bool = False,
    ):
        """Save embeddings to disk."""
        embeddings = np.vstack(embeddings_list)

        if self.output_format == "parquet":
            # Save as Parquet with metadata
            df = pd.DataFrame(
                {
                    "image_path": paths,
                    "embedding": list(embeddings),
                    "embedding_dim": embeddings.shape[1],
                }
            )

            if append and output_path.exists():
                # Append to existing file
                existing_df = pd.read_parquet(output_path)
                df = pd.concat([existing_df, df], ignore_index=True)

            df.to_parquet(output_path, index=False)

        else:  # npy
            # Save embeddings as .npy and metadata as .csv
            if append and output_path.exists():
                existing_embeddings = np.load(output_path)
                embeddings = np.vstack([existing_embeddings, embeddings])

            np.save(output_path, embeddings)

            # Save metadata
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.csv"
            df = pd.DataFrame(
                {
                    "image_path": paths,
                    "embedding_index": list(range(len(paths))),
                }
            )

            if append and metadata_path.exists():
                existing_df = pd.read_csv(metadata_path)
                df["embedding_index"] += len(existing_df)
                df = pd.concat([existing_df, df], ignore_index=True)

            df.to_csv(metadata_path, index=False)

    def _load_results(self, output_path: Path) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load embeddings from disk."""
        if self.output_format == "parquet":
            df = pd.read_parquet(output_path)
            embeddings = np.vstack(df["embedding"].values)
            metadata = df.drop(columns=["embedding"])
        else:  # npy
            embeddings = np.load(output_path)
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.csv"
            metadata = pd.read_csv(metadata_path)

        return embeddings, metadata
