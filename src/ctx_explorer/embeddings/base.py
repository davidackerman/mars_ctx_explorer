"""Base class for embedding extractors."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


class EmbeddingExtractor(ABC):
    """
    Base class for all embedding extractors.

    Provides a unified interface for extracting embedding vectors from images
    using various models (DINOv2, SimCLR, etc.).
    """

    @abstractmethod
    def extract(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract embeddings from a batch of images.

        Args:
            images: Tensor of shape (B, C, H, W) containing normalized images

        Returns:
            Array of shape (B, D) containing embedding vectors, where D is the
            embedding dimensionality
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of the embedding space.

        Returns:
            Integer dimensionality (e.g., 768 for DINOv2-B, 2048 for ResNet50)
        """
        pass

    def extract_single(self, image: torch.Tensor) -> np.ndarray:
        """
        Extract embedding from a single image.

        Args:
            image: Tensor of shape (C, H, W) containing normalized image

        Returns:
            Array of shape (D,) containing embedding vector
        """
        # Add batch dimension
        images_batch = image.unsqueeze(0)
        embeddings = self.extract(images_batch)
        return embeddings[0]
