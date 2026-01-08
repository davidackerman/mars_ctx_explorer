"""DINOv3 Vision Transformer embedding extractor."""

import logging
from typing import Optional

import numpy as np
import torch

from .base import EmbeddingExtractor

logger = logging.getLogger(__name__)


class DINOv3Extractor(EmbeddingExtractor):
    """
    DINOv3 Vision Transformer for embedding extraction.

    DINOv3 is a self-supervised ViT trained on ImageNet-22k and additional curated
    datasets. It produces high-quality visual embeddings suitable for unsupervised
    clustering and retrieval tasks.

    This class supports both DINOv2 and DINOv3 models, with DINOv3 as the recommended
    default for best performance.

    Supported model variants:
    - dinov3_vits14: Small (21M params, 384-dim embeddings)
    - dinov3_vitb14: Base (86M params, 768-dim embeddings) [RECOMMENDED]
    - dinov3_vitl14: Large (300M params, 1024-dim embeddings)
    - dinov3_vitg14: Giant (1.1B params, 1536-dim embeddings)

    Also supports legacy DINOv2 models:
    - dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14

    Args:
        model_name: DINO model variant name (dinov3_* or dinov2_*)
        device: Device to run inference on ('cuda' or 'cpu')
        use_half_precision: Use FP16 for faster inference (requires CUDA)
        repo: Torch hub repository ('facebookresearch/dinov3' or 'facebookresearch/dinov2')
    """

    MODEL_DIMS = {
        # DINOv2 models
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
        # DINOv3 models (same dimensions)
        "dinov3_vits14": 384,
        "dinov3_vitb14": 768,
        "dinov3_vitl14": 1024,
        "dinov3_vitg14": 1536,
    }

    def __init__(
        self,
        model_name: str = "dinov3_vitb14",
        device: str = "cuda",
        use_half_precision: bool = False,
        repo: Optional[str] = None,
    ):
        if model_name not in self.MODEL_DIMS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Supported models: {list(self.MODEL_DIMS.keys())}"
            )

        self.model_name = model_name
        self.device = device
        self.use_half_precision = use_half_precision and device == "cuda"

        # Auto-detect repository if not specified
        if repo is None:
            if model_name.startswith("dinov3"):
                repo = "facebookresearch/dinov3"
            else:
                repo = "facebookresearch/dinov2"

        self.repo = repo

        logger.info(f"Loading DINO model: {model_name} from {repo}")

        # Load model from torch.hub
        self.model = torch.hub.load(repo, model_name)
        self.model = self.model.to(device)
        self.model.eval()

        if self.use_half_precision:
            logger.info("Using FP16 half precision")
            self.model = self.model.half()

        logger.info(
            f"DINO {model_name} loaded successfully "
            f"(embedding dim: {self.get_embedding_dim()})"
        )

    def extract(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract [CLS] token embeddings from images.

        Args:
            images: Tensor of shape (B, C, H, W) with normalized images.
                   Expected normalization: ImageNet stats
                   (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        Returns:
            Array of shape (B, D) containing embedding vectors
        """
        with torch.no_grad():
            # Move to device and optionally convert to half precision
            images = images.to(self.device)
            if self.use_half_precision:
                images = images.half()

            # Extract features (returns [CLS] token by default)
            features = self.model(images)

            # Convert back to float32 and move to CPU
            if self.use_half_precision:
                features = features.float()

        return features.cpu().numpy()

    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality for this model variant."""
        return self.MODEL_DIMS[self.model_name]

    @staticmethod
    def get_default_transforms():
        """
        Get default image preprocessing transforms for DINOv3.

        Returns:
            torchvision.transforms composition for image preprocessing
        """
        from torchvision import transforms as T

        return T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
