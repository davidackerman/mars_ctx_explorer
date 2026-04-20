"""HuggingFace-hosted DINOv3 extractor (supports LVD-1689M and SAT-493M variants)."""

import logging
import os
from typing import Optional

import numpy as np
import torch

from .base import EmbeddingExtractor

logger = logging.getLogger(__name__)


class DINOv3HFExtractor(EmbeddingExtractor):
    """DINOv3 via HuggingFace transformers.

    Unlike the original DINOv3Extractor (torch.hub), this variant uses the
    HF-hosted safetensors weights. That matters because:
      * torch.hub weights (dl.fbaipublicfiles.com) are license-gated and return
        403 without Meta approval; HF variants are gated per-repo but behind a
        clickthrough license the user accepts in their HF account.
      * The satellite-pretrained SAT-493M variant (ViT-L/16 or ViT-7B/16) is
        only distributed via HF and is the far better fit for orbital imagery
        than the natural-image LVD-1689M pretraining.

    Supported repo IDs (output dim):
      - facebook/dinov3-vitb16-pretrain-lvd1689m     (768)
      - facebook/dinov3-vitl16-pretrain-lvd1689m     (1024)
      - facebook/dinov3-vitl16-pretrain-sat493m      (1024)  [satellite]
      - facebook/dinov3-vit7b16-pretrain-sat493m     (4096)  [satellite, huge]

    Returns the pooled CLS token embedding per image.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-sat493m",
        device: str = "cuda",
        use_half_precision: bool = False,
        cache_dir: Optional[str] = None,
    ):
        if cache_dir is None:
            cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")

        from transformers import AutoModel

        logger.info(f"Loading DINOv3 (HF): {model_name}")
        self.model_name = model_name
        self.device = device
        self.use_half_precision = use_half_precision and device == "cuda"
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = self.model.to(device)
        self.model.eval()
        if self.use_half_precision:
            self.model = self.model.half()

        self._embedding_dim = int(self.model.config.hidden_size)
        logger.info(f"DINOv3 HF loaded: dim={self._embedding_dim}")

    def extract(self, images: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            images = images.to(self.device)
            if self.use_half_precision:
                images = images.half()
            output = self.model(images)
            features = output.pooler_output
            if self.use_half_precision:
                features = features.float()
        return features.cpu().numpy()

    def extract_patches(self, images: torch.Tensor) -> np.ndarray:
        """Return per-patch embeddings, dropping CLS + register tokens.

        Layout of last_hidden_state: [CLS] + [R register tokens] + [P patch tokens].
        Returns shape (B, P, D) of patch embeddings only.
        """
        with torch.no_grad():
            images = images.to(self.device)
            if self.use_half_precision:
                images = images.half()
            output = self.model(images)
            hidden = output.last_hidden_state  # (B, 1 + R + P, D)
            num_skip = 1 + int(self.model.config.num_register_tokens)
            patches = hidden[:, num_skip:, :]
            if self.use_half_precision:
                patches = patches.float()
        return patches.cpu().numpy()

    def get_embedding_dim(self) -> int:
        return self._embedding_dim

    @staticmethod
    def get_default_transforms(image_size: int = 512):
        """Transforms compatible with DINOv3 ViT-16 (image_size multiple of 16)."""
        from torchvision import transforms as T

        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
