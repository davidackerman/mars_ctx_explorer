#!/usr/bin/env python3
"""Build a patch-level CTX similarity index from existing tile PNGs."""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from ctx_explorer.embeddings import DINOv3HFExtractor
from ctx_explorer.patch_retrieval import CTXPatchIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FAISS patch-level similarity index for CTX tiles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("outputs/ctx_similarity"),
        help="Directory holding the existing tiles/ subdir and where patch index is written",
    )
    parser.add_argument(
        "--tile-metadata",
        type=Path,
        default=None,
        help="Optional metadata.parquet to merge into tile-level metadata (defaults to {index-dir}/metadata.parquet if present)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/dinov3-vitl16-pretrain-sat493m",
        help="HuggingFace repo id for DINOv3 weights",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square input size fed to the backbone (multiple of patch size)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Tiles per forward pass"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-half", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", "/mnt/bigdisk/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/mnt/bigdisk/hf_cache/hub")

    tile_dir = args.index_dir / "tiles"
    if not tile_dir.exists():
        raise FileNotFoundError(f"No tile directory at {tile_dir}")
    tile_paths = sorted(tile_dir.glob("*.png"))
    if not tile_paths:
        raise ValueError(f"No tile PNGs found under {tile_dir}")
    logger.info(f"Found {len(tile_paths)} tiles under {tile_dir}")

    extractor = DINOv3HFExtractor(
        model_name=args.model_name,
        device=args.device,
        use_half_precision=args.use_half,
    )
    transform = DINOv3HFExtractor.get_default_transforms(image_size=args.image_size)

    extra_meta_path = args.tile_metadata or (args.index_dir / "metadata.parquet")
    extra_meta = None
    if extra_meta_path.exists():
        extra_meta = pd.read_parquet(extra_meta_path).drop_duplicates(
            subset="image_path", keep="first"
        )
        logger.info(
            f"Joining tile-level metadata from {extra_meta_path} "
            f"({len(extra_meta)} rows, columns={list(extra_meta.columns)})"
        )

    CTXPatchIndex.build_from_tiles(
        tile_paths=tile_paths,
        extractor=extractor,
        transform=transform,
        index_dir=args.index_dir,
        batch_size=args.batch_size,
        device=args.device,
        normalize=True,
        model_name=args.model_name,
        image_size=args.image_size,
        extra_tile_metadata=extra_meta,
    )


if __name__ == "__main__":
    main()
