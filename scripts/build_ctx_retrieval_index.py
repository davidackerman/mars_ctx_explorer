#!/usr/bin/env python3
"""Build a CTX image similarity index (embeddings + FAISS + metadata)."""

import argparse
import logging
from pathlib import Path

from scientific_pipelines.planetary.mars.ctx.retrieval import (
    CTXSimilarityIndex,
    build_ctx_embeddings,
    discover_images,
    generate_chunk_tiles,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FAISS similarity index for CTX images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/raw/ctx"),
        help="Directory containing CTX images",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("outputs/ctx_similarity"),
        help="Directory where embeddings and index artifacts will be stored",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Path to existing embeddings parquet (skip embedding extraction if provided)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional CTX manifest.json path for metadata enrichment (auto-detected if omitted)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dinov3_vitb14",
        help="Embedding backbone model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding extraction batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of image loading workers",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive image search",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Use L2 distance index instead of cosine similarity",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume mode for embedding extraction",
    )
    parser.add_argument(
        "--use-half",
        action="store_true",
        help="Enable fp16 inference for faster CUDA extraction",
    )
    parser.add_argument(
        "--chunkwise",
        action="store_true",
        help="Tile large images into chunk PNGs before embedding",
    )
    parser.add_argument(
        "--use-existing-tiles",
        action="store_true",
        help="Skip tile generation and re-embed the PNGs already in index-dir/tiles/",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Single-scale chunk tile size (ignored if --tile-scales is set)",
    )
    parser.add_argument(
        "--tile-stride",
        type=int,
        default=1024,
        help="Single-scale chunk tile stride (ignored if --tile-scales is set)",
    )
    parser.add_argument(
        "--tile-scales",
        type=str,
        default=None,
        help="Comma-separated tile sizes for multi-scale indexing (e.g. '256,512,1024')",
    )
    parser.add_argument(
        "--tile-stride-fraction",
        type=float,
        default=0.5,
        help="Stride fraction per scale (multi-scale only). 0.5 = half-tile overlap",
    )
    parser.add_argument(
        "--tile-min-std",
        type=float,
        default=5.0,
        help="Minimum grayscale stddev to keep tile (used with --chunkwise)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=518,
        help="Square input edge fed to the DINO backbone (multiple of patch size)",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="flat",
        choices=["flat", "ivfpq"],
        help=(
            "FAISS index type: 'flat' for exact search (small corpora), "
            "'ivfpq' for compressed index (~100x smaller, slight recall loss; "
            "required at global scale)"
        ),
    )
    parser.add_argument(
        "--pq-bytes",
        type=int,
        default=64,
        help="Product-quantization bytes per vector in ivfpq mode (must divide the embedding dim)",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=None,
        help="IVF coarse centroid count; default ~sqrt(N)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.index_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = args.embeddings or (args.index_dir / "embeddings.parquet")
    index_path = args.index_dir / "faiss.index"
    metadata_path = args.index_dir / "metadata.parquet"

    if args.embeddings is None:
        image_paths = discover_images(args.image_dir, recursive=not args.no_recursive)
        if len(image_paths) == 0:
            raise ValueError(f"No supported images found in {args.image_dir}")

        embedding_input_paths = image_paths
        if args.use_existing_tiles:
            chunk_dir = args.index_dir / "tiles"
            if not chunk_dir.exists():
                raise FileNotFoundError(f"--use-existing-tiles: no tiles dir at {chunk_dir}")
            embedding_input_paths = sorted(chunk_dir.glob("*.png"))
            if not embedding_input_paths:
                raise ValueError(f"--use-existing-tiles: zero PNGs under {chunk_dir}")
            logger.info(f"Reusing {len(embedding_input_paths)} existing tiles from {chunk_dir}")
        elif args.chunkwise:
            chunk_dir = args.index_dir / "tiles"
            tile_scales = (
                [int(s) for s in args.tile_scales.split(",") if s.strip()]
                if args.tile_scales
                else None
            )
            logger.info(
                "Generating chunk tiles before embedding "
                f"(scales={tile_scales if tile_scales else [args.tile_size]}, "
                f"stride_fraction={args.tile_stride_fraction}, min_std={args.tile_min_std})"
            )
            embedding_input_paths = generate_chunk_tiles(
                image_paths=image_paths,
                tile_output_dir=chunk_dir,
                tile_scales=tile_scales,
                stride_fraction=args.tile_stride_fraction,
                tile_size=args.tile_size,
                stride=args.tile_stride,
                min_std=args.tile_min_std,
            )
            if len(embedding_input_paths) == 0:
                raise ValueError("Chunkwise tiling produced zero tiles; adjust tile parameters")
            logger.info(f"Using {len(embedding_input_paths)} chunk tiles for embedding extraction")

        logger.info("Starting embedding extraction")
        build_ctx_embeddings(
            image_paths=embedding_input_paths,
            output_path=embeddings_path,
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_half_precision=args.use_half,
            resume=not args.no_resume,
            image_size=args.image_size,
        )
    else:
        logger.info(f"Using existing embeddings file: {embeddings_path}")

    logger.info("Building FAISS similarity index")
    CTXSimilarityIndex.build_from_embeddings(
        embeddings_path=embeddings_path,
        index_path=index_path,
        metadata_path=metadata_path,
        manifest_path=args.manifest,
        normalize=not args.no_normalize,
        model_name=args.model_name,
        image_size=args.image_size,
        index_type=args.index_type,
        pq_bytes=args.pq_bytes,
        nlist=args.nlist,
    )

    logger.info("Index build complete")
    logger.info(f"Embeddings: {embeddings_path}")
    logger.info(f"Index: {index_path}")
    logger.info(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
