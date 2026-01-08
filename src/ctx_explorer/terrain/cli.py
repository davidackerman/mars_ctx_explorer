"""CLI scripts for CTX terrain classification pipeline."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from .pipeline import CTXTerrainPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def run_pipeline():
    """Main entry point for CTX terrain classification pipeline."""
    parser = argparse.ArgumentParser(
        description='CTX Terrain Classification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with config
  ctx-terrain-pipeline --config configs/pipelines/ctx_terrain.yaml

  # Run with custom image directory
  ctx-terrain-pipeline --config configs/pipelines/ctx_terrain.yaml --images data/raw/ctx/

  # Skip tiling (use existing tiles)
  ctx-terrain-pipeline --config configs/pipelines/ctx_terrain.yaml --skip-tiling

  # Skip embedding extraction (use existing embeddings)
  ctx-terrain-pipeline --config configs/pipelines/ctx_terrain.yaml --skip-embedding
        """,
    )

    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to pipeline configuration YAML file',
    )

    parser.add_argument(
        '--images',
        type=Path,
        help='Directory containing CTX images (overrides config)',
    )

    parser.add_argument(
        '--output',
        type=Path,
        help='Output directory (overrides config)',
    )

    parser.add_argument(
        '--skip-tiling',
        action='store_true',
        help='Skip tiling step (use existing tiles)',
    )

    parser.add_argument(
        '--skip-embedding',
        action='store_true',
        help='Skip embedding extraction (use existing embeddings)',
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override output directory if specified
    output_dir = args.output or Path(config.get('output_dir', 'outputs/ctx_terrain'))

    # Get image paths
    image_paths = None
    if not args.skip_tiling:
        if args.images:
            image_dir = args.images
        elif 'image_dir' in config:
            image_dir = Path(config['image_dir'])
        else:
            logger.error(
                "Image directory required when not skipping tiling. "
                "Provide --images argument or set 'image_dir' in config."
            )
            sys.exit(1)

        # Find all images in directory (common formats: .jpg, .png, .tif)
        logger.info(f"Searching for images in {image_dir}")
        image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        image_paths += list(image_dir.glob('*.tif')) + list(image_dir.glob('*.tiff'))

        if not image_paths:
            logger.error(f"No images found in {image_dir}")
            sys.exit(1)

        logger.info(f"Found {len(image_paths)} images")

    # Create and run pipeline
    try:
        pipeline = CTXTerrainPipeline(config, output_dir)
        pipeline.run(
            image_paths=image_paths,
            skip_tiling=args.skip_tiling,
            skip_embedding=args.skip_embedding,
        )

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    run_pipeline()
