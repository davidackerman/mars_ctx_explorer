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
  # Download and process 100 images
  ctx-terrain-pipeline --config configs/pipelines/ctx_terrain.yaml --download --download-limit 100

  # Download images from specific region (Jezero crater)
  ctx-terrain-pipeline --config configs/pipelines/ctx_terrain.yaml --download --download-region -77.5 -77.0 18.0 18.5

  # Run with custom image directory (already downloaded)
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

    parser.add_argument(
        '--download',
        action='store_true',
        help='Download CTX images before processing',
    )

    parser.add_argument(
        '--download-limit',
        type=int,
        default=100,
        help='Maximum number of images to download (default: 100)',
    )

    parser.add_argument(
        '--download-region',
        nargs=4,
        metavar=('MIN_LON', 'MAX_LON', 'MIN_LAT', 'MAX_LAT'),
        type=float,
        help='Download images in geographic region (lon/lat bounds)',
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
        # Download images if requested
        if args.download:
            from .downloader import CTXDownloader

            logger.info("=" * 80)
            logger.info("DOWNLOAD STEP: Fetching CTX images from NASA PDS")
            logger.info("=" * 80)

            download_config = config.get('download', {})
            download_dir = Path(download_config.get('output_dir', 'data/raw/ctx'))

            # Get ISIS3 configuration
            isis3_config = download_config.get('isis3', {})
            use_isis3 = isis3_config.get('enabled', True)
            apply_calibration = isis3_config.get('apply_calibration', True)
            map_projection = isis3_config.get('map_projection', 'sinusoidal')
            map_resolution = isis3_config.get('map_resolution', None)

            downloader = CTXDownloader(
                output_dir=download_dir,
                use_isis3=use_isis3,
                apply_calibration=apply_calibration,
                map_projection=map_projection,
                map_resolution=map_resolution,
            )

            if args.download_region:
                min_lon, max_lon, min_lat, max_lat = args.download_region
                logger.info(
                    f"Downloading images for region: "
                    f"lon=[{min_lon}, {max_lon}], lat=[{min_lat}, {max_lat}]"
                )
                image_paths = downloader.download_region(
                    min_lon=min_lon,
                    max_lon=max_lon,
                    min_lat=min_lat,
                    max_lat=max_lat,
                    limit=args.download_limit,
                )
            else:
                logger.info(f"Downloading up to {args.download_limit} CTX images")
                image_list = downloader.search_images(limit=args.download_limit)
                image_paths = downloader.download_images(image_list)

            if not image_paths:
                logger.error("No images downloaded")
                sys.exit(1)

            logger.info(f"Downloaded {len(image_paths)} CTX images to {download_dir}")

        # Use existing image directory
        else:
            if args.images:
                image_dir = args.images
            elif 'image_dir' in config:
                image_dir = Path(config['image_dir'])
            else:
                logger.error(
                    "Image directory required when not skipping tiling. "
                    "Provide --images argument, --download flag, or set 'image_dir' in config."
                )
                sys.exit(1)

            # Find all images in directory (common formats: .jpg, .png, .tif, .IMG)
            logger.info(f"Searching for images in {image_dir}")
            image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
            image_paths += list(image_dir.glob('*.tif')) + list(image_dir.glob('*.tiff'))
            image_paths += list(image_dir.glob('*.IMG'))

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
