#!/usr/bin/env python3
"""Download CTX images into a local directory."""

import argparse
import logging
from pathlib import Path

from scientific_pipelines.planetary.mars.ctx.downloader import CTXDownloader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Mars CTX images from ODE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/ctx"),
        help="Directory to store downloaded CTX images",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of images to download",
    )
    parser.add_argument(
        "--min-lon",
        type=float,
        default=None,
        help="Minimum longitude for region filter",
    )
    parser.add_argument(
        "--max-lon",
        type=float,
        default=None,
        help="Maximum longitude for region filter",
    )
    parser.add_argument(
        "--min-lat",
        type=float,
        default=None,
        help="Minimum latitude for region filter",
    )
    parser.add_argument(
        "--max-lat",
        type=float,
        default=None,
        help="Maximum latitude for region filter",
    )
    parser.add_argument(
        "--no-isis3",
        action="store_true",
        help="Disable ISIS3 processing and keep basic GDAL fallback conversion",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable ISIS3 radiometric calibration step",
    )
    parser.add_argument(
        "--map-resolution",
        type=float,
        default=None,
        help="cam2map output resolution in meters/pixel (default: native ~6 m/px)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent ISIS3 pipelines (1 = serial)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    downloader = CTXDownloader(
        output_dir=args.output_dir,
        use_isis3=not args.no_isis3,
        apply_calibration=not args.no_calibration,
        map_resolution=args.map_resolution,
    )

    image_list = downloader.search_images(
        limit=args.limit,
        min_lon=args.min_lon,
        max_lon=args.max_lon,
        min_lat=args.min_lat,
        max_lat=args.max_lat,
    )

    if len(image_list) == 0:
        logger.warning("No CTX images found for the requested query")
        return

    downloaded_paths = downloader.download_images(image_list, max_workers=args.workers)
    logger.info(f"Downloaded {len(downloaded_paths)} CTX images into {args.output_dir}")
    logger.info(f"Manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
