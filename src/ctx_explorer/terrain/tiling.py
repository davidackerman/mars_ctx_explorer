"""Image tiling for CTX images with quality filtering."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CTXTiler:
    """
    Tile CTX images into patches with quality filtering.

    CTX (Context Camera) images are large orbital images of Mars surface.
    This class tiles them into fixed-size patches suitable for embedding
    extraction, while filtering out low-quality tiles (uniform terrain,
    NoData regions, etc.).

    Quality filters:
    - Contrast threshold: std(tile) > min_contrast_std (skip uniform areas)
    - Data fraction: >min_data_fraction non-NoData pixels
    - Edge filtering: Skip boundary artifacts

    Args:
        tile_size: Size of square tiles (default: 256x256)
        overlap: Overlap between adjacent tiles (default: 0)
        min_contrast_std: Minimum standard deviation for contrast (default: 10.0)
        min_data_fraction: Minimum fraction of valid data pixels (default: 0.9)
        nodata_value: Value representing NoData pixels (default: 0)
    """

    def __init__(
        self,
        tile_size: int = 256,
        overlap: int = 0,
        min_contrast_std: float = 10.0,
        min_data_fraction: float = 0.9,
        nodata_value: int = 0,
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_contrast_std = min_contrast_std
        self.min_data_fraction = min_data_fraction
        self.nodata_value = nodata_value

        self.stride = tile_size - overlap

        logger.info(
            f"CTXTiler initialized: tile_size={tile_size}, overlap={overlap}, "
            f"min_contrast_std={min_contrast_std}, min_data_fraction={min_data_fraction}"
        )

    def tile_image(
        self,
        image_path: Path,
        output_dir: Path,
        save_tiles: bool = True,
    ) -> List[Dict]:
        """
        Tile a single CTX image into patches.

        Args:
            image_path: Path to input CTX image
            output_dir: Directory to save tiles
            save_tiles: If True, save tiles to disk; if False, just return metadata

        Returns:
            List of tile metadata dictionaries:
            {
                'tile_path': Path to saved tile (or None if not saved),
                'source_image': Path to source image,
                'x_offset': X coordinate of tile top-left corner,
                'y_offset': Y coordinate of tile top-left corner,
                'contrast_std': Standard deviation of tile pixel values,
                'data_fraction': Fraction of non-NoData pixels,
                'passes_quality': Boolean indicating if tile passed quality filters,
            }
        """
        if save_tiles:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Load image
        img = Image.open(image_path)

        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')

        img_array = np.array(img)
        img_height, img_width = img_array.shape

        logger.info(
            f"Tiling image {image_path.name} "
            f"(size: {img_width}x{img_height})"
        )

        tiles_metadata = []
        tiles_passed = 0
        tiles_total = 0

        # Iterate over image in tiles
        for y in range(0, img_height - self.tile_size + 1, self.stride):
            for x in range(0, img_width - self.tile_size + 1, self.stride):
                tiles_total += 1

                # Extract tile
                tile = img_array[y : y + self.tile_size, x : x + self.tile_size]

                # Apply quality filters
                quality_result = self._check_tile_quality(tile)

                # Create metadata
                tile_metadata = {
                    'tile_path': None,
                    'source_image': str(image_path),
                    'x_offset': x,
                    'y_offset': y,
                    'contrast_std': quality_result['contrast_std'],
                    'data_fraction': quality_result['data_fraction'],
                    'passes_quality': quality_result['passes'],
                }

                # Save tile if it passes quality filters
                if quality_result['passes'] and save_tiles:
                    tile_filename = f"{image_path.stem}_tile_{y:06d}_{x:06d}.png"
                    tile_path = output_dir / tile_filename

                    # Save as PNG
                    Image.fromarray(tile).save(tile_path)
                    tile_metadata['tile_path'] = str(tile_path)
                    tiles_passed += 1

                tiles_metadata.append(tile_metadata)

        logger.info(
            f"Tiling complete: {tiles_passed}/{tiles_total} tiles passed quality filters "
            f"({100 * tiles_passed / tiles_total:.1f}%)"
        )

        return tiles_metadata

    def tile_dataset(
        self,
        image_paths: List[Path],
        output_dir: Path,
        save_tiles: bool = True,
    ) -> List[Dict]:
        """
        Tile multiple CTX images.

        Args:
            image_paths: List of paths to input CTX images
            output_dir: Directory to save all tiles
            save_tiles: If True, save tiles to disk

        Returns:
            Combined list of tile metadata from all images
        """
        all_tiles_metadata = []

        for image_path in tqdm(image_paths, desc="Tiling CTX images"):
            tiles = self.tile_image(image_path, output_dir, save_tiles=save_tiles)
            all_tiles_metadata.extend(tiles)

        total_tiles = len(all_tiles_metadata)
        passed_tiles = sum(1 for t in all_tiles_metadata if t['passes_quality'])

        logger.info(
            f"Dataset tiling complete: {passed_tiles}/{total_tiles} total tiles passed "
            f"({100 * passed_tiles / total_tiles:.1f}%)"
        )

        return all_tiles_metadata

    def _check_tile_quality(self, tile: np.ndarray) -> Dict:
        """
        Check if a tile passes quality filters.

        Args:
            tile: Tile as numpy array

        Returns:
            Dictionary with:
            - 'contrast_std': Standard deviation of pixel values
            - 'data_fraction': Fraction of non-NoData pixels
            - 'passes': Boolean indicating if tile passed all filters
        """
        # Compute contrast (standard deviation)
        contrast_std = float(tile.std())

        # Compute data fraction (non-NoData pixels)
        valid_pixels = (tile != self.nodata_value).sum()
        total_pixels = tile.size
        data_fraction = float(valid_pixels / total_pixels)

        # Check if passes all filters
        passes = (
            contrast_std >= self.min_contrast_std
            and data_fraction >= self.min_data_fraction
        )

        return {
            'contrast_std': contrast_std,
            'data_fraction': data_fraction,
            'passes': passes,
        }

    def get_tile_statistics(self, tiles_metadata: List[Dict]) -> Dict:
        """
        Compute statistics about tiles.

        Args:
            tiles_metadata: List of tile metadata dictionaries

        Returns:
            Dictionary with statistics
        """
        total = len(tiles_metadata)
        passed = sum(1 for t in tiles_metadata if t['passes_quality'])
        failed = total - passed

        contrasts = [t['contrast_std'] for t in tiles_metadata if t['passes_quality']]
        data_fractions = [
            t['data_fraction'] for t in tiles_metadata if t['passes_quality']
        ]

        return {
            'total_tiles': total,
            'passed_tiles': passed,
            'failed_tiles': failed,
            'pass_rate': passed / total if total > 0 else 0.0,
            'mean_contrast_std': float(np.mean(contrasts)) if contrasts else 0.0,
            'mean_data_fraction': float(np.mean(data_fractions))
            if data_fractions
            else 0.0,
        }
