"""End-to-end CTX terrain classification pipeline."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from scientific_pipelines.core.clustering import HDBSCANClusterer, NoveltyDetector
from scientific_pipelines.core.embeddings import DINOv3Extractor, EmbeddingPipeline

from .tiling import CTXTiler

logger = logging.getLogger(__name__)


class CTXTerrainPipeline:
    """
    End-to-end pipeline for CTX terrain classification.

    Pipeline steps:
    1. Tile CTX images into 256x256 patches with quality filtering
    2. Extract DINOv3 embeddings from tiles
    3. Cluster tiles with HDBSCAN
    4. Compute novelty scores
    5. Generate outputs (CSV files)

    Args:
        config: Configuration dictionary
        output_dir: Directory for all pipeline outputs
    """

    def __init__(self, config: Dict, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        logger.info("Initializing CTX terrain classification pipeline")

        # Tiling
        tiling_config = config.get('tiling', {})
        self.tiler = CTXTiler(**tiling_config)

        # Embedding extractor
        embedding_config = config.get('embedding', {})
        self.embedder = DINOv3Extractor(**embedding_config)

        # Embedding pipeline
        embedding_pipeline_config = config.get('embedding_pipeline', {})
        self.embedding_pipeline = EmbeddingPipeline(
            extractor=self.embedder,
            transform=DINOv3Extractor.get_default_transforms(),
            **embedding_pipeline_config,
        )

        # Clustering
        clustering_config = config.get('clustering', {})
        self.clusterer = HDBSCANClusterer(**clustering_config)

        # Novelty detection
        novelty_config = config.get('novelty', {})
        self.novelty_detector = NoveltyDetector(**novelty_config)

        logger.info("CTX terrain classification pipeline initialized")

    def run(
        self,
        image_paths: Optional[List[Path]] = None,
        skip_tiling: bool = False,
        skip_embedding: bool = False,
    ):
        """
        Execute the full CTX terrain classification pipeline.

        Args:
            image_paths: List of paths to CTX images (required if not skip_tiling)
            skip_tiling: If True, skip tiling step (assumes tiles already exist)
            skip_embedding: If True, skip embedding extraction (assumes embeddings exist)
        """
        logger.info("=" * 80)
        logger.info("Starting CTX Terrain Classification Pipeline")
        logger.info("=" * 80)

        # Step 1: Tile images
        if not skip_tiling:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 1: Tiling CTX Images")
            logger.info("=" * 80)

            if image_paths is None or len(image_paths) == 0:
                raise ValueError("image_paths required when skip_tiling=False")

            tile_dir = self.output_dir / "tiles"
            tiles_metadata = self.tiler.tile_dataset(
                image_paths, tile_dir, save_tiles=True
            )

            # Save tiles metadata
            tiles_df = pd.DataFrame(tiles_metadata)
            tiles_csv_path = self.output_dir / "tiles.csv"
            tiles_df.to_csv(tiles_csv_path, index=False)
            logger.info(f"Tiles metadata saved to {tiles_csv_path}")

            # Print statistics
            stats = self.tiler.get_tile_statistics(tiles_metadata)
            logger.info(f"Tiling statistics:")
            logger.info(f"  Total tiles: {stats['total_tiles']}")
            logger.info(f"  Passed quality filters: {stats['passed_tiles']}")
            logger.info(f"  Pass rate: {stats['pass_rate']:.1%}")
            logger.info(f"  Mean contrast std: {stats['mean_contrast_std']:.2f}")
        else:
            # Load existing tiles metadata
            tiles_csv_path = self.output_dir / "tiles.csv"
            logger.info(f"Skipping tiling, loading metadata from {tiles_csv_path}")
            tiles_df = pd.read_csv(tiles_csv_path)
            tiles_metadata = tiles_df.to_dict('records')

        # Filter to only tiles that passed quality
        valid_tiles = [t for t in tiles_metadata if t['passes_quality']]
        logger.info(f"Using {len(valid_tiles)} tiles that passed quality filters")

        # Step 2: Extract embeddings
        if not skip_embedding:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 2: Extracting DINOv3 Embeddings")
            logger.info("=" * 80)

            tile_paths = [Path(t['tile_path']) for t in valid_tiles]
            embeddings_output = self.output_dir / "embeddings.parquet"

            embeddings, emb_metadata = self.embedding_pipeline.extract_dataset(
                image_paths=tile_paths,
                output_path=embeddings_output,
                resume=True,
            )

            logger.info(
                f"Extracted embeddings: shape={embeddings.shape}, "
                f"dim={embeddings.shape[1]}"
            )
        else:
            # Load existing embeddings
            embeddings_output = self.output_dir / "embeddings.parquet"
            logger.info(f"Skipping embedding extraction, loading from {embeddings_output}")

            # Load from parquet
            import pyarrow.parquet as pq

            table = pq.read_table(embeddings_output)
            df = table.to_pandas()
            embeddings = pd.np.vstack(df['embedding'].values)
            emb_metadata = df.drop(columns=['embedding'])

            logger.info(f"Loaded embeddings: shape={embeddings.shape}")

        # Step 3: Cluster with HDBSCAN
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Clustering Tiles with HDBSCAN")
        logger.info("=" * 80)

        cluster_results = self.clusterer.fit_predict(embeddings)

        # Save cluster results
        clusters_df = pd.DataFrame(
            {
                'tile_path': [t['tile_path'] for t in valid_tiles],
                'source_image': [t['source_image'] for t in valid_tiles],
                'x_offset': [t['x_offset'] for t in valid_tiles],
                'y_offset': [t['y_offset'] for t in valid_tiles],
                'cluster_id': cluster_results['labels'],
                'cluster_probability': cluster_results['probabilities'],
                'outlier_score': cluster_results['outlier_scores'],
            }
        )

        clusters_csv_path = self.output_dir / "tile_clusters.csv"
        clusters_df.to_csv(clusters_csv_path, index=False)
        logger.info(f"Cluster results saved to {clusters_csv_path}")

        # Print cluster statistics
        logger.info(f"Clustering results:")
        logger.info(f"  Number of clusters: {cluster_results['n_clusters']}")
        logger.info(f"  Noise points: {cluster_results['noise_count']}")
        logger.info(f"  Noise fraction: {cluster_results['noise_fraction']:.1%}")

        for cluster_id in sorted(
            [k for k in cluster_results['cluster_sizes'].keys() if k != -1]
        ):
            size = cluster_results['cluster_sizes'][cluster_id]
            logger.info(f"  Cluster {cluster_id}: {size} tiles")

        # Step 4: Compute novelty scores
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Computing Novelty Scores")
        logger.info("=" * 80)

        self.novelty_detector.fit(embeddings, cluster_results['labels'])
        novelty_scores = self.novelty_detector.score(embeddings, cluster_results['labels'])

        # Get top novel tiles
        top_k = self.config.get('gallery', {}).get('top_n_outliers', 100)
        top_novel_indices = self.novelty_detector.get_top_k_novel(novelty_scores, k=top_k)

        # Save novelty results
        novelty_df = pd.DataFrame(
            {
                'tile_path': [t['tile_path'] for t in valid_tiles],
                'novelty_score': novelty_scores,
                'is_outlier': cluster_results['labels'] == -1,
                'is_top_novel': [
                    i in top_novel_indices for i in range(len(valid_tiles))
                ],
            }
        )

        novelty_csv_path = self.output_dir / "tile_novelty.csv"
        novelty_df.to_csv(novelty_csv_path, index=False)
        logger.info(f"Novelty scores saved to {novelty_csv_path}")

        logger.info(f"Novelty statistics:")
        logger.info(f"  Mean novelty score: {novelty_scores.mean():.3f}")
        logger.info(f"  Max novelty score: {novelty_scores.max():.3f}")
        logger.info(f"  Top {top_k} most novel tiles identified")

        # Pipeline complete
        logger.info("\n" + "=" * 80)
        logger.info("CTX Terrain Classification Pipeline Complete!")
        logger.info("=" * 80)
        logger.info(f"All outputs saved to: {self.output_dir}")
        logger.info(f"  - tiles.csv: Tile metadata")
        logger.info(f"  - embeddings.parquet: DINOv3 embeddings")
        logger.info(f"  - tile_clusters.csv: Cluster assignments")
        logger.info(f"  - tile_novelty.csv: Novelty scores")

        return {
            'tiles_metadata': tiles_metadata,
            'embeddings': embeddings,
            'cluster_results': cluster_results,
            'novelty_scores': novelty_scores,
        }
