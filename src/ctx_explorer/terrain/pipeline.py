"""End-to-end CTX terrain classification pipeline."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ctx_explorer.clustering import HDBSCANClusterer, NoveltyDetector
from ctx_explorer.embeddings import DINOv3Extractor, EmbeddingPipeline

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

        # The DINOv3 extractor is built lazily: run_from_embeddings() never
        # needs it, and torch.hub model loading is slow.
        self._embedding_config = config.get('embedding', {})
        self._embedding_pipeline_config = config.get('embedding_pipeline', {})
        self._embedding_pipeline: Optional[EmbeddingPipeline] = None

        # Clustering
        clustering_config = config.get('clustering', {})
        self.clusterer = HDBSCANClusterer(**clustering_config)

        # Novelty detection
        novelty_config = config.get('novelty', {})
        self.novelty_detector = NoveltyDetector(**novelty_config)

        logger.info("CTX terrain classification pipeline initialized")

    @property
    def embedding_pipeline(self) -> EmbeddingPipeline:
        if self._embedding_pipeline is None:
            embedder = DINOv3Extractor(**self._embedding_config)
            self._embedding_pipeline = EmbeddingPipeline(
                extractor=embedder,
                transform=DINOv3Extractor.get_default_transforms(),
                **self._embedding_pipeline_config,
            )
        return self._embedding_pipeline

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
            embeddings = np.vstack(df['embedding'].values)
            emb_metadata = df.drop(columns=['embedding'])

            logger.info(f"Loaded embeddings: shape={embeddings.shape}")

        meta_df = pd.DataFrame(
            {
                'tile_path': [t['tile_path'] for t in valid_tiles],
                'source_image': [t['source_image'] for t in valid_tiles],
                'x_offset': [t['x_offset'] for t in valid_tiles],
                'y_offset': [t['y_offset'] for t in valid_tiles],
            }
        )
        cluster_results, novelty_scores = self._cluster_and_score(
            embeddings, meta_df, key_column='tile_path'
        )

        logger.info("\n" + "=" * 80)
        logger.info("CTX Terrain Classification Pipeline Complete!")
        logger.info("=" * 80)
        logger.info(f"All outputs saved to: {self.output_dir}")
        logger.info("  - tiles.csv: Tile metadata")
        logger.info("  - embeddings.parquet: DINOv3 embeddings")
        logger.info("  - tile_clusters.csv: Cluster assignments")
        logger.info("  - tile_novelty.csv: Novelty scores")

        return {
            'tiles_metadata': tiles_metadata,
            'embeddings': embeddings,
            'cluster_results': cluster_results,
            'novelty_scores': novelty_scores,
        }

    def run_from_embeddings(
        self, embeddings_path: Path, key_column: Optional[str] = None
    ) -> Dict:
        """Cluster and novelty-score an existing embeddings parquet.

        Skips download, tiling and embedding entirely. Any parquet with an
        ``embedding`` column works, including ``embeddings.parquet`` written by
        the similarity-index builders (``build_ctx_retrieval_index.py``,
        ``run_ctx_pipeline.py``), so terrain clustering can be run over the
        same tiles the viewer searches.

        Args:
            embeddings_path: Parquet with an ``embedding`` column plus per-row
                metadata (``image_path``, ``tile_path``, lat/lon, ...).
            key_column: Column identifying each row in the outputs. Defaults to
                ``tile_path`` if present, else ``image_path``.
        """
        embeddings_path = Path(embeddings_path)
        logger.info(f"Loading embeddings from {embeddings_path}")
        df = pd.read_parquet(embeddings_path)
        if 'embedding' not in df.columns:
            raise ValueError(f"{embeddings_path} has no 'embedding' column")
        embeddings = np.vstack(df['embedding'].values).astype(np.float32)
        meta_df = df.drop(columns=['embedding']).reset_index(drop=True)
        if key_column is None:
            key_column = 'tile_path' if 'tile_path' in meta_df.columns else 'image_path'
        if key_column not in meta_df.columns:
            raise ValueError(f"key column '{key_column}' not in {list(meta_df.columns)}")
        logger.info(f"Loaded {len(meta_df)} embeddings of dim {embeddings.shape[1]}")

        cluster_results, novelty_scores = self._cluster_and_score(
            embeddings, meta_df, key_column=key_column
        )
        return {
            'embeddings': embeddings,
            'cluster_results': cluster_results,
            'novelty_scores': novelty_scores,
        }

    def _cluster_and_score(
        self, embeddings: np.ndarray, meta_df: pd.DataFrame, key_column: str
    ):
        """HDBSCAN clustering + novelty scoring; writes tile_clusters.csv and tile_novelty.csv.

        Cluster ids are arbitrary integers (-1 = noise). Nothing here names a
        cluster "crater" or "dune"; interpreting clusters is a manual step, and
        the similarity viewer is the tool for it.
        """
        logger.info("\n" + "=" * 80)
        logger.info("Clustering tiles with HDBSCAN")
        logger.info("=" * 80)
        cluster_results = self.clusterer.fit_predict(embeddings)

        clusters_df = meta_df.copy()
        clusters_df['cluster_id'] = cluster_results['labels']
        clusters_df['cluster_probability'] = cluster_results['probabilities']
        clusters_df['outlier_score'] = cluster_results['outlier_scores']
        clusters_csv_path = self.output_dir / "tile_clusters.csv"
        clusters_df.to_csv(clusters_csv_path, index=False)
        logger.info(f"Cluster results saved to {clusters_csv_path}")
        logger.info(f"  Number of clusters: {cluster_results['n_clusters']}")
        logger.info(f"  Noise points: {cluster_results['noise_count']}")
        logger.info(f"  Noise fraction: {cluster_results['noise_fraction']:.1%}")

        logger.info("\n" + "=" * 80)
        logger.info("Computing novelty scores")
        logger.info("=" * 80)
        self.novelty_detector.fit(embeddings, cluster_results['labels'])
        novelty_scores = self.novelty_detector.score(embeddings, cluster_results['labels'])
        top_k = self.config.get('gallery', {}).get('top_n_outliers', 100)
        top_novel = set(np.asarray(self.novelty_detector.get_top_k_novel(novelty_scores, k=top_k)).tolist())

        novelty_df = pd.DataFrame(
            {
                key_column: meta_df[key_column].values,
                'novelty_score': novelty_scores,
                'is_outlier': cluster_results['labels'] == -1,
                'is_top_novel': [i in top_novel for i in range(len(meta_df))],
            }
        )
        novelty_csv_path = self.output_dir / "tile_novelty.csv"
        novelty_df.to_csv(novelty_csv_path, index=False)
        logger.info(f"Novelty scores saved to {novelty_csv_path}")
        logger.info(f"  Mean novelty score: {novelty_scores.mean():.3f}")
        logger.info(f"  Max novelty score: {novelty_scores.max():.3f}")
        return cluster_results, novelty_scores
