"""HDBSCAN clustering with automatic parameter tuning."""

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class HDBSCANClusterer:
    """
    HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise).

    HDBSCAN is a density-based clustering algorithm that:
    - Automatically determines the number of clusters
    - Handles outliers/noise (assigns label -1)
    - Provides soft clustering probabilities
    - Works well with high-dimensional embeddings

    Parameter Guidelines:
    - min_cluster_size: 0.5-1% of dataset size (e.g., 50-100 for 10k samples)
    - min_samples: 5-20 for robustness (higher = more conservative clustering)
    - metric: 'euclidean' for most embeddings, 'cosine' for normalized vectors
    - cluster_selection_epsilon: 0.0 (auto) or small value for merging clusters

    Args:
        min_cluster_size: Minimum number of samples in a cluster
        min_samples: Number of samples in neighborhood for core point
        metric: Distance metric ('euclidean', 'cosine', 'manhattan', etc.)
        cluster_selection_epsilon: Distance threshold for cluster merging (0.0=auto)
        cluster_selection_method: 'eom' (excess of mass) or 'leaf'
    """

    def __init__(
        self,
        min_cluster_size: int = 50,
        min_samples: Optional[int] = None,
        metric: str = "euclidean",
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = "eom",
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size // 5  # Default heuristic
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method

        # Will be initialized in fit_predict
        self.clusterer = None

        logger.info(
            f"HDBSCANClusterer initialized: "
            f"min_cluster_size={min_cluster_size}, "
            f"min_samples={self.min_samples}, "
            f"metric={metric}"
        )

    def fit_predict(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Cluster embeddings using HDBSCAN.

        Args:
            embeddings: Array of shape (N, D) containing embedding vectors

        Returns:
            Dictionary containing:
            - labels: (N,) array of cluster IDs (-1 for outliers/noise)
            - probabilities: (N,) array of cluster membership probabilities
            - outlier_scores: (N,) array of GLOSH outlier scores
            - n_clusters: Number of clusters found (excluding noise)
            - cluster_sizes: Dict mapping cluster_id -> sample count
            - noise_count: Number of noise points (label=-1)
            - noise_fraction: Fraction of samples classified as noise
        """
        try:
            import hdbscan
        except ImportError:
            raise ImportError(
                "hdbscan is not installed. Install with: pip install hdbscan"
            )

        logger.info(f"Clustering {len(embeddings)} samples with HDBSCAN...")

        # Create and fit clusterer
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=True,  # Enable soft predictions
        )

        labels = self.clusterer.fit_predict(embeddings)

        # Compute statistics
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique_labels.tolist(), counts.tolist()))

        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_count = cluster_sizes.get(-1, 0)
        noise_fraction = noise_count / len(embeddings)

        logger.info(
            f"Clustering complete: {n_clusters} clusters found, "
            f"{noise_count} noise points ({noise_fraction:.1%})"
        )

        # Log cluster sizes
        for label in sorted(unique_labels):
            if label != -1:
                logger.info(f"  Cluster {label}: {cluster_sizes[label]} samples")

        result = {
            "labels": labels,
            "probabilities": self.clusterer.probabilities_,
            "outlier_scores": self.clusterer.outlier_scores_,
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_sizes,
            "noise_count": noise_count,
            "noise_fraction": noise_fraction,
        }

        return result

    def predict_soft(self, new_embeddings: np.ndarray) -> np.ndarray:
        """
        Predict soft cluster membership for new samples.

        Requires that fit_predict has been called first.

        Args:
            new_embeddings: Array of shape (M, D) containing new embedding vectors

        Returns:
            Array of shape (M, n_clusters) containing soft cluster probabilities
        """
        if self.clusterer is None:
            raise ValueError("Must call fit_predict before predict_soft")

        try:
            import hdbscan
        except ImportError:
            raise ImportError(
                "hdbscan is not installed. Install with: pip install hdbscan"
            )

        # Approximate predict using HDBSCAN's built-in method
        labels, strengths = hdbscan.approximate_predict(self.clusterer, new_embeddings)

        return labels, strengths

    @staticmethod
    def auto_select_parameters(n_samples: int) -> Dict[str, int]:
        """
        Automatically select HDBSCAN parameters based on dataset size.

        Args:
            n_samples: Number of samples in dataset

        Returns:
            Dictionary with suggested parameter values
        """
        # Heuristics for parameter selection
        min_cluster_size = max(50, int(n_samples * 0.005))  # 0.5% of dataset
        min_samples = max(5, min_cluster_size // 5)  # 20% of min_cluster_size

        logger.info(
            f"Auto-selected parameters for {n_samples} samples: "
            f"min_cluster_size={min_cluster_size}, min_samples={min_samples}"
        )

        return {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
        }
