"""Novelty/anomaly detection for embeddings."""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class NoveltyDetector:
    """
    Multi-method novelty/anomaly scoring for embedding vectors.

    Supported methods:
    1. kNN Distance: Average distance to k nearest neighbors (RECOMMENDED)
    2. Local Outlier Factor: Density-based outlier score
    3. Distance to Centroid: Distance to assigned cluster center
    4. Isolation Forest: Tree-based anomaly detection

    The kNN method is recommended for its simplicity, interpretability, and
    efficiency with FAISS acceleration.

    Args:
        method: Novelty detection method ('knn', 'lof', 'centroid', 'iforest')
        k: Number of nearest neighbors (for knn method)
        use_faiss: Use FAISS for fast kNN search (recommended for large datasets)
    """

    def __init__(
        self,
        method: str = "knn",
        k: int = 10,
        use_faiss: bool = True,
    ):
        if method not in ["knn", "lof", "centroid", "iforest"]:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Supported: 'knn', 'lof', 'centroid', 'iforest'"
            )

        self.method = method
        self.k = k
        self.use_faiss = use_faiss

        # Will be set in fit()
        self.fitted = False
        self.embeddings = None
        self.cluster_labels = None
        self.centroids = None
        self.index = None  # FAISS index
        self.model = None  # LOF or IForest model

        logger.info(
            f"NoveltyDetector initialized: method={method}, k={k}, use_faiss={use_faiss}"
        )

    def fit(
        self,
        embeddings: np.ndarray,
        cluster_labels: Optional[np.ndarray] = None,
    ):
        """
        Fit novelty detector on embeddings.

        Args:
            embeddings: Array of shape (N, D) containing embedding vectors
            cluster_labels: Optional array of shape (N,) with cluster assignments
                          (required for 'centroid' method)
        """
        self.embeddings = embeddings.astype('float32')  # FAISS requires float32
        self.cluster_labels = cluster_labels

        logger.info(f"Fitting {self.method} novelty detector on {len(embeddings)} samples")

        if self.method == "knn":
            self._fit_knn()
        elif self.method == "lof":
            self._fit_lof()
        elif self.method == "centroid":
            if cluster_labels is None:
                raise ValueError("cluster_labels required for 'centroid' method")
            self._fit_centroid()
        elif self.method == "iforest":
            self._fit_iforest()

        self.fitted = True
        logger.info("Novelty detector fitted successfully")

    def score(
        self,
        embeddings: Optional[np.ndarray] = None,
        cluster_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute novelty scores for embeddings.

        Higher scores indicate more novel/anomalous samples.

        Args:
            embeddings: Array of shape (M, D). If None, uses training embeddings.
            cluster_labels: Optional cluster labels (required for 'centroid' method)

        Returns:
            Array of shape (M,) containing novelty scores
        """
        if not self.fitted:
            raise ValueError("Must call fit() before score()")

        if embeddings is None:
            embeddings = self.embeddings
            cluster_labels = self.cluster_labels

        embeddings = embeddings.astype('float32')

        if self.method == "knn":
            return self._score_knn(embeddings)
        elif self.method == "lof":
            return self._score_lof(embeddings)
        elif self.method == "centroid":
            if cluster_labels is None:
                raise ValueError("cluster_labels required for 'centroid' method")
            return self._score_centroid(embeddings, cluster_labels)
        elif self.method == "iforest":
            return self._score_iforest(embeddings)

    def _fit_knn(self):
        """Build FAISS index for fast kNN search."""
        if self.use_faiss:
            try:
                import faiss
            except ImportError:
                logger.warning(
                    "FAISS not available, falling back to sklearn. "
                    "Install with: pip install faiss-cpu"
                )
                self.use_faiss = False

        if self.use_faiss:
            import faiss

            # Build FAISS index for L2 distance
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.embeddings)
            logger.info(f"Built FAISS index with {len(self.embeddings)} vectors")
        else:
            # Fallback to sklearn
            from sklearn.neighbors import NearestNeighbors

            self.model = NearestNeighbors(n_neighbors=self.k + 1, metric='euclidean')
            self.model.fit(self.embeddings)

    def _score_knn(self, embeddings: np.ndarray) -> np.ndarray:
        """Score using average distance to k nearest neighbors."""
        if self.use_faiss:
            import faiss

            # Search for k+1 neighbors (includes self if in training set)
            distances, indices = self.index.search(embeddings, self.k + 1)

            # Exclude self (first neighbor) if embeddings are from training set
            # Check if first neighbor distance is near-zero
            is_in_training = distances[:, 0] < 1e-6
            scores = np.where(
                is_in_training[:, None],
                distances[:, 1:],  # Exclude self
                distances[:, :self.k],  # Use all k neighbors
            ).mean(axis=1)
        else:
            # sklearn fallback
            distances, _ = self.model.kneighbors(embeddings)
            scores = distances[:, 1:].mean(axis=1)  # Exclude self

        return scores

    def _fit_lof(self):
        """Fit Local Outlier Factor model."""
        from sklearn.neighbors import LocalOutlierFactor

        self.model = LocalOutlierFactor(
            n_neighbors=self.k,
            novelty=True,  # Enable predict mode
            contamination='auto',
        )
        self.model.fit(self.embeddings)

    def _score_lof(self, embeddings: np.ndarray) -> np.ndarray:
        """Score using Local Outlier Factor."""
        # LOF returns negative scores (lower = more anomalous)
        # We negate to get positive novelty scores
        scores = -self.model.score_samples(embeddings)
        return scores

    def _fit_centroid(self):
        """Compute cluster centroids."""
        self.centroids = {}

        for label in np.unique(self.cluster_labels):
            if label != -1:  # Skip noise points
                mask = self.cluster_labels == label
                centroid = self.embeddings[mask].mean(axis=0)
                self.centroids[label] = centroid

        logger.info(f"Computed centroids for {len(self.centroids)} clusters")

    def _score_centroid(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> np.ndarray:
        """Score using distance to assigned cluster centroid."""
        scores = np.zeros(len(embeddings))

        for i, (emb, label) in enumerate(zip(embeddings, cluster_labels)):
            if label in self.centroids:
                # Distance to cluster centroid
                scores[i] = np.linalg.norm(emb - self.centroids[label])
            else:
                # Outlier/noise point - assign max score
                scores[i] = np.inf

        return scores

    def _fit_iforest(self):
        """Fit Isolation Forest model."""
        from sklearn.ensemble import IsolationForest

        self.model = IsolationForest(
            contamination='auto',
            random_state=42,
        )
        self.model.fit(self.embeddings)

    def _score_iforest(self, embeddings: np.ndarray) -> np.ndarray:
        """Score using Isolation Forest."""
        # IForest returns negative scores (lower = more anomalous)
        # We negate to get positive novelty scores
        scores = -self.model.score_samples(embeddings)
        return scores

    def get_top_k_novel(
        self,
        scores: np.ndarray,
        k: int = 100,
    ) -> np.ndarray:
        """
        Get indices of top-k most novel samples.

        Args:
            scores: Novelty scores from score() method
            k: Number of top samples to return

        Returns:
            Array of indices for top-k novel samples (sorted by score, descending)
        """
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return top_k_indices
