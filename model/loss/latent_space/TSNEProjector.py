from sklearn.manifold import TSNE
from enum import Enum, auto

import numpy as np

class TSNEMetric(Enum):
    """Distance metrics available for TSNE (sklearn pairwise distances)."""
    EUCLIDEAN = auto()
    """Standard L2 distance — fast, assumes locally linear structure."""
    MANHATTAN = auto() #
    """L1 distance — more robust to outliers than Euclidean."""
    COSINE = auto()
    """Angle-based similarity — good when magnitude is not meaningful."""

class TSNEProjector:
    """Computes t-SNE projection for visualization and analysis"""
    def __init__(self, n_components:int=2, metric:TSNEMetric=TSNEMetric.MANHATTAN, perplexity:float=30.0, max_iter:int=1000, random_state:int=None):
        """
        Computes t-SNE projection for visualization and analysis
        
        Args:
            n_components (int): Dimension of embedded space (usually 2 or 3)
            perplexity (int):   t-SNE perplexity parameter (5-50 typical)
            max_iter (int):     Maximum number of iterations for the optimization. Should be at least 250
            random_state (int): Random seed for reproducibility
        """
        if not isinstance(metric, TSNEMetric):
            raise ValueError(f"metric must be TSNEMetric, got {type(metric)}")
        
        self.n_components = n_components
        self.metric = metric
        self.perplexity = perplexity
        self.max_iter = max_iter
        self.random_state = random_state
    
    def project(self, features:np.ndarray, n_jobs:int) -> np.ndarray:
        """
        Compute t-SNE projection.
        
        Args:
            features (np.ndarray):  Array of shape (num_samples, feature_dim)
            n_jobs (int):           The number of parallel jobs to run for neighbors search. Set to -1 means using all processors. Used for parallelized :class:`TSNE`
        
        Returns:
            projected (np.ndarray): Projected features of shape (num_samples, n_components)
        """
        tsne = TSNE(
            n_components    = self.n_components,
            perplexity      = self.perplexity,
            max_iter        = self.max_iter,
            metric          = self.metric.name.lower(),
            
            n_jobs          = n_jobs,
            
            random_state    = self.random_state,
        )
        
        projected_subset = tsne.fit_transform(features)
        
        return projected_subset
