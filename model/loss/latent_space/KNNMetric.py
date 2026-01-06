from sklearn.neighbors import NearestNeighbors
import numpy as np

class KNNMetric:
    """Computes k-NN classification accuracy"""
    def __init__(self, k:int=5, metric:str='cosine'):
        """
        Computes k-NN classification accuracy to evaluate representation quality
        
        Args:
            k (int):        Number of nearest neighbors
            metric (str):   Distance metric ('cosine', 'euclidean', 'manhattan', etc.)
        """
        self.k = k
        self.metric = metric
    
    def compute_accuracy(self, features:np.ndarray, labels:np.ndarray, n_jobs:int) -> dict[str, float]:
        """
        Compute k-NN accuracy.
        
        Args:
            features (np.ndarray):  Feature array of shape (num_samples, feature_dim)
            labels (np.ndarray):    Labels array of shape (num_samples,)
            n_jobs (int):           The number of parallel jobs to run for neighbors search. Set to -1 means using all processors. Used for parallelized :class:`NearestNeighbors`
        
        Returns:
            out (dict[str, float]): Dictionary with 'accuracy' and optionally 'per_class_accuracy'
        """
        if len(features) != len(labels):
            raise ValueError(f"Features and labels length mismatch: {len(features)} vs {len(labels)}")
        
        # Fit k-NN (k+1 because nearest neighbor is the sample itself)
        knn = NearestNeighbors(n_neighbors=self.k + 1, metric=self.metric, n_jobs=n_jobs)
        knn.fit(features)
        
        # Find neighbors
        _, indices = knn.kneighbors(features)
        
        # Remove first neighbor (itself)
        neighbor_indices = indices[:, 1:]
        
        # Get neighbor labels
        neighbor_labels = labels[neighbor_indices]
        
        # Count correct predictions (neighbors with same label)
        labels_same_shape = np.broadcast_to(np.expand_dims(labels, axis=1), (len(labels), self.k))
        correct = (neighbor_labels == labels_same_shape).sum(axis=1)
        
        # Overall accuracy
        accuracy = (correct / self.k).mean()
        
        # Per-class accuracy
        unique_labels = np.unique(labels)
        per_class_acc = {}
        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() > 0:
                per_class_acc[f'class_{int(label)}'] = (correct[mask] / self.k).mean()
        
        return {
            'accuracy': float(accuracy),
            **per_class_acc
        }
