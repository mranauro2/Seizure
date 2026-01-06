from enum import Enum, auto
from torch import Tensor

import torch.nn.functional as F
import numpy as np
import torch

class ReductionStrategy(Enum):
    """Reduction strategies optimized for graph-based temporal EEG data"""
    TEMPORAL_SPATIAL_MEAN = auto()
    """Average across time, then across nodes - balanced approach"""
    LAST_TIME_SPATIAL_MEAN = auto() #
    """Take last time step, average across nodes - emphasizes recent activity"""
    ATTENTION_WEIGHTED = auto()
    """Attention-weighted pooling across nodes, then temporal mean"""

class FeatureReducer:
    """Reduces 4D encoder output (batch, seq_len, num_nodes, input_dim) to 2D features (batch, feature_dim)"""
    def __init__(self, strategy:ReductionStrategy=ReductionStrategy.LAST_TIME_SPATIAL_MEAN, num_samples:int=None):
        """
        Reduces 4D encoder output (batch, seq_len, num_nodes, input_dim) to 2D features (batch, feature_dim).
        Optimized for graph-based temporal EEG data.
    
        Args:
            strategy (ReductionStrategy):   Reduction strategy to use
            num_samples (int):              Number of samples the class should reduce. If it is set to None then the computation can be slower
        """
        if not isinstance(strategy, ReductionStrategy):
            raise ValueError(f"strategy must be ReductionStrategy, got {type(strategy)}")
        if (num_samples is not None) and (num_samples <= 0):
            raise ValueError(f"'num_samples' must be positive")
        self.strategy = strategy
        
        num_samples = num_samples if (num_samples is not None) else 1
        self.last_index_added = -1
        self.all_features= np.empty((num_samples, 1))
        self.all_labels=   np.empty((num_samples))
        self.feature_dim = 0
    
    def reduce(self, features:Tensor) -> Tensor:
        """
        Reduce 4D features to 2D
            :param features (Tensor):           Features with shape (batch, seq_len, num_nodes, input_dim)
            :returns reduced_features (Tensor): Reduced features with shape (batch, feature_dim)
        """
        if features.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {features.dim()}D")
        
        match self.strategy:
            case ReductionStrategy.TEMPORAL_SPATIAL_MEAN:
                # Average over time, then over nodes
                reduced = features.mean(dim=1).mean(dim=1)
                
            case ReductionStrategy.LAST_TIME_SPATIAL_MEAN:
                # Take last time step, average over nodes
                reduced = features[:, -1, :, :].mean(dim=1)
            
            case ReductionStrategy.ATTENTION_WEIGHTED:
                # Compute attention weights for each node based on feature magnitude. This gives more weight to nodes with stronger activations
                node_importance = features.norm(dim=-1, keepdim=True)  # (batch, seq_len, num_nodes, 1)
                attention_weights = F.softmax(node_importance, dim=2)  # Softmax over nodes
                
                # Apply attention weights
                weighted_features = features * attention_weights
                
                # Sum over nodes (already weighted), then mean over time
                reduced = weighted_features.sum(dim=2).mean(dim=1)
            
            case _:
                raise NotImplementedError(f"Strategy {self.strategy} not implemented yet")
        
        return reduced
    
    def _resize(self, num_samples:int, feature_dim:int):
        """Resize the `self.all_features` and `self.all_labels` parameters to match the size of the `num_samples` and `feature_dim`"""
        resize = False
        curr_num_samples = self.all_features.shape[0]
        
        if (self.feature_dim == 0):
            self.feature_dim = feature_dim
            resize = True
        elif (self.feature_dim != feature_dim):
            raise ValueError("Feature dimension has changed during execution from {} to {}".format(self.feature_dim, feature_dim))
        
        if (self.all_features.shape[0] < num_samples):
            curr_num_samples = num_samples
            resize = True
        
        if resize:
            self.all_features.resize(curr_num_samples, self.feature_dim)
            self.all_labels.resize(curr_num_samples)
    
    def get_features(self):
        """Returns an array of shape (num_samples, feature_dim)"""
        return self.all_features
    
    def get_labels(self):
        """Returns an array of shape (num_samples,)"""
        return self.all_labels
    
    @torch.no_grad()
    def add_feature(self, encoder_output:Tensor, labels:Tensor, normalize:bool=True):
        """
        Add to the class pre-computed encoder output to fixed-size features
        
        Args:
            encoder_output (Tensor):    Tensor of shape (batch, seq_len, num_nodes, input_dim)
            label (Tensor):             Target value of the model of shape (batch, 1)
            normalize (bool):           If True, L2-normalize features (useful for cosine similarity)
        """
        if (len(labels) != len(encoder_output)):
            raise ValueError("Length of `labels` is different from length of 'encoder_output'. {} vs {}".format(len(labels), len(encoder_output)))
        
        encoder_output = encoder_output.clone()
        labels = labels.clone()
        
        # Reduce to fixed-size features
        reduced = self.reduce(encoder_output)
        
        # Optional normalization
        if normalize:
            reduced = F.normalize(reduced, p=2, dim=1)
        
        # resize the array if needed
        self._resize(reduced.shape[0] + self.last_index_added + 2, reduced.shape[1])
        
        # add
        for index in range(len(reduced)):
            self.last_index_added += 1
            self.all_features[self.last_index_added] = reduced[index].cpu().numpy()
            self.all_labels[self.last_index_added] = labels[index].squeeze(-1).cpu().numpy()
