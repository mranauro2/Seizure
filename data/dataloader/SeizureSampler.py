from torch.utils.data import Sampler
from collections import defaultdict
import numpy as np

class SeizureSampler(Sampler):
    """Custom sampler for :class:`SeizureDataset`"""
    def __init__(self, labels:list[bool], valid_indices:list[int], batch_size:int, n_per_class:int, seed:int=None):
        """
        Custom Sampler ensuring each batch contains at least N samples of each class.
        
        Args:
            labels (list[bool]):        Boolean list representing class labels (True/False) for each sample
            valid_indices (list[int]):  List of valid indices that can be used for sampling
            batch_size (int):           The batch size for each iteration
            n_per_class (int):          Minimum number of samples per class in each batch
            seed (int):                 Random seed for reproducibility. If None the seed will be not custom initialize
        
        Raises:
            ValueError: If any class has fewer than `n_per_class` samples after filtering or `batch_size < (n_per_class * num_classes)`.
        """
        if seed:
            np.random.seed(seed)
        
        self.batch_size = batch_size
        self.n_per_class = n_per_class
        
        # Convert inputs to numpy arrays
        labels_array = np.array(labels)
        valid_indices_array = np.array(valid_indices)
        
        # Filter: only keep indices that are in valid_indices
        self.valid_indices = valid_indices_array
        self.labels = labels_array[valid_indices_array]         # Labels only for valid indices
        
        # Get unique classes (True/False) and organize indices by class
        self.classes = np.unique(self.labels)
        self.num_classes = len(self.classes)
        self.class_to_indices = defaultdict(list)
        
        # Map each class to its valid indices
        for i, label in enumerate(self.labels):
            original_idx = valid_indices_array[i]
            self.class_to_indices[label].append(original_idx)
        
        # Convert to numpy arrays for efficient sampling
        for label in self.class_to_indices:
            self.class_to_indices[label] = np.array(self.class_to_indices[label])
        
        # Validate constraints
        for label in self.classes:
            n_samples = len(self.class_to_indices[label])
            if n_samples < n_per_class:
                raise ValueError(f"Class {label} has only {n_samples} samples, but n_per_class={n_per_class} is required!")
        
        # Check if batch size is sufficient
        min_batch_size = n_per_class * self.num_classes
        if batch_size < min_batch_size:
            raise ValueError(f"Batch size ({batch_size}) must be at least ({min_batch_size}) to fit ({n_per_class}) samples of each of the ({self.num_classes}) classes")
        
        self.dataset_size = len(valid_indices_array)
        self.num_batches = self.dataset_size // self.batch_size
        
    def __iter__(self):
        """Iterator to yield indices for each batch while ensuring each batch has at least n_per_class samples from each class."""
        # Create shuffled indices for each class
        class_indices_shuffled = {}
        class_positions = {}
        
        for label in self.classes:
            indices = self.class_to_indices[label].copy()
            np.random.shuffle(indices)
            class_indices_shuffled[label] = indices
            class_positions[label] = 0
        
        # Track which indices have been used globally
        all_indices = self.valid_indices.copy()
        np.random.shuffle(all_indices)
        global_position = 0
        
        all_batches = []
        
        for _ in range(self.num_batches):
            batch = []
            used_in_batch = set()
            
            # First, ensure n_per_class samples from each class
            for label in self.classes:
                class_idx = class_indices_shuffled[label]
                pos = class_positions[label]
                
                # Use modulo to wrap around and reuse samples if needed
                for _ in range(self.n_per_class):
                    idx = class_idx[pos % len(class_idx)]
                    batch.append(idx)
                    used_in_batch.add(idx)
                    pos += 1
                
                class_positions[label] = pos
            
            # Fill remaining spots in batch with any samples
            remaining_slots = self.batch_size - len(batch)
            
            if remaining_slots > 0:
                # Get candidates from unused global indices
                end_pos = min(global_position + remaining_slots + self.batch_size, self.dataset_size)
                candidates = all_indices[global_position:end_pos]
                
                # Filter out indices already in batch
                available = np.setdiff1d(candidates, list(used_in_batch), assume_unique=True)
                
                # Take what we need
                n_to_take = min(remaining_slots, len(available))
                if n_to_take > 0:
                    batch.extend(available[:n_to_take].tolist())
                    global_position += len(candidates)
                
                # If still not enough, sample randomly from entire dataset
                remaining_slots = remaining_slots - n_to_take
                if remaining_slots > 0:
                    all_available = np.setdiff1d(all_indices, list(batch))
                    if len(all_available) > 0:
                        extra = np.random.choice(all_available, size=min(remaining_slots, len(all_available)), replace=False)
                        batch.extend(extra.tolist())
            
            all_batches.append(batch)
        
        # Flatten and yield indices
        for batch in all_batches:
            for idx in batch:
                yield idx
    
    def __len__(self):
        """Return the total number of samples that will be yielded."""
        return self.num_batches * self.batch_size
