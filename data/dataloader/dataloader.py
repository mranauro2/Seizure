from torch.utils.data import Dataset, Sampler
from torch import Tensor
import torch

from collections import defaultdict

from data.dataloader.SeizureDatasetMethod import SeizureDatasetMethod
from data.scaler.scaler import Scaler

import data.utils as utils
import numpy as np
import warnings
import os

class SeizureDataset(Dataset):
    def __init__(
            self,
            input_dir:str=None,
            files_record:list[str]=None,
            time_step_size:int=1,
            max_seq_len:int=12,
            use_fft:bool=True,
            preprocess_data:str=None,
            scaler:Scaler=None,
            method:SeizureDatasetMethod=SeizureDatasetMethod.CROSS,
            top_k:int=None,
            *,
            lambda_value:float=None
        ):
        """
        Args:
            input_dir (str):            Directory to data files
            files_record (list[str]):   List of simple files with line records like `(patient_id, file_name, index, bool_value)` if `preprocess_data` is None or `(patient_id, file_name, bool_value)` if `preprocess_data` is not None where
                - `file_name` is the name of a *.npy file
                - `index` is a number which is between `0` and `time_duration_file/max_seq_len`
                - `bool_value` is `0` or `1` and corrispond to the absence or presence of a seizure
            time_step_size (int):       Duration of each time step in seconds for FFT analysis. Used only if `use_fft` is True
            max_seq_len (int):          Total duration of the output EEG clip in seconds
            use_fft (bool):             Use the Fast Fourier Transform when obtain the slice from the file
            
            preprocess_data (str):      Directory to the preprocess data. If it is not None `input_dir`, `time_step_size`, `max_seq_len`, `use_fft` will not be considered
            
            scaler (Scaler):            Scaler to normalize the data. It will be applied after the Fast Fourier Transform (if present) and before the computation of the adjacency matrix
            
            method (str):               How to compute the adjacency matrix
            top_k (int):                Maintain only the `top_k` higher value when compute the adjacency matrix
            
            lambda_value (float):       Maximum eigenvalue for scaling the Laplacian matrix. If negative, computed automatically, if None compute only the Laplacian matrix 
        """
        if not(method==SeizureDatasetMethod.LAPLACIAN) and (lambda_value is not None):
            msg = "The parameter lambda_value is ignored because the attention is not set {}".format(SeizureDatasetMethod.LAPLACIAN.name)
            warnings.warn(msg)
        
        self.input_dir = input_dir
        self.preprocess_data = preprocess_data
        
        self.time_step_size = time_step_size
        self.max_seq_len = max_seq_len
        self.use_fft = use_fft
        
        self._scaler= scaler

        self.lambda_value= lambda_value
        self.top_k = top_k
        
        self.method= method
        match self.method:
            case SeizureDatasetMethod.CROSS:
                self._compute_adj = self._compute_cross
            case SeizureDatasetMethod.PLV:
                self._compute_adj = self._compute_plv
            case SeizureDatasetMethod.LAPLACIAN:
                self._compute_adj = self._compute_laplacian
            case _:
                raise NotImplementedError("Method {} is not implemented yet".format(self.method))
        
        self.file_info= list()
        for file in files_record:
            datas= self._read_preprocess_data_data(file) if (preprocess_data is not None) else self._read_input_dir_data(file)
            self.file_info.extend(datas)
        
        self._targets= defaultdict(list)
        for info in self.file_info:
            patient_id= info[0]
            target= info[-1]
            self._targets[patient_id].append(target)

    @property
    def scaler(self) -> Scaler:
        self._scaler
    
    @scaler.setter
    def scaler(self, value:Scaler) -> None:
        self._scaler= value

    def __len__(self):
        return len(self.file_info)
    
    def targets_dict(self) -> dict[str, list[int]]:
        """Returns target labels organized by patient where the key is the patient ID and the value is a list of binary labels (0 or 1) for each sample belonging to that patient"""
        return self._targets
    
    def targets_list(self) -> list[int]:
        """Returns all target labels as a flat list. The list contains binary labels (0 or 1) of all samples in the dataset, in the same order as dataset indices"""
        return [label for info_list in self._targets.values() for label in info_list]
    
    def targets_index_map(self) -> dict[str, list[int]]:
        """Returns dataset indices organized by patient where the key is the patient ID and the value is a list of of dataset indices for all samples belonging to that patient"""
        patient_to_indices = defaultdict(list)
        for idx,info in enumerate(self.file_info):
            patient_to_indices[info[0]].append(idx)
        return patient_to_indices
    
    def _read_input_dir_data(self, file:str):
        """Given a simple file returns the file informations accoring to the constructor when `preprocess_data` is None"""
        data= list()
        
        file_name:str=None
        index:str=None
        has_seizure:str=None
        
        with open(file, "r") as f:
            try:
                for line in f.readlines():
                    patient_id, file_name, index, has_seizure = line.split(",")
                    data.append((
                        patient_id.strip(),
                        file_name.strip(),
                        int(index.strip()),
                        bool(int(has_seizure.strip()))
                    ))
            except ValueError as e:
                if str(e).startswith("invalid literal for int() with base 10") or str(e).startswith("not enough values to unpack"):
                    e = ValueError("Expected format 'str, str, int, bool' for each line of file {}".format(file))
                raise e
        
        return data
    
    def _read_preprocess_data_data(self, file:str):
        """Given a simple file returns the file informations accoring to the constructor when `preprocess_data` is not None"""
        data= list()
        
        file_name:str=None
        has_seizure:str=None
        
        with open(file, "r") as f:
            try:
                for line in f.readlines():
                    patient_id, file_name, has_seizure = line.split(",")
                    data.append((
                        patient_id.strip(),
                        file_name.strip(),
                        bool(int(has_seizure.strip()))
                    ))
            except ValueError as e:
                if str(e).startswith("invalid literal for int() with base 10") or str(e).startswith("not enough values to unpack"):
                    e = ValueError("Expected format 'str, str, bool' for each line of file {}".format(file))
                raise e
        
        return data

    def _compute_cross(self, curr_feature:np.ndarray|Tensor):
        return utils.cross_correlation(curr_feature, self.top_k)
    
    def _compute_plv(self, curr_feature:np.ndarray|Tensor):
        curr_feature= curr_feature.numpy()
        curr_feature= curr_feature.transpose((1,0,2)).reshape(curr_feature.shape[1], -1)
        return utils.compute_plv_matrix(curr_feature)
    
    def _compute_laplacian(self, curr_feature:np.ndarray|Tensor):
        adj_cross_corr = self._compute_cross(curr_feature)
        return utils.normalize_laplacian_spectrum(adj_cross_corr, self.lambda_value)

    def __getitem__(self, index:int):
        """
        Args:
            index (int):    Index in [0, 1, ..., size_of_dataset-1]
            
        Returns:
            tuple (Tensor, Tensor, Tensor):     The triplets is:
                - Feature/node matrix with shape (max_seq_len, num_channels, feature_dim)
                - Target of the current graph
                - Adjacency matrix with shape (num_channels, num_channels)
        
        Notes:
        ------
            The number of channels depend on the file. If `use_fft` is False the `feature_dim` is equal to `frequency` otherwise it is equal to `frequency/2`
        """
        # compute EEG clip
        if (self.preprocess_data is not None):
            _, npy_file_name, has_seizure = self.file_info[index]    
            eeg_clip = np.load(npy_file_name)
        else:
            _, npy_file_name, clip_idx, has_seizure = self.file_info[index]
            resample_sig_dir = os.path.join(self.input_dir, npy_file_name)
            eeg_clip = utils.compute_slice_matrix(
                file_name       =   resample_sig_dir,
                clip_idx        =   clip_idx,
                time_step_size  =   self.time_step_size,
                clip_len        =   self.max_seq_len,
                use_fft         =   self.use_fft
            )
        
        curr_feature = eeg_clip.copy()
        
        if (self._scaler is not None):
            curr_feature= self._scaler.transform(curr_feature)

        adj = self._compute_adj(curr_feature)
        
        # transform in tensor all numpy arrays
        x = torch.FloatTensor(eeg_clip)
        y = torch.FloatTensor([has_seizure])
        adj= torch.FloatTensor(adj)

        return (x, y, adj)

class SeizureSampler(Sampler):
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
