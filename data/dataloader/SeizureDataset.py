from torch.utils.data import Dataset
from torch import Tensor
import torch

from collections import defaultdict
from enum import Enum, auto
from copy import deepcopy

from data.dataloader.SeizureUtilityType import SampleData, NO_AUGMENTATION
from data.dataloader.SeizureAugmentation import Augmentation
from data.scaler.Scaler import Scaler

import data.utils as utils
import numpy as np
import warnings
import os

class SeizureDatasetMethod(Enum):
    """Set the adjacency matrix method used in the `SeizureDataset` class"""
    CROSS = auto()
    """Use cross-correlation"""
    PLV = auto()
    """Use Phase Locking Value"""
    LAPLACIAN = auto()
    """Use Laplacian matrix after cross-correlation\\
    Can be add the parameter `lambda_value` to compute the scaled Laplacian matrix"""

class SeizureDataset(Dataset):
    """Custom dataset for seizure detection"""    
    def __init__(
            self,
            input_dir:str=None,
            files_record:list[str]=None,
            
            time_step_size:int=1,
            max_seq_len:int=12,
            use_fft:bool=True,
            
            preprocess_data:str=None,
            
            scaler:Scaler=None,
            augmentations:list[Augmentation]=None,
            method:SeizureDatasetMethod=SeizureDatasetMethod.CROSS,
            top_k:int=None,
            *,
            lambda_value:float=None
        ):
        """
        Args:
            input_dir (str):                    Directory to data files
            files_record (list[str]):           List of simple files with line records like `(patient_id, file_name, index, bool_value)` if `preprocess_data` is None or `(patient_id, file_name, bool_value)` if `preprocess_data` is not None where
                - `file_name` is the name of a *.npy file
                - `index` is a number which is between `0` and `time_duration_file/max_seq_len`
                - `bool_value` is `0` or `1` and corrispond to the absence or presence of a seizure

            time_step_size (int):               Duration of each time step in seconds for FFT analysis. Used only if `use_fft` is True
            max_seq_len (int):                  Total duration of the output EEG clip in seconds
            use_fft (bool):                     Use the Fast Fourier Transform when obtain the slice from the file
            
            preprocess_data (str):              Directory to the preprocess data. If it is not None `input_dir`, `time_step_size`, `max_seq_len`, `use_fft` will not be considered
            
            scaler (Scaler):                    Scaler to normalize the data. It will be applied after the Fast Fourier Transform (if present) and before the computation of the adjacency matrix
            augmentations (list[Augmentation]): Augmentations to use to augment the data. This classes will increase the number of samples adding trasformed ones
            method (str):                       How to compute the adjacency matrix
            top_k (int):                        Maintain only the `top_k` higher value when compute the adjacency matrix
            
            lambda_value (float):               Maximum eigenvalue for scaling the Laplacian matrix. If negative, computed automatically, if None compute only the Laplacian matrix 
        """
        # check warnings
        if not(method==SeizureDatasetMethod.LAPLACIAN) and (lambda_value is not None):
            msg = "The parameter lambda_value is ignored because the attention is not set {}".format(SeizureDatasetMethod.LAPLACIAN.name)
            warnings.warn(msg)
        if (preprocess_data is not None):
            ignored_parameters = []
            if (input_dir is not None):
                ignored_parameters.append('input_dir')
            if (time_step_size is not None):
                ignored_parameters.append('time_step_size')
            if (max_seq_len is not None):
                ignored_parameters.append('max_seq_len')
            if (use_fft is not None):
                ignored_parameters.append('use_fft')
            if len(ignored_parameters)>0:
                msg = "'preprocess_data' is not None. The passed following parameters are passed but ignored: '{}'".format("', '".join(ignored_parameters))
                warnings.warn(msg)
        
        # save parameters
        self.input_dir = input_dir
        self.preprocess_data = preprocess_data
        
        self.time_step_size = time_step_size
        self.max_seq_len = max_seq_len
        self.use_fft = use_fft
        
        self._scaler= scaler
        self.lambda_value= lambda_value
        self.top_k = top_k
        
        # check method to use
        match method:
            case SeizureDatasetMethod.CROSS:
                self._compute_adj = self._compute_cross
            case SeizureDatasetMethod.PLV:
                self._compute_adj = self._compute_plv
            case SeizureDatasetMethod.LAPLACIAN:
                self._compute_adj = self._compute_laplacian
            case _:
                raise NotImplementedError("Method '{}' is not implemented yet".format(method))
        
        # read files and initilize some variables
        self.file_info:list[SampleData] = []
        for file in files_record:
            data = self._read_preprocess_data_data(file) if (preprocess_data is not None) else self._read_input_dir_data(file)
            self.file_info.extend(data)
        self._targets = self._generate_targets_dict(self.file_info)
        self.apply_augmentations(augmentations)

    @property
    def scaler(self) -> Scaler:
        return self._scaler
    
    @scaler.setter
    def scaler(self, value:Scaler) -> None:
        self._scaler= value

    def apply_augmentations(self, augmentations:list[Augmentation]):
        """Apply the augmentation using the classes passed in the list"""
        self.augmentations = None
        
        # add to the variable all classes with probability not null
        if ((augmentations is not None) and (len(augmentations)>0)):
            self.augmentations = []
            for augmentation in augmentations:
                if not(augmentation.is_probability_zero()):
                    self.augmentations.append(augmentation)

            # check if at least one class was appendend
            if ( len(self.augmentations) == 0 ):
                self.augmentations = None
        
        # modify self values
        if (self.augmentations is not None):
            new_file_info:list[SampleData] = deepcopy(self.file_info)
            augmentation_index = deepcopy(NO_AUGMENTATION)
            
            for augmentation in self.augmentations:
                augmentation_index += 1
                new_file_info.extend( augmentation.generate_infos(self.file_info, augmentation_index=augmentation_index) )
            
            self.file_info = new_file_info
            self._targets = self._generate_targets_dict(self.file_info)
    
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
    
    def _generate_targets_dict(self, file_info:list[SampleData]) -> dict[str, list[int]]:
        """Generate variable used in the `targets_dict` function"""
        targets= defaultdict(list)
        for info in file_info:
            targets[info.patient_id].append(info.has_seizure)
        
        return targets
    
    def _read_input_dir_data(self, file:str):
        """Used when `preprocess_data` is None. Given a simple file returns the file informations accoring to `SampleData` class"""
        data:list[SampleData]= []
        
        with open(file, "r") as f:
            try:
                for line in f.readlines():
                    patient_id, file_name, index, has_seizure = line.split(",")
                    data.append(SampleData(
                        patient_id  = patient_id.strip(),
                        file_name   = file_name.strip(),
                        clip_index  = int(index.strip()),
                        has_seizure = bool(int(has_seizure.strip()))
                    ))
            except ValueError as e:
                if str(e).startswith("invalid literal for int() with base 10") or str(e).startswith("not enough values to unpack"):
                    e = ValueError("Expected format 'str, str, int, bool' for each line of file {}".format(file))
                raise e
        
        return data
    
    def _read_preprocess_data_data(self, file:str):
        """Used when `preprocess_data` is not None. Given a simple file returns the file informations accoring to `SampleData` class"""
        data:list[SampleData]= []
        
        with open(file, "r") as f:
            try:
                for line in f.readlines():
                    patient_id, file_name, has_seizure = line.split(",")
                    data.append(SampleData(
                        patient_id  = patient_id.strip(),
                        file_name   = file_name.strip(),
                        clip_index  = None,
                        has_seizure = bool(int(has_seizure.strip()))
                    ))
            except ValueError as e:
                if str(e).startswith("invalid literal for int() with base 10") or str(e).startswith("not enough values to unpack"):
                    e = ValueError("Expected format 'str, str, bool' for each line of file {}".format(file))
                raise e
        
        return data

    def _compute_cross(self, curr_feature:np.ndarray):
        return utils.cross_correlation(curr_feature, self.top_k)
    
    def _compute_plv(self, curr_feature:np.ndarray):
        curr_feature= curr_feature.transpose((1,0,2)).reshape(curr_feature.shape[1], -1)
        return utils.compute_plv_matrix(curr_feature)
    
    def _compute_laplacian(self, curr_feature:np.ndarray):
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
        # load or create the eeg_clip
        sample = self.file_info[index]
        if (self.preprocess_data is not None):
            eeg_clip = np.load(sample.file_name)
        else:
            resample_sig_dir = os.path.join(self.input_dir, sample.file_name)
            eeg_clip = utils.compute_slice_matrix(
                file_name       =   resample_sig_dir,
                clip_idx        =   sample.clip_index,
                time_step_size  =   self.time_step_size,
                clip_len        =   self.max_seq_len,
                use_fft         =   self.use_fft
            )
                    
        # apply augmentation if present
        if (self.augmentations is not None):
            for augmentation in self.augmentations:
                if (sample.augmentation == augmentation.index):
                    eeg_clip = augmentation.transform(eeg_clip)

        # apply scaler if present
        if (self._scaler is not None):
            eeg_clip = self._scaler.transform(eeg_clip)

        # construct adjacency matrix
        eeg_clip = eeg_clip.numpy() if isinstance(eeg_clip, Tensor) else eeg_clip
        curr_feature = eeg_clip.copy()
        adj = self._compute_adj(curr_feature)
        
        # retured values
        x = torch.FloatTensor(eeg_clip)
        y = torch.FloatTensor([sample.has_seizure])
        adj = torch.FloatTensor(adj)
        return (x, y, adj)
