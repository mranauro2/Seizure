from torch.utils.data import Dataset
from torch import Tensor, FloatTensor
import torch

from typing_extensions import override
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Any
from enum import Enum, auto
from copy import deepcopy

from data.dataloader.SeizureUtilityType import SampleSeizureData, NextTimeData, NO_AUGMENTATION
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ABSTRACT CLASS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class BaseSeizureDataset(Dataset, ABC):
    """Custom base dataset for seizure detection and next time prediction"""
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
            input_dir (str):                    Directory to data files
            files_record (list[str]):           List of simple files with line records structure according to the concrete class

            time_step_size (int):               Duration of each time step in seconds for FFT analysis. Used only if `use_fft` is True
            max_seq_len (int):                  Total duration of the output EEG clip in seconds
            use_fft (bool):                     Use the Fast Fourier Transform when obtain the slice from the file
            
            preprocess_data (str):              Directory to the preprocess data. If it is not None `input_dir`, `time_step_size`, `max_seq_len`, `use_fft` will not be considered
            
            scaler (Scaler):                    Scaler to normalize the data. It will be applied after the Fast Fourier Transform (if present) and before the computation of the adjacency matrix
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

        self._scaler = scaler
        self.top_k = top_k
        self.lambda_value = lambda_value

        self._configure_method(method)

        # parse metadata
        self.file_info = []
        for file in files_record or []:
            self.file_info.extend(self._parse_record_file(file))

        self._targets = self._generate_targets_dict(self.file_info)

    @property
    def scaler(self):
        return self._scaler

    @scaler.setter
    def scaler(self, value:Scaler):
        self._scaler = value
    
    def _configure_method(self, method:SeizureDatasetMethod):
        """Check the method to use and assign it"""
        match method:
            case SeizureDatasetMethod.CROSS:
                self._compute_adj = self._compute_cross
            case SeizureDatasetMethod.PLV:
                self._compute_adj = self._compute_plv
            case SeizureDatasetMethod.LAPLACIAN:
                self._compute_adj = self._compute_laplacian
            case _:
                raise NotImplementedError("Method '{}' is not implemented yet".format(method))

    def _compute_adj(self, x:np.ndarray) -> np.ndarray:
        """Compute the adjacency matrix"""
        raise NotImplementedError("This is an abstract function of an abstract class")
    
    def _compute_cross(self, x:np.ndarray):
        return utils.cross_correlation(x, self.top_k)

    def _compute_plv(self, x:np.ndarray):
        x = x.transpose((1, 0, 2)).reshape(x.shape[1], -1)
        return utils.compute_plv_matrix(x)

    def _compute_laplacian(self, x:np.ndarray):
        adj = self._compute_cross(x)
        return utils.normalize_laplacian_spectrum(adj, self.lambda_value)
    
    @abstractmethod
    def _parse_record_file(self, file:str) -> list:
        """Given a simple file returns the file informations according to the concrete class"""
        raise NotImplementedError("This is an abstract function of an abstract class")

    def _generate_targets_dict(self, file_info:list[SampleSeizureData]|list[NextTimeData]) -> dict[str, list[int]]:
        """Generate variable used in the `targets_dict` function"""
        targets= defaultdict(list)
        for info in file_info:
            targets[info.patient_id].append(info.has_seizure)
        
        return targets

    @abstractmethod
    def _build_target(self, sample:np.ndarray) -> Tensor:
        """Return the target tensor according to the concrete class"""
        raise NotImplementedError("This is an abstract function of an abstract class")

    def __len__(self):
        return len(self.file_info)

    def targets_dict(self) -> dict[str, list[Any]]:
        """Returns target labels organized by patient where the key is the patient ID and the value is a list of object according to the concrete class"""
        return self._targets

    def _load_clip(self, file_name:str, index:int) -> np.ndarray:
        """Load a clip given its file_name and its index (can be not present)"""
        if self.preprocess_data is not None:
            eeg_clip = np.load(file_name)
        else:
            eeg_clip = utils.compute_slice_matrix(
                file_name       = os.path.join(self.input_dir, file_name),
                clip_idx        = index,
                time_step_size  = self.time_step_size,
                clip_len        = self.max_seq_len,
                use_fft         = self.use_fft,
        )
        
        return eeg_clip
    
    def _apply_scaler(self, eeg_clip:np.ndarray) -> np.ndarray:
        if (self.scaler is not None):
            eeg_clip = self.scaler.transform(eeg_clip).numpy()
        
        return eeg_clip

    def __getitem__(self, index:int):
        """
        Args:
            index (int):    Index in [0, 1, ..., size_of_dataset-1]
        """
        raise NotImplementedError("This is an abstract function of an abstract class")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CONCRETE CLASSES
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class SeizureDatasetDetection(BaseSeizureDataset):
    """Custom dataset for seizure detection"""
    def __init__(self, **kwargs):
        """
        Args:
            files_record (list[str]): List of simple files with line records like `(patient_id, file_name, index, bool_value)` if `preprocess_data` is None or `(patient_id, file_name, bool_value)` if `preprocess_data` is not None where
                    - `file_name` is the name of a *.npy file
                    - `index` is a number which is between `0` and `time_duration_file/max_seq_len`
                    - `bool_value` is `0` or `1` and corrispond to the absence or presence of a seizure
        
        See:
        ----
            The others parameters are in the :class:`BaseSeizureDataset` class
        
        """
        super().__init__(**kwargs)
        self
        self.apply_augmentations(None)

    def apply_augmentations(self, augmentations:list[Augmentation], affected_patient_ids:list[str]=None):
        """
        Apply the augmentation
            :param augmentations (list[Augmentation]):  List of augmentation classes to use
            :param affected_patient_ids (list[str]):    List of patient ids which will be affected by the augmentation. Set to None to affect all
        """
        self.augmentations = None
        
        # add to the variable all classes with probability not null
        if ((augmentations is not None) and (len(augmentations)>0)):
            self.augmentations:list[Augmentation] = []
            for augmentation in augmentations:
                if not(augmentation.is_probability_zero()):
                    self.augmentations.append(augmentation)

            # check if at least one class was appendend
            if ( len(self.augmentations) == 0 ):
                self.augmentations = None
        
        # modify self values
        if (self.augmentations is not None):
            new_file_info:list[SampleSeizureData] = deepcopy(self.file_info)
            augmentation_index = deepcopy(NO_AUGMENTATION)
            
            for augmentation in self.augmentations:
                augmentation_index += 1
                new_file_info.extend( augmentation.generate_infos(self.file_info, affected_patient_ids, augmentation_index=augmentation_index) )
            
            self.file_info_before_augmentaion = self.file_info
            self.file_info = new_file_info
            self._targets = self._generate_targets_dict(self.file_info)
    
    def remove_augmentation(self):
        """Remove all augmentations applied. The dataset could be reduced"""
        if (self.augmentations is not None):
            self.file_info = self.file_info_before_augmentaion
            self.augmentations = None
    
    @override
    def _parse_record_file(self, file:str):
        """Given a simple file returns the file informations accoring to `SampleSeizureData` class"""
        data:list[SampleSeizureData]= []
        
        # case preprocess_data None
        if (self.preprocess_data is None):
            with open(file, "r") as f:
                try:
                    for line in f.readlines():
                        patient_id, file_name, index, has_seizure = line.split(",")
                        data.append(SampleSeizureData(
                            patient_id  = patient_id.strip(),
                            file_name   = file_name.strip(),
                            clip_index  = int(index.strip()),
                            has_seizure = bool(int(has_seizure.strip()))
                        ))
                except ValueError as e:
                    if str(e).startswith("invalid literal for int() with base 10") or str(e).startswith("not enough values to unpack") or str(e).startswith("too many values to unpack"):
                        e = ValueError("Expected format 'str, str, int, bool' for each line of file {}".format(file))
                    raise e
                
        # case preprocess_data not None
        else:
            with open(file, "r") as f:
                try:
                    for line in f.readlines():
                        patient_id, file_name, has_seizure = line.split(",")
                        data.append(SampleSeizureData(
                            patient_id  = patient_id.strip(),
                            file_name   = file_name.strip(),
                            clip_index  = None,
                            has_seizure = bool(int(has_seizure.strip()))
                        ))
                except ValueError as e:
                    if str(e).startswith("invalid literal for int() with base 10") or str(e).startswith("not enough values to unpack") or str(e).startswith("too many values to unpack"):
                        e = ValueError("Expected format 'str, str, bool' for each line of file {}".format(file))
                    raise e
        
        return data

    @override
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
            patient_to_indices[info.patient_id].append(idx)
        return patient_to_indices

    @override
    def _build_target(self, sample:SampleSeizureData):
        return torch.FloatTensor([sample.has_seizure])
    
    @override
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
        eeg_clip = self._load_clip(sample.file_name, sample.clip_index)
                    
        # apply augmentation if present
        if (self.augmentations is not None):
            for augmentation in self.augmentations:
                if (sample.augmentation == augmentation.index):
                    eeg_clip = augmentation.transform(eeg_clip)

        # apply scaler if present
        eeg_clip = self._apply_scaler(eeg_clip)

        # construct adjacency matrix
        adj = self._compute_adj(eeg_clip)
        
        # retured values
        x = torch.FloatTensor(eeg_clip)
        y = self._build_target(sample)
        adj = torch.FloatTensor(adj)
        
        return (x, y, adj)

class SeizureDatasetPrediction(BaseSeizureDataset):
    """Custom dataset for seizure next time prediction"""    
    def __init__(self, **kwargs):
        """
        Args:
            files_record (list[str]):   List of simple files with line records like `(patient_id, file_name, index, file_name_next, index_next, bool_value)` if `preprocess_data` is None or `(patient_id, file_name, file_name_next, bool_value)` if `preprocess_data` is not None where
                - `file_name` and `file_name_next` are the names of a *.npy file
                - `index` and `index_next` are the numbers which is between `0` and `time_duration_file/max_seq_len`
                - `bool_value` is `0` or `1` and corrispond to the absence or presence of a seizure
        
        See:
        ----
            The others parameters are in the :class:`BaseSeizureDataset` class
        
        """
        super().__init__(**kwargs)
    
    @override
    def _parse_record_file(self, file:str):
        """Given a simple file returns the file informations accoring to `NextTimeData` class"""
        data:list[NextTimeData]= []
        
        # case preprocess_data None
        if (self.preprocess_data is None):
            with open(file, "r") as f:
                try:
                    for line in f.readlines():
                        patient_id, file_name, index, file_name_next, index_next, has_seizure = line.split(",")
                        data.append(NextTimeData(
                            patient_id      = patient_id.strip(),
                            file_name       = file_name.strip(),
                            clip_index      = int(index.strip()),
                            file_name_next  = file_name_next.strip(),
                            clip_index_next = int(index_next.strip()),
                            has_seizure     = int(has_seizure.strip())
                        ))
                except ValueError as e:
                    if str(e).startswith("invalid literal for int() with base 10") or str(e).startswith("not enough values to unpack") or str(e).startswith("too many values to unpack"):
                        e = ValueError("Expected format 'str, str, int, str, int, int' for each line of file {}".format(file))
                    raise e
                
        # case preprocess_data not None
        else:
            with open(file, "r") as f:
                try:
                    for line in f.readlines():
                        patient_id, file_name, file_name_next, has_seizure = line.split(",")
                        data.append(NextTimeData(
                            patient_id      = patient_id.strip(),
                            file_name       = file_name.strip(),
                            clip_index      = None,
                            file_name_next  = file_name_next.strip(),
                            clip_index_next = None,
                            has_seizure     = int(has_seizure.strip())
                        ))
                except ValueError as e:
                    if str(e).startswith("invalid literal for int() with base 10") or str(e).startswith("not enough values to unpack") or str(e).startswith("too many values to unpack"):
                        e = ValueError("Expected format 'str, str, str, int' for each line of file {}".format(file))
                    raise e
        
        return data

    @override
    def targets_dict(self) -> dict[str, list[int]]:
        """Returns target labels organized by patient where the key is the patient ID and the value is a list of binary labels (0 or 1) for each sample belonging to that patient"""
        return self._targets
    
    def targets_index_map(self) -> dict[str, list[int]]:
        """Returns dataset indices organized by patient where the key is the patient ID and the value is a list of of dataset indices for all samples belonging to that patient"""
        patient_to_indices = defaultdict(list)
        for idx,info in enumerate(self.file_info):
            patient_to_indices[info.patient_id].append(idx)
        return patient_to_indices

    @override
    def _build_target(self, sample:NextTimeData):
        eeg_clip_next = self._load_clip(sample.file_name_next, sample.clip_index_next)
        eeg_clip_next = self._apply_scaler(eeg_clip_next)
        return torch.FloatTensor(eeg_clip_next), torch.FloatTensor([sample.has_seizure])
        
    @override
    def __getitem__(self, index:int):
        """
        Args:
            index (int):    Index in [0, 1, ..., size_of_dataset-1]
            
        Returns:
            tuple (Tensor, Tensor, Tensor):     The triplets is:
                - Feature/node matrix with shape (max_seq_len, num_channels, feature_dim)
                - Feature/node matrix to predict with shape (max_seq_len, num_channels, feature_dim)
                - Adjacency matrix with shape (num_channels, num_channels)
        
        Notes:
        ------
            The number of channels depend on the file. If `use_fft` is False the `feature_dim` is equal to `frequency` otherwise it is equal to `frequency/2`
        """
        # load or create the eeg_clip
        sample:NextTimeData = self.file_info[index]
        eeg_clip = self._load_clip(sample.file_name, sample.clip_index)
    
        # apply scaler if present
        eeg_clip = self._apply_scaler(eeg_clip)
        
        # construct adjacency matrix
        adj = self._compute_adj(eeg_clip)
        
        # retured values
        x = torch.FloatTensor(eeg_clip)
        x_next = self._build_target(sample)
        adj = torch.FloatTensor(adj)
        return (x, x_next, adj)
