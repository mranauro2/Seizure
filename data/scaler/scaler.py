import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
from typing import Callable
from abc import ABC, abstractmethod
from data.scaler.ScalerType import ScalerType

import os
import numpy as np

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ABSTRACT CLASS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class Scaler(ABC):
    """Abstract Scaler class to normalize the data"""
    def __init__(self, device:str=None):
        """
            :param device (str): Device where do computations on ('cuda', 'cpu', etc.)
        """
        self._device= device
    
    @property
    def device(self) -> str:
        return self._device
    
    @device.setter
    def device(self, value:str) -> None:
        self._device= value
    
    @abstractmethod
    def fit(self, dataset:Dataset, dataset_data_index:int=0, single_value:bool=False, func_operation:Callable[[Tensor], Tensor]=lambda t : t, use_tqdm:bool=False, device:str=None, batch_size:int=32, num_workers:int=0):
        """
        Fit the scaler by computing scaler values from the dataset

        Args:
            dataset (Dataset):                              Dataset to extract samples from
            dataset_data_index (int):                       Index of the element in dataset tuple to use
            single_value (bool):                            If True, compute single scaler values across all dimensions. If False, compute scaler values per feature (first dimension).
                                                            **Note**: Data must be 2D when `single_value=False`
            func_operation (Callable[[Tensor], Tensor]):    Optional preprocessing function to apply to each sample.\\
                                                            **Note**: The preprocessing is done on a batch of size `(batch_size, *original_shape)`
            use_tqdm (bool):                                Whether to display progress bar
            device (str):                                   Device to perform computations on ('cuda', 'cpu', etc.)
            batch_size (int):                               Number of samples to process at once
            num_workers (int):                              Number of subprocesses for data loading
        """
        pass
    
    @abstractmethod
    def transform(self, x:np.ndarray) -> Tensor:
        """
        Scale data using fitted scalar values.
            :param x (np.ndarray): Input array to scale
            :returns (Tensor): Scaled tensor
        """
        pass
    
    @abstractmethod
    def save(self, filepath:str) -> None:
        """
        Save the scalar values to a file
            :param filepath (str): Path where the scalar values will be saved
        """
        pass
    
    @staticmethod
    @abstractmethod
    def load(filepath:str, device:str=None):
        """
        Generate a new scalar using the values from a file
        
        Args:
            filepath (str): Path to the saved file
            device (str):   Device where do computations on
            
        Returns:
            Scaler:         Loaded Scaler instance
        """
        return Scaler()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ABSTRACT FACTORY CLASS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class ConcreteScaler():
    """Abstract factory class to normalize the data"""
    @staticmethod
    def create_scaler(scaler_type:ScalerType, device:str=None) -> Scaler:
        """
            :param scaler_type (ScalerType):    Type of scaler to generate
            :param device (str):                Device where do computations on ('cuda', 'cpu', etc.)
        """
        match scaler_type:
            case ScalerType.Z_SCORE:
                return StandardScaler(device=device)
            case ScalerType.MIN_MAX:
                return MinMaxScaler(device=device)
            case _:
                raise NotImplementedError("Scaler {} is not implemented yet".format(scaler_type))
                

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CONCRETE CLASSES
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class MinMaxScaler(Scaler):
    """MinMax scaler for normalizing tensor data"""
    def __init__(self, device:str=None):
        super().__init__(device=device)
        self.max_array= torch.asarray([float("-inf")])
        self.min_array= torch.asarray([float("+inf")])
    
    @property
    def max(self) -> torch.Tensor:
        return self.max_array.to(device=self.device)
    
    @max.setter
    def max(self, value:Tensor) -> None:
        self.max_array= value.to(device=self.device)
        
    @property
    def min(self) -> Tensor:
        return self.min_array.to(device=self.device)
    
    @min.setter
    def min(self, value:Tensor) -> None:
        self.min_array= value.to(device=self.device)
    
    def fit(self, dataset:Dataset, dataset_data_index:int=0, single_value:bool=False, func_operation:Callable[[Tensor], Tensor]=lambda t : t, use_tqdm:bool=False, device:str=None, batch_size:int=32, num_workers:int=0):
        """
        Fit the scaler by computing min/max values from the dataset

        Args:
            dataset (Dataset):                              Dataset to extract samples from
            dataset_data_index (int):                       Index of the element in dataset tuple to use
            single_value (bool):                            If True, compute single min/max across all dimensions. If False, compute min/max per feature (first dimension).
                                                            **Note**: Data must be 2D when `single_value=False`
            func_operation (Callable[[Tensor], Tensor]):    Optional preprocessing function to apply to each sample.\\
                                                            **Note**: The preprocessing is done on a batch of size `(batch_size, *original_shape)`
            use_tqdm (bool):                                Whether to display progress bar
            device (str):                                   Device to perform computations on ('cuda', 'cpu', etc.). If None use the default configuration
            batch_size (int):                               Number of samples to process at once
            num_workers (int):                              Number of subprocesses for data loading
        """
        if ( len(dataset)==0 ):
            raise ValueError("The dataset is empty")
        device= device if (device is not None) else self.device
        dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers)
        
        reduction_dim = None if single_value else 1
        
        if single_value:
            min_array = torch.tensor([float('+inf')])
            max_array = torch.tensor([float('-inf')])
        else:
            first_data = func_operation(next(iter(dataloader))[dataset_data_index])
            num_features = first_data.size(0)
            min_array = torch.full((num_features,1), float('+inf'))
            max_array = torch.full((num_features,1), float('-inf'))
        
        if device:
            min_array = min_array.to(device)
            max_array = max_array.to(device)
        
        for batch_tuple in (tqdm(dataloader, desc="Fitting scaler", leave=False) if use_tqdm else dataloader):
            data = func_operation(batch_tuple[dataset_data_index]).to(device)
            
            curr_min = data.amin(dim=reduction_dim).unsqueeze(-1)
            curr_max = data.amax(dim=reduction_dim).unsqueeze(-1)
            
            min_array = torch.minimum(min_array, curr_min)
            max_array = torch.maximum(max_array, curr_max)
        
        self.min = min_array
        self.max = max_array

    def transform(self, x:np.ndarray) -> Tensor:
        """
        Normalize data using fitted min/max values.
            :param x (np.ndarray): Input array to normalize
            :returns (Tensor): Normalized tensor with values in [0, 1]
        """        
        return (torch.from_numpy(x).to(device=self.device) - self.min) / (self.max - self.min + 1e-15)
    
    def save(self, filepath:str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state = {'min': self.min, 'max': self.max}
        torch.save(state, filepath)
        
    @staticmethod
    def load(filepath:str, device:str=None):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        dictionary:dict = torch.load(filepath, map_location=device)
        
        try:
            min_= dictionary['min']
            max_= dictionary['max']
        except KeyError as e:
            raise ValueError("The format of the file does not have the '{}' key".format(e.args[0]))
        
        scaler= MinMaxScaler(device)
        scaler.min= min_
        scaler.max= max_
        
        return scaler
    
    def __str__(self):
        return "Min-Max Scaler:\n   Min\t: {}\n   Max\t: {}".format(
            [v.item() for v in self.min_array],
            [v.item() for v in self.max_array]
        )

class StandardScaler(Scaler):
    """Standard scaler for normalizing tensor data using mean and standard deviation"""
    def __init__(self, device:str=None):
        super().__init__(device=device)
        self.mean_array = torch.tensor([0.0])
        self.std_array =  torch.tensor([1.0])
    
    @property
    def mean(self) -> Tensor:
        return self.mean_array.to(device=self.device)

    @mean.setter
    def mean(self, value:Tensor) -> None:
        self.mean_array= value.to(device=self.device)
    
    @property
    def std(self) -> Tensor:
        return self.std_array.to(device=self.device)
    
    @std.setter
    def std(self, value:Tensor) -> None:
        self.std_array= value.to(device=self.device)
    
    def fit(self, dataset:Dataset, dataset_data_index:int=0, single_value:bool=False, func_operation:Callable[[Tensor], Tensor]=lambda t : t, use_tqdm:bool=False, device:str=None, batch_size:int=32, num_workers:int=0):
        """
        Fit the scaler by computing mean and standard deviation values from the dataset

        Args:
            dataset (Dataset):                              Dataset to extract samples from
            dataset_data_index (int):                       Index of the element in dataset tuple to use
            single_value (bool):                            If True, compute single mean and standard deviation across all dimensions. If False, compute mean and standard deviation per feature (first dimension).\\
                                                            **Note**: Data must be 2D when `single_value=False`
            func_operation (Callable[[Tensor], Tensor]):    Optional preprocessing function to apply to each sample.\\
                                                            **Note**: The preprocessing is done on a batch of size `(batch_size, *original_shape)`
            use_tqdm (bool):                                Whether to display progress bar
            device (str):                                   Device to perform computations on ('cuda', 'cpu', etc.). If None use the default configuration
            batch_size (int):                               Number of samples to process at once
            num_workers (int):                              Number of subprocesses for data loading
        """
        device= device if (device is not None) else self.device
        dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers)
        
        reduction_dim = None if single_value else 1
        
        if single_value:
            sum_vals    = torch.tensor([0.0])
            sum_sq_vals = torch.tensor([0.0])
            count = 0
        else:
            first_data = func_operation(next(iter(dataloader))[dataset_data_index])
            num_features = first_data.size(0)
            sum_vals    = torch.zeros((num_features,1))
            sum_sq_vals = torch.zeros((num_features,1))
            count       = torch.zeros((num_features,1))
        
        if device:
            sum_vals = sum_vals.to(device)
            sum_sq_vals = sum_sq_vals.to(device)
            if isinstance(count, Tensor):
                count = count.to(device)
        
        for sample_tuple in (tqdm(dataloader, desc="Fitting scaler", leave=False) if use_tqdm else dataloader):
            data = func_operation(sample_tuple[dataset_data_index]).to(device)
            
            sum_vals    += data.sum(dim=reduction_dim).unsqueeze(-1)
            sum_sq_vals += (data ** 2).sum(dim=reduction_dim).unsqueeze(-1)
            
            count+= data.numel() if single_value else (data.size(0)*data.size(-1))
        
        # std = sqrt(E[X²] - E[X]²) = sqrt(sum_sq/count - (sum/count)²)
        mean_array = sum_vals / count
        variance = (sum_sq_vals / count) - (mean_array ** 2)
        std_array = torch.sqrt(variance)
        
        self.mean = mean_array
        self.std  = std_array
    
    def transform(self, x:np.ndarray) -> Tensor:
        """
        Standardize data using fitted mean and std values.
            :param x (np.ndarray): Input array to standardize
            :returns (Tensor): Standardized tensor with mean 0 and std 1
        """
        return (torch.from_numpy(x).to(device=self.device) - self.mean) / (self.std + 1e-15)
    
    def save(self, filepath:str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state = {'mean': self.mean, 'std': self.std}
        torch.save(state, filepath)
        
    @staticmethod
    def load(filepath:str, device:str=None):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        dictionary:dict = torch.load(filepath, map_location=device)
        
        try:
            mean_= dictionary['mean']
            std_=  dictionary['std']
        except KeyError as e:
            raise ValueError("The format of the file does not have the '{}' key".format(e.args[0]))
        
        scaler= StandardScaler(device)
        scaler.mean= mean_
        scaler.std= std_
        
        return scaler
    
    def __str__(self):
        return "Standard Scaler:\n   Mean\t\t\t: {}\n   Standard deviation\t: {}".format(
            [v.item() for v in self.mean_array],
            [v.item() for v in self.std_array]
        )
