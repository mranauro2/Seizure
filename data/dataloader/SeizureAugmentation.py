from data.dataloader.SeizureUtilityType import SampleSeizureData
from typing_extensions import override
from abc import ABC, abstractmethod
from typing import NamedTuple
import numpy as np

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ABSTRACT CLASS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class Augmentation(ABC):
    """Abstract augmentation class"""
    def __init__(self, labels:list, p:float, seed:int=0):
        """
        Args:
            labels (list):  List of labels to augment. All labels no found in the dataset are ignored
            p (float):      Probability of a sample to be augmented
            seed (int):     Seed to initialize the random generator
        """
        if not(0 <= p <= 1):
            raise ValueError("'p' must be between 0 and 1")
        self.random_generator = np.random.default_rng(seed)
        self.labels = labels
        self.p = p
    
    def generate_infos(self, infos:list[SampleSeizureData], affected_patient_ids:list[str], augmentation_index:int) -> list[SampleSeizureData]:
        """
        Generate an augmented list with the same file in the dataset
        Args:
            infos (list[SampleSeizureData]):        List of informations according to `SampleSeizureData` class
            affected_patient_ids (list[str]):       Which patient ids will be affected by the augmentation. Set to None to affect all
            augmentation_index (int):               Index which replace the existing one in the `SampleSeizureData` class
        Returns:
            new_infos (list[SampleSeizureData]):    New list of informations according to `SampleSeizureData` class where the augmentation_index is replaced when appropriate
        """
        self._index = augmentation_index
        
        new_infos:list[SampleSeizureData] = []
        for sample in infos:
            if (sample.has_seizure in self.labels) and ((affected_patient_ids is None) or (sample.patient_id in affected_patient_ids)):
                if (self.random_generator.random() < self.p):
                    new_infos.append(sample._replace(augmentation=augmentation_index))
        
        return new_infos
    
    @property
    def index(self) -> any:
        return self._index
    
    def is_probability_zero(self) -> bool:
        """Returns True if the probability is zero, False otherwise"""
        return (self.p <= 0)
    
    def is_probability_one(self) -> bool:
        """Returns True if the probability is one, False otherwise"""
        return (self.p >= 1)
    
    @abstractmethod
    def transform(self, eeg_clip:np.ndarray) -> np.ndarray:
        """
        Transform the input according to the class description. If the probability is 0 no transformation is applied
            :param eeg_clip (np.ndarray):               EEG clip with shape (clip_len, num_channels, frequency)
            :returns eeg_clip_transform (np.ndarray):   EEG clip transformed
        """
        raise NotImplementedError("Function in abstract class not implemented")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CONCRETE CLASSES
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class SwapFixedChannels(Augmentation):
    """Swap the channels of the data"""
    class Channels(NamedTuple):
        index_0:int
        index_1:int
    
    def __init__(self, channels_to_swap:list[tuple[int,int]], **kwargs):
        """
        Swap the channels of the data
        
        Args:
            channels_to_swap (list[tuple[int,int]): List of the indices of the channels to swap. The indices are the channels to swap
            **kwargs:
                see :class:`Augmentation` for required parameters 
        """
        super().__init__(**kwargs)
        
        if (channels_to_swap is None) or len(channels_to_swap)==0:
            raise ValueError("'channels_to_swap' must be not empty")
        
        self.channels:list[SwapFixedChannels.Channels] = []
        for item in channels_to_swap:
            channel = self.Channels(item[0], item[1])
            if (channel.index_0 == channel.index_1):
                raise ValueError("Channels to swap are identical: ({}) and ({})".format(channel.index_0, channel.index_1))
            self.channels.append(channel)
            
    @override
    def transform(self, eeg_clip:np.ndarray):
        # if the probability is null do nothing
        if self.is_probability_zero():
            return eeg_clip
        
        eeg_clip = eeg_clip.copy()
        for channel in self.channels:
            eeg_clip[:, [channel.index_0, channel.index_1], :] = eeg_clip[:, [channel.index_1, channel.index_0], :]
        
        return eeg_clip

class SwapRandomChannels(Augmentation):
    """Swap the channels of the data"""    
    def __init__(self, num_channels:int, **kwargs):
        """
        Swap a default number of channels of the data
        
        Args:
            num_channels (int): Number of channels to swap. The channels swapped can be any
            **kwargs:
                see :class:`Augmentation` for required parameters
        """
        super().__init__(**kwargs)
        
        if (num_channels is None) or (num_channels < 2):
            raise ValueError("'num_channels' must have at least 2 channels to swap")

        self.num_channels_to_swap = num_channels
            
    @override
    def transform(self, eeg_clip:np.ndarray):
        # if the probability is null do nothing
        if self.is_probability_zero():
            return eeg_clip
        
        num_channels = eeg_clip.shape[1]
        channels_to_swap = self.random_generator.choice(num_channels, self.num_channels_to_swap, replace=False)
        
        eeg_clip = eeg_clip.copy()
        for index in range(len(channels_to_swap)-1):
            eeg_clip[:, [index, index+1], :] = eeg_clip[:, [index+1, index], :]
        
        return eeg_clip

class Scale(Augmentation):
    """Scale the input"""
    OPERATIONS = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: x / y,
        '//': lambda x, y: x // y,
        '**': lambda x, y: x ** y,
    }
    
    def __init__(self, interval:list[int,int], operation:str, **kwargs):
        """
        Scale the input using a value choosen in the interval

        Args:
            interval (list[int,int]):   Interval to use
            operation (str):            Operation to use for the scale, like '+', '-', ...
            **kwargs:
                see :class:`Augmentation` for required parameters
        """
        super().__init__(**kwargs)
        self.min = interval[0]
        self.max = interval[1]
        
        if self.min >= self.max:
            raise ValueError("The max value ({}) is not greather than the min value ({})".format(self.max, self.min))

        if (operation not in Scale.OPERATIONS.keys()):
            raise ValueError("Operation value '{}' not acceptable. Use one of '{}'".format("', '".join(Scale.OPERATIONS.keys())))
        
        self.operation = Scale.OPERATIONS[operation]
    
    @override
    def transform(self, eeg_clip:np.ndarray):
        # if the probability is null do nothing
        if self.is_probability_zero():
            return eeg_clip
        
        scale = self.random_generator.uniform(self.min, self.max)
        
        eeg_clip = eeg_clip.copy()
        eeg_clip = self.operation(eeg_clip, scale)
        
        return eeg_clip
    