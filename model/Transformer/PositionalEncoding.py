from abc import ABC, abstractmethod
from enum import Enum, auto
from torch import Tensor

import torch
import math

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TYPE ENUMERATION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class PositionalEncodingType(Enum):
    """Set the positional encoding used in the `Transformer` class"""
    SINUSOIDAL = auto()
    """Fixed sinusoidal positional encoding from 'Attention is All You Need (https://arxiv.org/abs/1706.03762)'. \\
    The parameter `num_inputs` must be set"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# INTERFACE
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class PositionalEncoding(ABC):
    @abstractmethod
    def __getitem__(self, inputs:Tensor) -> Tensor:
        pass

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CONCRETE CLASSES
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class SinusoidalPositionalEncoding(PositionalEncoding):
    def __init__(self, d_model:int, max_len:int, device:str=None):
        """
        Fixed sinusoidal positional encoding from 'Attention is All You Need (https://arxiv.org/abs/1706.03762)'. \\
        Adds position-dependent signals to token embeddings so the model can reason about word order

        Formula:
            PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

        Args:
            d_model (int): Input dimension
            max_len (int): Maximum sequence length this class will support
            device (str):  Device where do computations on ('cuda', 'cpu', etc.)
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension: shape becomes [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=device)
    
    def __getitem__(self, x:Tensor) -> Tensor:
        """
        Add positional encodings to the input
            :param x (Tensor):          Input of shape of shape `(batch_size, seq_len, d_model)` where `seq_len` can be at max equal to `max_len`
            :returns output (Tensor):   Output of same shape of the input with positional encodings added
        """
        _, seq_len, d_model = x.shape
        
        pe_value = self.pe[:, 0:seq_len, :]     # extract positional encoding values from the class
        pe_value = pe_value[:, :, 0:d_model]    # make possible to have d_model odd by cutting the last value if necessary

        return (x + pe_value)
