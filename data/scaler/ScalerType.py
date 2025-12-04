from enum import Enum, auto

class ScalerType(Enum):
    """Set the scaler to use in the `SeizureDataset` class"""
    Z_SCORE = auto()
    """Rescale features to have a mean of 0 and a standard deviation of 1"""
    MIN_MAX = auto()
    """Rescale features to a fixed range [0, 1]"""