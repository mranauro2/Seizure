from enum import Enum, auto

class SeizureDatasetMethod(Enum):
    """Set the adjacency matrix method used in the `SeizureDataset` class"""
    CROSS = auto()
    """Use cross-correlation"""
    PLV = auto()
    """Use Phase Locking Value"""
    LAPLACIAN = auto()
    """Use Laplacian matrix after cross-correlation\\
    Can be add the parameter `lambda_value` to compute the scaled Laplacian matrix"""