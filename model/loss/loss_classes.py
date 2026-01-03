from torch.nn.functional import one_hot, binary_cross_entropy_with_logits, cross_entropy, l1_loss, mse_loss
from torchvision.ops import sigmoid_focal_loss

from typing_extensions import override
from abc import ABC, abstractmethod
from torch import Tensor
import torch

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ABSTRACT CLASS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class Loss(ABC):    
    """Abstract Scaler class to normalize the data"""
    @abstractmethod
    def compute_loss(self, result:Tensor, target:Tensor) -> Tensor:
        raise NotImplementedError("This is an abstract function of an abstract class")

    def parameters(self):
        """Retuns all parameters of the class"""
        return vars(self)

class LossDetection(Loss):
    @abstractmethod
    def compute_loss(self, result:Tensor, target:Tensor) -> Tensor:
        """
        Use a specific loss class to compute the loss
        
        Args:
            result (Tensor):    Result of the model with size (batch_size, num_classes)
            target (Tensor):    Target value with size (batch_size, 1)
        
        Returns:
            loss (Tensor):      Loss of shape (batch_size)
        """
        raise NotImplementedError("This is an abstract function of an abstract class")

class LossPrediction(Loss):
    @abstractmethod
    def compute_loss(self, result:Tensor, target:Tensor) -> Tensor:
        """
        Use a specific loss class to compute the loss
        
        Args:
            result (Tensor):    Result of the model of any size
            target (Tensor):    Target value with the same size
        
        Returns:
            loss (Tensor):      Scalar loss
        """
        raise NotImplementedError("This is an abstract function of an abstract class")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CONCRETE CLASSES OF DETECTION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class FocalLoss(LossDetection):
    """Use `torchvision.ops.sigmoid_focal_loss`"""
    def __init__(self, num_classes:int, alpha:float, gamma:float):
        """
            :param num_classes (int):  Total numeber of classes
            :param alpha (float):      Weighting factor in range [0, 1] to balance positive vs negative examples or -1 for ignore. See `sigmoid_focal_loss` for more detail
            :param gamma (float):      Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples. See `sigmoid_focal_loss` for more detail
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
    
    @override
    def compute_loss(self, result:Tensor, target:Tensor):
        target_one_hot= one_hot(target.squeeze(-1).to(dtype=torch.int64), num_classes=self.num_classes)
        target_one_hot= target_one_hot.to(dtype=target.dtype)

        return sigmoid_focal_loss(inputs=result, targets=target_one_hot, alpha=self.alpha, gamma=self.gamma, reduction='none').sum(dim=1)

class BCE_Logits(LossDetection):
    """Use `torch.nn.functional.binary_cross_entropy_with_logits`"""
    def __init__(self, num_classes:int, pos_weight:Tensor=None):
        """
            :param num_classes (int):   Total numeber of classes
            :param pos_weight (Tensor): A weight of positive examples to be broadcasted with target. Must be a tensor with equal size along the class dimension to the number of classes. See `binary_cross_entropy_with_logits` for more detail
        """
        if (pos_weight is not None) and (list(pos_weight.shape)!=[num_classes]):
                msg = "pos_weight if not None must have shape equal to the number of classes ({}) but got ({})".format(num_classes, list(pos_weight.shape))
                raise ValueError(msg)
        
        super().__init__()
        self.num_classes = num_classes
        self.pos_weight = pos_weight
        
    @override
    def compute_loss(self, result:Tensor, target:Tensor):
        target_one_hot= one_hot(target.squeeze(-1).to(dtype=torch.int64), num_classes=self.num_classes)
        target_one_hot= target_one_hot.to(dtype=target.dtype)
        
        return binary_cross_entropy_with_logits(result, target_one_hot, pos_weight=self.pos_weight, reduction="none").sum(dim=1)

class CrossEntropy(LossDetection):
    """Use `torch.nn.functional.cross_entropy`"""
    def __init__(self, weight:Tensor=None):
        """:param weight (Tensor): a manual rescaling weight given to each class. If given, must have size equal to the number of classes"""
        super().__init__()
        self.weight = weight
    
    @override
    def compute_loss(self, result:Tensor, target:Tensor):
        return cross_entropy(result, target.squeeze(-1).to(dtype=torch.int64), weight=self.weight, reduction='none')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CONCRETE CLASSES OF DETECTION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class MAE(LossPrediction):
    """Mean absolute error: use `torch.nn.functional.l1_loss`"""
    @override
    def compute_loss(self, result:Tensor, target:Tensor):
        return l1_loss(result, target)

class MSE(LossPrediction):
    """Mean square error: use `torch.nn.functional.mse_loss`"""
    @override
    def compute_loss(self, result:Tensor, target:Tensor):
        return mse_loss(result, target)
