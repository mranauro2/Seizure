from enum import Enum, auto

class LossType(Enum):
    """Chose the loss for the training"""
    FOCAL_LOSS = auto()
    """Use `torchvision.ops.sigmoid_focal_loss`\\
    Can be add the parameters `num_classes`, `alpha` and `gamma`"""
    CROSS_ENTROPY = auto()
    """USe `torch.nn.functional.cross_entropy`\\
    Can be add the parameter `weight`"""
    BCE_LOGITS = auto()
    """Use `torch.nn.functional.binary_cross_entropy_with_logits`\\
    Can be add the parameters `num_classes` and `pos_weight`"""
