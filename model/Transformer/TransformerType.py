from enum import Enum, auto

class TransformerType(Enum):
    """Set the torch Transformer used in the `Transformer` class"""
    TRANSFORMER         = auto()
    """Use `torch.nn.Transformer`"""
    TRANSFORMER_ENCODER = auto()
    """Use `torch.nn.TransformerEncoder` and apply the mean of the result\\
    The parameter `num_decoder_layers` will be ignored"""
    TRANSFORMER_DECODER = auto()
    """Use `torch.nn.TransformerDecoderLayer`\\
    The parameter `num_encoder_layers` will be ignored"""
