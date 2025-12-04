from enum import Enum, auto

class GraphLearnerAttention(Enum):
    """Set the attention used in the `GraphLearner` class"""
    GRAPH_ATTENTION_LAYER   = auto()
    """Use a custom attention layer similato to `GAT`\\
    Can be add the parameter `v2` in the `GraphLearner` class to choose if the logic must be of `GATv2Conv` rather than `GATConv`"""
    GAT                     = auto()
    """Use `GAT` attention from `torch_geometric.nn`\\
    Can be add the parameter `v2` in the `GraphLearner` class to choose if use `GATv2Conv` rather than `GATConv`"""
    TRANSFORMER_CONV        = auto()
    """Use `TransformerConv` as attention from `torch_geometric.nn.conv`\\
    Can be added the parameters `concat` to concatenate the multi-head attention and `beta` to combine aggregation and skip information"""