import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import warnings
from utils.constants_eeg import INF

from torch_geometric.nn import GAT
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.utils import dense_to_sparse

class GraphLearner(nn.Module):
    """
    Graph learning layer that computes a new learned adjacency matrix
    """
    def __init__(self, input_size:int, hidden_size:int, num_nodes:int=21, num_heads:int=16, use_GATv2:bool=False, num_layers:int=3, use_Transformer:bool=False, concat:bool=False, beta:bool=False, dropout:float=0.5, epsilon:float=None, device:str=None):
        """
        Use the multi-head GAT layers to learn a new representation of the adjacency matrix.
        Args:
            input_size (int):       Dimension of input node features
            hidden_size (int):      Hidden dimension for attention computation
            num_nodes (int):        Number of nodes in input graph
            num_heads (int):        Number of heads for multi-head attention
            
            use_GATv2 (bool):       Use GATV2 instead of GAT for the multi-head attention
            num_layers (int):       Used only if `use_Transformer` is False. Number of message passing layers in the GAT module
            
            use_Transformer (bool): Use `TransformerConv` for multi-head attention instead of GAT. If True the parameter `use_GATv2` and `num_layers` are ignored
            concat (bool):          Used only if `use_Transformer` is True. If True the multi-head attentions are concatenated, otherwise are averaged
            beta (bool):            Used only if `use_Transformer` is True. If True will combine aggregation and skip information
            
            dropout (float):        Dropout probability applied in the attention layer
            epsilon (float):        Threshold for deleting weak connections in the learned graph. If None, no deleting is applied
            device (str):           Device to place the model on
        """
        super(GraphLearner, self).__init__()
        self.epsilon = epsilon
        
        if use_Transformer:
            if concat:
                for curr_num_heads in range(num_heads, 0, -1):
                    if (num_nodes % curr_num_heads == 0):
                        if (curr_num_heads != num_heads):
                            warnings.warn("The number of heads in reduced from ({}) to ({}) because concat is True, otherwise the output it would have been ({}) instead of ({})".format(num_heads, curr_num_heads, num_heads * (num_nodes//num_heads), num_nodes))
                            num_heads= curr_num_heads
                            break
            
            out_channels= num_nodes//num_heads if concat else num_nodes
            self.att= TransformerConv(
                in_channels=input_size,
                out_channels=out_channels,
                heads=num_heads,
                concat=concat,
                beta=beta,
                dropout=dropout,
                edge_dim=1
            ).to(device=device)
        
        else:
            self.att= GAT(
                in_channels=input_size,
                hidden_channels=hidden_size,
                num_layers=num_layers,
                out_channels=num_nodes,
                dropout=dropout,
                v2=use_GATv2,
                heads=num_heads,
                edge_dim=1
            ).to(device=device)
        
        for param in self.att.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def forward(self, context:Tensor, adj:Tensor) -> Tensor:
        """
        Compute the attention scores for each head and for each batch by:
        1. Compute attention scores for each head independently
        2. Average attention across all heads
        3. Apply epsilon thresholding if specified (delete weak edges)
        4. Apply sigmoid activation for non-negative edge weights
        
        Args:
            context (Tensor):   Node features matrix with size (batch_size, num_nodes*input_size)
            adj (Tensor):       Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            
        Returns:
            attention (Tensor): Adjacency attention matrix with size (batch_size, num_nodes, num_nodes)
        """
        # Reshape: (batch_size, num_nodes*input_size) --> (batch_size, num_nodes, input_size)
        context = context.reshape(adj.size(0), adj.size(1), -1)
        
        batch_size, num_nodes, _ = context.shape
    
        # Create batched graph in PyG format
        x_list = []
        edge_index_list = []
        edge_attr_list = []
        
        for batch_idx in range(batch_size):
            edge_index, edge_attr = dense_to_sparse(adj[batch_idx])
            edge_index_offset = edge_index + batch_idx * num_nodes          # Offset edge indices for this graph in the batch
            
            x_list.append(context[batch_idx])
            edge_index_list.append(edge_index_offset)
            edge_attr_list.append(edge_attr)
        
        # Concatenate all graphs
        x_batched = torch.cat(x_list, dim=0)                        # (batch_size * num_nodes, input_size)
        edge_index_batched = torch.cat(edge_index_list, dim=1)      # (2, total_edges)
        edge_attr_batched = torch.cat(edge_attr_list, dim=0)        # (total_edges, 1)
        
        attention_batched = self.att.forward(x_batched, edge_index_batched, edge_attr=edge_attr_batched)
        
        # Reshape (batch_size * num_nodes, num_nodes) --> (batch_size, num_nodes, num_nodes)
        attention = attention_batched.reshape(batch_size, num_nodes, -1)

        # Modify the attention matrix by setting values below epsilon to a marker
        if self.epsilon is not None:
            mask= (attention > self.epsilon).detach().float()
            attention= torch.where(mask==1, attention, -INF)

        attention= F.sigmoid(attention)

        return attention
