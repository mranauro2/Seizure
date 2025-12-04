import torch
import torch.nn as nn

from utils.constant.constants_eeg import INF
from model.EEGGraphAttentionLayer import EEGGraphAttentionLayer
from model.GraphLearner.GraphLearnerAttention import GraphLearnerAttention

from torch_geometric.nn import GAT
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.resolver import activation_resolver

from torch import Tensor
from typing import Callable

import warnings

class GraphLearner(nn.Module):
    """
    Graph learning layer that computes a new learned adjacency matrix
    """
    def __init__(
            self,
            input_size:int,
            hidden_size:int,
            num_nodes:int=21,
            num_heads:int=16,
            attention:GraphLearnerAttention=GraphLearnerAttention.GRAPH_ATTENTION_LAYER,
            num_layers:int=3,
            dropout:float=0.5,
            epsilon:float=None,
            device:str=None,
            *,
            act:str|Callable='relu',
            v2:bool=False,
            concat:bool=False,
            beta:bool=False
        ):
        """
        Use the multi-head GAT layers to learn a new representation of the adjacency matrix.
        Args:
            input_size (int):                   Dimension of input node features
            hidden_size (int):                  Hidden dimension for attention computation
            num_nodes (int):                    Number of nodes in input graph
            num_heads (int):                    Number of heads for multi-head attention
            attention (GraphLearnerAttention):  Type of attention used
            num_layers (int):                   Number of message passing layers in the module
            
            dropout (float):                    Dropout probability applied in the attention layer
            epsilon (float):                    Threshold for deleting weak connections in the learned graph. If None, no deleting is applied
            device (str):                       Device to place the model on
            
            act (str|Callable):                 The non-linear activation function to use
            v2 (bool):                          Use GATV2 instead of GAT for the multi-head attention
            concat (bool):                      If True the multi-head attentions are concatenated, otherwise are averaged
            beta (bool):                        If True will combine aggregation and skip information
        """
        super(GraphLearner, self).__init__()
        self.epsilon = epsilon
        self.attention_type= attention
        
        # Check errors
        if (num_layers < 1):
            raise ValueError("Number of layer less than 1")
        if (act is None):
            raise TypeError("act must be not None")
        act = activation_resolver(act)
        
        # print warnings
        if not(attention==GraphLearnerAttention.GAT or attention==GraphLearnerAttention.GRAPH_ATTENTION_LAYER) and (v2==True):
            msg = "The parameter v2 is ignored because the attention is not set {} or {}".format(GraphLearnerAttention.GAT.name, GraphLearnerAttention.GRAPH_ATTENTION_LAYER.name)
            warnings.warn(msg)
        if not(attention==GraphLearnerAttention.TRANSFORMER_CONV):
            if (concat==True):
                msg = "The parameter concat is ignored because the attention is not set {}".format(GraphLearnerAttention.TRANSFORMER_CONV.name)
                warnings.warn(msg)
            if (beta==True):
                msg = "The parameter beta is ignored because the attention is not set {}".format(GraphLearnerAttention.TRANSFORMER_CONV.name)
                warnings.warn(msg)
        
        # set attention type
        match self.attention_type:
            case GraphLearnerAttention.GRAPH_ATTENTION_LAYER:
                self.att = self._build_graph_attention(input_size, hidden_size, num_layers, num_nodes, num_heads, dropout, act, v2, device=device)
                self.forward = self._forward_graph_attention
            case GraphLearnerAttention.GAT:
                self.att = self._build_gat(input_size, hidden_size, num_layers, num_nodes, num_heads, dropout, act, v2, device=device)
                self.forward = self._forward_gat
            case GraphLearnerAttention.TRANSFORMER_CONV:
                self.att = self._build_transformers(input_size, hidden_size, num_layers, num_nodes, num_heads, dropout, act, concat, beta, device=device)
                self.forward = self._forward_transformer
            case _:
                raise NotImplementedError("Attention {} is not implemented yet".format(self.attention_type))
        
        # initialize
        for param in self.att.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def _build_graph_attention(self, input_size:int, hidden_size:int, num_layers:int, num_nodes:int, num_heads:int, dropout:float, act:Callable, v2:bool, edge_dim:int=1, device:str=None):
        """Builds a stack of EEGGraphAttentionLayer layers based on the GAT pattern"""
        layers = nn.ModuleList()
        for _ in range(num_heads):
            layers.append(
                EEGGraphAttentionLayer(
                    in_features  = input_size,
                    out_features = hidden_size,
                    dropout      = dropout,
                    act          = act,
                    use_v2       = v2
                )
            )
            
        return layers.to(device=device)
    
    def _build_gat(self, input_size:int, hidden_size:int, num_layers:int, num_nodes:int, num_heads:int, dropout:float, act:Callable, v2:bool, edge_dim:int=1, device:str=None):
        """Builds a GAT layer(s)"""
        # num heads  reduction if not compliant
        if (num_layers>1):
            for curr_num_heads in range(num_heads, 0, -1):
                if (hidden_size % curr_num_heads == 0):
                    if (curr_num_heads != num_heads):
                        msg = "There are more layers, the number of heads is reduced from ({}) to ({}), otherwise the intermediate output would have been ({}) instead of ({})".format(num_heads, curr_num_heads, num_heads * (hidden_size//num_heads), hidden_size)
                        warnings.warn(msg)
                        num_heads = curr_num_heads
                    break
        
        return GAT(
            in_channels     = input_size,
            hidden_channels = hidden_size,
            num_layers      = num_layers,
            out_channels    = num_nodes,
            dropout         = dropout,
            act             = act,
            v2              = v2,
            heads           = num_heads,
            edge_dim        = edge_dim,
        ).to(device=device)
    
    def _build_transformers(self, input_size:int, hidden_size:int, num_layers:int, num_nodes:int, num_heads:int, dropout:float, act:Callable, concat:bool, beta:bool, edge_dim:int=1, device:str=None) -> nn.ModuleList:
        """Builds a stack of TransformerConv layers based on the GAT pattern"""
        # num heads and hidden size reduction if not compliant
        if concat:
            for curr_num_heads in range(num_heads, 0, -1):
                if (num_nodes % curr_num_heads == 0):
                    if (curr_num_heads != num_heads):
                        msg = "concat is True, the number of heads is reduced from ({}) to ({}), otherwise the output would have been ({}) instead of ({})".format(num_heads, curr_num_heads, num_heads * (num_nodes//num_heads), num_nodes)
                        warnings.warn(msg)
                        num_heads = curr_num_heads
                    break
            for curr_hidden_size in range(hidden_size, 0, -1):
                if (curr_hidden_size % num_heads == 0):
                    if (curr_hidden_size != hidden_size):
                        msg = "concat is True, hidden size is reduced from ({}) to ({}) because it was not divisible by the number of heads ({})".format(hidden_size, curr_hidden_size, num_heads)
                        warnings.warn(msg)
                        hidden_size = curr_hidden_size
                    break
        
        layers = nn.ModuleList()
        out_channels = hidden_size//num_heads if concat else hidden_size
        
        # First layer
        block = nn.ModuleList()
        block.append(
            TransformerConv(
                in_channels  = input_size,
                out_channels = out_channels,
                heads        = num_heads,
                concat       = concat,
                beta         = beta,
                dropout      = dropout,
                edge_dim     = edge_dim
            )
        )
        if (num_layers > 1):
            block.append(act)
        layers.append(block)
        
        # Intermediate layers
        for _ in range(num_layers-1):
            block= nn.ModuleList([
                TransformerConv(
                    in_channels  = hidden_size,
                    out_channels = out_channels,
                    heads        = num_heads,
                    concat       = concat,
                    beta         = beta,
                    dropout      = dropout,
                    edge_dim     = edge_dim
                ),
                act
            ])
            layers.append(block)
        
        # Last layer
        if (num_layers > 1):
            block = nn.ModuleList([
                TransformerConv(
                    in_channels  = hidden_size,
                    out_channels = num_nodes//num_heads if concat else num_nodes,
                    heads        = num_heads,
                    concat       = concat,
                    beta         = beta,
                    dropout      = dropout,
                    edge_dim     = edge_dim
                )
            ])
            layers.append(block)
            
        return layers.to(device=device)

    def forward(self, context:Tensor, adj:Tensor) -> Tensor:
        """
        Compute the attention scores
            :param context (Tensor):    Node features matrix with size (batch_size, num_nodes*input_size)
            :param adj (Tensor):        Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            :return attention (Tensor): Adjacency attention matrix with size (batch_size, num_nodes, num_nodes)
        """
        raise NotImplementedError("This function is only a decoration")
    
    def _forward_graph_attention(self, context:Tensor, adj:Tensor) -> Tensor:
        """
        Compute the attention scores
            :param context (Tensor):    Node features matrix with size (batch_size, num_nodes*input_size)
            :param adj (Tensor):        Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            :return attention (Tensor): Adjacency attention matrix with size (batch_size, num_nodes, num_nodes)
        """
        # Reshape: (batch_size, num_nodes*input_size) --> (batch_size, num_nodes, input_size)
        context = context.reshape(adj.size(0), adj.size(1), -1)
        
        attention = []
        graph_attention:EEGGraphAttentionLayer= None
        for graph_attention in self.att:
            attention_head = []
            for i in range(context.size(0)):
                h = context[i]
                ad = adj[i]
                attention_ = graph_attention.forward(h, ad)
                attention_head.append(attention_)
            attention_head = torch.stack(attention_head, 0)
            attention.append(attention_head)

        attention = torch.mean(torch.stack(attention, 0), 0)

        # Modify the attention matrix by setting values below epsilon to a marker
        if self.epsilon is not None:
            attention = self._build_epsilon_neighbourhood(attention)

        return attention

    def _forward_gat(self, context:Tensor, adj:Tensor) -> Tensor:
        """
        Compute the attention scores
            :param context (Tensor):    Node features matrix with size (batch_size, num_nodes*input_size)
            :param adj (Tensor):        Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            :return attention (Tensor): Adjacency attention matrix with size (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = adj.shape
        x_batched, edge_index_batched, edge_attr_batched = self._create_batch(context, adj)
        
        attention_batched = self.att.forward(x_batched, edge_index_batched, edge_attr=edge_attr_batched)
        
        # Reshape (batch_size * num_nodes, num_nodes) --> (batch_size, num_nodes, num_nodes)
        attention = attention_batched.reshape(batch_size, num_nodes, -1)

        # Modify the attention matrix by setting values below epsilon to a marker
        if self.epsilon is not None:
            attention = self._build_epsilon_neighbourhood(attention)
        
        return attention
    
    def _forward_transformer(self, context:Tensor, adj:Tensor) -> Tensor:
        """
        Compute the attention scores
            :param context (Tensor):    Node features matrix with size (batch_size, num_nodes*input_size)
            :param adj (Tensor):        Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            :return attention (Tensor): Adjacency attention matrix with size (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = adj.shape
        x_batched, edge_index_batched, edge_attr_batched = self._create_batch(context, adj)
        
        layer:nn.Module= None
        attention_batched = x_batched
        for block in self.att:
            transformer:TransformerConv = block[0]
            attention_batched = transformer.forward(attention_batched, edge_index_batched, edge_attr=edge_attr_batched)
            
            for layer in block[1:]:
                attention_batched = layer.forward(attention_batched)
        
        attention = attention_batched.reshape(batch_size, num_nodes, -1)

        # Modify the attention matrix by setting values below epsilon to a marker
        if self.epsilon is not None:
            attention = self._build_epsilon_neighbourhood(attention)
        
        return attention
    
    def _create_batch(self, context:Tensor, adj:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Generate a batch given the node feature matrix and the adjacency matrix
        
        Args:
            context (Tensor):    Node features matrix with size (batch_size, num_nodes*input_size)
            adj (Tensor):        Adjacency matrix with size (batch_size, num_nodes, num_nodes)
        
        Returns:
            tuple(Tensor, Tensor, Tensor):
                - `context_batch`: Node features matrix in a batch with size (batch_size * num_nodes, input_size)
                - `edge_index_batch`: Edge indices of a sparse adjacency matrix with size (2, total_edges)
                - `edge_attr_batch`: Edge attributes of a sparse adjacency matrix with size (total_edges, 1)
        """
        # Reshape: (batch_size, num_nodes*input_size) --> (batch_size, num_nodes, input_size)
        context = context.reshape(adj.size(0), adj.size(1), -1)
        batch_size, num_nodes, _ = context.shape
        
        # Create batched graph
        x_list = []
        edge_index_list = []
        edge_attr_list = []
        # Create batched graph
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
        x_batched = torch.cat(x_list, dim=0)                                    # (batch_size * num_nodes, input_size)
        edge_index_batched = torch.cat(edge_index_list, dim=1)                  # (2, total_edges)
        edge_attr_batched = torch.cat(edge_attr_list, dim=0).unsqueeze(-1)      # (total_edges, 1)
        
        return x_batched, edge_index_batched, edge_attr_batched

    def _build_epsilon_neighbourhood(self, attention:Tensor, markoff_value:float=-INF):
        """Modify the attention matrix by setting values below epsilon to a marker"""
        mask = (attention > self.epsilon).detach().float()
        attention = torch.where(mask==1, attention, markoff_value)
        return attention
