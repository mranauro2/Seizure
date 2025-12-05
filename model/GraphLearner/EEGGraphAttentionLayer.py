
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constant.constants_eeg import VERY_SMALL_NUMBER
from typing import Callable
from torch import Tensor

class EEGGraphAttentionLayer(nn.Module):
    """
    Unified Graph Attention Layer supporting both GAT and GATv2 architectures, similar to https://arxiv.org/abs/1710.10903 and https://arxiv.org/abs/2105.14491
    """
    def __init__(self, in_features:int, out_features:int, dropout:float=0.5, act:Callable=nn.LeakyReLU(0.2, inplace=True), use_v2:bool=False, device:str=None):
        """
        If GAT layer, similar to https://arxiv.org/abs/1710.10903
        - Generate two learnable matrix `W` and `a` which have dimension `(in_features, out_features)` and `(2*out_features, 1)` to learn where focus the attention.
        
        If GATv2 layer, similar to https://arxiv.org/abs/2105.14491
        - Generate two learnable matrix `W` and `a` which have dimension `(2*in_features, out_features)` and `(2*out_features, 1)` to learn where focus the attention.
        Args:
            in_features (int):      Dimension of input node features
            out_features (int):     Dimension chosen for hidden representation
            dropout (float):        Dropout probability applied to the final normalized attention scores
            act (Callable):         The non-linear activation function to use
            use_v2 (bool):          If True uses GATv2, otherwise uses GAT
            device (str):           Device to place the model on
        """
        super(EEGGraphAttentionLayer, self).__init__()
        if (act is None):
            raise TypeError("act must be not None")

        if use_v2:
            self._compute_attention = self._compute_attention_v2
            self.w = nn.Parameter(torch.empty(size=(2*in_features, out_features), device=device))
            self.a = nn.Parameter(torch.empty(size=(out_features, 1), device=device))
        else:
            self._compute_attention = self._compute_attention_v1
            self.w = nn.Parameter(torch.empty(size=(in_features, out_features), device=device))
            self.a = nn.Parameter(torch.empty(size=(2*out_features, 1), device=device))
        
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.dropout= nn.Dropout(p=dropout)
        self.activation_function = act

    def _prepare_attentional_mechanism_input(self, tensor:Tensor):
        """
        Generates a tensor containing the Cartesian product of the first dimension of the input
            :param  tensor (Tensor):                Tensor of shape `(num_nodes, other_dimension)`
            :return Cartesian_product (Tensor):     Tensor of shape `(num_nodes, num_nodes, 2*other_dimension)`
        """
        N = tensor.size()[0]
        tensor_repeated_in_chunks = tensor.repeat_interleave(N, dim=0)
        tensor_repeated_alternating = tensor.repeat(N, 1)

        all_combinations_matrix = torch.cat([tensor_repeated_in_chunks, tensor_repeated_alternating], dim=1)
        
        return all_combinations_matrix.view(N, N, -1)
    
    def _compute_attention_v1(self, h: Tensor) -> Tensor:
        """
        Attention mechanism of GAT.\\
        Calculate the attention as a matrix of size `(num_nodes, num_nodes)` calculating for each element of the matrix `e_(i,j)=activation_function(a^T*[W*h_i||W*h_j])`:
        - `Wh = h × W` --> `(num_nodes, in_features) × (in_features, out_features)`
        - `cartesian_product` as the cartesian product of `Wh` --> `(num_nodes, num_nodes, 2*out_features)`
        - `coefficient = activation_function( cartesian_product × a )`
        
            :param  h (Tensor):         Node features matrix with size (num_nodes, in_features)
            :return attention (Tensor): Adjacency attention matrix with size (num_nodes, num_nodes)
        """
        Wh = torch.matmul(h, self.w)
        cartesian_product = self._prepare_attentional_mechanism_input(Wh)
        e = self.activation_function( torch.matmul(cartesian_product, self.a ).squeeze(2))
        
        return e
    
    def _compute_attention_v2(self, h: Tensor) -> Tensor:
        """
        Attention mechanism of GATv2.\\
        Calculate the attention as a matrix of size `(num_nodes, num_nodes)` calculating for each element of the matrix `e_(i,j)=a^T*activation_function(W*[h_i||h_j])`:
        - `cartesian_product` as the cartesian product of `h` --> `(num_nodes, num_nodes, 2*in_features)`
        - `Wh = activation_function( cartesian_product × W )` --> `(num_nodes, num_nodes, 2*in_features) × (2*in_features, out_features)`
        - `coefficient =  Wh × a`
        
            :param  h (Tensor):         Node features matrix with size (num_nodes, in_features)
            :return attention (Tensor): Adjacency attention matrix with size (num_nodes, num_nodes)
        """
        cartesian_product = self._prepare_attentional_mechanism_input(h)
        Wh = self.activation_function( torch.matmul(cartesian_product, self.w) )
        e = torch.matmul(Wh, self.a).squeeze(2)
        
        return e
    
    def forward(self, h:Tensor, adj:Tensor) -> Tensor:
        """
        Calculate the attention as a matrix of size `(num_nodes, num_nodes)` calculating for each element of the matrix:
        - `e_(i,j)=activation_function(a^T*[W*h_i||W*h_j])` if use GAT:
            - `Wh = h × W` --> `(num_nodes, in_features) × (in_features, out_features)`
            - `cartesian_product` as the cartesian product of `Wh` --> `(num_nodes, num_nodes, 2*out_features)`
            - `coefficient = activation_function( cartesian_product × a )`
        - `e_(i,j)=a^T*activation_function(W*[h_i||h_j])` if use GATv2
            - `cartesian_product` as the cartesian product of `h` --> `(num_nodes, num_nodes, 2*in_features)`
            - `Wh = activation_function( cartesian_product × W )` --> `(num_nodes, num_nodes, 2*in_features) × (2*in_features, out_features)`
        
        - make zero all values corresponding to not positive value in the adjacency matrix (it is used a low value but not zero)
        
        Args:
            h (Tensor):         Node features matrix with size (num_nodes, in_features)
            adj (Tensor):       Adjacency matrix with size (num_nodes, num_nodes)
            
        Returns:
            attention (Tensor): Normalized djacency attention matrix with size (num_nodes, num_nodes)
        """
        e= self._compute_attention(h)

        zero_vec = -VERY_SMALL_NUMBER * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        return attention
