from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GAT
from typing import Callable
from enum import Enum, auto

from torch import Tensor
import torch.nn as nn
import torch

import warnings

class GGNNType(Enum):
    """Set the module used in the `GGNNLayer` class"""
    PROPAGATOR = auto()
    """"Use custom Propagator module"""
    GRU = auto()
    """Use `GRU` module from `torch.nn.GRUCell`"""
    GAT = auto()
    """Use `GAT` attention from `torch_geometric.nn`\\
    Can be add the parameter `v2` to choose if use `GATv2Conv` rather than `GATConv` and `num_heads`"""

class Gate(nn.Module):
    def __init__(self, input_dim_1:int, input_dim_2:int, act:Callable=None, common_weights:bool=True, device:str=None):
        """
        Aggregate the inputs using two weight matrices

        Args:
            input_dim_1 (int):      Input of the first matrix passed in the forward method
            input_dim_2 (int):      Input of the second matrix passed in the forward method
            act (Callable):         Activation function to use. If None, do not use any activation function
            common_weights (bool):  Use only one common weight matrix instead of two
            device (str):           Device to place the model on
        """
        super(Gate, self).__init__()
        if common_weights:
            self.linear = nn.Linear(input_dim_1 + input_dim_2, input_dim_2, device=device)
        else:
            self.linear_1 = nn.Linear(input_dim_1, input_dim_2, device=device)
            self.linear_2 = nn.Linear(input_dim_2, input_dim_2, device=device)
        
        self.common_weights = common_weights
        self.act = act if (act is not None) else nn.Identity()
        self.act = self.act.to(device=device)
    
    def forward(self, input_1:Tensor, input_2:Tensor):        
        if self.common_weights:
            output = self.linear(torch.cat([input_1, input_2], dim=2))
        else:
            output_1 = self.linear_1(input_1)
            output_2 = self.linear_2(input_2)
            output = output_1 + output_2
            
        return self.act(output)

class Propagator(nn.Module):
    """
    Gated Propagator for GGNN using GRU-style gating mechanism
    """
    def __init__(self, state_dim:int, common_weights:bool=True, device:str=None):
        """
        Propagates the input through the adjacency matrix using both incoming and outgoing edges, controlled by learned reset and update gates.
        
        Args:
            state_dim (int):        Dimension of node features used in the `forward` method
            common_weights (bool):  Use a common weight matrix instead of different matrices
            device (str):           Device to place the model on
        """
        super(Propagator, self).__init__()
        self.state_dim = state_dim

        self.reset_gate =   Gate(state_dim * 2, state_dim, nn.Sigmoid(), common_weights, device)
        self.update_gate =  Gate(state_dim * 2, state_dim, nn.Sigmoid(), common_weights, device)
        self.transform =    Gate(state_dim * 2, state_dim, nn.Tanh(),    common_weights, device)

    def forward(self, x:Tensor, supports:Tensor) -> Tensor:
        """
        Compute the new representation of the input matrix by:
        1. Aggregate features from incoming (A·x) and outgoing (A^T·x) neighbors
        2. Compute reset (r) and update (z) gates
        3. Compute candidate state h_hat
        4. Update: output = (1-z)⊙x + z⊙h_hat
        
        Args:
            x (Tensor):         Node features matrix with size (batch_size, num_nodes, state_dim)
            supports (Tensor):  Adjacency matrix with size (batch_size, num_nodes, num_nodes)

        Returns:
            Tensor:             New representation of the node features matrix (batch_size, num_nodes, state_dim)
        """
        a_in =  torch.matmul(supports, x)
        a_out = torch.matmul(supports.transpose(1, 2), x)   # transposing to obtain a new (batch_size, num_nodes, num_nodes) matrix

        a = torch.cat((a_in, a_out), dim=2)

        r = self.reset_gate(a, x)
        z = self.update_gate(a, x)
        
        h_hat = self.transform(a, (r * x))

        output = (1 - z) * x + z * h_hat

        return output

class GGNNLayer(nn.Module):
    """
    Gated Graph Neural Networks (GGNN) Layer for learning the feature/node matrix
    """
    def __init__(
            self,
            input_dim:int,
            num_nodes:int,
            output_dim:int,
            
            type:GGNNType,
            num_steps:int,
            num_layers:int=1,
            
            act_mid:str|Callable=None,
            act_last:str|Callable=None,
            common_weights:bool=True,
            
            seed:int=None,
            device:str=None,
            
            *,
            
            v2:bool=False,
            num_heads:int=0
        ):
        """
        Use iterative propagation with the GRU mechanism to learn a new representation of the feature/node matrix.
        Args:
            input_dim (int):            Dimension of input node features
            num_nodes (int):            Number of nodes in both input graph and hidden state
            output_dim (int):           Dimension chosen for the output of the new feature/node matrix
            
            type (GGNNType):            Type of module to use
            num_steps (int):            Number of propagation iterations
            num_layers (int):           Number of Propagation modules
            
            act_mid (str|Callable):     The non-linear activation function to use between the two fully-connected layers, if provided
            act_last (str|Callable):    The non-linear activation function to use after the second fully-connected layers, if provided
            common_weights (bool):      Use a common weight matrix instead of different matrices in the Propagator modules
            seed (int):                 Sets the seed for the weights initializations. If None, don't use any seed
            device (str):               Device to place the model on
            
            v2 (bool):                  Use GATV2 instead of GAT for the multi-head attention
            num_heads (int):            Number of heads for multi-head attention
        """
        super(GGNNLayer, self).__init__()
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.num_steps = num_steps
        if (act_mid is not None):
            act_mid = activation_resolver(act_mid)
        if (act_last is not None):
            act_last = activation_resolver(act_last)
        
        # print errors
        if (type == GGNNType.GAT) and ( (num_heads is None) or (num_heads <= 0) ):
            raise ValueError("'num_heads' must be positive")
        if (num_layers < 1):
            raise ValueError("Number of layer less than 1")
        
        # print warnings
        if not(type == GGNNType.GAT):
            if (v2==True):
                msg = "The parameter 'v2' is ignored because the type is not set {}".format(GGNNType.GAT.name)
                warnings.warn(msg)
            if (num_heads is not None) and (num_heads != 0):
                msg = "The parameter 'num_heads' is ignored because the type is not set {}".format(GGNNType.GAT.name)
                warnings.warn(msg)
        if not( (type == GGNNType.PROPAGATOR) or (type == GGNNType.GRU) ) and ( (num_steps is not None) and (num_steps != 0) ):
            msg = "The parameter 'num_steps' is ignored because the type is not set {} or {}".format(GGNNType.PROPAGATOR.name, GGNNType.GRU.name)
            warnings.warn(msg)
        if not(type == GGNNType.PROPAGATOR) and (common_weights):
            msg = "'common_weights' cannot be used when the type is not {}".format(GGNNType.PROPAGATOR.name)
            warnings.warn(msg)
        
        match type:
            case GGNNType.PROPAGATOR:
                self._forward = self._forward_propagator
                self.propagators = self._build_propagator(input_dim, num_layers, common_weights, device=device)
                
            case GGNNType.GRU:
                self._forward = self._forward_GRU
                self.propagators = self._build_GRU(input_dim, num_layers, device=device)
                
            case GGNNType.GAT:
                self._forward = self._forward_GAT
                self.propagators = self._build_GAT(input_dim, input_dim, num_layers, num_heads, v2, device=device)
                
            case _:
                raise NotImplementedError("Type {} is not implemented yet".format(type))
        
        modules = []
        modules.append(nn.Linear(input_dim, output_dim*4, device=device))
        if (act_mid is not None):
            modules.append(act_mid)
        modules.append(nn.Linear(output_dim*4, output_dim, device=device))
        if (act_last is not None):
            modules.append(act_last)
        self.fc = nn.Sequential(*modules)
        
        if (seed is not None):
            torch.manual_seed(seed)
        for param in self.propagators.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        for param in self.fc.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def _build_propagator(self, input_dim:int, num_layers:int, common_weights:bool, device:str):
        """Builds standard Propagator layer(s)"""
        layers = nn.ModuleList()
        for _ in range(num_layers):
            layers.append(
                Propagator(input_dim, common_weights, device=device)
            )
        return layers
    
    def _build_GRU(self, input_dim:int, num_layers:int, device:str):
        """Builds GRU layer(s)"""
        layers = nn.ModuleList()
        for _ in range(num_layers):
            layers.append(
                nn.GRUCell(
                    input_size = 2*input_dim,
                    hidden_size = input_dim,
                    device = device
                ))
        return layers
    
    def _build_GAT(self, input_size:int, hidden_size:int, num_layers:int, num_heads:int, v2:bool, edge_dim:int=1, device:str=None):
        """Builds a GAT layer(s)"""
        # num heads  reduction if not compliant
        if (num_layers>1):
            for curr_num_heads in range(num_heads, 0, -1):
                if (hidden_size % curr_num_heads == 0):
                    if (curr_num_heads != num_heads):
                        msg = "There are {} layers, the number of heads is reduced from ({}) to ({}), otherwise the intermediate output would have been ({}) instead of ({})".format(num_layers, num_heads, curr_num_heads, num_heads * (hidden_size//num_heads), hidden_size)
                        warnings.warn(msg)
                        num_heads = curr_num_heads
                    break
        
        return GAT(
            in_channels     = input_size,
            hidden_channels = hidden_size,
            num_layers      = num_layers,
            out_channels    = input_size,
            v2              = v2,
            heads           = num_heads,
            edge_dim        = edge_dim,
        ).to(device=device)
    
    def _forward_GRU(self, x:Tensor, supports:Tensor):
        """
        Compute the new representation of the feature/node matrix using GRU cells
            :param x (Tensor):          Node features matrix with size (batch_size, num_nodes*input_dim)
            :param supports (Tensor):   Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            :returns Tensor:            New representation of the feature/node matrix with size (batch_size, num_nodes*output_dim)
        """
        batch_size = x.shape[0] 
        x = x.reshape(batch_size*self.num_nodes, self.input_dim)            # x in 2D
        for _ in range(self.num_steps):
            for propagator in self.propagators:
                x_3D = x.reshape(batch_size, self.num_nodes, self.input_dim)
                
                a_in =  torch.matmul(supports, x_3D)
                a_out = torch.matmul(supports.transpose(1, 2), x_3D)            # transposing to obtain a new (batch_size, num_nodes, num_nodes) matrix
                a = torch.cat((a_in, a_out), dim=2)
                a = a.reshape(batch_size*self.num_nodes, 2*self.input_dim)      # a in 2D
                
                x = propagator(a, x)

        x = x.reshape(batch_size, self.num_nodes, self.input_dim)           # x in 3D
        return x
    
    def _forward_propagator(self, x:Tensor, supports:Tensor):
        """
        Compute the new representation of the feature/node matrix using standard propagation
            :param x (Tensor):          Node features matrix with size (batch_size, num_nodes*input_dim)
            :param supports (Tensor):   Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            :returns Tensor:            New representation of the feature/node matrix with size (batch_size, num_nodes*output_dim)
        """
        for _ in range(self.num_steps):
            for propagator in self.propagators:
                x = propagator(x, supports)
            
        return x
    
    def _forward_GAT(self, x:Tensor, supports:Tensor):
        """
        Compute the new representation of the feature/node matrix using GAT attention mechanism
            :param x (Tensor):          Node features matrix with size (batch_size, num_nodes*input_dim)
            :param supports (Tensor):   Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            :returns Tensor:            New representation of the feature/node matrix with size (batch_size, num_nodes*output_dim)
        """    
        # Reshape: (batch_size, num_nodes*input_size) --> (batch_size, num_nodes, input_size)
        x = x.reshape(supports.size(0), supports.size(1), -1)
        batch_size, num_nodes, _ = x.shape
        
        # Create batched graph
        x_list = []
        edge_index_list = []
        edge_attr_list = []
        
        for batch_idx in range(batch_size):
            edge_index, edge_attr = dense_to_sparse(supports[batch_idx])
            edge_index_offset = edge_index + batch_idx * num_nodes              # Offset edge indices for this graph in the batch
            
            x_list.append(x[batch_idx])
            edge_index_list.append(edge_index_offset)
            edge_attr_list.append(edge_attr)
        
        # Concatenate all graphs
        x_batched = torch.cat(x_list, dim=0)                                    # (batch_size * num_nodes, input_size)
        edge_index_batched = torch.cat(edge_index_list, dim=1)                  # (2, total_edges)
        edge_attr_batched = torch.cat(edge_attr_list, dim=0).unsqueeze(-1)      # (total_edges, 1)
  
        # calculate 
        x = self.propagators(x_batched, edge_index_batched, edge_attr=edge_attr_batched)
        
        # Reshape (batch_size * num_nodes, num_nodes) --> (batch_size, num_nodes, num_nodes)
        x = x.reshape(batch_size, num_nodes, -1)
        
        return torch.sigmoid(x)
    
    def forward(self, inputs:Tensor, supports:Tensor) -> Tensor:
        """
        Compute the new representation of the feature/node matrix. The computation method depends on the `GGNNType` specified during initialization.
        1. Apply the selected propagation method
        2. Apply fully-connected network to transform to output dimension
        
        Args:
            inputs (Tensor):        Node features matrix with size (batch_size, num_nodes*input_dim)
            supports (Tensor):      Adjacency matrix with size (batch_size, num_nodes, num_nodes)

        Returns:
            Tensor:                 New representation of the feature/node matrix with size (batch_size, num_nodes*output_dim)
        """        
        batch_size = inputs.shape[0]
        inputs = inputs.reshape(batch_size, self.num_nodes, self.input_dim)

        x = self._forward(inputs, supports)

        # (batch_size, num_nodes, output_dim)
        x:Tensor = self.fc(x)  

        x = x.reshape(batch_size, self.num_nodes*self.output_dim)
        return x
