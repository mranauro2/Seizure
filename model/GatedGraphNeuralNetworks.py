from torch_geometric.nn.resolver import activation_resolver
from typing import Callable

from torch import Tensor
import torch.nn as nn
import torch

import warnings

class Propogator(nn.Module):
    """
    Gated Propagator for GGNN using GRU-style gating mechanism
    """
    def __init__(self, state_dim:int, device:str=None):
        """
        Propagates the input through the adjacency matrix using both incoming and outgoing edges, controlled by learned reset and update gates.
        
        Args:
            state_dim (int):    Dimension of node features used in the `forward` method
            device (str):       Device to place the model on
        """
        super(Propogator, self).__init__()
        self.state_dim = state_dim

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        ).to(device=device)
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        ).to(device=device)
        self.transform = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Tanh()
        ).to(device=device)

    def forward(self, x:Tensor, supports:Tensor) -> Tensor:
        """
        Compute the new representation of the input matrix by:
        1. Aggregate features from incoming (A·x) and outgoing (A^T·x) neighbors
        2. Concatenate with current features: [A·x || A^T·x || x]
        3. Compute reset (r) and update (z) gates
        4. Compute candidate state: h_hat = tanh(W·[A·x || A^T·x || r⊙x])
        5. Update: output = (1-z)⊙x + z⊙h_hat
        
        Args:
            x (Tensor):         Node features matrix with size (batch_size, num_nodes, state_dim)
            supports (Tensor):  Adjacency matrix with size (batch_size, num_nodes, num_nodes)

        Returns:
            Tensor:             New representation of the node features matrix (batch_size, num_nodes, state_dim)
        """
        a_in =  torch.matmul(supports, x)
        a_out = torch.matmul(supports.transpose(1, 2), x)   # transposing to obtain a new (batch_size, num_nodes, num_nodes) matrix

        a = torch.cat((a_in, a_out, x), dim=2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        
        joined_input = torch.cat((a_in, a_out, r * x), dim=2)
        
        h_hat = self.transform(joined_input)

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
            
            num_steps:int,
            num_layers:int=1,
            
            act:str|Callable=nn.LeakyReLU(negative_slope=0.2),
            use_GRU:bool=True,
            
            seed:int=None,
            device:str=None
        ):
        """
        Use iterative propagation with the GRU mechanism to learn a new representation of the feature/node matrix.
        Args:
            input_dim (int):        Dimension of input node features
            num_nodes (int):        Number of nodes in both input graph and hidden state
            output_dim (int):       Dimension chosen for the output of the new feature/node matrix
            
            num_steps (int):        Number of propagation iterations
            num_layers (int):       Number of Propagation modules
            
            act (str|Callable):     The non-linear activation function to use inside the linear activation function. The last layer of has not activation function
            use_GRU (bool):         Use the GRU module instead of the standard propagator
            seed (int):             Sets the seed for the weights initializations. If None, don't use any seed
            device (str):           Device to place the model on
        """
        super(GGNNLayer, self).__init__()
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.num_steps = num_steps
        self.use_GRU= use_GRU
        act = activation_resolver(act)
        
        if (num_layers < 1):
            raise ValueError("Number of layer less than 1")
        
        self.propagators = nn.ModuleList()
        if use_GRU:
            self._forward = self._forward_GRU
            for _ in range(num_layers):
                self.propagators.append(
                    nn.GRUCell(
                        input_size = 2*input_dim,
                        hidden_size = input_dim,
                        device = device
                    ))
        else:
            self._forward = self._forward_propagator
            for _ in range(num_layers):
                self.propagators.append(
                    Propogator(input_dim, device=device)
                )
                
        self.fc = nn.Sequential(nn.Linear(input_dim, output_dim*4, device=device), act, nn.Linear(input_dim*4, output_dim, device=device))
        
        if (seed is not None):
            torch.manual_seed(seed)
        for param in self.propagators.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        for param in self.fc.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def _forward_GRU(self, x:Tensor, supports:Tensor, batch_size:int):
        """
        Compute the new representation of the feature/node matrix using standard propagation
            :param x (Tensor):          Node features matrix with size (batch_size, num_nodes*input_dim)
            :param supports (Tensor):   Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            :param batch_size (int):    Size of the batch
            :returns Tensor:            New representation of the feature/node matrix with size (batch_size, num_nodes*output_dim)
        """        
        x = x.reshape(batch_size*self.num_nodes, self.input_dim)            # x in 2D
        for _ in range(self.num_steps):
            for propagator in self.propagators:
                x_3D = x.reshape(batch_size, self.num_nodes, self.input_dim)
                
                a_in =  torch.matmul(supports, x_3D)
                a_out = torch.matmul(supports.transpose(1, 2), x_3D)            # transposing to obtain a new (batch_size, num_nodes, num_nodes) matrix
                a = torch.cat((a_in, a_out), dim=2)
                a = a.reshape(batch_size*self.num_nodes, 2*self.input_dim)      # a in 2D
                
                x = propagator.forward(a, x)

        x = x.reshape(batch_size, self.num_nodes, self.input_dim)           # x in 3D
        return x
    
    def _forward_propagator(self, x:Tensor, supports:Tensor, batch_size:int=None):
        """
        Compute the new representation of the feature/node matrix using standard propagation
            :param x (Tensor):          Node features matrix with size (batch_size, num_nodes*input_dim)
            :param supports (Tensor):   Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            :param batch_size (int):    Not used. Present for compatibility
            :returns Tensor:            New representation of the feature/node matrix with size (batch_size, num_nodes*output_dim)
        """
        propagator:nn.GRUCell= None
        for _ in range(self.num_steps):
            for propagator in self.propagators:
                x = propagator(x, supports)
            
        return x
    
    def forward(self, inputs:Tensor, supports:Tensor) -> Tensor:
        """
        Compute the new representation of the feature/node matrix via iterative propagation:
        1. Concatenate input features and hidden state
        2. Apply propagator num_steps times
        3. Apply the fully-connected network
        
        Args:
            inputs (Tensor):        Node features matrix with size (batch_size, num_nodes*input_dim)
            supports (Tensor):      Adjacency matrix with size (batch_size, num_nodes, num_nodes)

        Returns:
            Tensor:                 New representation of the feature/node matrix with size (batch_size, num_nodes*output_dim)
        """        
        batch_size = inputs.shape[0]
        inputs = inputs.reshape(batch_size, self.num_nodes, self.input_dim)

        x = self._forward(inputs, supports, batch_size)

        # (batch_size, num_nodes, output_dim)
        x:Tensor = self.fc(x)  

        x = x.reshape(batch_size, self.num_nodes*self.output_dim)
        return x
