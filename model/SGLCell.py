import torch
import torch.nn as nn
from torch import Tensor

import warnings
from typing import Callable

from model.GraphLearner.GraphLearner import GraphLearner
from model.GatedGraphNeuralNetworks import GGNNLayer, GGNNType
from model.GraphLearner.GraphLearnerAttention import GraphLearnerAttention

class SGLC_Cell(nn.Module):
    """
    Spatio-Graph Learning Cell (SGLC) with Graph Learner, the Gated Graph Neural Networks and the GRU module
    """
    def __init__(
            self,
            input_dim:int,
            num_nodes:int,
            
            graph_skip_conn:float=0.3,
            use_GRU:bool=False,
            
            hidden_dim_GL:int=100,
            attention_type:GraphLearnerAttention=GraphLearnerAttention.TRANSFORMER_CONV,
            num_GL_layers:int=3,
            num_heads:int=8,
            dropout:float=0.0,
            epsilon:float=None,
            
            hidden_dim_GGNN:int=None,
            type_GGNN:GGNNType=GGNNType.PROPAGATOR,
            num_steps:int=5,
            num_GGNN_layers:int=1,
            act_GGNN:str|Callable=None,
            v2_GGNN:bool=False,
            num_GGNN_heads:int=0,
            
            seed:int=None,
            device:str=None,
            **kwargs
        ):
        """
        Use the Graph Learner, the Gated Graph Neural Networks and the GRU module to obtain new representations
        
        Args:
            input_dim (int):                        Feature dimension of input nodes
            num_nodes (int):                        Number of nodes in both input graph and hidden state
            
            graph_skip_conn (float):                Skip connection weight for adjacency matrix updates
            use_GRU (bool):                         Use GRU to compute a hidden state used in the Gated Graph Neural Networks module
            
            hidden_dim_GL (int):                    Hidden dimension for Graph Learner module
            attention_type (GraphLearnerAttention): Type of attention used for the Graph Learner module
            num_GL_layers (int):                    Number of message passing layers in the GAT or Transformer module for the Graph Learner module
            num_heads (int):                        Number of heads for multi-head attention in the Graph Learner module
            dropout (float):                        Dropout probability applied in the attention layer of the Graph Learner module
            epsilon (float):                        Threshold for deleting weak connections in the learned graph in the Graph Learner module. If None, no deleting is applied
            
            hidden_dim_GGNN (int):                  Hidden dimension of the hidden state for Gated Graph Neural Networks module (only if `use_GRU` is True)
            type_GGNN (GGNNType):                   Type of module to use in the Gated Graph Neural Networks module
            num_steps (int):                        Number of propagation steps in the Gated Graph Neural Networks module
            num_GGNN_layers (int):                  Number of Propagation modules in the Gated Graph Neural Networks module
            act_GGNN (str|Callable):                The non-linear activation function to use inside the linear activation function in the Gated Graph Neural Networks module. If None use the default class value
            v2_GGNN (bool):                         Use GATV2 instead of GAT for the multi-head attention in the Gated Graph Neural Networks module
            num_GGNN_heads (int):                   Number of heads for multi-head attention in the Gated Graph Neural Networks module
            
            seed (int):                             Sets the seed for the weights initializations. If None, don't use any seed
            device (str):                           Device to place the model on
            **kwargs:                               Additional arguments of `GraphLearner`
        """
        super(SGLC_Cell, self).__init__()
        self._num_nodes= num_nodes
        self._hidden_dim_GGNN= hidden_dim_GGNN
        self.use_GRU= use_GRU
        self.graph_skip_conn= graph_skip_conn
        
        if use_GRU and ((hidden_dim_GGNN is None) or (hidden_dim_GGNN<=0)):
            raise ValueError("hidden_dim_GGNN must be positive")
        if not(use_GRU) and ((hidden_dim_GGNN is not None) and (hidden_dim_GGNN != 0)):
            warnings.warn("hidden_dim_GGNN is not used because use_GRU is False")

        keys_graph_learner = ["use_sigmoid", "act", "v2", "concat", "beta"]
        kwargs_graph_learner = {key:value for key,value in kwargs.items() if key in keys_graph_learner}
        self.graph_learner = GraphLearner(
            input_size  = input_dim,
            hidden_size = hidden_dim_GL,
            num_nodes   = num_nodes,
            num_heads   = num_heads,
            attention   = attention_type,
            num_layers  = num_GL_layers,
            dropout     = dropout,
            epsilon     = epsilon,
            device      = device,
            seed        = seed,
            **kwargs_graph_learner
        )
        
        GGNN_kwargs = {}
        if act_GGNN is not None:
            GGNN_kwargs['act'] = act_GGNN
        GGNN_input= (input_dim + hidden_dim_GGNN) if use_GRU else (input_dim)
        self.ggnn = GGNNLayer(
            input_dim   = GGNN_input,
            num_nodes   = num_nodes,
            output_dim  = input_dim,
            type        = type_GGNN,
            num_steps   = num_steps,
            num_layers  = num_GGNN_layers,
            v2          = v2_GGNN,
            num_heads   = num_GGNN_heads,
            seed        = seed,
            device      = device,
            **GGNN_kwargs
        )
        
        if self.use_GRU:
            self.gru = nn.GRUCell(
                input_size  = num_nodes*input_dim,
                hidden_size = num_nodes*hidden_dim_GGNN,
                device      = device
            )
            if (seed is not None):
                torch.manual_seed(seed)
            for param in self.gru.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def forward(self, inputs:Tensor, supports:Tensor, state:Tensor=None) -> tuple[Tensor, Tensor]|tuple[Tensor, Tensor, Tensor]:
        """
        Compute the new representation of the parameters by:
        1. Learn new adjacency matrix via Graph Learner with skip connections
        2. Update node features via Gated Graph Neural Networks with sigmoid activation
        3. Update hidden state via GRUCell (if `use_GRU` is True)
        
        Args:
            inputs (Tensor):                Matrix of node with size (batch_size, num_nodes*input_dim)
            supports (Tensor):              Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            state (Tensor):                 To pass only if `use_GRU` is True. Hidden state matrix with size (batch_size, num_nodes*hidden_dim)
            
        Returns:
            tuple(Tensor, Tensor, Tensor):  A tuple containing:
                - Update matrix of node with same size
                - Update adjacency matrix with same size
                - Update hidden state matrix with same size (present only if `use_GRU` is True)
        """
        adj = self.graph_learner(inputs, supports)
        supports = self.graph_skip_conn * supports + (1 - self.graph_skip_conn) * adj
        
        # if use_GRU compute the concatenation of inputs and state, then use the GRU module
        if self.use_GRU:            
            batch_size= inputs.size(0)
            ggnn_input= torch.cat([
                inputs.reshape(batch_size, self._num_nodes, -1),
                state.reshape(batch_size, self._num_nodes, -1)
            ], dim=2)
            ggnn_input= ggnn_input.reshape(batch_size, -1)
            
            inputs = torch.sigmoid(self.ggnn(ggnn_input, supports))
            state= self.gru(inputs, state)
            
            return inputs, supports, state
        
        # if not use_GRU
        else:
            inputs = torch.sigmoid(self.ggnn(inputs, supports))
            return inputs, supports

    def hidden_state_empty(self, batch_size:int) -> Tensor:
        """
        Create an uninitialized hidden state tensor
            :param batch_size (int):   The size of the batch dimension
            :return Tensor:            Hidden state tensor with size (batch_size, num_nodes * hidden_dim_GGNN)
        """
        return torch.empty([batch_size, self._num_nodes * self._hidden_dim_GGNN])
