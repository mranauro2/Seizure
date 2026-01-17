import torch
import torch.nn as nn
from torch import Tensor

from model.SGLCell import SGLC_Cell
from model.GatedGraphNeuralNetworks import GGNNType
from model.GraphLearner.GraphLearnerAttention import GraphLearnerAttention

import warnings
from typing import Callable

class SGLC_Encoder(nn.Module):
    """
    Spatio-Graph Learning Cell (SGLC) Encoder using multiple SGLCell layers
    """
    def __init__(
            self,
            num_cells:int,
            input_dim:int,
            num_nodes:int,
            
            graph_skip_conn:float=0.3,
            use_GRU:bool=False,
            hidden_per_step:bool=True,
            
            hidden_dim_GL:int=100,
            attention_type:GraphLearnerAttention=None,
            num_GL_layers:int=3,
            num_heads:int=16,
            dropout:float=0,
            epsilon:float=None,
            
            hidden_dim_GGNN:int=None,
            type_GGNN:GGNNType=GGNNType.PROPAGATOR,
            num_steps:int=5,
            num_GGNN_layers:int=1,
            act_mid_GGNN:str|Callable=None,
            act_last_GGNN:str|Callable=None,
            v2_GGNN:bool=False,
            num_GGNN_heads:int=0,
            
            seed:int=None,
            device:str=None,
            **kwargs
        ):
        """
        Use a stack of SGLCell to learn from the data. Each SGLCell use the GL, the GGNN and the GRUCell module
        
        Args:
            num_cells (int):                        Number of the SGLCell layers in the stack
            input_dim (int):                        Feature dimension of input nodes
            num_nodes (int):                        Number of nodes in both input graph and hidden state
            
            graph_skip_conn (float):                Skip connection weight for adjacency matrix updates
            use_GRU (bool):                         Use GRU to compute a hidden state used in the Gated Graph Neural Networks module
            hidden_per_step (bool):                 Use a new hidden state for each time step (only if `use_GRU` is True)
            
            hidden_dim_GL (int):                    Hidden dimension for the Graph Learner module
            attention_type (GraphLearnerAttention): Type of attention used for the Graph Learner module
            num_GL_layers (int):                    Number of message passing layers in the GAT or Transformer module for the Graph Learner module
            num_heads (int):                        Number of heads for multi-head attention in the Graph Learner module
            dropout (float):                        Dropout probability applied in the attention layer of the Graph Learner module
            epsilon (float):                        Threshold for deleting weak connections in the learned graph in the Graph Learner module. If None, no deleting is applied
            
            hidden_dim_GGNN (int):                  Hidden dimension of the hidden state for Gated Graph Neural Networks module (only if `use_GRU` is True)
            type_GGNN (GGNNType):                   Type of module to use in the Gated Graph Neural Networks module
            num_steps (int):                        Number of propagation steps in the Gated Graph Neural Networks module
            num_GGNN_layers (int):                  Number of Propagation modules in the Gated Graph Neural Networks module
            act_mid_GGNN (str|Callable):            The non-linear activation function to use between the two fully-connected layers in the Gated Graph Neural Networks module, if provided
            act_last_GGNN (str|Callable):           The non-linear activation function to use after the second fully-connected layers in the Gated Graph Neural Networks module, if provided
            v2_GGNN (bool):                         Use GATV2 instead of GAT for the multi-head attention in the Gated Graph Neural Networks module
            num_GGNN_heads (int):                   Number of heads for multi-head attention in the Gated Graph Neural Networks module
            
            seed (int):                             Sets the seed for the weights initializations. If None, don't use any seed
            device (str):                           Device to place the model on
            **kwargs:                               Additional arguments of `SGLC_Cell`
        """
        super(SGLC_Encoder, self).__init__()

        self.use_GRU = use_GRU
        self.hidden_per_step = hidden_per_step
        
        if not(use_GRU) and (hidden_per_step):
            warnings.warn("Parameter 'hidden_per_step' is ignored because 'use_GRU' is False")
        
        if not(use_GRU):
            self._forward = self._cell_forward
        elif not(hidden_per_step):
            self._forward = self._cell_forward_GRU
        else:
            self._forward = self._cell_forward_GRU_hidden_per_step
        
        encoding_cells = list()
        for _ in range(num_cells):
            encoding_cells.append(
                SGLC_Cell(
                    input_dim       = input_dim,
                    num_nodes       = num_nodes,
                    
                    graph_skip_conn = graph_skip_conn,
                    use_GRU         = use_GRU,
                    
                    hidden_dim_GL   = hidden_dim_GL,
                    attention_type  = attention_type,
                    num_GL_layers   = num_GL_layers,
                    num_heads       = num_heads,
                    dropout         = dropout,
                    epsilon         = epsilon,
                    
                    hidden_dim_GGNN = hidden_dim_GGNN,
                    type_GGNN       = type_GGNN,
                    num_steps       = num_steps,
                    num_GGNN_layers = num_GGNN_layers,
                    act_mid_GGNN    = act_mid_GGNN,
                    act_last_GGNN   = act_last_GGNN,
                    v2_GGNN         = v2_GGNN,
                    num_GGNN_heads  = num_GGNN_heads,
                    
                    seed            = seed,
                    device          = device,
                    **kwargs
                )
            )
        self.encoding_cells = nn.ModuleList(encoding_cells)
    
    def _cell_forward(self, input_t:Tensor, supports:Tensor, hidden_state:Tensor=None):
        """
        Compute the new representation of the parameters using the stack of SGLCell modules
            :param inputs (Tensor):                 Matrix of node with size (sequential_length, batch_size, num_nodes, input_dim)
            :param supports (Tensor):               Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            :param initial_hidden_state (Tensor):   Not used. Present for compatibility
            :returns tuple(Tensor, Tensor, Tensor): Update matrices of the input
        """
        for cell in self.encoding_cells:
            input_t, supports = cell(input_t, supports)
        
        return input_t, supports, hidden_state
    
    def _cell_forward_GRU(self, input_t:Tensor, supports:Tensor, hidden_state:Tensor=None):
        """
        Compute the new representation of the parameters using the stack of SGLCell modules
            :param inputs (Tensor):                 Matrix of node with size (sequential_length, batch_size, num_nodes, input_dim)
            :param supports (Tensor):               Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            :param initial_hidden_state (Tensor):   Hidden state matrix with size (batch_size, num_nodes\*hidden_dim)
            :returns tuple(Tensor, Tensor, Tensor): Update matrices of the input
        """
        for cell in self.encoding_cells:
            input_t, supports, hidden_state = cell(input_t, supports, hidden_state)
        
        return input_t, supports, hidden_state
    
    def _cell_forward_GRU_hidden_per_step(self, input_t:Tensor, supports:Tensor, hidden_state:Tensor=None):
        """
        Compute the new representation of the parameters using the stack of SGLCell modules
            :param inputs (Tensor):                 Matrix of node with size (sequential_length, batch_size, num_nodes, input_dim)
            :param supports (Tensor):               Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            :param initial_hidden_state (Tensor):   Hidden state matrix with size (num_cells, batch_size, num_nodes\*hidden_dim)
            :returns tuple(Tensor, Tensor, Tensor): Update matrices of the input
        """
        updated_states = []
        
        for cell_idx,cell in enumerate(self.encoding_cells):
            current_hidden = hidden_state[cell_idx]
            input_t, supports, current_hidden = cell(input_t, supports, current_hidden)
            updated_states.append(current_hidden)
        
        hidden_state = torch.stack(updated_states, dim=0)
        
        return input_t, supports, hidden_state
    
    def forward(self, inputs:Tensor, supports:Tensor, initial_hidden_state:Tensor=None):
        """
        Compute the new representation of the parameters using the stack of SGLCell modules
        
        Args:
            inputs (Tensor):                Matrix of node with size (sequential_length, batch_size, num_nodes, input_dim)
            supports (Tensor):              Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            initial_hidden_state (Tensor):  To pass only if `use_GRU` is True. Hidden state matrix with size (num_cells, batch_size, num_nodes\*hidden_dim) if `hidden_per_step` is True, otherwise size (batch_size, num_nodes\*hidden_dim)

        Returns:
            tuple(Tensor, Tensor):  A tuple containing:
                - Update matrix of node with same size
                - Update adjacency matrix with same size
                - Update hidden state matrix with same size (present only if `use_GRU` is True)
        """        
        seq_length = inputs.size(0)
        original_shape = inputs.shape
        
        # Reshape: (sequential_length, batch_size, num_nodes*input_dim)
        inputs = inputs.reshape(inputs.size(0), inputs.size(1), -1)
        
        # Initialize outputs to avoid modifying in place
        processed_inputs = []
        current_supports = supports.clone()
        current_hidden = initial_hidden_state

        # Process each timestep
        for t in range(seq_length):
            input_t = inputs[t]
            input_t, current_supports, current_hidden = self._forward(input_t, current_supports, current_hidden)
            processed_inputs.append(input_t)
  
        # Reconstruct output tensor with original spatial dimensions
        new_input_representation = torch.stack(processed_inputs, dim=0)
        new_input_representation = new_input_representation.reshape(*original_shape)

        if self.use_GRU:
            return new_input_representation, current_supports, current_hidden
        else:
            return new_input_representation, current_supports

    def hidden_state_empty(self, batch_size:int) -> Tensor:
        """
        Create an uninitialized hidden state tensor for all SGLCell layers
            :param batch_size (int):   The size of the batch dimension
            :return Tensor:            Hidden state tensor, calculated as stack of the hidden state for each SGLCell
        """
        init_states = []
        sglc_cell:SGLC_Cell= None
        for sglc_cell in self.encoding_cells:
            init_states.append(sglc_cell.hidden_state_empty(batch_size))
            
        return torch.stack(init_states, dim=0)
