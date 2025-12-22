import torch
import torch.nn as nn
from torch import Tensor

from model.SGLCell import SGLC_Cell
from model.GraphLearner.GraphLearnerAttention import GraphLearnerAttention

import os
import warnings
from enum import Enum
from types import NoneType

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
            num_layers:int=3,
            num_heads:int=16,
            dropout:float=0,
            epsilon:float=None,
            
            hidden_dim_GGNN:int=None,
            num_steps:int=5,
            use_GRU_in_GGNN:bool=True,
            
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
            num_layers (int):                       Number of message passing layers in the GAT or Transformer module for the Graph Learner module
            num_heads (int):                        Number of heads for multi-head attention in the Graph Learner module
            dropout (float):                        Dropout probability applied in the attention layer of the Graph Learner module
            epsilon (float):                        Threshold for deleting weak connections in the learned graph in the Graph Learner module. If None, no deleting is applied
            
            hidden_dim_GGNN (int):                  Hidden dimension of the hidden state for Gated Graph Neural Networks module (only if `use_GRU` is True)
            num_steps (int):                        Number of propagation steps in the Gated Graph Neural Networks module
            use_GRU_in_GGNN (bool):                 Use the GRU module instead of the standard propagator in the Gated Graph Neural Networks module
            
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
                    input_dim=input_dim,
                    num_nodes=num_nodes,
                    
                    graph_skip_conn=graph_skip_conn,
                    use_GRU=use_GRU,
                    
                    hidden_dim_GL=hidden_dim_GL,
                    attention_type=attention_type,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                    epsilon=epsilon,
                    
                    hidden_dim_GGNN=hidden_dim_GGNN,
                    num_steps=num_steps,
                    use_GRU_in_GGNN=use_GRU_in_GGNN,
                    
                    seed=seed,
                    device=device,
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

class SGLC_Classifier(nn.Module):
    """
    Classification model using SGLC Encoder with fully connected output layer
    """
    def __init__(
            self,
            num_classes:int=1,
            num_cells:int=1,
            input_dim:int=100,
            num_nodes:int=10,
            
            graph_skip_conn:float=0.3,
            use_GRU:bool=False,
            hidden_per_step:bool=True,
            
            hidden_dim_GL:int=100,
            attention_type:GraphLearnerAttention=GraphLearnerAttention.GRAPH_ATTENTION_LAYER,
            num_layers:int=3,
            num_heads:int=16,
            dropout:float=0,
            epsilon:float=None,

            hidden_dim_GGNN:int=None,
            num_steps:int=5,
            use_GRU_in_GGNN:bool=False,
            
            seed:int=None,
            device:str=None,
            **kwargs
        ):
        """
        Use a stack of SGLCell to learn from the data. Each SGLCell use the GL, the GGNN and the GRUCell module
        
        Args:
            num_classes (int):                      Number of output classes
        
            num_cells (int):                        Number of the SGLCell layers in the encoder stack
            input_dim (int):                        Feature dimension of input nodes
            num_nodes (int):                        Number of nodes in both input graph and hidden state
            
            graph_skip_conn (float):                Skip connection weight for adjacency updates
            use_GRU (bool):                         Use GRU to compute a hidden state used in the Gated Graph Neural Networks module
            hidden_per_step (bool):                 Use a new hidden state for each time step (only if `use_GRU` is True)
            
            hidden_dim_GL (int):                    Hidden dimension for Graph Learner module
            attention_type (GraphLearnerAttention): Type of attention used for the Graph Learner module
            num_layers (int):                       Number of message passing layers in the GAT or Transformer module for the Graph Learner module
            num_heads (int):                        Number of heads for multi-head attention in the Graph Learner module
            dropout (float):                        Dropout probability applied in the attention layer of the Graph Learner module
            epsilon (float):                        Threshold for deleting weak connections in the learned graph in the Graph Learner module. If None, no deleting is applied
            
            hidden_dim_GGNN (int):                  Hidden dimension of the hidden state for Gated Graph Neural Networks module (only if `use_GRU` is True)
            num_steps (int):                        Number of propagation steps in the Gated Graph Neural Networks module
            use_GRU_in_GGNN (bool):                 Use the GRU module instead of the standard propagator in the Gated Graph Neural Networks module
            
            seed (int):                             Sets the seed for the weights initializations. If None, don't use any seed
            device (str):                           Device to place the model on
            **kwargs:                               Additional arguments of `SGLC_Encoder`
        """
        super(SGLC_Classifier, self).__init__()

        self.params = locals().copy()
        self.params.update(self.params.pop("kwargs"))
        self.params.pop('self')
        self.params.pop('__class__')
        
        self.use_GRU= use_GRU
        self.hidden_per_step= hidden_per_step
        self.device= device
        self.encoder = SGLC_Encoder(
            num_cells=num_cells,
            input_dim=input_dim,
            num_nodes=num_nodes,
            
            graph_skip_conn=graph_skip_conn,
            use_GRU=use_GRU,
            hidden_per_step=hidden_per_step,

            hidden_dim_GL=hidden_dim_GL,
            attention_type=attention_type,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            epsilon=epsilon,
            
            hidden_dim_GGNN=hidden_dim_GGNN,
            num_steps=num_steps,
            use_GRU_in_GGNN=use_GRU_in_GGNN,
            
            seed=seed,
            device=device,
            **kwargs
        )

        self.fc= nn.Sequential(
            nn.Linear(num_nodes*input_dim, num_classes*4, device=device),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(num_classes*4, num_classes, device=device),
        )
        
        if (seed is not None):
            torch.manual_seed(seed)
        for param in self.fc.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def forward(self, input_seq:Tensor, supports:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Use the SGLCEncoder to calculate the new representations of the feature/node matrix and the adjacency matrix using a hidden state initialize at all zeros
        
        Args:
            input_seq (Tensor):     Input features matrix with size shape (batch_size, sequential_length, num_nodes, input_dim)
            supports (Tensor):      Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            
        Returns:
            tuple(Tensor, Tensor, Tensor): A tuple containing:
                - result: Class probability distribution with shape (batch_size, num_classes)
                - input_seq: Encoded feature sequence with shape (batch_size, seq_length, num_nodes, input_dim)
                - supports: Learned adjacency matrix with shape (batch_size, num_nodes, num_nodes)
        """
        # Transpose: (batch_size, seq_length, num_nodes, input_dim) --> (seq_length, batch_size, num_nodes, input_dim)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)
        
        # case GRU and hidden per step
        if self.use_GRU:
            if self.hidden_per_step:
                init_hidden_state = torch.zeros_like( self.encoder.hidden_state_empty(batch_size=input_seq.size(1)) )
        
        # case GRU and single hidden
            else:
                init_hidden_state = torch.zeros_like( self.encoder.hidden_state_empty(batch_size=input_seq.size(1))[0] )
            init_hidden_state = init_hidden_state.to(device=self.device)
            input_seq, supports, _ = self.encoder(input_seq, supports, init_hidden_state)
        
        # case no GRU
        else:
            input_seq, supports = self.encoder(input_seq, supports)
        
        features_mean= input_seq[-1]
        features_mean= features_mean.reshape(features_mean.size(0), -1)
        result = self.fc(features_mean)

        # Transpose: (seq_length, batch_size, num_nodes, input_dim) --> (batch_size, seq_length, num_nodes, input_dim)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)
        
        return result, input_seq, supports

    def save(self, filepath:str) -> None:
        """
        Save the model's state dictionary and configuration to a file.
            :param filepath (str): Path where the model will be saved (e.g., 'model.pth')
        """
        def _from_type_to_value(key:str, value:any, accepted_types:list=[int, float, str, bool, NoneType]):
            """Check if a `value` type is in the `accepted_types`. If not try to modify it, otherwise raise a `TypeError`. It is the opposite of `_from_value_to_type`"""
            accepted = any(isinstance(value, accepted_type) for accepted_type in accepted_types )
            if not(accepted):
                if isinstance(value, Enum):
                    value = value.name
                    accepted = True
            if not(accepted):
                raise TypeError("Key '{}' has class {} but one of the following classes are expected {}".format(key, type(value), ", ".join([str(_type) for _type in accepted_types])))
            
            return value
        
        dict_to_save:dict[str,any] = {}
        for key,value in self.params.items():
            dict_to_save[key] = _from_type_to_value(key, value)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.state_dict(),
            'config': dict_to_save
        }
        
        torch.save(save_dict, filepath)
    
    def load(self, filepath:str, strict:bool=False, device:str=None):
        """
        Load model weights from a file, validating that the architecture matches.
        
        Args:
            filepath (str): Path to the saved model file
            strict (bool):  If True, raise exception for any mismatches, otherwise print a warning
            device (str):   Device to place the model on. If None, uses the device from saved config
            
        Returns:
            SGLCModel_classification: Loaded model instance
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
        """
        def _from_value_to_type(key:str, value:any, value_class:type):    
            """Check if a `value` is different from its original class. If yes try to modified it. If it cannot, raise a TypeError. It is the opposite of `_from_type_to_value`"""
            if isinstance(value, value_class):
                return value
            if issubclass(value_class, Enum) and isinstance(value, str):
                return value_class[value]
            raise TypeError("Key '{}' cannot be converted".format(key))
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=device)
        conf:dict[str,any] = checkpoint['config']
        conf['device'] = device if (device is not None) else conf['device']
        
        conf_set = set(conf.keys())
        param_set = set(self.params.keys())
        for key in conf_set.difference(param_set):
            raise ValueError("Key '{}' is loaded but not present in the model".format(key))
        for key in param_set.difference(conf_set):
            msg ="Key '{}' is in the model but not loaded".format(key)
            if strict:
                raise ValueError(msg)
            warnings.warn(msg)
        
        dict_to_load:dict[str,any] = {}
        for key in conf.keys():
            conf_value = _from_value_to_type(key, conf[key], type(self.params[key]))
            if conf_value != self.params[key]:
                msg ="Key '{}' has value ({}) but ({}) was expected".format(key, self.params[key], conf_value)
                if strict:
                    raise ValueError(msg)
                warnings.warn(msg)
            dict_to_load[key] = conf_value
        
        model = SGLC_Classifier(**dict_to_load)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model

# if __name__=="__main__":
#     def count_parameters(model:nn.Module):
#         return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     num_classes     = 2
#     num_cells       = 2
#     input_dim       = 256//2
#     num_nodes       = 21
#     hidden_dim_GL   = 192
#     hidden_dim_GGNN = 192
#     dropout         = 0.5
#     num_heads       = 8
#     use_GATv2       = False
#     use_Transformer = True
#     concat          = True
#     num_layers      = 3
#     use_propagator  = True
#     use_GRU         = True
    
#     model= SGLC_Classifier(
#         num_classes     = num_classes,
#         num_cells       = num_cells,
#         input_dim       = input_dim,
#         num_nodes       = num_nodes,
#         hidden_dim_GL   = hidden_dim_GL,
#         hidden_dim_GGNN = hidden_dim_GGNN,
#         dropout         = dropout,
#         num_heads       = num_heads,
#         use_GATv2       = use_GATv2,
#         use_Transformer = use_Transformer,
#         concat          = concat,
#         use_GRU         = use_GRU,
#         use_propagator  = use_propagator
#     )
    
#     list_to_print= [(key,value) for key,value in model.config.items()]
#     string= ""
#     ljust_value= max([len(item) for item,_ in list_to_print])
#     for name,value in list_to_print:
#         string += "{} : {}\n".format(name.ljust(ljust_value), value)
    
#     print(f"Total size model : {count_parameters(model):,}")
#     print(f"\nUsing parametrs:\n{string}")
    
#     print()
    
#     from torchinfo import summary
#     BATCH_SIZE= 64
#     SEQ_LEN= 4
#     summary(model, 
#         input_data={
#             'input_seq': torch.rand((BATCH_SIZE, SEQ_LEN, num_nodes, input_dim)),
#             'supports': torch.rand((BATCH_SIZE, num_nodes, num_nodes))
#         },
#         depth=1,  # Shows more detailed structure
#         col_names=['input_size', 'output_size', 'num_params', 'trainable'],
#         verbose=1
#     )