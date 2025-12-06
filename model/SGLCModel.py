import torch
import torch.nn as nn
from torch import Tensor

from model.SGLCell import SGLC_Cell
from model.GraphLearner.GraphLearnerAttention import GraphLearnerAttention

import os
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
            
            hidden_dim_GL:int=100,
            attention_type:GraphLearnerAttention=None,
            num_layers:int=3,
            num_heads:int=16,
            dropout:float=0,
            epsilon:float=None,
            
            hidden_dim_GGNN:int=None,
            num_steps:int=5,
            use_GRU_in_GGNN:bool=True,
            
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
            
            hidden_dim_GL (int):                    Hidden dimension for the Graph Learner module
            attention_type (GraphLearnerAttention): Type of attention used for the Graph Learner module
            num_layers (int):                       Number of message passing layers in the GAT or Transformer module for the Graph Learner module
            num_heads (int):                        Number of heads for multi-head attention in the Graph Learner module
            dropout (float):                        Dropout probability applied in the attention layer of the Graph Learner module
            epsilon (float):                        Threshold for deleting weak connections in the learned graph in the Graph Learner module. If None, no deleting is applied
            
            hidden_dim_GGNN (int):                  Hidden dimension of the hidden state for Gated Graph Neural Networks module (only if `use_GRU` is True)
            num_steps (int):                        Number of propagation steps in the Gated Graph Neural Networks module
            use_GRU_in_GGNN (bool):                 Use the GRU module instead of the standard propagator in the Gated Graph Neural Networks module
            
            device (str):                           Device to place the model on
            **kwargs:                               Additional arguments of `SGLC_Cell`
        """
        super(SGLC_Encoder, self).__init__()

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
                    
                    device=device,
                    **kwargs
                )
            )
        self.encoding_cells = nn.ModuleList(encoding_cells)
        self.use_GRU= use_GRU
    
    def forward(self, inputs: torch.Tensor, supports: torch.Tensor, initial_hidden_state: torch.Tensor = None):
        """
        Compute the new representation of the parameters using the stack of SGLCell modules
        
        Args:
            inputs (Tensor):                Matrix of node with size (sequential_length, batch_size, num_nodes, input_dim)
            supports (Tensor):              Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            initial_hidden_state (Tensor):  To pass only if `use_GRU` is True. Hidden state matrix with size (num_cells, batch_size, num_nodes*hidden_dim)

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
        cell:SGLC_Cell= None
        for t in range(seq_length):
            input_t = inputs[t]
            updated_states = []

            # Pass through each encoding cell (using the hidden state)
            if self.use_GRU:
                for cell_idx,cell in enumerate(self.encoding_cells):
                    hidden_state = current_hidden[cell_idx]
                    input_t, current_supports, hidden_state = cell.forward(input_t, current_supports, hidden_state)
                    updated_states.append(hidden_state)
                
                # Update store output
                current_hidden = torch.stack(updated_states, dim=0)
                processed_inputs.append(input_t)
            
            # Pass through each encoding cell
            else:
                for cell in self.encoding_cells:                
                    input_t, current_supports = cell.forward(input_t, current_supports)

                # Update store output
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
        Create an initialized hidden state tensor for all SGLCell layers
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
            
            hidden_dim_GL:int=100,
            attention_type:GraphLearnerAttention=GraphLearnerAttention.GRAPH_ATTENTION_LAYER,
            num_layers:int=3,
            num_heads:int=16,
            dropout:float=0,
            epsilon:float=None,
            
                        
            hidden_dim_GGNN:int=None,
            num_steps:int=5,
            use_GRU_in_GGNN:bool=False,
            
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
            
            hidden_dim_GL (int):                    Hidden dimension for Graph Learner module
            attention_type (GraphLearnerAttention): Type of attention used for the Graph Learner module
            num_layers (int):                       Number of message passing layers in the GAT or Transformer module for the Graph Learner module
            num_heads (int):                        Number of heads for multi-head attention in the Graph Learner module
            dropout (float):                        Dropout probability applied in the attention layer of the Graph Learner module
            epsilon (float):                        Threshold for deleting weak connections in the learned graph in the Graph Learner module. If None, no deleting is applied
            
            hidden_dim_GGNN (int):                  Hidden dimension of the hidden state for Gated Graph Neural Networks module (only if `use_GRU` is True)
            num_steps (int):                        Number of propagation steps in the Gated Graph Neural Networks module
            use_GRU_in_GGNN (bool):                 Use the GRU module instead of the standard propagator in the Gated Graph Neural Networks module
            
            device (str):                           Device to place the model on
            **kwargs:                               Additional arguments of `SGLC_Encoder`
        """
        super(SGLC_Classifier, self).__init__()

        self.params = locals().copy()
        self.params.update(self.params.pop("kwargs"))
        self.params.pop('self')
        self.params.pop('__class__')

        # Store configuration for saving with key : (value, critical_value)
        self.config:dict[str, tuple[any,bool]] = {
            'num_classes':      (num_classes, True),
            'num_cells':        (num_cells, True),
            'input_dim':        (input_dim, True),
            'num_nodes':        (num_nodes, True),
            'graph_skip_conn':  (graph_skip_conn, False),
            'use_GRU':          (use_GRU, True),
            'hidden_dim_GL':    (hidden_dim_GL, True),
            'attention_type':   (attention_type, True),
            'num_layers':       (num_layers, True),
            'num_heads':        (num_heads, True),
            'dropout':          (dropout, False),
            'epsilon':          (epsilon, False),
            'hidden_dim_GGNN':  (hidden_dim_GGNN, True),
            'num_steps':        (num_steps, False),
            'use_GRU_in_GGNN':  (use_GRU_in_GGNN, True),
            'device':           (device, False)
        }
        self.config.update(kwargs)
        
        self.use_GRU= use_GRU
        self.device= device
        self.encoder = SGLC_Encoder(
            num_cells=num_cells,
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
            
            device=device,
            **kwargs
        )

        self.fc= nn.Sequential(
            nn.Linear(num_nodes*input_dim, num_classes*4, device=device),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(num_classes*4, num_classes, device=device),
        )
    
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
        
        if self.use_GRU:
            init_hidden_state = torch.zeros_like( self.encoder.hidden_state_empty(batch_size=input_seq.size(1)) )            
            init_hidden_state = init_hidden_state.to(device=self.device)
            input_seq, supports, _ = self.encoder.forward(input_seq, supports, init_hidden_state)
        else:
            input_seq, supports = self.encoder.forward(input_seq, supports)
        
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
        
        dict_to_save = {}
        for (conf_key, _), (local_key, local_value) in zip(self.config.items(), self.params.items()):
            if conf_key != local_key:
                raise ValueError("Key '{}' not found in the function __init__".format(conf_key))
            dict_to_save[local_key] = _from_type_to_value(local_key, local_value)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.state_dict(),
            'config': dict_to_save
        }
        
        torch.save(save_dict, filepath)

    def old_load(self, filepath:str, device:str=None):
        """
        Load a saved model from a file.
        
        Args:
            filepath (str): Path to the saved model file
            device (str):   Device to place the model on. If None, uses the device from saved config
            
        Returns:
            SGLCModel_classification: Loaded model instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=device)
        
        config = checkpoint['config']
        
        # Convert attention_type string back to enum
        if isinstance(config['attention_type'], str):
            config['attention_type'] = GraphLearnerAttention[config['attention_type']]
        
        # Override device if specified
        if device:
            config['device'] = device
        
        model = SGLC_Classifier(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def load(self, filepath:str, strict:bool=False, device:str=None):
        """
        Load model weights from a file, validating that the architecture matches.
        
        Args:
            filepath (str): Path to the saved model file
            strict (bool):  If True, raise exception for any mismatches. If False, raises exception on critical mismatches
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
        checkpoint_config = checkpoint['config']
        checkpoint_config['device'] = device if (device is not None) else checkpoint_config['device']
        
        try:
            for (checkpoint_key,checkpoint_value),(conf_value,conf_critical) in zip(checkpoint_config.items(), self.config.values()):
                pass
        except ValueError:
            return self.old_load(filepath, device)
        
        dict_to_load = {}
        for (checkpoint_key,checkpoint_value),(conf_value,conf_critical) in zip(checkpoint_config.items(), self.config.values()):
            new_value =  _from_value_to_type(checkpoint_key, checkpoint_value, type(conf_value))
            if (strict or conf_critical) and (new_value!=conf_value):
                raise ValueError("Key '{}' ({}critical) should be '{}' but got '{}'".format(checkpoint_key, '' if conf_critical else 'not ' ,new_value, conf_value))
            dict_to_load[checkpoint_key]= new_value
        
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