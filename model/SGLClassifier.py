import torch
import torch.nn as nn
from torch import Tensor

from model.SGLCEncoder import SGLC_Encoder
from model.GraphLearner.GraphLearnerAttention import GraphLearnerAttention
from model.Transformer.Transformer import Transformer, TransformerType, PositionalEncodingType

import os
import warnings
from enum import Enum
from copy import deepcopy
from types import NoneType
from typing import Callable

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
            pretrain_with_decoder:bool=False,
            new_pretrain_hidden:bool=False,
            
            hidden_dim_GL:int=100,
            attention_type:GraphLearnerAttention=GraphLearnerAttention.GRAPH_ATTENTION_LAYER,
            num_GL_layers:int=3,
            num_GL_heads:int=16,
            dropout_GL:float=0,
            epsilon:float=None,

            hidden_dim_GGNN:int=None,
            num_steps:int=5,
            num_GGNN_layers:int=1,
            act_GGNN:str|Callable=None,
            use_GRU_in_GGNN:bool=False,
            
            transformer_type:TransformerType=None,
            num_transf_heads:int=None,
            num_encoder_layers:int=None,
            num_decoder_layers:int=None,
            positional_encoding:PositionalEncodingType=None,
            dim_feedforward:int=None,
            dropout_transf=None,
            act_transf:str|Callable=None,
            
            seed:int=None,
            device:str=None,
            **kwargs
        ):
        """
        Use a stack of SGLCell to learn from the data. Each SGLCell use the GL, the GGNN and the GRUCell module.
        After the SGLCell a Transformer class can be applied to generate a single output from the sequence,
        then a fully connected layer compute the real output
        
        Args:
            num_classes (int):                              Number of output classes
        
            num_cells (int):                                Number of the SGLCell layers in the encoder stack
            input_dim (int):                                Feature dimension of input nodes
            num_nodes (int):                                Number of nodes in both input graph and hidden state
            
            graph_skip_conn (float):                        Skip connection weight for adjacency updates
            use_GRU (bool):                                 Use GRU to compute a hidden state used in the Gated Graph Neural Networks module
            hidden_per_step (bool):                         Use a new hidden state for each time step (only if `use_GRU` is True)
            pretrain_with_decoder (bool):                   Pretrain the model using an encoder-decoder architecture. The output will be the next predicion instead of a classification
            new_pretrain_hidden (bool):                     Use a new hidden state instead of the encoder hidden state (only if `use_GRU` and `pretrain_with_decoder` are True)
            
            hidden_dim_GL (int):                            Hidden dimension for Graph Learner module
            attention_type (GraphLearnerAttention):         Type of attention used for the Graph Learner module
            num_GL_layers (int):                            Number of message passing layers in the GAT or Transformer module for the Graph Learner module
            num_GL_heads (int):                             Number of heads for multi-head attention in the Graph Learner module
            dropout_GL (float):                             Dropout probability applied in the attention layer of the Graph Learner module
            epsilon (float):                                Threshold for deleting weak connections in the learned graph in the Graph Learner module. If None, no deleting is applied
            
            hidden_dim_GGNN (int):                          Hidden dimension of the hidden state for Gated Graph Neural Networks module (only if `use_GRU` is True)
            num_steps (int):                                Number of propagation steps in the Gated Graph Neural Networks module
            num_GGNN_layers (int):                          Number of Propagation modules in the Gated Graph Neural Networks module
            act_GGNN (str|Callable):                        The non-linear activation function to use inside the linear activation function in the Gated Graph Neural Networks module. If None use the default class value
            use_GRU_in_GGNN (bool):                         Use the GRU module instead of the standard propagator in the Gated Graph Neural Networks module
            
            transformer_type (TransformerType):             Type of transformer to use for the Transformer module
            num_transf_heads (int):                         Number of heads in the multi-heads attention in the Transformer module (only if `transformer_type` is not None)
            num_encoder_layers (int):                       Number of sub-encoder layers in the encoder in the Transformer module (only if `transformer_type` is not None)
            num_decoder_layers (int):                       Number of sub-decoder layers in the decoder in the Transformer module (only if `transformer_type` is not None)
            positional_encoding (PositionalEncodingType):   Type of positional encoder to use in the Transformer module (only if `transformer_type` is not None)
            dim_feedforward (int):                          Dimension of the feedforward network model in the Transformer module (only if `transformer_type` is not None)
            dropout_transf (float):                         Dropout value in the Transformer module (only if `transformer_type` is not None)
            act_transf (str|Callable):                      Activation funcion of the encoder/decoder intermediate layer in the Transformer module. If None use the default class value (only if `transformer_type` is not None)
            
            seed (int):                                     Sets the seed for the weights initializations. If None, don't use any seed
            device (str):                                   Device to place the model on
            **kwargs:                                       Additional arguments of `SGLC_Encoder` and `Transformer`
        """
        super(SGLC_Classifier, self).__init__()

        self.params = locals().copy()
        self.params.update(self.params.pop("kwargs"))
        self.params.pop('self')
        self.params.pop('__class__')
        
        if (pretrain_with_decoder) and (transformer_type is not None):
            raise ValueError("'pretrain_with_decoder' and 'transformer_type' are True but in conflict. Choose one")            
        if (new_pretrain_hidden):
            if not(pretrain_with_decoder):
                warnings.warn("'new_pretrain_hidden' parameter is ignored because 'pretrain_with_decoder' is False")
            if (pretrain_with_decoder) and not(use_GRU):
                warnings.warn("'new_pretrain_hidden' parameter is ignored because 'use_GRU' is False")

        self.device= device
        self.use_GRU= use_GRU
        self.hidden_per_step= hidden_per_step
        self.new_pretrain_hidden= new_pretrain_hidden
        self.pretrain_with_decoder= pretrain_with_decoder
        self.encoder = SGLC_Encoder(
            num_cells       = num_cells,
            input_dim       = input_dim,
            num_nodes       = num_nodes,
            
            graph_skip_conn = graph_skip_conn,
            use_GRU         = use_GRU,
            hidden_per_step = hidden_per_step,

            hidden_dim_GL   = hidden_dim_GL,
            attention_type  = attention_type,
            num_GL_layers   = num_GL_layers,
            num_heads       = num_GL_heads,
            dropout         = dropout_GL,
            epsilon         = epsilon,
            
            hidden_dim_GGNN = hidden_dim_GGNN,
            num_steps       = num_steps,
            num_GGNN_layers = num_GGNN_layers,
            act_GGNN        = act_GGNN,
            use_GRU_in_GGNN = use_GRU_in_GGNN,
            
            seed            = seed,
            device          = device,
            **kwargs
        )
        
        
        keys_transformer = ["num_inputs"]
        kwargs_transformer = {key:value for key,value in kwargs.items() if (key in keys_transformer) and (value is not None)}
        transformer_params = (
            (num_transf_heads    is not None) or
            (num_encoder_layers  is not None) or
            (num_decoder_layers  is not None) or
            (positional_encoding is not None) or
            (dim_feedforward     is not None) or
            (dropout_transf      is not None) or
            (act_transf          is not None)
        )
        
        if (pretrain_with_decoder):
            self.decoder = deepcopy(self.encoder)
            if ( transformer_params or ( len(kwargs_transformer)!=0 ) ):
                others_msg = ". Some of them are : '{}'".format("', '".join(kwargs_transformer.keys())) if (len(kwargs_transformer)!=0) else ""
                msg = "'transformer_type' is None but some parameters regarding to it have been passed{}".format(others_msg)
                raise ValueError(msg)
        
        elif (transformer_type is None):
            if ( transformer_params or ( len(kwargs_transformer)!=0 ) ):
                others_msg = ". Some of them are : '{}'".format("', '".join(kwargs_transformer.keys())) if (len(kwargs_transformer)!=0) else ""
                msg = "'transformer_type' is None and some parameters regarding to it have been ignored{}".format(others_msg)
                warnings.warn(msg)
            self.transf = None
            
        else:
            if act_transf is not None:
                kwargs_transformer['activation'] = act_transf
            self.transf = Transformer(
                transformer_type    = transformer_type,
                input_shape         = [num_nodes, input_dim],
                num_heads           = num_transf_heads,
                num_encoder_layers  = num_encoder_layers,
                num_decoder_layers  = num_decoder_layers,
                positional_encoding = positional_encoding,
                dim_feedforward     = dim_feedforward,
                dropout             = dropout_transf,
                seed                = seed,
                device              = device,
                **kwargs_transformer,
            )
        
        if not(self.pretrain_with_decoder):
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
    
    def _forward_encoder(self, input_seq:Tensor, supports:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Process input according to the encoder using the `self.forward` parameters"""
        # Transpose: (batch_size, seq_length, num_nodes, input_dim) --> (seq_length, batch_size, num_nodes, input_dim)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)
        
        # case GRU and hidden per step
        if self.use_GRU:
            if self.hidden_per_step:
                init_hidden_state = torch.zeros_like( self.encoder.hidden_state_empty(batch_size=input_seq.size(1)) )
                init_hidden_state = init_hidden_state.to(device=self.device)
                input_seq, supports, hidden_state = self.encoder(input_seq, supports, init_hidden_state)
        
        # case GRU and single hidden
            else:
                init_hidden_state = torch.zeros_like( self.encoder.hidden_state_empty(batch_size=input_seq.size(1))[0] )
                init_hidden_state = init_hidden_state.to(device=self.device)
                input_seq, supports, hidden_state = self.encoder(input_seq, supports, init_hidden_state)
        
        # case no GRU
        else:
            input_seq, supports = self.encoder(input_seq, supports)
            hidden_state = None
        
        # Transpose: (seq_length, batch_size, num_nodes, input_dim) --> (batch_size, seq_length, num_nodes, input_dim)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)
        
        return input_seq, supports, hidden_state
    
    def _forward_decoder(self, input_seq:Tensor, supports:Tensor, hidden_state:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Process input according to the decoder using the `self.forward` parameters and hidden_state"""        
        # Transpose: (batch_size, seq_length, num_nodes, input_dim) --> (seq_length, batch_size, num_nodes, input_dim)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)
        
        # case GRU and hidden per step
        if self.use_GRU:
            if self.hidden_per_step:
                init_hidden_state = hidden_state if not(self.new_pretrain_hidden) else torch.zeros_like( self.decoder.hidden_state_empty(batch_size=input_seq.size(1)) )
                init_hidden_state = init_hidden_state.to(device=self.device)
                input_seq, supports, hidden_state = self.decoder(input_seq, supports, init_hidden_state)
        
        # case GRU and single hidden
            else:
                init_hidden_state = hidden_state if not(self.new_pretrain_hidden) else torch.zeros_like( self.decoder.hidden_state_empty(batch_size=input_seq.size(1))[0] )
                init_hidden_state = init_hidden_state.to(device=self.device)
                input_seq, supports, hidden_state = self.decoder(input_seq, supports, init_hidden_state)
        
        # case no GRU
        else:
            input_seq, supports = self.decoder(input_seq, supports)
            hidden_state = None
        
        # Transpose: (seq_length, batch_size, num_nodes, input_dim) --> (batch_size, seq_length, num_nodes, input_dim)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)
        
        return input_seq, supports, hidden_state
    
    def _forward_result(self, input_seq:Tensor) -> Tensor:
        """Process input according to the transformer and last fully connected layer using the `self.forward` parameters"""
        if (self.transf is not None):
            input_clone = input_seq.clone()
            # Reshape: (batch_size, sequential_length, num_nodes, input_dim) --> (batch_size, seq_length*num_nodes, input_dim)
            input_clone = input_clone.reshape(input_clone.size(0), -1, input_clone.size(-1))
            last_feature = self.transf(input_clone)
        else:
            last_feature = input_seq[:, -1, :, :]
        
        features_mean= last_feature.reshape(last_feature.size(0), -1)
        result = self.fc(features_mean)
        
        return result
    
    def forward(self, input_seq:Tensor, supports:Tensor) -> tuple[Tensor,Tensor]|tuple[Tensor, Tensor, Tensor]:
        """
        Use the SGLCEncoder to calculate the new representations of the feature/node matrix and the adjacency matrix using a hidden state initialize at all zeros
        
        Args:
            input_seq (Tensor):     Input features matrix with size (batch_size, sequential_length, num_nodes, input_dim)
            supports (Tensor):      Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            
        :returns out (tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]):
        - tuple[Tensor, Tensor]: if `pretrain_with_decoder` is True
            - encoder_input_seq:    Updated Tensor after the encoder model with shape (batch_size, seq_length, num_nodes, input_dim)
            - input_seq:            Updated Tensor after encoder-decoder structure with shape (batch_size, seq_length, num_nodes, input_dim)
        - tuple[Tensor, Tensor, Tensor]: if `pretrain_with_decoder` is False
            - result:    Class probability distribution with shape (batch_size, num_classes)
            - input_seq: Encoded feature sequence with shape (batch_size, seq_length, num_nodes, input_dim)
            - supports:  Learned adjacency matrix with shape (batch_size, num_nodes, num_nodes)
        """
        input_seq, supports, hidden = self._forward_encoder(input_seq, supports)
        
        # case pretraining
        if self.pretrain_with_decoder:
            encoder_input_seq = input_seq.clone()
            input_seq, _, _ = self._forward_decoder(input_seq, supports, hidden)
            return encoder_input_seq, input_seq
        
        # case fine-tuning
        else:
            result = self._forward_result(input_seq)    
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
            if (value_class==NoneType) and not(isinstance(value, NoneType)):
                raise TypeError("Key '{}' is found None in the model, change it".format(key))
            
            if isinstance(value, value_class):
                return value
            if issubclass(value_class, Enum) and isinstance(value, str):
                return value_class[value]
            if isinstance(value, NoneType):
                return None
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
    
    def load_from_pretraining(self, pretrained_path:str, device:str=None):
        """
        Load encoder weights from a pretrained model and initialize fine-tuning components
        
        Args:
            pretrained_path (str):      Path to the pretrained model file
            device (str):               Device to place the model on. If None, uses the device from saved config
        
        Returns:
            SGLCModel_classification:   Returns itself (modifies self in-place)
        """
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained model file not found: {pretrained_path}")
        
        # Load pretrained checkpoint
        checkpoint = torch.load(pretrained_path, map_location=device or self.device)
        pretrained_state = checkpoint['model_state_dict']
        
        # Extract only encoder weights
        encoder_state = {key.replace('encoder.', ''):value for key,value in pretrained_state.items() if key.startswith('encoder.')}
        
        # Load encoder weights
        self.encoder.load_state_dict(encoder_state, strict=True)

        return self