from model.Transformer.TransformerType import TransformerType
from model.Transformer.PositionalEncoding import *
from torch import nn, Tensor

from torch_geometric.nn.resolver import activation_resolver

from typing import Callable, Sequence
import warnings

class Transformer(nn.Module):
    def __init__(
            self,
            transformer_type:TransformerType,
            input_shape:Sequence[int],
            num_heads:int,
            
            num_encoder_layers:int,
            num_decoder_layers:int,
            
            positional_encoding:PositionalEncodingType=None,
            dim_feedforward:int=2048,
            dropout:float=0.1,
            activation:str|Callable='relu',
            
            seed:int=None,
            device:str=None,
            
            *,
            
            spread_sequence_factor:int=1
        ):
        """
        Generate the transformer class according to the type passed as parameter. The output will have the same size as a single input

        Args:
            transformer_type (TransformerType):             Type of transformer class to use
            input_shape (Sequence[int]):                    Shape of the 3D input [sequence_length, num_nodes, input_dim]
            num_heads (int):                                Number of heads in the multi-heads attention models
            
            num_encoder_layers (int):                       Number of sub-encoder layers in the encoder
            num_decoder_layers (int):                       Number of sub-decoder layers in the decoder
            
            positional_encoding (PositionalEncodingType):   Type of positional encoder to use
            dim_feedfarward (int):                          Dimension of the feedforward network model
            dropout (float):                                Dropout value
            activation (str|Callable):                      Activation funcion of the encoder/decoder intermediate layer
            
            seed (int):                                     Sets the seed for the weights initializations. If None, don't use any seed
            device (str):                                   Device to place the model on

            spread_sequence_factor (int):                   Each time step of the `sequence_length` will be divided in more time steps
        See:
        -----
            :func:`torch.nn.Transformer` for more details
        """
        super(Transformer, self).__init__()
        
        if (len(input_shape) != 3):
            raise ValueError("Expected 'input_shape' as 3D but got {}-D".format(len(input_shape)))
        if (spread_sequence_factor <= 0):
            raise ValueError("'spread_information_factor' must be positve")
        
        if (transformer_type == TransformerType.TRANSFORMER_ENCODER) and (num_decoder_layers is not None) and (num_decoder_layers != 0):
            msg = "'num_decoder_layers' will be ignored because the type is set to {}".format(transformer_type.name)
            warnings.warn(msg)
        if (transformer_type == TransformerType.TRANSFORMER_DECODER) and (num_encoder_layers is not None) and (num_encoder_layers != 0):
            msg = "'num_encoder_layers' will be ignored because the type is set to {}".format(transformer_type.name)
            warnings.warn(msg)
        
        sequence_length, num_nodes, input_dim = input_shape
        self.new_sequence_length = spread_sequence_factor * sequence_length
        self.new_input_dim = input_dim // spread_sequence_factor
        if (input_dim % spread_sequence_factor != 0):
            raise ValueError("The sequence cannot be extended from ({}) to ({}) because the input dimension ({}) is not divisible by ({})".format(
                sequence_length, self.new_sequence_length, input_dim, spread_sequence_factor
            ))
        
        d_model = num_nodes * self.new_input_dim
        for curr_num_heads in range(num_heads, 0, -1):
            if (d_model % curr_num_heads == 0):
                if (curr_num_heads != num_heads):
                    msg = "The number of heads is reduced from ({}) to ({}), otherwise second dimension of 'd_model' ({}*{}) was not divisible by the number of heads".format(num_heads, curr_num_heads, num_nodes, self.new_input_dim)
                    warnings.warn(msg)
                    num_heads = curr_num_heads
                break
        
        if (seed is not None):
            torch.manual_seed(seed)
        self.sos_token = torch.nn.Parameter(torch.randn(1, 1, d_model)).to(device=device)
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, d_model)).to(device=device)
        activation = activation_resolver(activation)
        
        match positional_encoding:
            case None:
                self.pe = None
            case PositionalEncodingType.SINUSOIDAL:
                self.pe = SinusoidalPositionalEncoding(d_model=d_model, max_len=self.new_sequence_length, device=device)
            case _:
                raise NotImplementedError("Positional encoder {} is not implemented yet".format(positional_encoding))
        
        match transformer_type:
            case TransformerType.TRANSFORMER:
                self._forward = self._forward_transformer
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd")
                    warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.activation_relu_or_gelu was not True")
                    self.transf = nn.Transformer(
                        d_model              = d_model,
                        nhead                = num_heads,
                        num_encoder_layers   = num_encoder_layers,
                        num_decoder_layers   = num_decoder_layers,
                        dim_feedforward      = dim_feedforward,
                        dropout              = dropout,
                        activation           = activation,
                        batch_first          = True,
                        device               = device
                    )
            case TransformerType.TRANSFORMER_ENCODER:
                self._forward = self._forward_transformer_encoder
                self.transf = nn.TransformerEncoder(
                    enable_nested_tensor = False,
                    num_layers    = num_encoder_layers,
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model         = d_model,
                        nhead           = num_heads,
                        dim_feedforward = dim_feedforward,
                        dropout         = dropout,
                        activation      = activation,
                        batch_first     = True,
                        device          = device
                    )
                )
            case TransformerType.TRANSFORMER_DECODER:
                self._forward = self._forward_transformer_decoder
                self.transf = nn.TransformerDecoder(
                    num_layers    = num_decoder_layers,
                    decoder_layer = nn.TransformerDecoderLayer(
                        d_model         = d_model,
                        nhead           = num_heads,
                        dim_feedforward = dim_feedforward,
                        dropout         = dropout,
                        activation      = activation,
                        batch_first     = True,
                        device          = device
                    )
                )
            case _:
                raise NotImplementedError("Transformer {} is not implemented yet".format(transformer_type))

        for param in self.transf.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def _forward_preprocessing(self, inputs:Tensor):
        """Preprocessing to apply to the input before the forward method. See :func:`self.forward` for more detail about the parameters"""
        inputs = inputs.transpose(-1, -2)                                                                           # (batch_size, sequence_length,     input_dim,          num_nodes)
        inputs = inputs.reshape(inputs.shape[0], self.new_sequence_length, self.new_input_dim, inputs.shape[-1])    # (batch_size, new_sequence_length, new_input_dim,      num_nodes)
        inputs = inputs.transpose(-1, -2)                                                                           # (batch_size, new_sequence_length, num_nodes,          new_input_dim)
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)                                               # (batch_size, new_sequence_length, num_nodes*new_input_dim)
        
        if (self.pe is not None):
            inputs = self.pe[inputs]
        return inputs
    
    def _forward_postprocessing(self, output:Tensor):
        """Postprocessing to apply at the output of the Transformer"""
        return output.squeeze(1).reshape(output.shape[0], -1, self.new_input_dim)
    
    def _forward_transformer(self, inputs:Tensor) -> Tensor:
        """See :func:`self.forward` for more detail"""
        batch_size = inputs.size(0)
        repetaed_sos_token = self.sos_token.repeat(batch_size, 1, 1)
        
        output = self.transf(src=inputs, tgt=repetaed_sos_token)
        return output
    
    def _forward_transformer_encoder(self, inputs:Tensor) -> Tensor:
        """See :func:`self.forward` for more detail"""
        batch_size = inputs.size(0)
        repeated_cls_token = self.cls_token.repeat(batch_size, 1, 1)
        
        new_input = torch.cat([repeated_cls_token, inputs], dim=1)
        output = self.transf(src=new_input)
        return output[:, 0:self.cls_token.size(1), :]
    
    def _forward_transformer_decoder(self, inputs:Tensor) -> Tensor:
        """See :func:`self.forward` for more detail"""
        batch_size = inputs.size(0)
        repeated_sos_token = self.sos_token.repeat(batch_size, 1, 1)
        
        output = self.transf(tgt=repeated_sos_token, memory=inputs)
        return output
    
    def forward(self, inputs:Tensor) -> Tensor:
        """
        Given a sequence of token classify the next token
            :param inputs (Tensor): Sequence of input with size `(batch_size, sequence_length, num_nodes, input_dim)`
            :returns cls (Tensor):  Next input token with size `(batch_size, num_nodes, input_dim//spread_sequence_factor)`
        """
        inputs = self._forward_preprocessing(inputs)
        output = self._forward(inputs)
        return self._forward_postprocessing(output)

    def _forward(self, inputs:Tensor):
        raise NotImplementedError("This function is only a decoration")
