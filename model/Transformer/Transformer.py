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
            
            num_inputs:int=None
        ):
        """
        Generate the transformer class according to the type passed as parameter. The output will have the same size as a single input

        Args:
            transformer_type (TransformerType):             Type of transformer class to use
            input_shape (Sequence[int]):                    Shape of the 2D input where the number of expected features in the input is the second dimension. It is [input_dim, d_model]
            num_heads (int):                                Number of heads in the multi-heads attention models
            
            num_encoder_layers (int):                       Number of sub-encoder layers in the encoder
            num_decoder_layers (int):                       Number of sub-decoder layers in the decoder
            
            positional_encoding (PositionalEncodingType):   Type of positional encoder to use
            dim_feedfarward (int):                          Dimension of the feedforward network model
            dropout (float):                                Dropout value
            activation (str|Callable):                      Activation funcion of the encoder/decoder intermediate layer
            
            seed (int):                                     Sets the seed for the weights initializations. If None, don't use any seed
            device (str):                                   Device to place the model on

            num_inputs (int):                               Number of inputs passed. Necessary only for some types of positional encodings
        See:
        -----
            :func:`torch.nn.Transformer` for more details
        """
        super(Transformer, self).__init__()
        
        if (len(input_shape) != 2):
            raise ValueError("Expected 'input_shape' as 2D but got {}-D".format(len(input_shape)))
        
        if (transformer_type == TransformerType.TRANSFORMER_ENCODER) and (num_decoder_layers is not None) and (num_decoder_layers != 0):
            msg = "'num_decoder_layers' will be ignored because the type is set to {}".format(transformer_type.name)
            warnings.warn(msg)
        if (transformer_type == TransformerType.TRANSFORMER_DECODER) and (num_encoder_layers is not None) and (num_encoder_layers != 0):
            msg = "'num_encoder_layers' will be ignored because the type is set to {}".format(transformer_type.name)
            warnings.warn(msg)
        
        d_model = input_shape[1]
        for curr_num_heads in range(num_heads, 0, -1):
            if (d_model % curr_num_heads == 0):
                if (curr_num_heads != num_heads):
                    msg = "The number of heads is reduced from ({}) to ({}), otherwise second dimension of 'input_shape' ({}) was not divisible by the number of heads".format(num_heads, curr_num_heads, d_model)
                    warnings.warn(msg)
                    num_heads = curr_num_heads
                break
        
        if (seed is not None):
            torch.manual_seed(seed)
        self.sos_token = torch.nn.Parameter(torch.randn(1, input_shape[0], input_shape[1])).to(device=device)
        self.cls_token = torch.nn.Parameter(torch.randn(1, input_shape[0], input_shape[1])).to(device=device)
        activation = activation_resolver(activation)
        
        match positional_encoding:
            case None:
                self.pe = None
            case PositionalEncodingType.SINUSOIDAL:
                if (num_inputs is None) or (num_inputs <= 0):
                    raise ValueError("'num_inputs' must be positive")
                self.pe = SinusoidalPositionalEncoding(d_model=input_shape[1], max_len=num_inputs*input_shape[0], device=device)
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
        if (self.pe is not None):
            inputs = self.pe[inputs]
        return inputs
      
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
            :param inputs (Tensor): Sequence of input with size `(batch_size, num_inputs*input_dim, d_model)` where `input_dim` and `d_model` are respectively the first and the second dimension of `input_shape`
            :returns cls (Tensor):  Next input token with size `(batch_size, input_dim, d_model)`
        """
        inputs = self._forward_preprocessing(inputs)
        return self._forward(inputs)

    def _forward(self, inputs:Tensor):
        raise NotImplementedError("This function is only a decoration")
