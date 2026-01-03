"""Contains constant useful to the computation of EEG files"""
from model.GraphLearner.GraphLearnerAttention import GraphLearnerAttention
from model.Transformer.PositionalEncoding import PositionalEncodingType
from model.Transformer.TransformerType import TransformerType
from data.dataloader.SeizureAugmentation import *

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# UTILS INFO
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

FREQUENCY_CHB_MIT= 256
"""Frequency of the data"""

VERY_SMALL_NUMBER = 1e-12
"""Value used to avoid zero division"""

INF = 1e25
"""Value used to avoid `float('Inf')`"""

USE_CUDA= True
"""Use cuda if available otherwise use cpu"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# INFO FOR data.dataloader.SeizureDataset
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

TIME_STEP_SIZE= 1
"""Used in :data:`data.dataloader.SeizureDataset` \\
Duration of each time step in seconds for FFT analysis"""

MAX_SEQ_LEN= 4
"""Used in :data:`data.dataloader.SeizureDataset` \\
Total duration of the output EEG clip in seconds"""

USE_FFT= True
"""Used in :data:`data.dataloader.SeizureDataset` \\
Use the Fast Fourier Transform when obtain the slice from the file"""

TOP_K= None
"""Used in :data:`data.dataloader.SeizureDataset` \\
Maintain only the `top_k` higher value when compute the adjacency matrix"""

seed = 1559
classes_to_use = [True]
AUGMENTATIONS = [
    SwapChannels(labels=classes_to_use, seed=seed, p=0.0, channels_to_swap=[(0,1)])
]
"""Used in :data:`data.dataloader.SeizureDataset` \\
Set the augmentation to use inside the dataset"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# DAMPING FACTOR FOR LOSS FUNCTIONS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

DAMP_DEGREE= 1.0
"""Damping factor for :func:`model.loss_functions.degree_regularization_loss_func`. Set to 0 to not use it"""

DAMP_SMOOTH= 1.0
"""Damping factor for :func:`model.loss_functions.smoothness_loss_func`. Set to 0 to not use it"""

DAMP_SPARSITY= 1.0
"""Damping factor for :func:`model.loss_functions.sparsity_loss_func`. Set to 0 to not use it"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# INFO FOR model.SGLClassifier.SGLC_Classifier
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

NUM_CLASSES= 2
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Number of output classes"""

NUM_CELLS= 2
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Number of the :data:`model.SGLCell.SGLCell` layers in the encoder stack"""

GRAPH_SKIP_CONN= 0.3
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Skip connection weight for adjacency updates"""

USE_GRU= False
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Use GRU module in the :data:`model.SGLCell.SGLC_Cell` and hidden state in the :data:`model.GatedGraphNeuralNetworks.GGNNLayer` module"""

HIDDEN_PER_STEP= False
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Use a new hidden state for each time step (only if `USE_GRU` is True) in the :data:`model.SGLClassifier.SGLC_Encoder` module"""

PRETRAIN= False
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Use an encoder-decoder architecture to do next output prediction using self-supervised learning"""

PRETRAIN_NEW_HIDDEN= False
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Use a new hidden state instead of the encoder hidden state. To use (only if `USE_GRU` and `PRETRAIN` are True)"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GRAPH LEARNER VALUES

HIDDEN_DIM_GL= 192
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Hidden dimension for the :data:`model.GraphLearner.GraphLearner` module"""

ATTENTION_TYPE= GraphLearnerAttention.GAT
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Type of attention used in the :data:`model.GraphLearner.GraphLearner` module"""

NUM_GL_LAYERS= 3
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Number of message passing layers in the GAT or Transformer module for the :data:`model.GraphLearner.GraphLearner` module"""

NUM_GL_HEADS= 8
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Number of heads for multi-head attention in the :data:`model.GraphLearner.GraphLearner` module"""

GL_DROPOUT= 0.4
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Dropout probability applied in the attention layer for the :data:`model.GraphLearner.GraphLearner` module"""

EPSILON= 1e-10
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Threshold for deleting weak connections in the learned graph for the :data:`model.GraphLearner.GraphLearner` module. If None, no deleting is applied"""

GL_ACT= 'relu'
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Non-linear activation function to use in the :data:`model.GraphLearner.GraphLearner` module"""

USE_SIGMOID= True
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Use the sigmoid as activation function after the computation of the attention in the :data:`model.GraphLearner.GraphLearner` module"""

USE_GATv2= True
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Use GATV2 instead of GAT for the multi-head attention in the :data:`model.GraphLearner.GraphLearner` module"""

CONCAT= False
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Concatenate (True) or average (False) the multi-head attention in the :data:`model.GraphLearner.GraphLearner` module"""

BETA= False
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
If True will combine aggregation and skip information in the :data:`model.GraphLearner.GraphLearner` module"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TRANSFORMER VALUES

TRANSFORMER_TYPE= None # TransformerType.TRANSFORMER_ENCODER
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Type of transformer used in the :data:`model.Transformer.Transformer` module"""

TRANSFORMER_NUM_HEADS= None # 8
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Number of heads for multi-head attention in the :data:`model.Transformer.Transformer` module"""

NUM_ENCODER_LAYERS= None # 3
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Number of sub-encoder layers in the encoder in the :data:`model.Transformer.Transformer` module"""

NUM_DECODER_LAYERS= None # 3
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Number of sub-decoder layers in the decoder in the :data:`model.Transformer.Transformer` module"""

POSITIONAL_ENCODING= None # PositionalEncodingType.SINUSOIDAL
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Type of positional encoder to use in the :data:`model.Transformer.Transformer` module. It can be None if don't use any"""

DIM_FEEDFORWARD= None # 2048
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Dimension of the feedforward network model in the :data:`model.Transformer.Transformer` module"""

TRANSFORMER_DROPOUT= None # 0.3
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Dropout probability applied in the :data:`model.Transformer.Transformer` module"""

TRANSFORMER_ACT= None
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Non-linear activation function to use in the :data:`model.Transformer.Transformer` module"""

NUM_INPUTS= MAX_SEQ_LEN if (TRANSFORMER_TYPE is not None) else None
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Number of inputs passed during the forward method in the :data:`model.Transformer.Transformer` module"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GATED GRAPH NEURAL NETWORKS VALUES

HIDDEN_DIM_GGNN= 0
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Hidden dimension in the :data:`model.GatedGraphNeuralNetworks.GGNNLayer` module. Only if `USE_GRU` is True"""

NUM_STEPS= 5
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Number of propagation steps in the :data:`model.GatedGraphNeuralNetworks.GGNNLayer` module"""

NUM_GGNN_LAYERS= 1
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Number of propagation modules in the :data:`model.GatedGraphNeuralNetworks.GGNNLayer` module"""

ACT_GGNN= None
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Non-linear activation function to use inside the linear activation in the :data:`model.GatedGraphNeuralNetworks.GGNNLayer` module. If None use the default class value"""

USE_GRU_IN_GGNN= False
"""Used in :data:`model.SGLClassifier.SGLC_Classifier` \\
Use the GRU module instead of the standard propagator in the :data:`model.GatedGraphNeuralNetworks.GGNNLayer` module"""
