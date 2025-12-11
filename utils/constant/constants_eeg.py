"""Contains constant useful to the computation of EEG files"""
from model.GraphLearner.GraphLearnerAttention import GraphLearnerAttention

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# UTILS INFO
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

FREQUENCY_CHB_MIT= 256
"""Frequency of the data"""

VERY_SMALL_NUMBER = 1e-12
"""Value used to avoid zero division"""

INF = 1e20
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# DAMPING FACTOR FOR LOSS FUNCTIONS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

DAMP_SMOOTH= 0.0
"""Damping factor for :func:`model.loss_functions.smoothness_loss_func`. Set to 0 to not use it"""

DAMP_DEGREE= 0.0
"""Damping factor for :func:`model.loss_functions.degree_regularization_loss_func`. Set to 0 to not use it"""

DAMP_SPARSITY= 0.0
"""Damping factor for :func:`model.loss_functions.sparsity_loss_func`. Set to 0 to not use it"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# INFO FOR model.ASGPFmodel.SGLCModel_classification
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

NUM_CLASSES= 2
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Number of output classes"""

NUM_CELLS= 2
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Number of the :data:`model.SGLCell.SGLCell` layers in the encoder stack"""

GRAPH_SKIP_CONN= 0.3
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Skip connection weight for adjacency updates"""

USE_GRU= False
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Use GRU module in the :data:`model.SGLCell.SGLC_Cell` and hidden state in the :data:`model.GatedGraphNeuralNetworks.GGNNLayer` module"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GRAPH LEARNER VALUES

HIDDEN_DIM_GL= 192
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Hidden dimension for the :data:`model.GraphLearner.GraphLearner` module"""

ATTENTION_TYPE= GraphLearnerAttention.GRAPH_ATTENTION_LAYER
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Type of attention used in the :data:`model.GraphLearner.GraphLearner` module"""

NUM_LAYERS= 3
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Number of message passing layers in the GAT or Transformer module for the :data:`model.GraphLearner.GraphLearner` module"""

NUM_HEADS= 8
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Number of heads for multi-head attention in the :data:`model.GraphLearner.GraphLearner` module"""

DROPOUT= 0.4
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Dropout probability applied in the attention layer for the :data:`model.GraphLearner.GraphLearner` module"""

EPSILON= None
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Threshold for deleting weak connections in the learned graph for the :data:`model.GraphLearner.GraphLearner` module. If None, no deleting is applied"""

ACT= 'relu'
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Non-linear activation function to use in the :data:`model.GraphLearner.GraphLearner` module"""

USE_SIGMOID= False
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Use the sigmoid as activation function after the computation of the attention in the :data:`model.GraphLearner.GraphLearner` module"""

USE_GATv2= False
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Use GATV2 instead of GAT for the multi-head attention in the :data:`model.GraphLearner.GraphLearner` module"""

CONCAT= False
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Concatenate (True) or average (False) the multi-head attention in the :data:`model.GraphLearner.GraphLearner` module"""

BETA= False
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
If True will combine aggregation and skip information in the :data:`model.GraphLearner.GraphLearner` module"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GATED GRAPH NEURAL NETWORKS VALUES

HIDDEN_DIM_GGNN= 192
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Hidden dimension in the :data:`model.GatedGraphNeuralNetworks.GGNNLayer` module. Only if `USE_GRU` is True"""

NUM_STEPS= 6
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Number of propagation steps in the :data:`model.GatedGraphNeuralNetworks.GGNNLayer` module"""

USE_GRU_IN_GGNN= False
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Use the GRU module instead of the standard propagator in the :data:`model.GatedGraphNeuralNetworks.GGNNLayer` module"""
