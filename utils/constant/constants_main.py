"""Contains constant useful to the main functions instad of have a lot of parameters passed in line"""
from utils.constant.constants_eeg import PRETRAIN
import os

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FOLDERS AND FILE NAMES
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

PRETRAIN_SUFFIX= "_pretrain"
"""Suffix to add if the `PRETRAIN` constant is True"""

METRICS_SAVE_FOLDER= "./metrics{}".format(PRETRAIN_SUFFIX if PRETRAIN else "")
"""Folder where save the metric values"""

MODEL_SAVE_FOLDER= "./weights{}".format(PRETRAIN_SUFFIX if PRETRAIN else "")
"""Folder where save the model weights"""

TSNE_SAVE_FOLDER= "./t-SNE"
"""Folder where the t-SNE projection will be saved"""

ANALYZER_SAVE_FOLDER= "./analyzer"
"""Folder where the inference-time predictions analysis and ROC curve will be saved"""

SCALER_SAVE_FOLDER= "./scalers_weight"
"""Folder where save the values calculates with the scalar"""

MODEL_NAME= "Model{}".format(PRETRAIN_SUFFIX if PRETRAIN else "")
"""Name of the file where the model will be saved"""

EPOCH_FOLDER_NAME= "epoch"
"""Name of the folder where the metrics will be saved"""

TSNE_NAME= "t-SNE"
"""Name of the t-SNE projection file where the image will be saved"""

ANALYZER_NAME= "analysis"
"""Name of the inference-time predictions analysis file as numpy array will be saved"""

ANALYZER_HISTOGRAM_NAME= "analysis_hist"
"""Name of the inference-time predictions analysis file where the image will be saved"""

ANALYZER_ROC_NAME= "analysis_ROC"
"""Name of the ROC curve file where the image will be saved"""

MODEL_EXTENTION= "pth"
"""Extention of the model weights"""

METRICS_EXTENTION= "npz"
"""Extention of the metrics"""

MODEL_PARTIAL_PATH= os.path.join(MODEL_SAVE_FOLDER, MODEL_NAME)
"""Model path without extention"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# INFO DATASET INITIALIZATION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

TEST_PATIENT_IDS= None
"""List of ids of patients to use for test"""

VAL_PATIENT_IDS= None # ["chb10", "chb13", "chb14"]
"""List of ids of patients to use fot valdidation"""

EXCEPT_DATA= ["chb16"]
"""List of ids of patient which are not allowed to be in the dataset"""

K_FOLD_VAL= 2
"""Number of patients to use for each fold in the k-fold cross validation"""

K_FOLD_TEST= 1
"""Number of patients to use for each fold in the k-fold cross test""" 

PATIENT_SPECIFIC= None # "chb01"
"""Patient specific to use for a k-fold training"""

PERCENTAGE_TRAINING_SPLIT= 0.8
"""How much of the dataset must be used only for training"""

PERCENTAGE_BOTH_CLASS_IN_BATCH= None
"""Min percentage number of both class in a batch. If set to None does not use a sampler in the DataLoader"""

BATCH_SIZE= 96
"""Batch size used during the training. The last batch can have different size"""

NUM_WORKERS= 12
"""How many subprocesses to use for data loading"""

RANDOM_STATE= 13
"""Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls"""

RANDOM_SEED= 0
"""Seed to controls the weights initializations. If None, don't use any seed"""

DATASET_REDUCTION= 9
"""Ratio of negative samples to maintain after the reduction. If not set, no reduction will be applied"""

DATASET_REDUCTION_SEED= 0
"""Seed for reduction sampling reproducibility"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# LOSSES INFO
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

TAU= 0.5
"""Threshold for binary classification"""

LEARNING_RATE= 1e-5
"""Learning rate of the model"""

USE_WEIGHT= True
"""If True, will be use weighted loss if available"""

FOCAL_LOSS_APLHA= 0.25
"""Weighting factor in range [0, 1] to balance positive vs negative examples. High weight for positive class. If None are set based on the representativeness of the classes"""

FOCAL_LOSS_GAMMA= 2.0
"""Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples. Higher is more focus on hard examples"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CHECKPOINT INFO
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

BEST_K_MODELS= 0
"""Maximum number of model maintained as best model in performance"""

MAX_NUM_EPOCHS= 5
"""Maximum number of epochs after which the model will be save"""

PERCENTAGE_MARGIN= 0.00
"""Percentage of margin needed with respect to the metric considered to save the model"""

EARLY_STOP_PATIENCE= None
"""Number of epochs to wait to early stop the model training"""

START_USE_EARLY_STOP= 20
"""Number of epochs to wait before using the early stop"""
