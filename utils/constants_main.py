"""Contains constant useful to the main functions instad of have a lot of parameters passed in line"""
import os

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FOLDERS AND FILE NAMES
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

METRICS_SAVE_FOLDER= "./metrics"
"""Folder where save the metric values"""

MODEL_SAVE_FOLDER= "./weights"
"""Folder where save the model weights"""

SCALER_SAVE_FOLDER= "./scalars_weight"
"""Folder where save the values calculates with the scalar"""

MODEL_NAME= "Model"
"""Name of the file where the model will be saved"""

MODEL_EXTENTION= "pth"
"""Extention of the model weights"""

METRICS_EXTENTION= "npz"
"""Extention of the metrics"""

MODEL_PARTIAL_PATH= os.path.join(MODEL_SAVE_FOLDER, MODEL_NAME)
"""Model path without extention"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# INFO DATASET INITIALIZATION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

TEST_PATIENT_IDS= ["chb02"]
"""List of ids of patient to use for test"""

PERCENTAGE_TRAINING_SPLIT= 0.8
"""How much of the dataset must be used only for training"""

PERCENTAGE_BOTH_CLASS_IN_BATCH= 1
"""Min percentage number of both class in a batch"""

BATCH_SIZE= 64
"""Batch size used during the training. The last batch can have different size"""

NUM_WORKERS= 16
"""How many subprocesses to use for data loading"""

RANDOM_STATE= 13
"""Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# OTHERS INFO
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

LEARNING_RATE= 5e-6
"""Learning rate of the model"""

FOCAL_LOSS_APLHA= None
"""Weighting factor in range [0, 1] to balance positive vs negative examples. High weight for positive class. If None are set the default"""

FOCAL_LOSS_GAMMA= 1.5
"""Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples. Higher is more focus on hard examples. If None are set the default"""

BEST_K_MODELS= 0
"""Maximum number of model maintained as best model in performance"""

MAX_NUM_EPOCHS= 5
"""Maximum number of epochs after which the model will be save"""

PERCENTAGE_MARGIN= 0.00
"""Percentage of margin needed with respect to the metric considered to save the model"""

EARLY_STOP_PATIENCE= 20 
"""Number of epochs to wait to early stop the model training"""

START_USE_EARLY_STOP= 20
"""Number of epochs to wait before using the early stop"""
