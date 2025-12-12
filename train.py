# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# IMPORT
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
from torch.utils.data import DataLoader, Subset

from utils.classes.Checkpoint_manager import CheckPoint
from utils.constant.constants_eeg import *
from utils.constant.constants_main import *
from utils.classes.Metrics_classes import *
from utils.classes.Metric_manager import Metrics
from data.scaler.scaler import *
from data.utils import *

from data.dataloader.dataloader import SeizureDataset, SeizureSampler

from model.loss.loss_regularization import *
from model.SGLCModel import SGLC_Classifier
from model.loss.loss_classes import *

from torch.nn.functional import one_hot
from train_args import parse_arguments

from datetime import datetime, timedelta
from tqdm.auto import tqdm
import numpy as np
import logging
import random
import string
import os
import re


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GLOBAL VARIABLE
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

MIN_SAMPLER_PER_BATCH= max(1, round( PERCENTAGE_BOTH_CLASS_IN_BATCH/100 * BATCH_SIZE )) if (PERCENTAGE_BOTH_CLASS_IN_BATCH is not None) else 0
"""If not None, min number of both class in a batch"""

START_EPOCH= 0
"""Number of train epochs of the loaded model before the training"""

def generate_id(char_num:int, chars:str=string.ascii_lowercase+string.digits):
    """Generate a random id with a certain number of chars using a string"""
    return ''.join(random.choices(chars, k=char_num))
STOP_FILE = "./stop_execution_{}.txt".format(generate_id(3))
"""Stop file to use to stop the execution and save the model without wait the checkpoint"""

DEVICE= "cpu"
NUM_SEIZURE_DATA= 0
NUM_NOT_SEIZURE_DATA= 0

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# UTILS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class TqdmMinutes(tqdm):
    """
    Custom tqdm:
      - switches s/it → min/it when iteration time > 60 sec
      - adds ETA clock time in HH:MM:SS
    """
    def __init__(self, *args, bar_format=None, **kwargs):
        # if caller does not supply bar_format → use default
        if bar_format is None:
            bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] ETA: {eta_clock} LAST_UPDATE: {last_update}"
        super().__init__(*args, bar_format=bar_format, **kwargs)
    
    @property
    def format_dict(self):
        d = super().format_dict

        # ---- determine if switching to minutes ----
        rate = d["rate"]  # iterations per second
        if rate and rate > 0:
            sec_per_iter = 1.0 / rate
            if sec_per_iter > 60:
                min_per_iter = sec_per_iter / 60
                d["rate_fmt"] = f"{min_per_iter:0.2f}min/{d['unit']}"

        # ---- ETA as a clock time ----
        remaining = (self.total - self.n) / rate if rate and self.total else None
        if remaining is not None:
            finish = datetime.now() + timedelta(seconds=remaining)
            d["eta_clock"] = finish.strftime("%H:%M:%S")
            d["last_update"] = datetime.now().strftime("%H:%M:%S")
        else:
            d["eta_clock"] = "--:--:--"
            d["last_update"] = "--:--:--"
        
        return d

def get_model(dir:str, specific_num:int=None) -> tuple[str, int]:
    """
    Search for model files with numeric identifiers in a directory tree.\\
    Recursively searches the specified directory and all subdirectories for files matching the pattern: `<prefix>_<number>.<extension>`\\
    The function extracts the numeric identifier from each matching filename and either:
    - Returns the file with the highest number (default behavior)
    - Returns the file with a specific number if requested
    
    Args:
        dir (str):              Root directory path to search. All subdirectories will be recursively traversed.
        specific_num (int):     If provided, returns the first file found with this exact number instead of searching for the maximum
    
    Returns:
        tuple(str, int):
            - filename (str):   Full filepath (including directory path) of the matching model file. If no matching files exist returnsa an empty string
            - epoch (int):      Number found for the full filepath
    """
    pattern= re.compile(r'_(\d+)\.(\D+)$')
    
    curr_epoch= 0
    output_filename= ""
    
    for path, _, files in os.walk(dir):
        for filename in files:
            match = re.search(pattern, filename)
            if match:
                number = int(match.group(1))
                if number > curr_epoch:
                    curr_epoch= number
                    output_filename= os.path.join(path, filename)
                if (specific_num is not None) and (specific_num==number):
                    return os.path.join(path, filename), curr_epoch
    
    return output_filename, curr_epoch

def scaler_file_patient_ids(dictionary:dict[str, list[int]], separator:str="-") -> str:
    """
    Generate a string using the `separator` to divide the numbers of the patient ids.
    
    Args:
        dictionary (dict[str, list[int]]):  Dictionary with patient_id as key and list of labels of integers as value
        separator (str):                    Separator to use to divide the patient ids
    
    Examples:
        >>> scaler_file_patient_ids(dictionary, separator="_")
        >>> '02_04_05_06_07_09_10_11_12_14_15_17_18_20_21_22_23_24'
        >>> scaler_file_patient_ids(dictionary, separator="-")
        >>> '02-04-05-06-07-09-10-11-12-14-15-17-18-20-21-22-23-24'
        >>> scaler_file_patient_ids(dictionary, separator=" - ")
        >>> '02 - 04 - 05 - 06 - 07 - 09 - 10 - 11 - 12 - 14 - 15 - 17 - 18 - 20 - 21 - 22 - 23 - 24'
    """
    return separator.join(sorted([key.replace("chb", "") for key in dictionary.keys()], key=lambda x : int(x)))

def check_stop_file():
    """Check if the stop file exists"""
    return os.path.exists(STOP_FILE)

def delete_stop_file():
    """Delete the stop file if exists"""
    if check_stop_file():
        os.remove(STOP_FILE)

def func_operation(x:Tensor) -> Tensor:
    """Operation to apply at the scaler"""
    return x.transpose(dim0=0, dim1=2).reshape(x.size(2), -1)

def dict_to_str(list_to_print:list[tuple[str,any]], print_none:bool=False, print_zero:bool=False):
    """Generate a string given the input"""
    string= ""
    ljust_value= max([len(item) for item,_ in list_to_print])
    for name,value in list_to_print:
        if not(print_none) and (value is None):
            continue
        if not(print_zero) and (isinstance(value, int) or isinstance(value, float)) and (value==0.0):
            continue
        string += "\t{} : {}\n".format(name.ljust(ljust_value), value)
    return string

def additional_info(preprocessed_data:bool, dataset_data=list[tuple[str,any]]) -> str:
    """Extract static additionl info"""
    # DATASET
    dataset_tuple_no_preprocess = [
        ("MAX_SEQ_LEN", MAX_SEQ_LEN),
        ("TIME_STEP_SIZE", TIME_STEP_SIZE),
        ("USE_FFT", USE_FFT)
    ]
    dataset_tuple = [
        ("TOP_K", TOP_K),
        *dataset_data
    ]
    if not(preprocessed_data):
        dataset_tuple.extend(dataset_tuple_no_preprocess)
    dataset_str = "Dataset info:\n{}".format(dict_to_str(dataset_tuple))
    
    # MODEL
    model_tuple = [
        ("NUM_CELLS", NUM_CELLS),
        ("GRAPH_SKIP_CONN", GRAPH_SKIP_CONN),
        ("USE_GRU", USE_GRU)
    ]
    model_str = "Model info:\n{}".format(dict_to_str(model_tuple))
    
    # GRAPH LEARNER
    GL_tuple = [
        ("HIDDEN_DIM_GL", HIDDEN_DIM_GL),
        ("NUM_HEADS", NUM_HEADS),
        ("ATTENTION_TYPE", ATTENTION_TYPE),
        ("ACT", ACT),
        ("NUM_LAYERS", NUM_LAYERS),
        ("DROPOUT", DROPOUT),
        ("EPSILON", EPSILON),
        ("USE_SIGMOID", USE_SIGMOID),
        ("USE_GATv2", USE_GATv2),
        ("CONCAT", CONCAT),
        ("BETA", BETA)
    ]
    GL_str = "GL info:\n{}".format(dict_to_str(GL_tuple))
    
    # GATED GRAPH NEURAL NETWORK
    GGNN_tuple = [("HIDDEN_DIM_GGNN", HIDDEN_DIM_GGNN)] if USE_GRU else []
    GGNN_tuple.extend([
        ("NUM_STEPS", NUM_STEPS),
        ("USE_GRU_IN_GGNN", USE_GRU_IN_GGNN)
    ])
    GGNN_str = "GGNN info:\n{}".format(dict_to_str(GGNN_tuple))
    
    # LOSSES
    loss_tuple = [
        ("LEARNING_RATE", LEARNING_RATE),
        ("DAMP_SMOOTH", DAMP_SMOOTH),
        ("DAMP_DEGREE", DAMP_DEGREE),
        ("DAMP_SPARSITY", DAMP_SPARSITY)
    ]
    loss_str = "Losses info:\n{}".format(dict_to_str(loss_tuple))
    
    total_str = "\n".join([dataset_str, model_str, GL_str, GGNN_str, loss_str])
        
    return total_str

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TRAINING & EVALUATION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def train_or_eval(data_loader:DataLoader, model:SGLC_Classifier, prediction_loss:Loss, optimizer:torch.optim.Optimizer, show_progress:bool=False, verbose:bool=False) -> list[tuple[str, float]]:
    """
    Unique function to train and evaluate the model. The operation is only one, so the training is for only one epoch.\\
    The use of training mode or evaluation mode depend on the `optimizer` parameter. During the evaluation mode there are no
    optimizer so, when the parameter is None the function create a dummy optimizer and set to `eval()` the model instead of `train()`. 

    Args:
        data_loader (DataLoader):           Data on which compute operations for the model
        model (SGLC_Classifier):            Model to train or to evaluate
        prediction_loss (Loss):             Prediction loss class to train/evaluate the model
        optimizer (torch.optim.Optimizer):  Optimizer used for training. If it is None then the evaluation is applyed
        show_progress (bool):               Show the progress bar. The progress bar is removed when terminated
        verbose (bool):                     Useful for printing information during the execution

    Returns:
        list(tuple(str, float)):            The list has as many tuples as used metrics. Each tuple have the name of the metric and its value
    """
    is_training = optimizer is not None
    model.train(is_training)
    
    # init metrics
    average_smooth   = Loss_Meter("smooth")   if (DAMP_SMOOTH!=0)   else None
    average_degree   = Loss_Meter("degree")   if (DAMP_DEGREE!=0)   else None
    average_sparsity = Loss_Meter("sparsity") if (DAMP_SPARSITY!=0) else None
    average_pred     = Loss_Meter("pred")     if (average_smooth or average_degree or average_sparsity) else None
    average_total    = Loss_Meter()
    accuracy         = Accuracy_Meter([1.0, NUM_NOT_SEIZURE_DATA/NUM_SEIZURE_DATA], num_classes=NUM_CLASSES)
    conf_matrix      = ConfusionMatrix_Meter(NUM_CLASSES)
    
    # enable or not the gradients
    with (torch.enable_grad() if is_training else torch.no_grad()):
        for x,target,adj in (tqdm(data_loader, desc=f"{'Train' if model.training else 'Eval'} current epoch", leave=False) if show_progress else data_loader):
            x:Tensor= x.to(device=DEVICE)
            target:Tensor= target.to(device=DEVICE)
            adj:Tensor= adj.to(device=DEVICE)
            
            result, node_matrix, adj_matrix = model.forward(x, adj)
            
            # reshape from (batch_size, seq_length, num_nodes, input_dim) to (batch_size, num_nodes, seq_length*input_dim) for smoothness_loss_func
            node_matrix_for_smooth= node_matrix.transpose(dim0=1, dim1=2)
            node_matrix_for_smooth= node_matrix_for_smooth.reshape(node_matrix_for_smooth.size(0), node_matrix_for_smooth.size(1), -1)
            
            loss_pred = prediction_loss.compute_loss(result, target)
            
            loss_smooth= smoothness_loss_func(node_matrix_for_smooth, adj_matrix)
            loss_degree= degree_regularization_loss_func(adj_matrix)
            loss_sparsity= sparsity_loss_func(adj_matrix)
            
            total_loss= loss_pred + DAMP_SMOOTH*loss_smooth + DAMP_DEGREE*loss_degree + DAMP_SPARSITY*loss_sparsity
            
            if is_training:
                optimizer.zero_grad()
                total_loss.mean().backward()
                optimizer.step()
            
            # update metrics from (batch_size,1) to (batch_size)
            target= target.squeeze(-1)
            average_total.update(total_loss)
            accuracy.update(result, target)
            conf_matrix.update(result, target)
            
            loss_tuple:list[tuple[Loss_Meter,Tensor]]=[
                (average_pred,     loss_pred),
                (average_smooth,   DAMP_SMOOTH*loss_smooth),
                (average_degree,   DAMP_DEGREE*loss_degree),
                (average_sparsity, DAMP_SPARSITY*loss_sparsity)
            ]
            for average,loss in loss_tuple:
                if (average is not None):
                    average.update(loss)
    
    model.eval()
    
    metrics= [
        average_total.get_metric(),
        accuracy.get_metric(),
        *accuracy.get_class_accuracy(),
        *accuracy.get_avg_target_prob(),
        conf_matrix.get_precision(),
        conf_matrix.get_recall(),
        conf_matrix.get_f1_score()
    ]
    for average,_ in loss_tuple:
        if (average is not None):
            metrics.extend([average.get_metric()])
    
    # print metrics during execution
    if verbose:
        mode= "Train" if is_training else "Eval"
        max_len= max(len(name) for name,_ in metrics)
        print(f"\n{mode} mode:")
        for name,value in metrics:
            print(f"{name:<{max_len}} --> {value:.6f}")
    
    return metrics

def eval(data_loader:DataLoader, model:SGLC_Classifier, prediction_loss:Loss, verbose:bool=True, show_progress:bool=False):
    """For more info see the function :func:`train_or_eval`"""
    return train_or_eval(data_loader=data_loader, model=model, prediction_loss=prediction_loss, optimizer=None, verbose=verbose, show_progress=show_progress)

def train_epoch(data_loader:DataLoader, model:SGLC_Classifier, prediction_loss:Loss, optimizer:torch.optim.Optimizer, verbose:bool=True, show_progress:bool=False):
    """For more info see the function :func:`train_or_eval`"""
    return train_or_eval(data_loader=data_loader, model=model, prediction_loss=prediction_loss, optimizer=optimizer, verbose=verbose, show_progress=show_progress)

def train(train_loader:DataLoader, val_loader:DataLoader, test_loader:DataLoader, model:SGLC_Classifier, prediction_loss:Loss, optimizer:torch.optim.Optimizer, num_epochs:int, verbose:bool=True, show_epoch_progress:bool=False):
    """
    Train the model and evaluate its performance. Use static parameters from :data:`utils.constants_main` and :data:`utils.constants_eeg`
    to implement some operations.\\
    Each time the model improves by :const:`PERCENTAGE_MARGIN`% both model and metrics will be saved. They will be saved also
    if for :const:`MAX_NUM_EPOCHS` the model has no saves and at the last iteration. The path where model will be saved 
    is defined by :const:`MODEL_PARTIAL_PATH`\\_ `epoch_number` with extention :const:`MODEL_EXTENTION` and the path where the metrics 
    will be saved will be `name_metric`\\_ `epoch_number` with extention :const:`METRICS_EXTENTION`.\\
    The metric used to is the first returned by :func:`train_or_eval`

    Args:
        train_loader (DataLoader):          Data on which train the model
        val_loader (DataLoader):            Data on which evaluate the model
        test_loader (DataLoader):           Data on which test the model
        model (SGLCModel_classification):   Model to train or to evaluate
        prediction_loss (Loss):             Prediction loss class to train the model
        optimizer (torch.optim.Optimizer):  Optimizer used for training
        num_epochs (int):                   Number of epochs for trainig
        verbose (bool):                     Useful for printing information during the execution of the evaluation method
        show_progress (bool):               Show the inner progress bar. The progress bar is removed when terminated
    """
    # print the information about the stop file to interrupt the training
    LOGGER.info("To stop the execution create the file '{}' in the current folder".format(os.path.basename(STOP_FILE)))
    delete_stop_file()
    
    # using a dataloader of one batch with one item to compute dynamically the number of metrics and the name of metrics
    single_item_dataset= Subset(val_loader.dataset, indices=[0])
    single_item_dataloader = DataLoader(single_item_dataset, batch_size=1, shuffle=False)
    dummy_metrics= eval(single_item_dataloader, model, prediction_loss, verbose=False, show_progress=False)
    metrics_name= [name for name,_ in dummy_metrics]
    num_metrics= len( metrics_name )
    
    # generate the metrics array
    array_train= np.empty((START_EPOCH+num_epochs, num_metrics), dtype=np.float64)
    array_val=   np.empty((START_EPOCH+num_epochs, num_metrics), dtype=np.float64)
    array_test=  np.empty((START_EPOCH+num_epochs, num_metrics), dtype=np.float64)
    
    # fusion with old metrics if exist
    if START_EPOCH!=0:
        epoch_folder = os.path.join(METRICS_SAVE_FOLDER, f"{EPOCH_FOLDER_NAME}_{START_EPOCH}")
        for index,name in enumerate(metrics_name):
            position = os.path.join(epoch_folder, f"{name}_{START_EPOCH}.{METRICS_EXTENTION}")
            array_train[0:START_EPOCH, index], array_val[0:START_EPOCH, index], array_test[0:START_EPOCH, index] = Metrics.load(position)
    
    # real training
    higher_is_better= False
    early_stop_start= (START_USE_EARLY_STOP-START_EPOCH) if (START_USE_EARLY_STOP-START_EPOCH)>0 else 0
    checkpoint_observer= CheckPoint(best_k=BEST_K_MODELS, each_spacing=MAX_NUM_EPOCHS, total_epochs=num_epochs, higher_is_better=higher_is_better, early_stop_patience=EARLY_STOP_PATIENCE, early_stop_start=early_stop_start)
    checkpoint_observer.margin= PERCENTAGE_MARGIN
    
    LOGGER.info(
        "CheckPoint will save {} values of '{}':".format('higher' if higher_is_better else 'lower', dummy_metrics[0][0]) +
        "\n" +
        "\tbest K model     : {} with margin of {:.2f}%".format(BEST_K_MODELS, 100*PERCENTAGE_MARGIN) +
        "\n" +
        "\teach epochs      : {}".format(MAX_NUM_EPOCHS) +
        "\n"+
        "\tearly stop       : {}".format(EARLY_STOP_PATIENCE) +
        "\n"+
        "\tearly_stop_start : {}".format(early_stop_start)
    )
    
    for epoch_num in TqdmMinutes(range(num_epochs), desc="Progress", unit="epoch"):
        metrics_train= train_epoch(train_loader, model, prediction_loss, optimizer, verbose=False, show_progress=(epoch_num==0))
        metrics_val=   eval(val_loader,  model, prediction_loss, verbose=verbose, show_progress=(epoch_num==0))
        metrics_test=  eval(test_loader, model, prediction_loss, verbose=verbose, show_progress=(epoch_num==0))

        current_idx = START_EPOCH + epoch_num
        array_train[current_idx] = np.array([value for _,value in metrics_train])
        array_val[current_idx]   = np.array([value for _,value in metrics_val])
        array_test[current_idx]  = np.array([value for _,value in metrics_test])
        
        used_metric= metrics_val[0][1]
        
        # save the metrics at each iteration
        for index,name in enumerate(metrics_name):
            position= os.path.join(METRICS_SAVE_FOLDER, f"{EPOCH_FOLDER_NAME}_{epoch_num+START_EPOCH+1}", f"{name}_{epoch_num+START_EPOCH+1}.{METRICS_EXTENTION}")
            until_epoch = START_EPOCH+epoch_num+1
            Metrics.save(position, train_metric=array_train[0:until_epoch, index], val_metric=array_val[0:until_epoch, index], test_metric=array_test[0:until_epoch, index])
        
        # conditions to save the model
        saved_files= []
        if checkpoint_observer.check_saving(used_metric) or checkpoint_observer.check_early_stop() or check_stop_file():
            file_path= f"{MODEL_PARTIAL_PATH}_{epoch_num+START_EPOCH+1}.{MODEL_EXTENTION}"
            saved_files.append(file_path)
            model.save(file_path)
            
            checkpoint_observer.update_saving(used_metric, saved_files)
        
        # delete obsolete model and check saved
        checkpoint_observer.delete_obsolete_checkpoints(auto_delete=True)
        
        # check the early stop
        if checkpoint_observer.check_early_stop():
            LOGGER.warning(f"Reached early stop at epoch {epoch_num+1}")
            break
        
        # check stop file
        if check_stop_file():
            LOGGER.warning("Stop file '{}' found. Stopped at epoch {}".format(os.path.basename(STOP_FILE), epoch_num+1))
            delete_stop_file()
            break

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MAIN
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def main():
    # take input from command line and print some informations
    loss_type, input_dir, files_record, method, lambda_value, scaler, single_scaler, save_num, do_train, num_epochs, verbose, preprocess_dir = parse_arguments()
    string_additional_info= additional_info(
        preprocessed_data=(preprocess_dir is not None),
        dataset_data=[
            ('method', method),
            ('lambda_value', lambda_value),
            ('scaler', scaler)
        ]
    )
    string= "{}{}".format("\n\t", "\n\t".join([item for item in string_additional_info.split("\n")]))
    LOGGER.info(string)
    
    # global variables
    global DEVICE, START_EPOCH, NUM_NOT_SEIZURE_DATA, NUM_SEIZURE_DATA
    
    # load dataset
    LOGGER.info("Loading dataset with at least ({}) samples for class in a batch of ({}) [min positive ratio {:.3f}%]...".format(MIN_SAMPLER_PER_BATCH, BATCH_SIZE, 100 * MIN_SAMPLER_PER_BATCH / BATCH_SIZE))
    dataset= SeizureDataset(
        input_dir= input_dir,
        files_record= files_record,
        
        time_step_size= TIME_STEP_SIZE,
        max_seq_len= MAX_SEQ_LEN,
        use_fft= USE_FFT,
        
        preprocess_data=preprocess_dir,
        
        method=method,
        top_k= TOP_K,
        lambda_value=lambda_value
    )
    
    # splitting data
    remaining_data, test_dict = split_patient_data_specific(dataset.targets_dict(), TEST_PATIENT_IDS)
    train_dict, val_dict = split_patient_data(remaining_data, split_ratio=PERCENTAGE_TRAINING_SPLIT)
    
    train_set= subsets_from_patient_splits(dataset, dataset.targets_index_map(), train_dict)
    val_set=   subsets_from_patient_splits(dataset, dataset.targets_index_map(), val_dict)
    test_set=  subsets_from_patient_splits(dataset, dataset.targets_index_map(), test_dict)
    
    # generating new scaler
    if (scaler is not None):
        LOGGER.info(f"Loading scaler '{scaler.name}'...")
        scaler_name= "{}{}_{}.{}".format(scaler.name, '_single' if single_scaler else "", scaler_file_patient_ids(train_dict, separator="-"), MODEL_EXTENTION)
        scaler_path= os.path.join(SCALER_SAVE_FOLDER, scaler_name)
        scaler= ConcreteScaler.create_scaler(scaler, device=DEVICE)
    
    if (scaler is not None):
        if os.path.exists(scaler_path):
            scaler= scaler.load(scaler_path, device=DEVICE)
        else:
            scaler.fit(train_set, single_value=single_scaler, func_operation=func_operation, use_tqdm=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
            scaler.save(scaler_path)
        dataset.scaler= scaler
    
    # generate dataloaders
    train_sampler = None
    if (MIN_SAMPLER_PER_BATCH != 0):
        train_sampler= SeizureSampler(dataset.targets_list(), train_set.indices, batch_size=BATCH_SIZE, n_per_class=MIN_SAMPLER_PER_BATCH, seed=RANDOM_STATE)
        LOGGER.warning("train_sampler is OK")

    train_loader= DataLoader(dataset,  sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader=  DataLoader(test_set, sampler=None,          batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader=   DataLoader(val_set,  sampler=None,          batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    
    # print on screen some informations
    def pos_neg_samples(dictionary:dict[str, list[int]]):
        l= [label for value_list in dictionary.values() for label in value_list]
        return sum(l), len(l)-sum(l)
    
    names= ["test", "train", "validation"]
    dictionaries= [test_dict, train_dict, val_dict]
    ljust_value= len(max(names))
    
    for name,dictionary in zip(names,dictionaries):
        string = "" 
        samples_pos, samples_neg = pos_neg_samples(dictionary)
        string+= "Using patient(s) for {} : '{}'".format(name.ljust(ljust_value), ", ".join(dictionary.keys()))
        string+= "\n\tTotal positive samples  : {:>{}}/{:,}".format(samples_pos, len(str(samples_pos+samples_neg)), samples_pos+samples_neg)
        string+= "\n\tTotal negative samples  : {:>{}}/{:,}".format(samples_neg, len(str(samples_pos+samples_neg)), samples_pos+samples_neg)
        string+= "\n\tPositive ratio          : {:.3f}%".format(100 * samples_pos / (samples_pos+samples_neg))
        LOGGER.info(string)
    
    # load model if exists or create a new model
    LOGGER.info("Loading model...")
    DEVICE= 'cuda' if (torch.cuda.is_available() and USE_CUDA) else 'cpu'
    LOGGER.info(f"Using {DEVICE} device...")
    
    feature_matrix, _, _ = dataset[0]
    num_nodes= feature_matrix.size(1)
    input_dim= feature_matrix.size(2)
            
    model= SGLC_Classifier(
        num_classes     = NUM_CLASSES,
        
        num_cells       = NUM_CELLS,
        input_dim       = input_dim,
        num_nodes       = num_nodes,
        
        graph_skip_conn = GRAPH_SKIP_CONN,
        use_GRU         = USE_GRU,
        
        hidden_dim_GL   = HIDDEN_DIM_GL,
        attention_type  = ATTENTION_TYPE,
        num_layers      = NUM_LAYERS,
        num_heads       = NUM_HEADS,
        dropout         = DROPOUT,
        epsilon         = EPSILON,
        
        hidden_dim_GGNN = HIDDEN_DIM_GGNN,
        num_steps       = NUM_STEPS,
        use_GRU_in_GGNN = USE_GRU_IN_GGNN,
        
        use_sigmoid     = USE_SIGMOID,
        act             = ACT,
        v2              = USE_GATv2,
        concat          = CONCAT,
        beta            = BETA,
        
        seed            = RANDOM_SEED,
        device          = DEVICE
    )
    
    filename, num_epoch = get_model(MODEL_SAVE_FOLDER, specific_num=save_num)
    START_EPOCH= num_epoch
    if len(filename)==0 and (not do_train):
        raise ValueError(f"Evaluation stopped, model not present in the '{MODEL_SAVE_FOLDER}' folder")
    if len(filename)!=0:
        model= model.load(filename, device=DEVICE)
        LOGGER.info(f"Loaded '{os.path.basename(filename)}'...")
    
    # set the number of seizure and not seizure data
    NUM_SEIZURE_DATA, NUM_NOT_SEIZURE_DATA = pos_neg_samples(train_dict)
    if do_train and (NUM_NOT_SEIZURE_DATA==0):
        raise ValueError(f"Training aborted, no data without seizure")
    if do_train and (NUM_SEIZURE_DATA==0):
        raise ValueError(f"Training aborted, no data with seizure")
    
    # create loss to use
    # neg_weight = 1.0
    # pos_weight = NUM_NOT_SEIZURE_DATA/NUM_SEIZURE_DATA
    # weight = torch.Tensor([neg_weight/(pos_weight+neg_weight), pos_weight/(pos_weight+neg_weight)]).to(device=DEVICE) if USE_WEIGHT else None
    weight = torch.Tensor([1.0, NUM_NOT_SEIZURE_DATA/NUM_SEIZURE_DATA]).to(device=DEVICE) if USE_WEIGHT else None
    alpha = FOCAL_LOSS_APLHA if (FOCAL_LOSS_APLHA is not None) else NUM_NOT_SEIZURE_DATA / (NUM_NOT_SEIZURE_DATA + NUM_SEIZURE_DATA)
    match loss_type:
        case LossType.CROSS_ENTROPY:
            loss = CrossEntropy(weight=weight)
        case LossType.BCE_LOGITS:
            loss = BCE_Logits(num_classes=NUM_CLASSES, pos_weight=weight)
        case LossType.FOCAL_LOSS:
            loss = FocalLoss(num_classes=NUM_CLASSES, alpha=alpha, gamma=FOCAL_LOSS_GAMMA)
        case _:
            raise NotImplementedError("Loss {} is not implemented yet".format(loss_type))
    loss_params_str = dict_to_str(list(loss.parameters().items()))
    LOGGER.info("Using loss type '{}'{}".format(loss_type.name, '' if loss_params_str=="" else f' with parameters :\n{loss_params_str}'))
    
    # start train or evaluation
    if do_train:
        LOGGER.info(f'Start training : {datetime.now().strftime("%d/%m/%Y at %H:%M:%S")}\n')
        optimizer= torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train(train_loader, val_loader, test_loader, model=model, prediction_loss=loss, optimizer=optimizer, num_epochs=num_epochs, verbose=verbose, show_epoch_progress=False)
    else:
        eval(test_loader, model, prediction_loss=loss, verbose=True, show_progress=True)

if __name__=='__main__':
    main()
