# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# IMPORT
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from utils.Checkpoint_manager import CheckPoint
from utils.constants_eeg import *
from utils.constants_main import *
from utils.metrics_classes import *
from utils.metric import Metrics
from data.scaler import *
from data.utils import *

from data.dataloader import SeizureDataset, SeizureSampler

from model.ASGPFmodel import SGLCModel_classification
from model.loss_functions import *

from torchvision.ops import sigmoid_focal_loss
from torch.nn.functional import one_hot
from train_args import parse_arguments

from datetime import datetime
from tqdm.auto import tqdm
import numpy as np
import logging
import os
import re


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GLOBAL VARIABLE
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

MIN_SAMPLE_PER_CLASS=  max(1, round( PERCENTAGE_BOTH_CLASS_IN_BATCH/100 * BATCH_SIZE ))
MIN_SAMPLER_PER_BATCH= max(1, round( PERCENTAGE_BOTH_CLASS_IN_BATCH/100 * BATCH_SIZE ))

DEVICE= "cpu"
NUM_SEIZURE_DATA= 0
NUM_NOT_SEIZURE_DATA= 0
START_EPOCH= 0

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# UTILS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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

def func_operation(x:Tensor) -> Tensor:
    """Operation to apply at the scaler"""
    return x.transpose(dim0=0, dim1=2).reshape(x.size(2), -1)

def additional_info(*parameters_to_process:tuple[str,any]) -> str:
    """Extract static additionl info"""    
    list_to_print= [
        ("NUM_CELLS", NUM_CELLS),
        ("LEARNING_RATE", LEARNING_RATE), 
        ("DAMP_SMOOTH", DAMP_SMOOTH),
        ("DAMP_DEGREE", DAMP_DEGREE),
        ("DAMP_SPARSITY", DAMP_SPARSITY),
        
        ("GRAPH_SKIP_CONN", GRAPH_SKIP_CONN),
        ("DROPOUT", DROPOUT),
        ("USE_GATv2", USE_GATv2),
        ("USE_TRANSFORMER", USE_TRANSFORMER),
        ("CONCAT", CONCAT),
        ("USE_STANDARD_PROPAGATOR", USE_STANDARD_PROPAGATOR),
        ('NUM_LAYERS', NUM_LAYERS),
        ("USE_GRU", USE_GRU)
    ]
    list_to_print.extend(parameters_to_process)
    
    string= ""
    ljust_value= max([len(item) for item,_ in list_to_print])
    for name,value in list_to_print:
        string += "{} : {}\n".format(name.ljust(ljust_value), value)
        
    return string

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TRAINING & EVALUATION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

ALPHA_PRINT_INFO = False
def train_or_eval(data_loader:DataLoader, model:SGLCModel_classification, optimizer:torch.optim.Optimizer, show_progress:bool=False, verbose:bool=False) -> list[tuple[str, float]]:
    """
    Unique function to train and evaluate the model. The operation is only one, so the training is for only one epoch.\\
    The use of training mode or evaluation mode depend on the `optimizer` parameter. During the evaluation mode there are no
    optimizer so, when the parameter is None the function create a dummy optimizer and set to `eval()` the model instead of `train()`. 

    Args:
        data_loader (DataLoader):           Data on which compute operations for the model
        model (SGLCModel_classification):   Model to train or to evaluate
        optimizer (torch.optim.Optimizer):  Optimizer used for training. If it is None then the evaluation is applyed
        show_progress (bool):               Show the progress bar. The progress bar is removed when terminated
        verbose (bool):                     Useful for printing information during the execution

    Returns:
        list(tuple(str, float)):            The list has as many tuples as used metrics. Each tuple have the name of the metric and its value
    """
    is_training = optimizer is not None
    model.train(is_training)
    
    # pos_weight= torch.Tensor([NUM_NOT_SEIZURE_DATA / NUM_SEIZURE_DATA]).to(device=DEVICE)
    
    alpha = FOCAL_LOSS_APLHA if (FOCAL_LOSS_APLHA is not None) else NUM_NOT_SEIZURE_DATA / (NUM_SEIZURE_DATA+NUM_NOT_SEIZURE_DATA)
    gamma = FOCAL_LOSS_GAMMA if (FOCAL_LOSS_GAMMA is not None) else 1.0
    global ALPHA_PRINT_INFO
    if not(ALPHA_PRINT_INFO):
        LOGGER.info("Using alpha {:.3f} and gamma {:.3f}".format(alpha, gamma))
        ALPHA_PRINT_INFO = True
    
    # init metrics
    average_total= Average_Meter("total")
    average_pred= Average_Meter("pred")
    average_smooth= Average_Meter("smooth")
    average_degree= Average_Meter("degree")
    average_sparsity= Average_Meter("sparsity")
    accuracy= Accuracy_Meter([NUM_NOT_SEIZURE_DATA, NUM_SEIZURE_DATA], num_classes=NUM_CLASSES)
    conf_matrix= ConfusionMatrix_Meter(NUM_CLASSES)
    
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
            
            # reshape from (batch_size, 1) to (batch_size) and transform in one-hot encoder for the focal loss (the original dtype is keep)
            target_one_hot= one_hot(target.squeeze(-1).to(dtype=torch.int64), num_classes=NUM_CLASSES)
            target_one_hot= target_one_hot.to(dtype=target.dtype)

            #loss_pred= F.binary_cross_entropy_with_logits(result, target_one_hot, pos_weight=pos_weight, reduction="none").sum(dim=1)
            loss_pred = sigmoid_focal_loss(inputs=result, targets=target_one_hot, alpha=alpha, gamma=gamma, reduction='none').sum(dim=1)
            
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
            average_pred.update(loss_pred)
            average_smooth.update(DAMP_SMOOTH*loss_smooth)
            average_degree.update(DAMP_DEGREE*loss_degree)
            average_sparsity.update(DAMP_SPARSITY*loss_sparsity)
            accuracy.update(result, target)
            conf_matrix.update(result, target)
    
    model.eval()
    
    metrics= [
        average_total.get_metric(),
        average_pred.get_metric(),
        average_smooth.get_metric(),
        average_degree.get_metric(),
        average_sparsity.get_metric(),
        accuracy.get_metric(),
        *accuracy.get_class_accuracy(),
        *accuracy.get_avg_target_prob(),
        conf_matrix.get_precision(),
        conf_matrix.get_recall(),
        conf_matrix.get_f1_score()
    ]
    
    # print metrics during execution
    if verbose:
        mode= "Train" if is_training else "Eval"
        max_len= max(len(name) for name,_ in metrics)
        print(f"\n{mode} mode:")
        for name,value in metrics:
            print(f"{name:<{max_len}} --> {value:.6f}")
    
    return metrics

def eval(data_loader:DataLoader, model:SGLCModel_classification, verbose:bool=True, show_progress:bool=False):
    """For more info see the function :func:`train_or_eval`"""
    return train_or_eval(data_loader=data_loader, model=model, optimizer=None, verbose=verbose, show_progress=show_progress)

def train_epoch(data_loader:DataLoader, model:SGLCModel_classification, optimizer:torch.optim.Optimizer, verbose:bool=True, show_progress:bool=False):
    """For more info see the function :func:`train_or_eval`"""
    return train_or_eval(data_loader=data_loader, model=model, optimizer=optimizer, verbose=verbose, show_progress=show_progress)

def train(train_loader:DataLoader, val_loader:DataLoader, test_loader:DataLoader, model:SGLCModel_classification, optimizer:torch.optim.Optimizer, num_epochs:int, verbose:bool=True, show_epoch_progress:bool=False):
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
        optimizer (torch.optim.Optimizer):  Optimizer used for training
        num_epochs (int):                   Number of epochs for trainig
        verbose (bool):                     Useful for printing information during the execution of the evaluation method
        show_progress (bool):               Show the inner progress bar. The progress bar is removed when terminated
    """
    # using a dataloader of one batch with one item to compute dynamically the number of metrics and the name of metrics
    single_item_dataset= Subset(val_loader.dataset, indices=[0])
    single_item_dataloader = DataLoader(single_item_dataset, batch_size=1, shuffle=False)
    dummy_metrics= eval(single_item_dataloader, model, verbose=False, show_progress=False)
    metrics_name= [name for name,_ in dummy_metrics]
    num_metrics= len( metrics_name )
    
    # generate the metrics array
    array_train= np.empty((num_epochs, num_metrics), dtype=np.float64)
    array_val=   np.empty((num_epochs, num_metrics), dtype=np.float64)
    array_test=  np.empty((num_epochs, num_metrics), dtype=np.float64)
    
    # real training
    higher_is_better= False
    early_stop_start= (START_USE_EARLY_STOP-START_EPOCH) if (START_USE_EARLY_STOP-START_EPOCH)>0 else 0
    checkpoint_observer= CheckPoint(best_k=BEST_K_MODELS, each_spacing=MAX_NUM_EPOCHS, total_epochs=num_epochs, higher_is_better=higher_is_better, early_stop_patience=EARLY_STOP_PATIENCE, early_stop_start=early_stop_start)
    checkpoint_observer.margin= PERCENTAGE_MARGIN
    
    LOGGER.info(
        "CheckPoint will save {} values:".format('higher' if higher_is_better else 'lower') +
        "\n" +
        "\tbest K model     : {} with margin of {:.2f}%".format(BEST_K_MODELS, 100*PERCENTAGE_MARGIN) +
        "\n" +
        "\teach epochs      : {}".format(MAX_NUM_EPOCHS) +
        "\n"+
        "\tearly stop       : {}".format(EARLY_STOP_PATIENCE) +
        "\n"+
        "\tearly_stop_start : {}".format(early_stop_start)
    )
    
    for epoch_num in tqdm(range(num_epochs), desc="Progress", unit="epoch"):
        metrics_train= train_epoch(train_loader, model, optimizer, verbose=False, show_progress=(epoch_num==0))
        metrics_val=   eval(val_loader,  model, verbose=verbose, show_progress=False)
        metrics_test=  eval(test_loader, model, verbose=verbose, show_progress=False)

        array_train[epoch_num]= np.array([value for _,value in metrics_train])
        array_val[epoch_num]=   np.array([value for _,value in metrics_val])
        array_test[epoch_num]=  np.array([value for _,value in metrics_test])
        
        used_metric= metrics_val[0][1]
        
        # conditions to save the model
        saved_files= []
        if checkpoint_observer.check_saving(used_metric) or checkpoint_observer.check_early_stop():
            file_path= f"{MODEL_PARTIAL_PATH}_{epoch_num+START_EPOCH+1}.{MODEL_EXTENTION}"
            saved_files.append(file_path)
            model.save(file_path)
            
            for index,name in enumerate(metrics_name):
                position= os.path.join(METRICS_SAVE_FOLDER, f"epoch_{epoch_num+START_EPOCH+1}", f"{name}_{epoch_num+START_EPOCH+1}.{METRICS_EXTENTION}")
                saved_files.append(position)
                Metrics.save(position, train_metric=array_train[0:epoch_num+1, index], val_metric=array_val[0:epoch_num+1, index], test_metric=array_test[0:epoch_num+1, index])
            
            checkpoint_observer.update_saving(used_metric, saved_files)
        
        # delete obsolete model and check saved
        checkpoint_observer.delete_obsolete_checkpoints(auto_delete=True)
        
        # check the early stop
        if checkpoint_observer.check_early_stop():
            LOGGER.warning(f"Reached early stop at epoch {epoch_num}")
            break

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MAIN
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def main():
    # take input from command line
    input_dir, files_record, method, scaler, single_scaler, save_num, do_train, num_epochs, verbose, preprocess_dir = parse_arguments()
    string_additional_info= additional_info(('scaler', scaler), ('method', method))
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
        
        preprocess_data=preprocess_dir,
        
        method=method,
        use_fft= USE_FFT,
        top_k= TOP_K
    )
    
    # splitting data
    remaining_data, test_dict = split_patient_data_specific(dataset.targets_dict(), TEST_PATIENT_IDS)
    train_dict, val_dict = split_patient_data(remaining_data, split_ratio=PERCENTAGE_TRAINING_SPLIT)

    test_set=  subsets_from_patient_splits(dataset, dataset.targets_index_map(), test_dict)
    trian_set= subsets_from_patient_splits(dataset, dataset.targets_index_map(), train_dict)
    val_set=   subsets_from_patient_splits(dataset, dataset.targets_index_map(), val_dict)
    
    # generating new scaler
    if (scaler is not None):
        LOGGER.info(f"Loading scaler '{scaler}'...")
        scaler_name= "{}{}_{}.{}".format(scaler, '_single' if single_scaler else "",scaler_file_patient_ids(train_dict, separator="-"), MODEL_EXTENTION)
        scaler_path= os.path.join(SCALER_SAVE_FOLDER, scaler_name)
        if  (scaler=='z-score'):
            scaler= StandardScaler()
        elif (scaler=='min-max'):
            scaler= MinMaxScaler()
        else:
            raise ValueError(f"scaler '{scaler}' is not defined")
    
    if (scaler is not None):
        if os.path.exists(scaler_path):
            scaler= scaler.load(scaler_path, device=DEVICE)
        else:
            scaler.fit(trian_set, single_value=single_scaler, func_operation=func_operation, use_tqdm=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
            scaler.save(scaler_path)
        dataset.scaler= scaler
    
    # generate dataloaders
    test_sampler=  SeizureSampler(dataset.targets_list(), test_set.indices,  batch_size=BATCH_SIZE, n_per_class=MIN_SAMPLER_PER_BATCH, seed=RANDOM_STATE)
    train_sampler= SeizureSampler(dataset.targets_list(), trian_set.indices, batch_size=BATCH_SIZE, n_per_class=MIN_SAMPLER_PER_BATCH, seed=RANDOM_STATE)
    val_sampler=   SeizureSampler(dataset.targets_list(), val_set.indices,   batch_size=BATCH_SIZE, n_per_class=MIN_SAMPLER_PER_BATCH, seed=RANDOM_STATE)

    test_loader=  DataLoader(dataset, sampler=test_sampler,  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=True)
    train_loader= DataLoader(dataset, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=True)
    val_loader=   DataLoader(dataset, sampler=val_sampler,   batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=True)
    
    # print on screen some informations
    def pos_neg_samples(dictionary:dict[str, list[int]]):
        l= [label for value_list in dictionary.values() for label in value_list]
        return sum(l), len(l)-sum(l)
    
    names= ["test", "train", "validation"]
    dictionaries= [test_dict, train_dict, val_dict]
    ljust_value= len(max(names))
    
    for name,dictionary in zip(names,dictionaries):    
        samples_pos, samples_neg = pos_neg_samples(dictionary)
        LOGGER.info("Using patient(s) for {} : '{}'".format(name.ljust(ljust_value), ", ".join(dictionary.keys())))
        LOGGER.info("\tTotal positive samples  : {:>{}}/{:,}".format(samples_pos, len(str(samples_pos+samples_neg)), samples_pos+samples_neg))
        LOGGER.info("\tTotal negative samples  : {:>{}}/{:,}".format(samples_neg, len(str(samples_pos+samples_neg)), samples_pos+samples_neg))
        LOGGER.info("\tPositive ratio          : {:.3f}%".format(100 * samples_pos / (samples_pos+samples_neg)))
    
    # load model if exists
    LOGGER.info("Loading model...")
    DEVICE= 'cuda' if (torch.cuda.is_available() and USE_CUDA) else 'cpu'
    LOGGER.info(f"Using {DEVICE} device...")
    
    filename, num_epoch = get_model(MODEL_SAVE_FOLDER, specific_num=save_num)
    START_EPOCH= num_epoch
    if len(filename)==0 and (not do_train):
        raise ValueError(f"Evaluation stopped, model not present in the '{MODEL_SAVE_FOLDER}' folder")
    if len(filename)!=0:
        model= SGLCModel_classification.load(filename, device=DEVICE)
        LOGGER.info(f"Loaded '{os.path.basename(filename)}'...")
    else:        
        feature_matrix, _, _ = dataset[0]
        num_nodes= feature_matrix.size(1)
        input_dim= feature_matrix.size(2)
                
        model= SGLCModel_classification(
            num_classes= NUM_CLASSES,
            
            num_cells= NUM_CELLS,
            input_dim= input_dim,
            num_nodes= num_nodes,
            
            hidden_dim_GL= HIDDEN_DIM_GL,
            hidden_dim_GGNN=HIDDEN_DIM_GGNN,
            
            graph_skip_conn= GRAPH_SKIP_CONN,
            
            dropout= DROPOUT,
            epsilon= EPSILON,
            num_heads= NUM_HEADS,
            num_steps= NUM_STEPS,
            use_GATv2= USE_GATv2,
            use_Transformer=USE_TRANSFORMER,
            concat=CONCAT,
            use_propagator=USE_STANDARD_PROPAGATOR,
            num_layers=NUM_LAYERS,
            use_GRU= USE_GRU,
            
            device= DEVICE
        )    
    
    # set the number of seizure and not seizure data
    NUM_SEIZURE_DATA, NUM_NOT_SEIZURE_DATA = pos_neg_samples(train_dict)
    if do_train and (NUM_NOT_SEIZURE_DATA==0):
        raise ValueError(f"Training aborted, no data without seizure")
    if do_train and (NUM_SEIZURE_DATA==0):
        raise ValueError(f"Training aborted, no data with seizure")
    
    # start train or evaluation
    if do_train:
        LOGGER.info(f'Start training : {datetime.now().strftime("%d/%m/%Y at %H:%M:%S")}\n')
        optimizer= torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train(train_loader, val_loader, test_loader, model=model, optimizer=optimizer, num_epochs=num_epochs, verbose=verbose, show_epoch_progress=True)
    else:
        eval(test_loader, model, verbose=True, show_progress=True)

if __name__=='__main__':
    main()
