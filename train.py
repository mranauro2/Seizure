# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# IMPORT
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
from torch.utils.data import DataLoader, Subset

from utils.classes.Checkpoint_manager import CheckPoint
from utils.classes.Metric_manager import Metrics
from utils.classes.Metrics_classes import *
from utils.constant.constants_main import *
from utils.constant.constants_eeg import *
from data.scaler.Scaler import *
from data.utils import *
from train_utils import *

from data.dataloader.SeizureDataset import SeizureDataset
from data.dataloader.SeizureSampler import SeizureSampler

from model.SGLClassifier import SGLC_Classifier
from model.loss.loss_regularization import *
from model.loss.loss_classes import *

from train_args import parse_arguments

from datetime import datetime
from tqdm.auto import tqdm
import numpy as np
import logging
import random
import string
import os


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
            
            result, node_matrix, adj_matrix = model(x, adj)
            
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

def train(
        train_loader:DataLoader,
        val_loader:DataLoader,
        test_loader:DataLoader,
        
        model:SGLC_Classifier,
        prediction_loss:Loss,
        optimizer:torch.optim.Optimizer,
        
        num_epochs:int,
        
        verbose:bool=True,
        evaluation_verbose:bool=False,
        show_progress:bool=True,
        show_inner_progress:bool|str=False,
        
        *,
        
        folder_number:str=None
    ) -> bool:
    """
    Train the model and evaluate its performance. Use static parameters from :data:`utils.constant.constants_main` and :data:`utils.constant.constants_eeg`
    to implement some operations.\\
    Each time the model improves by :const:`PERCENTAGE_MARGIN`% both model and metrics will be saved. They will be saved also
    if for :const:`MAX_NUM_EPOCHS` the model has no saves and at the last iteration. The path where model will be saved 
    is defined by :const:`MODEL_PARTIAL_PATH`\\_ `epoch_number` with extention :const:`MODEL_EXTENTION` and the path where the metrics 
    will be saved will be `name_metric`\\_ `epoch_number` with extention :const:`METRICS_EXTENTION`.\\
    If the train is done using k-fold cross validation, the model path folder and the metrics path folder will have inside of them
    each folder for each fold. Inside each folder there will be the epochs.\\
    The metric used in the checkpoint is the first returned by :func:`train_or_eval`

    Args:
        train_loader (DataLoader):          Data on which train the model
        val_loader (DataLoader):            Data on which evaluate the model
        test_loader (DataLoader):           Data on which test the model (it's None when :const:`K_FOLD` is not None)
        
        model (SGLCModel_classification):   Model to train or to evaluate
        prediction_loss (Loss):             Prediction loss class to train the model
        optimizer (torch.optim.Optimizer):  Optimizer used for training
        
        num_epochs (int):                   Number of epochs for trainig
        
        verbose (bool):                     Useful for printing informations of the training phase
        evaluation_verbose (bool):          Useful for printing informations during the execution of the evaluation method
        show_progress (bool):               Show the progress bar with the number of epochs. The progress bar is removed when terminated
        show_inner_progress (bool|str):     Show the inner progress bar. Can show also only the first iteration. The progress bar is removed when terminated
        
        folder_number (str):                Parameter used when :const:`K_FOLD` is not None to found the correct folder
    
    Returns:
        interrupt (bool):                   If the execution is interrupted by checkpoint or stop file
    """
    # check variable
    possibilities = [True, False, "first"]
    if show_inner_progress not in possibilities:
        raise ValueError("The value '{}' in show_inner_progress does not exist. Choose between: '{}'".format(show_inner_progress, "', '".join([str(p) for p in possibilities])))
    
    # print the information about the stop file to interrupt the training
    if verbose:
        LOGGER.info("To stop the execution create the file '{}' in the current folder".format(os.path.basename(STOP_FILE)))
    delete_stop_file(STOP_FILE)
    
    # using a dataloader of one batch with one item to compute dynamically the number of metrics and the name of metrics
    single_item_dataset= Subset(val_loader.dataset, indices=[0])
    single_item_dataloader = DataLoader(single_item_dataset, batch_size=1, shuffle=False)
    dummy_metrics= eval(single_item_dataloader, model, prediction_loss, verbose=False, show_progress=False)
    metrics_name= [name for name,_ in dummy_metrics]
    num_metrics= len( metrics_name )
    
    # generate the metrics array (the test is present only if test_loader is not None)
    array_train= np.empty((START_EPOCH+num_epochs, num_metrics), dtype=np.float64)
    array_val=   np.empty((START_EPOCH+num_epochs, num_metrics), dtype=np.float64)
    if (test_loader is not None):
        array_test=  np.empty((START_EPOCH+num_epochs, num_metrics), dtype=np.float64)
    
    # change MODEL_PARTIAL_PATH if needed
    if K_FOLD:
        global MODEL_PARTIAL_PATH
        MODEL_PARTIAL_PATH= os.path.join(MODEL_SAVE_FOLDER, f"{os.path.basename(MODEL_SAVE_FOLDER)}_{folder_number}", MODEL_NAME)
    
    # fusion with old metrics if exist (the folder position and metrics retrieved can be different)
    if START_EPOCH != 0:
        if (TEST_PATIENT_IDS or VAL_PATIENT_IDS):
            epoch_folder = os.path.join(METRICS_SAVE_FOLDER, f"{EPOCH_FOLDER_NAME}_{START_EPOCH}")
        if K_FOLD:
            epoch_folder = os.path.join(METRICS_SAVE_FOLDER, f"{os.path.basename(METRICS_SAVE_FOLDER)}_{folder_number}", f"{EPOCH_FOLDER_NAME}_{START_EPOCH}")
            
        for index,name in enumerate(metrics_name):
            position = os.path.join(epoch_folder, f"{name}_{START_EPOCH}.{METRICS_EXTENTION}")
            if (test_loader is not None):
                array_train[0:START_EPOCH, index], array_val[0:START_EPOCH, index], array_test[0:START_EPOCH, index] = Metrics.load(position)
            else:
                array_train[0:START_EPOCH, index], array_val[0:START_EPOCH, index], _ = Metrics.load(position)
    
    # checkpoint init
    higher_is_better= not( dummy_metrics[0][0].lower().startswith("loss") )
    checkpoint_observer= CheckPoint(best_k=BEST_K_MODELS, each_spacing=MAX_NUM_EPOCHS, total_epochs=num_epochs, higher_is_better=higher_is_better, early_stop_patience=EARLY_STOP_PATIENCE, early_stop_start=START_USE_EARLY_STOP)
    checkpoint_observer.margin= PERCENTAGE_MARGIN
    
    if verbose:
        LOGGER.info(
            "CheckPoint will save {} values of '{}':".format('higher' if higher_is_better else 'lower', dummy_metrics[0][0]) +
            "\n" +
            "\tbest K model     : {} with margin of {:.2f}%".format(BEST_K_MODELS, 100*PERCENTAGE_MARGIN) +
            "\n" +
            "\teach epochs      : {}".format(MAX_NUM_EPOCHS) +
            "\n"+
            "\tearly stop       : {}".format(EARLY_STOP_PATIENCE) +
            "\n"+
            "\tearly stop start : {}".format(START_USE_EARLY_STOP)
        )
    
    for index in range(START_EPOCH):
        checkpoint_observer.update_saving(array_val[index, 0])
    
    # real training
    interrupt = False
    for epoch_num in (TqdmMinutesAndHours(range(num_epochs), desc="Progress", unit="epoch", leave=False) if show_progress else range(num_epochs)):
        show = show_inner_progress if isinstance(show_inner_progress, bool) else (epoch_num==0)
        
        metrics_train= train_epoch(train_loader, model, prediction_loss, optimizer, verbose=False, show_progress=show)
        metrics_val=   eval(val_loader,  model, prediction_loss, verbose=evaluation_verbose, show_progress=show)
        if (test_loader is not None):
            metrics_test=  eval(test_loader, model, prediction_loss, verbose=evaluation_verbose, show_progress=show)

        current_idx = START_EPOCH + epoch_num
        array_train[current_idx] = np.array([value for _,value in metrics_train])
        array_val[current_idx]   = np.array([value for _,value in metrics_val])
        if (test_loader is not None):
            array_test[current_idx]  = np.array([value for _,value in metrics_test])
        
        used_metric= metrics_val[0][1]
        
        # save the metrics at each iteration
        for index,name in enumerate(metrics_name):
            if (TEST_PATIENT_IDS or VAL_PATIENT_IDS):
                intermediate = ""
            if K_FOLD:
                intermediate = f"{os.path.basename(METRICS_SAVE_FOLDER)}_{folder_number}"
            
            position= os.path.join(METRICS_SAVE_FOLDER, intermediate, f"{EPOCH_FOLDER_NAME}_{epoch_num+START_EPOCH+1}", f"{name}_{epoch_num+START_EPOCH+1}.{METRICS_EXTENTION}")
            until_epoch = START_EPOCH+epoch_num+1
            Metrics.save(
                file_path    = position,
                train_metric = array_train[0:until_epoch, index],
                val_metric   = array_val[0:until_epoch, index],
                test_metric  = array_test[0:until_epoch, index] if (test_loader is not None) else None
            )
        
        # conditions to save the model
        saved_files= []
        if checkpoint_observer.check_saving(used_metric) or checkpoint_observer.check_early_stop() or check_stop_file(STOP_FILE):
            file_path= f"{MODEL_PARTIAL_PATH}_{epoch_num+START_EPOCH+1}.{MODEL_EXTENTION}"            
            saved_files.append(file_path)
            model.save(file_path)
            
            checkpoint_observer.update_saving(used_metric, saved_files)
        
        # delete obsolete model and check saved
        checkpoint_observer.delete_obsolete_checkpoints(auto_delete=True)
        
        # check the early stop
        if checkpoint_observer.check_early_stop():
            LOGGER.warning(f"Reached early stop at epoch {epoch_num+1}")
            interrupt = True
        
        # check stop file
        if check_stop_file(STOP_FILE):
            LOGGER.warning("Stop file '{}' found. Stopped at epoch {}".format(os.path.basename(STOP_FILE), epoch_num+1))
            delete_stop_file(STOP_FILE)
            interrupt = True
        
        if interrupt and (K_FOLD is None):
            break
    
    return interrupt

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MAIN
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def main_k_fold():
    """Main to evaluate the performance with k-fold cross validation"""
    # take input from command line and print some informations
    loss_type, input_dir, files_record, method, lambda_value, scaler_type, single_scaler, save_num, do_train, num_epochs, verbose, preprocess_dir = parse_arguments()
    dataset:SeizureDataset = generate_dataset(LOGGER, input_dir, files_record, method, lambda_value, scaler_type, preprocess_dir)
    
    # removing unwanted patients
    remaining_data = dataset.targets_dict()
    if (EXCEPT_DATA is not None):
        remaining_data, _ = split_patient_data_specific(dataset.targets_dict(), EXCEPT_DATA)
    
    # splitting data
    k_fold = k_fold_split_patient_data(remaining_data, val_remaining_patients=K_FOLD)
    
    # print on screen some informations    
    names= ["train", "validation"]
    ljust_value= len(max(names))
    
    for index,item in enumerate(k_fold):
        string = ""
        string+= "Fold number {}".format(index+1)
        for name,dictionary in zip(names,item):
            if (name=="train"):
                dictionary = augment_dataset_train(LOGGER, dataset, dictionary)
                samples_pos, samples_neg = pos_neg_samples(dictionary)
                dictionary = augment_dataset_train(None, dataset, dictionary, remove=True)
            else:
                samples_pos, samples_neg = pos_neg_samples(dictionary)
            string+= "\n\tUsing patient(s) for {} : '{}'".format(name.ljust(ljust_value), ", ".join(dictionary.keys()))
            string+= "\n\t\tTotal samples           : {:>{}}/{:,} (positive) & {:>{}}/{:,} (negative)".format(
                samples_pos,
                len(str(samples_pos+samples_neg)), samples_pos+samples_neg,
                
                samples_neg,
                len(str(samples_pos+samples_neg)), samples_pos+samples_neg
                )
            string+= "\n\t\tPositive ratio          : {:.3f}%".format(100 * samples_pos / (samples_pos+samples_neg))
        LOGGER.info(string+"\n")
    
    # global variables
    global DEVICE, START_EPOCH, NUM_NOT_SEIZURE_DATA, NUM_SEIZURE_DATA
    
    # print some informations
    DEVICE= 'cuda' if (torch.cuda.is_available() and USE_CUDA) else 'cpu'
    LOGGER.info(f"Using {DEVICE} device...")
    if (MIN_SAMPLER_PER_BATCH != 0):
        LOGGER.info("Loading dataset with at least ({}) samples for class in a batch of ({}) [min positive ratio {:.3f}%]...".format(MIN_SAMPLER_PER_BATCH, BATCH_SIZE, 100 * MIN_SAMPLER_PER_BATCH / BATCH_SIZE))
    if (scaler_type is not None):
        LOGGER.info(f"Loading scaler '{scaler_type.name}'...")
    _ = generate_loss(LOGGER, k_fold[0][0], do_train, loss_type, DEVICE)
    LOGGER.info("To stop the execution create the file '{}' in the current folder".format(os.path.basename(STOP_FILE)))
    
    # training at most MAX_NUM_EPOCHS iterative for each fold
    print_info = True
    epoch_interrupted = False
    total_training = ( num_epochs // MAX_NUM_EPOCHS ) + int( num_epochs % MAX_NUM_EPOCHS != 0 )    
    LOGGER.info(f'Start training : {datetime.now().strftime("%d/%m/%Y at %H:%M:%S")}\n')
    
    for index in TqdmMinutesAndHours(range(1, total_training+1), desc="Iteration"):
        current_epochs = MAX_NUM_EPOCHS if (index*MAX_NUM_EPOCHS <= num_epochs) else (num_epochs % MAX_NUM_EPOCHS)
        
        for train_dict,val_dict in TqdmMinutesAndHours(k_fold, desc="K-Fold", leave=False):
            train_dict = augment_dataset_train(None, dataset, train_dict)
            train_set= subsets_from_patient_splits(dataset, dataset.targets_index_map(), train_dict)
            val_set=   subsets_from_patient_splits(dataset, dataset.targets_index_map(), val_dict)
            
            # generating new scaler
            scaler = scaler_load_and_save(None, scaler_type, single_scaler, train_dict, train_set, device='cpu')
            if (scaler is not None):
                dataset.scaler= scaler
            
            # generate dataloaders
            train_sampler = None
            if (MIN_SAMPLER_PER_BATCH != 0):
                train_sampler= SeizureSampler(dataset.targets_list(), train_set.indices, batch_size=BATCH_SIZE, n_per_class=MIN_SAMPLER_PER_BATCH, seed=RANDOM_STATE)

            train_loader= DataLoader(dataset,  sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=False)
            val_loader=   DataLoader(val_set,  sampler=None,          batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=False)
            
            # load model if exists or create a new model            
            model= generate_model(dataset, DEVICE)
            
            folder_number = scaler_file_patient_ids(val_dict, separator="_")
            filename, num_epoch = get_model(os.path.join(MODEL_SAVE_FOLDER, f"{os.path.basename(MODEL_SAVE_FOLDER)}_{folder_number}"), specific_num=save_num)
            
            START_EPOCH= num_epoch
            if len(filename)==0 and (not do_train):
                raise ValueError(f"Evaluation stopped, model not present in the '{MODEL_SAVE_FOLDER}' folder")
            if len(filename)!=0:
                model= model.load(filename, device=DEVICE)
                
            loss, NUM_SEIZURE_DATA, NUM_NOT_SEIZURE_DATA = generate_loss(None, train_dict, do_train, loss_type, DEVICE)
            
            # start train or evaluation
            if do_train:
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)                
                interrupt = train(
                                    train_loader, val_loader, None,
                                    model=model, prediction_loss=loss, optimizer=optimizer, num_epochs=current_epochs,
                                    verbose=False, evaluation_verbose=False,
                                    show_progress=True, show_inner_progress="first" if print_info else False,
                                    folder_number=folder_number
                                )
                print_info = False
                epoch_interrupted = (epoch_interrupted and interrupt)
            else:
                raise NotImplementedError("Evaluation method with k-fold is not implemented yet")
            
            # remove the augmentation before pass to the next fold
            _ = augment_dataset_train(None, dataset, train_dict, remove=True)
            
        if epoch_interrupted:
            LOGGER.warning(f"Interruption detected after completing epoch for all folds")
            break

    LOGGER.info(f'Stop training  : {datetime.now().strftime("%d/%m/%Y at %H:%M:%S")}\n')
    
def main_test_set():
    """Main to evaluate the patients in the test"""
    # take input from command line and print some informations
    loss_type, input_dir, files_record, method, lambda_value, scaler, single_scaler, save_num, do_train, num_epochs, verbose, preprocess_dir = parse_arguments()
    dataset:SeizureDataset = generate_dataset(LOGGER, input_dir, files_record, method, lambda_value, scaler, preprocess_dir)
    
    # splitting data, augment train set and removing unwanted patients
    remaining_data = split_patient_data_specific(dataset.targets_dict(), EXCEPT_DATA)[0] if (EXCEPT_DATA is not None) else dataset.targets_dict()
    
    test_dict = None
    if (TEST_PATIENT_IDS is not None):
        remaining_data, test_dict = split_patient_data_specific(remaining_data, TEST_PATIENT_IDS)
    train_dict, val_dict = split_patient_data(remaining_data, split_ratio=PERCENTAGE_TRAINING_SPLIT) if (VAL_PATIENT_IDS is None) else split_patient_data_specific(remaining_data, VAL_PATIENT_IDS)
    
    train_dict = augment_dataset_train(LOGGER, dataset, train_dict)
    
    train_set= subsets_from_patient_splits(dataset, dataset.targets_index_map(), train_dict)
    val_set=   subsets_from_patient_splits(dataset, dataset.targets_index_map(), val_dict)
    test_set=  subsets_from_patient_splits(dataset, dataset.targets_index_map(), test_dict) if (test_dict is not None) else None
    
    # global variables
    global DEVICE, START_EPOCH, NUM_NOT_SEIZURE_DATA, NUM_SEIZURE_DATA
    
    # generating new scaler
    scaler = scaler_load_and_save(LOGGER, scaler, single_scaler, train_dict, train_set, device='cpu')
    if (scaler is not None):
        dataset.scaler= scaler
    
    # generate dataloaders
    train_sampler = None
    if (MIN_SAMPLER_PER_BATCH != 0):
        train_sampler= SeizureSampler(dataset.targets_list(), train_set.indices, batch_size=BATCH_SIZE, n_per_class=MIN_SAMPLER_PER_BATCH, seed=RANDOM_STATE)
        LOGGER.info("Loading dataset with at least ({}) samples for class in a batch of ({}) [min positive ratio {:.3f}%]...".format(MIN_SAMPLER_PER_BATCH, BATCH_SIZE, 100 * MIN_SAMPLER_PER_BATCH / BATCH_SIZE))

    train_loader= DataLoader(dataset,  sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader=   DataLoader(val_set,  sampler=None,          batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader=  DataLoader(test_set, sampler=None,          batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True) if (test_set is not None) else None
    
    # print on screen some informations
    names= ["test", "train", "validation"]
    dictionaries= [test_dict, train_dict, val_dict]
    ljust_value= len(max(names))
    
    for name,dictionary in zip(names,dictionaries):
        if (dictionary is None):
            continue
        string = "" 
        samples_pos, samples_neg = pos_neg_samples(dictionary)
        string+= "Using patient(s) for {} : '{}'".format(name.ljust(ljust_value), ", ".join(dictionary.keys()))
        string+= "\n\tTotal positive samples  : {:>{}}/{:,}".format(samples_pos, len(str(samples_pos+samples_neg)), samples_pos+samples_neg)
        string+= "\n\tTotal negative samples  : {:>{}}/{:,}".format(samples_neg, len(str(samples_pos+samples_neg)), samples_pos+samples_neg)
        string+= "\n\tPositive ratio          : {:.3f}%".format(100 * samples_pos / (samples_pos+samples_neg))
        LOGGER.info(string)
    
    # load model if exists or create a new model
    DEVICE= 'cuda' if (torch.cuda.is_available() and USE_CUDA) else 'cpu'
    LOGGER.info(f"Using {DEVICE} device...")
    LOGGER.info("Loading model...")
    model= generate_model(dataset, DEVICE)
    
    filename, num_epoch = get_model(MODEL_SAVE_FOLDER, specific_num=save_num)
    START_EPOCH= num_epoch
    if len(filename)==0 and (not do_train):
        raise ValueError(f"Evaluation stopped, model not present in the '{MODEL_SAVE_FOLDER}' folder")
    if len(filename)!=0:
        model= model.load(filename, device=DEVICE)
        LOGGER.info(f"Loaded '{os.path.basename(filename)}'...")
    
    loss, NUM_SEIZURE_DATA, NUM_NOT_SEIZURE_DATA = generate_loss(LOGGER, train_dict, do_train, loss_type, DEVICE)
    
    # start train or evaluation
    if do_train:
        LOGGER.info(f'Start training : {datetime.now().strftime("%d/%m/%Y at %H:%M:%S")}\n')
        optimizer= torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train(  
                train_loader, val_loader, test_loader,
                model=model, prediction_loss=loss, optimizer=optimizer, num_epochs=num_epochs,
                verbose=True, evaluation_verbose=verbose,
                show_progress=True, show_inner_progress="first"
            )
        LOGGER.info(f'Stop training  : {datetime.now().strftime("%d/%m/%Y at %H:%M:%S")}\n')
    else:
        eval(test_loader, model, prediction_loss=loss, verbose=True, show_progress=True)

if __name__=='__main__':
    if K_FOLD and (TEST_PATIENT_IDS or VAL_PATIENT_IDS):
        raise ValueError("The variables K_FOLD, TEST_PATIENT_IDS, VAL_PATIENT_IDS are not None. Only one can be not None")
    
    if K_FOLD:
        LOGGER.info("Execution with k-fold cross validation\n")
        main_k_fold()
        
    elif (TEST_PATIENT_IDS or VAL_PATIENT_IDS):
        LOGGER.info("Execution with patients as {}{}{} set\n".format(
            "test" if (TEST_PATIENT_IDS) else "",
            " and " if (TEST_PATIENT_IDS and VAL_PATIENT_IDS) else "",
            "validation" if (VAL_PATIENT_IDS) else ""
        ))
        main_test_set()
        
    else:
        raise ValueError("All choises are set to None")
