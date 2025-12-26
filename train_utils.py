from model.SGLClassifier import SGLC_Classifier
from model.loss.loss_classes import *
from torch.utils.data import Subset

from data.dataloader.SeizureDataset import SeizureDataset, SeizureDatasetMethod
from data.scaler.Scaler import *

from utils.constant.constants_main import *
from utils.constant.constants_eeg import *

from datetime import datetime, timedelta
from tqdm.auto import tqdm
from logging import Logger
import os
import re

class TqdmMinutesAndHours(tqdm):
    """
    Custom tqdm:
      - switches s/it → min/it when iteration time > 60 sec
      - switches min/it → hour/it when iteration time > 60 min
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
                if min_per_iter > 60:
                    hour_per_iter = min_per_iter / 60
                    d["rate_fmt"] = f"{hour_per_iter:0.2f}hour/{d['unit']}"

        # ---- ETA as a clock time ----
        d["last_update"] = datetime.now().strftime("%H:%M:%S")
        remaining = (self.total - self.n) / rate if rate and self.total else None
        if remaining is not None:
            finish = datetime.now() + timedelta(seconds=remaining)
            d["eta_clock"] = finish.strftime("%H:%M:%S")
        else:
            d["eta_clock"] = "--:--:--"
        
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
            - filename (str):   Full filepath (including directory path) of the matching model file. If no matching files exist returns an empty string
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

def scaler_load_and_save(logger:Logger|None, scaler:ScalerType|None, single_scaler:bool, train_dict:dict[str,list[int]], train_set:Subset, device:str) -> Scaler|None:
    """Load the scaler if `scaler` is not None and save it"""
    if (scaler is not None):
        if (logger is not None):
            logger.info(f"Loading scaler '{scaler.name}'...")
        scaler_name= "{}{}_{}.{}".format(scaler.name, '_single' if single_scaler else "", scaler_file_patient_ids(train_dict, separator="-"), MODEL_EXTENTION)
        scaler_path= os.path.join(SCALER_SAVE_FOLDER, scaler_name)
        scaler:Scaler= ConcreteScaler.create_scaler(scaler, device=device)
    
        if os.path.exists(scaler_path):
            scaler= scaler.load(scaler_path, device=device)
        else:
            scaler.fit(train_set, single_value=single_scaler, func_operation=func_operation, use_tqdm=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
            scaler.save(scaler_path)
    
    return scaler

def generate_model(dataset:SeizureDataset, device:str):
    """Generate the model using the constant in the constant files"""
    feature_matrix, _, _ = dataset[0]
    num_nodes= feature_matrix.size(1)
    input_dim= feature_matrix.size(2)
            
    model= SGLC_Classifier(
        num_classes         = NUM_CLASSES,
        
        num_cells           = NUM_CELLS,
        input_dim           = input_dim,
        num_nodes           = num_nodes,
        
        graph_skip_conn     = GRAPH_SKIP_CONN,
        use_GRU             = USE_GRU,
        hidden_per_step     = HIDDEN_PER_STEP,
        
        hidden_dim_GL       = HIDDEN_DIM_GL,
        attention_type      = ATTENTION_TYPE,
        num_GL_layers       = NUM_GL_LAYERS,
        num_GL_heads        = NUM_GL_HEADS,
        dropout_GL          = GL_DROPOUT,
        epsilon             = EPSILON,
        
        hidden_dim_GGNN     = HIDDEN_DIM_GGNN,
        num_steps           = NUM_STEPS,
        num_GGNN_layers     = NUM_GGNN_LAYERS,
        act_GGNN            = ACT_GGNN,
        use_GRU_in_GGNN     = USE_GRU_IN_GGNN,
        
        transformer_type    = TRANSFORMER_TYPE,
        num_transf_heads    = TRANSFORMER_NUM_HEADS,
        num_encoder_layers  = NUM_ENCODER_LAYERS,
        num_decoder_layers  = NUM_DECODER_LAYERS,
        positional_encoding = POSITIONAL_ENCODING,
        dim_feedforward     = DIM_FEEDFORWARD,
        dropout_transf      = TRANSFORMER_DROPOUT,
        act_transf          = TRANSFORMER_ACT,
        
        num_inputs          = NUM_INPUTS,
        
        use_sigmoid         = USE_SIGMOID,
        act                 = GL_ACT,
        v2                  = USE_GATv2,
        concat              = CONCAT,
        beta                = BETA,
        
        seed                = RANDOM_SEED,
        device              = device
    )
    
    return model

def generate_dataset(logger:Logger, input_dir:str, files_record:list[str], method:SeizureDatasetMethod, lambda_value:float|None, scaler:ScalerType|None, preprocess_dir:str|None):
    """Generate the dataset"""
    string_additional_info= additional_info(
        preprocessed_data=(preprocess_dir is not None),
        dataset_data=[
            ('method', method),
            ('lambda_value', lambda_value),
            ('scaler', scaler)
        ]
    )
    string= "{}{}".format("\n\t", "\n\t".join([item for item in string_additional_info.split("\n")]))
    logger.info(string)
    
    # load dataset
    dataset= SeizureDataset(
        input_dir       = input_dir,
        files_record    = files_record,
        time_step_size  = TIME_STEP_SIZE if (preprocess_dir is None) else None,
        max_seq_len     = MAX_SEQ_LEN    if (preprocess_dir is None) else None,
        use_fft         = USE_FFT        if (preprocess_dir is None) else None,
        preprocess_data = preprocess_dir,
        method          = method,
        top_k           = TOP_K,
        lambda_value    = lambda_value
    )
    pos_samples_before = sum(dataset.targets_list())
    neg_samples_before = len(dataset.targets_list()) - pos_samples_before
    
    dataset.apply_augmentations(AUGMENTATIONS)
    pos_samples_after = sum(dataset.targets_list())
    neg_samples_after = len(dataset.targets_list()) - pos_samples_after
    
    if (pos_samples_before != pos_samples_after):
        logger.info("Positive samples are augmented from {:,} to {:,} [{:+.2f}%]".format(
            pos_samples_before,
            pos_samples_after,
            100*(pos_samples_after - pos_samples_before) / pos_samples_before
        ))
    if (neg_samples_before != neg_samples_after):
        logger.info("Negative samples are augmented from {:,} to {:,} [{:+.2f}%]".format(
            neg_samples_before,
            neg_samples_after,
            100*(neg_samples_after - neg_samples_before) / neg_samples_before
        ))
    
    return dataset

def generate_loss(logger:Logger|None, train_dict:dict[str, list[int]], do_train:bool, loss_type:LossType, device:str) -> tuple[Loss, int, int]:
    """
    Modify NUM_SEIZURE_DATA, NUM_NOT_SEIZURE_DATA and generate the loss
        :returns tuple(Loss, int, int): Loss function, num_seizure_data, num_not_seizure_data
    """
    NUM_SEIZURE_DATA = 0
    NUM_NOT_SEIZURE_DATA = 0
    
    # set the number of seizure and not seizure data
    NUM_SEIZURE_DATA, NUM_NOT_SEIZURE_DATA = pos_neg_samples(train_dict)
    if do_train and (NUM_NOT_SEIZURE_DATA==0):
        raise ValueError(f"Training aborted, no data without seizure")
    if do_train and (NUM_SEIZURE_DATA==0):
        raise ValueError(f"Training aborted, no data with seizure")
    
    # create loss to use
    weight = torch.Tensor([1.0, NUM_NOT_SEIZURE_DATA/NUM_SEIZURE_DATA]).to(device=device) if USE_WEIGHT else None
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
    if (logger is not None):
        logger.info("Using loss type '{}'{}".format(loss_type.name, '' if loss_params_str=="" else f' with parameters :\n{loss_params_str}'))
    
    return loss, NUM_SEIZURE_DATA, NUM_NOT_SEIZURE_DATA

def pos_neg_samples(dictionary:dict[str, list[int]]):
    """Returns the positive and negative samples for the dictionary passed"""
    l= [label for value_list in dictionary.values() for label in value_list]
    return sum(l), len(l)-sum(l)

def check_stop_file(stop_file:str):
    """Check if the stop file exists"""
    return os.path.exists(stop_file)

def delete_stop_file(stop_file:str):
    """Delete the stop file if exists"""
    if check_stop_file(stop_file):
        os.remove(stop_file)

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
        ("USE_GRU", USE_GRU),
        ("HIDDEN_PER_STEP", HIDDEN_PER_STEP)
    ]
    model_str = "Model info:\n{}".format(dict_to_str(model_tuple))
    
    # GRAPH LEARNER
    GL_tuple = [
        ("HIDDEN_DIM_GL", HIDDEN_DIM_GL),
        ("ATTENTION_TYPE", ATTENTION_TYPE),
        ("NUM_GL_LAYERS", NUM_GL_LAYERS),
        ("NUM_GL_HEADS", NUM_GL_HEADS),
        ("GL_ACT", GL_ACT),
        ("GL_DROPOUT", GL_DROPOUT),
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
        ("NUM_GGNN_LAYERS", NUM_GGNN_LAYERS),
        ("ACT_GGNN", ACT_GGNN),
        ("USE_GRU_IN_GGNN", USE_GRU_IN_GGNN)
    ])
    GGNN_str = "GGNN info:\n{}".format(dict_to_str(GGNN_tuple))
    
    transformer_tuple = [
        ("TRANSFORMER_TYPE", TRANSFORMER_TYPE),
        ("TRANSFORMER_NUM_HEADS", TRANSFORMER_NUM_HEADS),
        ("NUM_ENCODER_LAYERS", NUM_ENCODER_LAYERS),
        ("NUM_DECODER_LAYERS", NUM_DECODER_LAYERS),
        ("DIM_FEEDFORWARD", DIM_FEEDFORWARD),
        ("TRANSFORMER_DROPOUT", TRANSFORMER_DROPOUT),
        ("TRANSFORMER_ACT", TRANSFORMER_ACT),
        ("POSITIONAL_ENCODING", POSITIONAL_ENCODING),
        ("NUM_INPUTS", NUM_INPUTS)
    ]
    transformer_str = "Tansformer info:\n{}".format(dict_to_str(transformer_tuple))
    
    # LOSSES
    loss_tuple = [
        ("LEARNING_RATE", LEARNING_RATE),
        ("DAMP_SMOOTH", DAMP_SMOOTH),
        ("DAMP_DEGREE", DAMP_DEGREE),
        ("DAMP_SPARSITY", DAMP_SPARSITY)
    ]
    loss_str = "Losses info:\n{}".format(dict_to_str(loss_tuple))
    
    total_str = "\n".join([dataset_str, model_str, GL_str, GGNN_str, transformer_str, loss_str])
        
    return total_str
