import argparse
from data.scaler.ScalerType import ScalerType
from data.dataloader.SeizureDatasetMethod import SeizureDatasetMethod

def parse_arguments() -> tuple[str, list[str], SeizureDatasetMethod, float|None, ScalerType|None, bool, int, bool, int, bool, str|None]:
    """
    Parses command-line arguments
    
    Returns:
        tuple:
            - input_dir (str):\\
                Path to the directory containing resampled files
            - files_record (list[str]):\\
                A list of one or more simple file names with line records as described in `data.dataloade.SeizureDataset` to process
            - method (SeizureDatasetMethod):\\
                How to compute the adjacency matrix
            - lambda_value (float):\\
                Maximum eigenvalue for scaling the Laplacian matrix
            - scaler (ScalerType):\\
                If use scaler and which one
            - single_scaler (bool):\\
                If True, compute single scaler values across all dimensions instead of compute scaler values per feature
            - save_num (int):\\
                Numeric identifier to search for model files inside the standard folder. Can be None
            - train (bool):\\
                Run in training mode
            - epochs (int):\\
                The number of epochs to train for. Required for training.
            - verbose (bool):\\
                Enable more detailed verbose logging and console output
            - preprocess_dir (str):\\
                Directory to the preprocess data
    """
    
    list_scaler_type= [scaler.name.lower() for scaler in ScalerType]
    list_seizuredataset_methods= [method.name.lower() for method in SeizureDatasetMethod]
    
    parser = argparse.ArgumentParser(
        description="Train or evaluate the `model.ASGPFmodel.SGLCModel_classification` on specified files. Other parameters are hardcoded inside constants files. The files are `utils.constants_eeg.py` and `utils.constants_main.py`",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Required Positional Arguments ---
    parser.add_argument('input_dir',    type=str,            help="Path to the directory containing resampled files")
    parser.add_argument('files_record', type=str, nargs='+', help="A list of one or more simple file names with line records as described in `data.dataloade.SeizureDataset` to process")

    # --- Optional Arguments ---
    parser.add_argument('--preprocess_dir', action='store_true',                                help="If the `input_dir` is the directory to the preprocess data and not to the resampled files")
    parser.add_argument('--method',   '-m', type=str,   default=list_seizuredataset_methods[0], help="How to compute the adjacency matrix: {}".format(", ".join(list_seizuredataset_methods)))
    parser.add_argument('--lambda_value',   type=float, default=None,                           help="Maximum eigenvalue for scaling the Laplacian matrix. If negative, computed automatically, if None compute only the Laplacian matrix")
    parser.add_argument('--scaler',   '-s', type=str,   default=None,                           help="If use or not a scaler. It can be: {}".format(", ".join(list_scaler_type)))
    parser.add_argument('--single_scaler',  action='store_true',                                help="If True, compute single scaler values across all dimensions instead of compute scaler values per feature")
    parser.add_argument('--save_num', '-n', type=int,   default=None,                           help="Numeric identifier to search for model files inside the standard folder (e.g., checkpoint or epoch number)")
    parser.add_argument('--train',    '-t', action='store_true',                                help="Run in training mode")
    parser.add_argument('--epochs',   '-e', type=int,   default=None,                           help="Number of epochs to train for. Required if --train is set.")
    parser.add_argument('--verbose',  '-v', action='store_true',                                help="Enable more detailed verbose logging and console output")

    args = parser.parse_args()
    
    if (args.train) and (args.epochs is None):
        parser.error("The argument --epochs is required when --train is set")

    possibilities= list_seizuredataset_methods
    if (args.method) not in possibilities:
        parser.error("The value '{}' in argument --method do not exists. Choose between: '{}'".format(args.method, "', '".join(possibilities)))

    possibilities= list_scaler_type
    possibilities.extend([None])
    if (args.scaler) not in possibilities:
        parser.error("The value '{}' in argument --scaler do not exists. Choose between: '{}'".format(str(args.scaler), "', '".join([str(item) for item in possibilities])))

    preprocess_dir= None
    if (args.preprocess_dir):
        preprocess_dir= args.input_dir
        args.input_dir= None
    
    method = SeizureDatasetMethod[args.method.upper()]
    scaler = None if (args.scaler is None) else ScalerType[args.scaler.upper()]
    
    return args.input_dir, args.files_record, method, args.lambda_value, scaler, args.single_scaler, args.save_num, args.train, args.epochs, args.verbose, preprocess_dir

if __name__=="__main__":
    args= parse_arguments()
    print("* "*30)
    for arg in args:
        print(arg)