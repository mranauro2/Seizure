import argparse

def parse_arguments() -> tuple[str|None, list[str], str, str|None, bool, int, bool, int, bool, str|None]:
    """
    Parses command-line arguments
    
    Returns:
        tuple:
            - input_dir (str):\\
                Path to the directory containing resampled files
            - files_record (list[str]):\\
                A list of one or more simple file names with line records as described in `data.dataloade.SeizureDataset` to process
            - method (str):\\
                How to compute the adjacency matrix. The values can be `cross` or `plv`
            - scaler (str):\\
                If use scaler and which one. The values can be `z-score` or `min-max` (if it is not None)
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
    
    parser = argparse.ArgumentParser(
        description="Train or evaluate the `model.ASGPFmodel.SGLCModel_classification` on specified files. Other parameters are hardcoded inside constants files. The files are `utils.constants_eeg.py` and `utils.constants_main.py`",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Required Positional Arguments ---
    parser.add_argument('input_dir',    type=str,            help="Path to the directory containing resampled files")
    parser.add_argument('files_record', type=str, nargs='+', help="A list of one or more simple file names with line records as described in `data.dataloade.SeizureDataset` to process")

    # --- Optional Arguments ---
    parser.add_argument('--preprocess_dir', action='store_true',    help="If the `input_dir` is the directory to the preprocess data and not to the resampled files")
    parser.add_argument('--method',   '-m', type=str, default='plv',help="How to compute the adjacency matrix: cross (for the scaled Laplacian matrix of the normalized cross-correlation), plv (for the Phase Locking Value)")
    parser.add_argument('--scaler',   '-s', type=str, default=None, help="If use or not a scaler. It can be: z-score, min-max")
    parser.add_argument('--single_scaler',  action='store_true',    help="If True, compute single scaler values across all dimensions instead of compute scaler values per feature")
    parser.add_argument('--save_num', '-n', type=int, default=None, help="Numeric identifier to search for model files inside the standard folder (e.g., checkpoint or epoch number)")
    parser.add_argument('--train',    '-t', action='store_true',    help="Run in training mode")
    parser.add_argument('--epochs',   '-e', type=int, default=None, help="Number of epochs to train for. Required if --train is set.")
    parser.add_argument('--verbose',  '-v', action='store_true',    help="Enable more detailed verbose logging and console output")

    args = parser.parse_args()
    
    if (args.train) and (args.epochs is None):
        parser.error("The argument --epochs is required when --train is set")

    possibilities= ["cross", "plv"]
    if (args.method) not in possibilities:
        parser.error("The value '{}' in argument --method do not exists. Choose between: '{}'".format(args.method, "', '".join(possibilities)))

    possibilities= [None, "z-score", "min-max"]
    if (args.scaler) not in possibilities:
        parser.error("The value '{}' in argument --scaler do not exists. Choose between: '{}'".format(str(args.scaler), "', '".join([str(item) for item in possibilities])))

    preprocess_dir= None
    if (args.preprocess_dir):
        preprocess_dir= args.input_dir
        args.input_dir= None
    
    return args.input_dir, args.files_record, args.method, args.scaler, args.single_scaler, args.save_num, args.train, args.epochs, args.verbose, preprocess_dir

if __name__=="__main__":
    args= parse_arguments()
    print("* "*30)
    for arg in args:
        print(arg)