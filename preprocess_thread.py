from tqdm.auto import tqdm
import numpy as np
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import tempfile
import shutil

from data.utils import compute_slice_matrix

# for printing aesthetic
import locale
locale.setlocale(locale.LC_ALL, '')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# STATIC VARIABLE
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# chb13_18.edf;921600;17;no seizure;0;0;0
# chb13_19.edf;921600;21;seizure;1;2077;2121
# chb13_55.edf;921600;21;seizure;2;458-2436;478-2454
IDX_FILE_NAME= 0
IDX_NUM_SAMPLES= 1
IDX_NUM_SEIZURES= 4
IDX_START_SEIZURE= 5
IDX_STOP_SEIZURE= 6

FREQUENCY_CHB_MIT= 256

# Thread-safe file writing
file_lock = Lock()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# UTILS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_static_info(file:str, data_files:list[str]) -> list[dict]:
    """
    Get the information from the CHB-MIT summary file
    
    Args:
        file (str):             File of the CHB-MIT summary
        data_files (list[str]): List of all present file. Some files can be present in the summary but not available
    Returns:
        info (list[dist]):      The info are stored with the keys: "file_name", "duration", "num_seizure", "start_seizure", "finish_seizure".\\
                                If there are no seizure the "start_seizure", "finish_seizure" are set to Inf
    """    
    with open(file, "r") as f:
        lines= f.readlines()
    
    # not consider head line)
    lines= lines[1:]
    
    data_files_basename= [os.path.basename(data_file) for data_file in data_files]
    dict_list= list()
    
    for line in lines:
        values= line.split(";")
        file_name= values[IDX_FILE_NAME].replace(".edf", ".npy")
        
        if file_name not in data_files_basename:
            continue
        
        duration= int(values[IDX_NUM_SAMPLES]) / FREQUENCY_CHB_MIT
        num_seizures= int(values[IDX_NUM_SEIZURES])
        
        if num_seizures==0:
            start_seizure= [float('inf')]
            finish_seizure= [float('inf')]
        else:
            start_seizure= [int(number) for number in values[IDX_START_SEIZURE].split("-")]
            finish_seizure= [int(number) for number in values[IDX_STOP_SEIZURE].split("-")]
        
        dict_list.append({
            "file_name": data_files[ data_files_basename.index(file_name) ],
            "duration": duration,
            "num_seizure": num_seizures,
            "start_seizure": start_seizure,
            "finish_seizure": finish_seizure
        })
    
    return dict_list

def is_overlap(interval_1:list[int, int], interval_2:list[int, int], full_overlap:bool=False) -> bool:
    """
    Verify if two interval are overlapping.
    
    Args:
        interval_1 (list[int, int]):    First interval
        interval_2 (list[int, int]):    Second interval
        full_overlap (bool):            If True the overlap must be complete. The first interval must be inside the second interval

    Returns:
        overlap (bool):                 True if the two intervals overlaps, otherwise False
    """
    a, b = interval_1
    c, d = interval_2
    
    if full_overlap:
        return ( (c<=a) and (b<=d) )
    else:
        are_disjoint= (b < c) or (d < a)
        return (not are_disjoint)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# PARALLEL PROCESSING FUNCTION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def process_segment(args):
    """Process a single segment - designed to run in parallel"""
    info_dict, index, seq_len, z_fill, save_dir, data_dir, use_fft, overwrite_segments, full_overlap, patient_id = args
    
    name = os.path.abspath(info_dict['file_name'])
    curr_start = index * seq_len
    curr_stop = curr_start + seq_len
    
    # Check for overlap with seizures
    overlap = False
    for start, stop in zip(info_dict["start_seizure"], info_dict["finish_seizure"]):
        overlap = overlap or is_overlap([curr_start, curr_stop], [start, stop], full_overlap=full_overlap)
    
    skipped = False
    
    # Save segment if save_dir is specified
    if save_dir is not None:
        basename = os.path.basename(name)
        name_without_ext, extention = os.path.splitext(basename)
        indexed_filename = f"{name_without_ext}_{str(index).zfill(z_fill)}{extention}"
        relative_parent_dir = os.path.dirname(os.path.relpath(name, os.path.abspath(data_dir)))
        output_dir = os.path.join(save_dir, relative_parent_dir, name_without_ext)
        new_path = os.path.abspath(os.path.join(output_dir, indexed_filename))
        
        if os.path.exists(new_path) and (not overwrite_segments):
            skipped = True
        else:
            eeg_clip = compute_slice_matrix(name, index, time_step_size=1, clip_len=seq_len, use_fft=use_fft)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            np.save(new_path, eeg_clip)
        
        output_string = f"{patient_id}, {new_path}, {int(overlap)}\n"
    else:
        output_string = f"{patient_id}, {name}, {str(index).zfill(z_fill)}, {int(overlap)}\n"
    
    return output_string, overlap, skipped

def process_batch_with_file(batch_tasks, temp_file_path, progress_callback=None):
    """Process a batch of tasks and write results to a temporary file"""
    results = []
    
    with open(temp_file_path, 'w') as f:
        for task in batch_tasks:
            output_string, overlap, skipped = process_segment(task)
            f.write(output_string)
            results.append((overlap, skipped))
            if progress_callback:
                progress_callback()
    
    return results

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MAIN PROCESSING FUNCTION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def main(summary_dir:str, data_dir:str, file_name:str, save_dir:str, seq_len:int, z_fill:int, check_file:bool=True, use_fft:bool=True, overwrite_segments:bool=False, full_overlap:bool=False, max_workers:int=4, verbose:bool=True) -> None:
    """
    Process CHB-MIT dataset and generate segmented window labels using parallel processing.
    
    Args:
        summary_dir (str):          Root directory containing CHB-MIT csv summary files
        data_dir (str):             Root directory containing all CHB-MIT available file
        save_dir (str):             Root directory where save the preprocessed data
        file_name (str):            Output file path where segmentation results will be saved
        seq_len (int):              Length of each segment window in seconds
        z_fill (int):               Number of digits for zero-padding segment indices
        check_file (bool):          If True stop the execution when the file already exists
        use_fft (bool):             Use the Fast Fourier Transform when obtain the slice
        overwrite_segments (bool):  If True, overwrite existing segment files
        full_overlap (bool):        Used in `is_overlap`: if True the overlap must be complete. The first interval must be inside the second interval
        max_workers (int):          Number of parallel workers (threads)
        verbose (bool):             Print information at the end of the execution
    """
    # stop the execution or delete the output file if it exists
    if os.path.exists(file_name):
        if check_file:
            raise FileExistsError(f"File <{file_name}> already exists, execution stopped")
        else:
            os.remove(file_name)
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    os.makedirs(save_dir, exist_ok=True) if (save_dir is not None) else None
            
    # find all summary files
    summary_dir = os.path.abspath(summary_dir)
    summary_files = []
    for path, _, files in os.walk(summary_dir):
        for name in files:
            if name.endswith(".csv"):
                summary_files.append(os.path.join(path, name))
    
    # find all data files
    data_files = []
    for path, _, files in os.walk(data_dir):
        for name in files:
            if name.endswith(".npy"):
                data_files.append(os.path.join(path, name))
    
    total_segments = 0
    seizure_segments = 0
    skipped_segments = 0
    
    # Collect all tasks to process
    tasks = []
    for summary in summary_files:
        info_list = get_static_info(summary, data_files)
        patient_id = os.path.splitext(os.path.basename(summary))[0]
        
        for info_dict in info_list:
            num_segments = int(info_dict['duration'] // seq_len)
            for index in range(num_segments):
                tasks.append((info_dict, index, seq_len, z_fill, save_dir, data_dir, use_fft, overwrite_segments, full_overlap, patient_id))

    # Create temporary directory for worker files
    temp_dir = tempfile.mkdtemp()
    temp_files = []
    
    try:
        # Split tasks into batches for each worker
        batch_size = max(1, len(tasks) // max_workers)
        batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
        
        # Create progress bar for individual segments
        with tqdm(total=len(tasks), desc="Processing segments", unit="segment") as pbar:
            def update_progress():
                pbar.update(1)
        
            # Process batches in parallel, each writing to its own temp file
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for i, batch in enumerate(batches):
                    temp_file = os.path.join(temp_dir, f"worker_{i}.txt")
                    temp_files.append(temp_file)
                    futures[executor.submit(process_batch_with_file, batch, temp_file, update_progress)] = i
                
                # Collect results as they complete
                for future in as_completed(futures):
                    results = future.result()
                    for overlap, skipped in results:
                        total_segments += 1
                        if overlap:
                            seizure_segments += 1
                        if skipped:
                            skipped_segments += 1
        
        # Merge all temporary files into the final output file
        with open(file_name, 'w') as outfile:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    with open(temp_file, 'r') as infile:
                        outfile.write(infile.read())
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    # # Process segments in parallel
    # with open(file_name, "a") as f:
    #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         # Submit all tasks
    #         futures = {executor.submit(process_segment, task): task for task in tasks}
            
    #         # Process results as they complete
    #         for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing segments"):
    #             output_string, overlap, skipped = future.result()
                
    #             # Thread-safe file writing
    #             with file_lock:
    #                 f.write(output_string)
                
    #             total_segments += 1
    #             if overlap:
    #                 seizure_segments += 1
    #             if skipped:
    #                 skipped_segments += 1
    
    if verbose: 
        z_fill_print = len(str("{0:n}".format(total_segments)))
        non_seizure_segments = "{0:n}".format(total_segments-seizure_segments).rjust(z_fill_print, " ")
        seizure_segments_str = "{0:n}".format(seizure_segments).rjust(z_fill_print, " ")
        total_segments_str = "{0:n}".format(total_segments)
        print("SEGMENTATION COMPLETE")
        print(f"Informations saved at <{file_name}>")
        if (save_dir is not None):
            print(f"Data saved at         <{save_dir}>")
            if (skipped_segments > 0):
                print(f"Skipped segments:     {skipped_segments:n} (already existed)")
        print(f"Seizure segments:     {seizure_segments_str}/{total_segments_str}")
        print(f"Non-seizure segments: {non_seizure_segments}/{total_segments_str}")
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MAIN
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment CHB-MIT dataset into fixed-length windows and label seizure activity", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("summary_dir",      type=str,                                 help="Root directory containing CHB-MIT csv summary files")
    parser.add_argument("data_dir",         type=str,                                 help="Root directory containing all CHB-MIT available file")
    parser.add_argument("file_name",        type=str,                                 help="Output file path for segmentation results")
    
    parser.add_argument("--save_dir",       type=str,            default=None,        help="Root directory where save the preprocessed data")
    parser.add_argument("--use_fft",        action='store_true', default=False,       help="Use the Fast Fourier Transform when obtain the slice from the file. Used only if --save_dir is not None")
    parser.add_argument("--overwrite",      action='store_true', default=False,       help="Overwrite existing segment files. If not set, existing files will be skipped")
    parser.add_argument("--partial_overlap",action='store_true', default=False,       help="If True the overlap must be complete. The interval of the segmented data must be completely inside the seizure event to be labeled as seizure")
    parser.add_argument("--seq_len",        type=int,            default=4,           help="Length of each segment window in seconds")
    parser.add_argument("--z_fill",         type=int,            default=3,           help="Number of digits for zero-padding segment indices")
    parser.add_argument("--delete",         action='store_true', default=False,       help="Delete the output file if already exists")
    parser.add_argument("--no_verbose",     action='store_true', default=False,       help="Disable printing information at the end of the execution")
    parser.add_argument("--workers",        type=int,            default=4,           help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Validate directories exist
    if not os.path.exists(args.summary_dir):
        raise ValueError(f"Root directory does not exist: {args.summary_dir}")
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Root directory does not exist: {args.data_dir}")
    
    # Run main processing
    main(
        summary_dir=args.summary_dir,
        data_dir=args.data_dir,
        file_name=args.file_name,
        save_dir=args.save_dir,
        seq_len=args.seq_len,
        z_fill=args.z_fill,
        check_file=not(args.delete),
        use_fft=args.use_fft,
        overwrite_segments=args.overwrite,
        full_overlap=not(args.partial_overlap),
        max_workers=args.workers,
        verbose=not(args.no_verbose)
    )