import warnings
import itertools
import numpy as np

from scipy.fftpack import fft
from scipy.sparse import linalg
from scipy.sparse.csgraph import laplacian
from mne_features.bivariate import compute_phase_lock_val

from collections import defaultdict
from torch.utils.data import Dataset, Subset

FREQUENCY_CHB_MIT= 256

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# DATASET UTILS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def normalize_laplacian_spectrum(adj_mx:np.ndarray, lambda_value:float=2) -> np.ndarray:
    """
    Compute Laplacian matrix for graph convolutional networks. It can also compute the scaled Laplacian matrix\\
    The scaled Laplacian is defined as:\\
    `(2 / lambda_max) * L - I`\\
    where:
    - `L` is the normalized Laplacian
    - `I` is the identity matrix
    
    Args:
        adj_mx (np.ndarray):        Adjacency matrix with shape (num_nodes, num_nodes)
        lambda_value (float):       Maximum eigenvalue for scaling. If negative, computed automatically, if None compute only the Laplacian matrix
    
    Returns:
        laplacian (np.ndarray):     Scaled Laplacian matrix with shape (num_nodes, num_nodes)
    """
    is_symmetric= np.allclose(adj_mx, adj_mx.T, rtol=1e-10, atol=1e-12)
    L= laplacian(adj_mx, normed=True, symmetrized=(not is_symmetric))
    
    if (lambda_value is None):
        return L
    
    lambda_value= lambda_value if (lambda_value >= 0) else linalg.eigsh(L, 1, which='LM', return_eigenvectors=False)[0]
    lambda_value= max(lambda_value, 1e-8) # for numerical stability

    I = np.eye(L.shape[0])
    L = (2 / lambda_value) * L - I
    
    return L

def keep_topk(adj_mat:np.ndarray, top_k:int, directed:bool=True) -> np.ndarray:
    """
    Helper function to sparsen the adjacency matrix by keeping top-k neighbors for each node.
    
    Args:
        adj_mat (np.ndarray):   Adjacency matrix with size (num_nodes, num_nodes)
        top_k (int):            Number of higher value neighbors for each node to maintain
        directed (bool):        If the graph is direct or undirect
    Returns:
        adj_mat (np.ndarray):   Sparse adjacency matrix with size (num_nodes, num_nodes)
    """
    num_nodes = adj_mat.shape[0]
    
    # Set values that are not of top-k neighbors to 0
    adj_mat_noSelfEdge = adj_mat.copy()
    np.fill_diagonal(adj_mat_noSelfEdge, 0.0)

    # Find top-k indices for each row
    top_k_idx = np.argsort(-adj_mat_noSelfEdge, axis=-1)[:, :top_k]
    
    # Create mask for the top_k indeces
    mask = np.zeros_like(adj_mat, dtype=bool)
    row_indices = np.arange(num_nodes).reshape((num_nodes, 1))
    mask[row_indices, top_k_idx] = True
    np.fill_diagonal(mask, True)
    
    # If the graph is undirect the mask must be symmetric
    mask= mask if directed else np.logical_or(mask, mask.T)
    
    return (mask * adj_mat)

def cross_correlation(eeg_clip:np.ndarray, top_k:int=None) -> np.ndarray:
    """
    Compute adjacency matrix using normalized cross-correlation between EEG channels
    
    Args:
        eeg_clip (np.ndarray):  EEG signal with shape (seq_len, num_nodes, input_dim)
        top_k (int):            Number of strongest connections to maintain per node. If None, all connections are maintained
        
    Returns:
        adj_mat (np.ndarray):   Absolute normalized cross-correlation adjacency matrix with shape (num_nodes, num_nodes)
    
    Notes:
    ------
        The resulting adjacency matrix is symmetric with self-connections set to 1. Cross-correlations are computed between flattened time-feature dimensions
    """
    num_nodes = eeg_clip.shape[1]
    adj_mat = np.eye(num_nodes, num_nodes, dtype=np.float32)  # diagonal is 1

    # reshape from (seq_len, num_nodes, input_dim) to (num_nodes, seq_len*input_dim)
    eeg_flat = np.transpose(eeg_clip, (1, 0, 2)).reshape((num_nodes, -1))

    # Pre-normalize all signals, handling zero-norm cases
    signals_norm = eeg_flat / np.linalg.norm(eeg_flat, axis=1, keepdims=True)
    signals_norm = np.where(np.isnan(signals_norm), 0, signals_norm)
    
    # Compute cross-correlation between pre-normalized signals ensuring diagonal self-correlation equal to 1
    adj_mat = np.abs( np.dot(signals_norm, signals_norm.T) )
    np.fill_diagonal(adj_mat, 1.0)

    if top_k is not None:
        adj_mat = keep_topk(adj_mat, top_k=top_k, directed=True)

    return adj_mat

def compute_FFT(signals:np.ndarray, n:int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the FFT of the signal
    
    Args:
        signals (np.ndarray):   Signals with size (num_channels, num_data_points)
        n (int):                Length of positive frequency terms of fourier transform
    Returns:
        tuple (np.ndarray, np.ndarray): 
            - Log amplitude of FFT of signals with size (num_channels, num_data_points//2)
            - Phase spectrum of FFT of signals with size (num_channels, num_data_points//2)
    """
    # fourier transform
    fourier_signal = fft(signals, n=n, axis=-1)  # FFT on the last dimension

    # only take the positive freq part
    idx_pos = n // 2
    fourier_signal = fourier_signal[:, :idx_pos]
    amp = np.abs(fourier_signal)
    amp[amp == 0.0] = 1e-8  # avoid log of 0
    
    FT = np.log(amp)
    P = np.angle(fourier_signal)

    return FT, P

def compute_slice_matrix(file_name:str, clip_idx:int, time_step_size:int=1, clip_len:int=60, use_fft:bool=False) -> np.ndarray:
    """
    Extract and process an EEG clip from an HDF5 file.
    The function extracts a clip of specified length from resampled EEG data and optionally applies FFT processing to generate time-frequency representations.
    
    Args:
        file_name (str):        Path to *.npy file containing the signal
        clip_idx (int):         Index of the clip to extract (0-based). Maximum value depends on signal length and clip duration
        time_step_size (int):   Duration of each time step in seconds for FFT analysis
        clip_len (int):         Total duration of the EEG clip in seconds
        use_fft (bool):         If True, apply FFT to generate time-frequency representation
    Returns:
        eeg_clip (np.ndarray):  EEG clip with shape:
            - Without FFT: (clip_len, num_channels, FREQUENCY_CHB_MIT)
            - With FFT: (clip_len, num_channels, FREQUENCY_CHB_MIT//2)
    """
    signal_array:np.ndarray= np.load(file_name)

    # calculate physical dimensions
    physical_clip_len = FREQUENCY_CHB_MIT * clip_len
    physical_time_step_size = FREQUENCY_CHB_MIT * time_step_size
    
    start_window = clip_idx * physical_clip_len
    end_window = start_window + physical_clip_len
    
    # extract clipped signal (num_channels, physical_clip_len)
    clipped_signal = signal_array[:, start_window:end_window]
    
    # create empty clip of size (clip_len, num_channels, feature_dim)
    feature_dim= FREQUENCY_CHB_MIT//2 if use_fft else FREQUENCY_CHB_MIT
    eeg_clip= np.empty((clip_len, signal_array.shape[0], feature_dim))
    
    # if not use the FFT then the output has only different shape and different order of the axis
    # reshape from (num_signal, clip_len*feature_dim) to (clip_len, num_signal, feature_dim)
    if not use_fft:
        eeg_clip= clipped_signal.reshape(signal_array.shape[0], clip_len, feature_dim).transpose((1, 0, 2))

    # if use the FFT then is necessary to compute the FFT for each time step
    else:
        for t in range(clip_len):
            start_time_step = t*physical_time_step_size
            end_time_step = start_time_step + physical_time_step_size
            
            eeg_clip[t], _ = compute_FFT(signals=clipped_signal[:, start_time_step:end_time_step], n=physical_time_step_size)

    return eeg_clip

def compute_plv_matrix(graph: np.ndarray) -> np.ndarray:
    """Compute connectivity matrix via usage of PLV from MNE implementation.
    Args:
        graph: (np.ndarray) Single graph with shape [nodes,features] where features represent consecutive time samples and nodes represent electrodes in EEG.
        
    Returns:
        plv_matrix: (np.ndarray) PLV matrix of the input graph.
        
    Notes:
    -----
        See https://github.com/szmazurek/sano_eeg/blob/main/src/utils/utils.py#L695 for more detail
    """
    plv_conn_vector = compute_phase_lock_val(graph)

    n = int(np.sqrt(2 * len(plv_conn_vector))) + 1

    # Reshape the flattened array into a square matrix
    upper_triangular = np.zeros((n, n))
    upper_triangular[np.triu_indices(n, k=1)] = plv_conn_vector

    # Create an empty matrix for the complete symmetric matrix
    symmetric_matrix = np.zeros((n, n))

    # Fill the upper triangular part (including the diagonal)
    symmetric_matrix[np.triu_indices(n)] = upper_triangular[np.triu_indices(n)]

    # Fill the lower triangular part by mirroring the upper triangular
    plv_matrix = ( symmetric_matrix + symmetric_matrix.T - np.diag(np.diag(symmetric_matrix)) )

    # Add 1 to the diagonal elements
    np.fill_diagonal(plv_matrix, 1)
    return plv_matrix

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# DATASET SPLIT UTILS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def split_patient_data(
        patient_data: dict[str, list[int]],
        split_ratio: float = 0.8,
        
        val_remaining_patients: int = None,
        except_data: list[str] = None,
        
        size_tolerance: float = 1e-4,
        combinations_needed: int = 10,
        max_combinations: int = 100_000,
        tolerance_growth_factor: float = 2,
        
        return_tolerance: bool = False,
        verbose: bool = False
    ) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """
    Split patient data in train and validation to maintain positive/negative ratio using optimized search
    
    Args:
        patient_data (dict[str,list[int]]): Dictionary with patient_id as key and list of labels of integers as value
        split_ratio (float):                Desired proportion of *total samples* that should appear in the first set. Must be between 0 and 1
        
        val_remaining_patients (int):       If specified, the validation set will contain exactly this many patients, overriding the `split_ratio` parameter
        except_data (list[str]):            Key as patient_id which are not allowed to be in the validation set
        
        size_tolerance (float):             Allowed deviation from the exact target size
        combinations_needed (int):          Combinations to found before checking for the best one
        max_combinations (int):             Maximum number of combinations to check before stopping (if `combinations_needed` combinations are not found before)
        tolerance_growth_factor (float):    If no valid combination is found, expand size_tolerance by this factor and retry instead of raising an error
        
        return_tolerance (bool):            If True, return (set1, set2, final_tolerance), otherwise return only the sets
        verbose (bool):                     Whether to print detailed progress information
    
    Returns:
        tuple(dict(str, list(int)), dict(str, list(int))):  Each set has the same structure as `patient_data`
        
    Raises:
        ValueError: If no combinations are found with `size_tolerance`
    """
    # Input validation
    if (combinations_needed <= 0):
        raise ValueError("combinations_needed must be positive.")
    if (max_combinations <= 0):
        raise ValueError("max_combinations must be positive.")
    if not (0 < size_tolerance < 1):
        raise ValueError("size_tolerance must be between 0 and 1.")
    if not (0 < split_ratio < 1):
        raise ValueError("split_ratio must be between 0 and 1.")
    if (tolerance_growth_factor <= 0):
        raise ValueError("tolerance_growth_factor must be positive.")
    
    # Validate target_remaining_patients
    total_patients = len(patient_data)
    if (val_remaining_patients is not None):
        if (val_remaining_patients >= total_patients):
            msg = f"target_remaining_patients ({val_remaining_patients}) must be less than the total number of patients ({total_patients})."
            raise ValueError(msg)
        if val_remaining_patients <= 0:
            raise ValueError("target_remaining_patients must be positive.")
    
    # Check if except_data items exist in patient_data
    if (except_data is not None):
        missing_patients = [pid for pid in except_data if pid not in patient_data.keys()]
        msg = f"The following patient IDs in 'except_data' were not found in 'patient_data': {', '.join(missing_patients)}"
        if missing_patients:
            warnings.warn(msg)    
        except_data = [pid for pid in except_data if pid in patient_data.keys()]
    
    # Precompute basic statistics
    subjects = []
    for patient_id, labels in patient_data.items():
        subjects.append({
            'id': patient_id,
            'total': len(labels),
            'positives': sum(labels)
        })
    
    total_pos = sum(s['positives'] for s in subjects)
    total_samples = sum(s['total'] for s in subjects)
    global_ratio = total_pos / total_samples if total_samples > 0 else 0
    
    target_sample_count = split_ratio * total_samples
    allowed_abs_deviation = size_tolerance * total_samples
    
    use_patient_count_mode = (val_remaining_patients is not None)
    
    if verbose:
        print(f"=== Split Search Start ===")
        print(f"Total patients:        {len(subjects)}")
        print(f"Total samples:         {total_samples}")
        print(f"Total positives:       {total_pos}")
        print(f"Global ratio:          {global_ratio:.6f}")
        if use_patient_count_mode:
            print(f"Target remaining:      {val_remaining_patients} patients in validation set")
        else:
            print(f"Target samples:        {target_sample_count:.1f} ± {allowed_abs_deviation:.1f} [{100 * size_tolerance:.4f}%]")
        if (except_data is not None):
            print(f"Required in train:     {', '.join(except_data)}")

    # Sort by sample count (large subjects first --> better pruning)
    subjects_sorted = sorted(subjects, key=lambda x: x['total'], reverse=True)
    
    # Start from target length and search nearby sizes
    num_subjects = len(subjects)
    target_length = int(num_subjects * split_ratio)
    search_order = []
    
    # Create search order: target_length, target_length+1, target_length-1, etc.
    for i in range(num_subjects):
        up = target_length + i
        down = target_length - i

        if (0 <= up <= num_subjects):
            search_order.append(up)
        if (0 <= down <= num_subjects) and (down not in search_order):
            search_order.append(down)
    
    # Only search for exact size (override search_order)
    if use_patient_count_mode:
        search_order = [num_subjects - val_remaining_patients]
    
    valid_candidates = []
    tested = 0
    
    # Combination search
    for subset_size in search_order:
        for combo in itertools.combinations(subjects_sorted, subset_size):
            tested += 1
            
            # Early stopping if we've checked too many
            if tested > max_combinations:
                break
            
            total_c = sum(s["total"] for s in combo)
            
            # sample count too far from target
            if not(use_patient_count_mode) and abs(total_c - target_sample_count) > allowed_abs_deviation:
                continue
            # key not allowed in the second set --> must be in the first set
            if (except_data is not None) and (not(all(expection in list(s['id'] for s in combo) for expection in except_data))):
                    continue
            
            pos_c = sum(s["positives"] for s in combo)
            ratio = pos_c / total_c if total_c > 0 else 0

            valid_candidates.append({
                "ids": [s["id"] for s in combo],
                "ratio_diff": abs(ratio - global_ratio)
            })
        
        # If we found valid combinations and checked enough, we can stop
        if (tested > max_combinations) or (len(valid_candidates) >= combinations_needed):
            break
    
    if verbose:
        print(f"Checked {tested:,} combinations")
        print(f"Valid candidates: {len(valid_candidates)}")
        if valid_candidates:
            for index,candidate in enumerate(valid_candidates[:3], 1):
                aux = {pid: patient_data[pid] for pid in patient_data.keys() if pid not in set(candidate["ids"])}
                print(f"\tCandidate {index} - validation patients: {', '.join(aux.keys())}")
        
    # If no valid combination found, expand tolerance and retry
    if not valid_candidates:
        if tolerance_growth_factor > 1:
            new_tol = size_tolerance * tolerance_growth_factor
            if verbose:
                print(f"No valid combination found. Expanding tolerance to {new_tol:.6f} and retrying...")
            return split_patient_data(
                patient_data            = patient_data,
                split_ratio             = split_ratio,
                except_data             = except_data,
                val_remaining_patients  = val_remaining_patients,
                size_tolerance          = new_tol,
                combinations_needed     = combinations_needed,
                max_combinations        = max_combinations,
                tolerance_growth_factor = tolerance_growth_factor,
                return_tolerance        = return_tolerance,
                verbose                 = verbose
            )
        raise ValueError("No valid combination found within tolerance.")
    
    # Select best candidate
    valid_candidates.sort(key=lambda x: x['ratio_diff'])
    best_ids = set(valid_candidates[0]["ids"])
    
    set1 = {pid: patient_data[pid] for pid in best_ids}
    set2 = {pid: patient_data[pid] for pid in patient_data.keys() if pid not in best_ids}
    
    if verbose:
        set_1_total=    len([value for value_list in set1.values() for value in value_list])
        set_1_positive= sum([value for value_list in set1.values() for value in value_list])
        
        set_2_total=    len([value for value_list in set2.values() for value in value_list])
        set_2_positive= sum([value for value_list in set2.values() for value in value_list])
        
        print(f"=== Split Search Results ===")
        print("Train set positive ratio {:.5f} [{:.5f} expected] and length {:.2f} % [{} % expected]".format(
            set_1_positive/set_1_total,
            global_ratio,
            100 * set_1_total / (set_1_total+set_2_total), 
            "???" if use_patient_count_mode else f"{(100 * split_ratio):.2f}"
        ))
        print("Valid set positive ratio {:.5f} [{:.5f} expected] and length {:.2f} % [{} % expected]".format(
            set_2_positive/set_2_total,
            global_ratio,
            100 * set_2_total / (set_1_total+set_2_total),
            "???" if use_patient_count_mode else f"{(100 * (1-split_ratio)):.2f}"
        ))
        
    if return_tolerance:
        return set1, set2, size_tolerance
    return set1, set2

def k_fold_split_patient_data(
        patient_data: dict[str, list[int]],
        
        val_remaining_patients: int,
        except_data: list[str] = None,
        
        combinations_needed: int = 10,
        max_combinations: int = 100_000,
        
        verbose: bool = False,
        verbose_inner:bool=False
    ) -> list[tuple[dict[str, list[int]], dict[str, list[int]]]]:
    """
    Use `split_patient_data` to generate K-fold train and validation maintaining positive/negative ratio using optimized search.
    The last fold could have a inferior number of subjects
    
    Args:
        patient_data (dict[str,list[int]]): Dictionary with patient_id as key and list of labels of integers as value
        
        val_remaining_patients (int):       The validation set will contain exactly this many patients
        except_data (list[str]):            Key as patient_id which are not allowed to be in the second set
        
        combinations_needed (int):          Combinations to found before checking for the best one
        max_combinations (int):             Maximum number of combinations to check before stopping (if `combinations_needed` combinations are not found before)
        
        verbose (bool):                     Whether to print detailed progress information
        verbose_inner (bool):               Whether to print detailed progress information of the `split_patient_data` function
    
    Returns:
        list(dict(str, list(int)), dict(str, list(int))):  Each dict has the same structure as `patient_data`
    """
    if (except_data is not None):
        missing_patients = [pid for pid in except_data if pid not in patient_data.keys()]
        msg = f"The following patient IDs in 'except_data' were not found in 'patient_data': {', '.join(missing_patients)}"
        if missing_patients:
            warnings.warn(msg)    
        except_data = [pid for pid in except_data if pid in patient_data.keys()]
    
    total_subjects = len(patient_data) - len(except_data)
    total_folds = ( total_subjects // val_remaining_patients ) + int( total_subjects % val_remaining_patients != 0 )
    except_data = except_data if (except_data is not None) else []
    k_fold = []
    
    for index in range(1, total_folds+1):
        fold_size = val_remaining_patients if (index*val_remaining_patients <= total_subjects) else (total_subjects % val_remaining_patients)
        
        train_dict, val_dict = split_patient_data(
                                    patient_data           = patient_data,
                                    val_remaining_patients = fold_size,
                                    except_data            = except_data,
                                    combinations_needed    = combinations_needed,
                                    max_combinations       = max_combinations,
                                    verbose                = verbose_inner
                                )
        k_fold.append((train_dict, val_dict))
        except_data.extend(val_dict.keys())
        if verbose:
            print("Fold {} has patitents : {}".format(index, ", ".join(val_dict.keys())))
    
    return k_fold

def split_patient_data_specific(patient_data:dict[str,list[int]], patient_ids:list[str]) -> tuple[dict[str, list[int]], dict[str,list[int]]]:
    """
    This function removes a subset of patients from the original dictionary and returns two dictionaries:
      - one containing the extracted patient entries
      - one containing the remaining patient entries
    It performs a safe copy of the input dictionary so the original data is not modified

    Args:
        patient_data (dict[str,list[int]]): Dictionary with patient_id as key and list of labels of integers as value
        patient_ids (list[str]):            List of keys to extract from `patient_data`

    Returns:
        tuple (dict[str, list[int]], dict[str,list[int]]): 
            - `remaining_data`: contains all entries from `patient_data` except those whose IDs appear in `patient_ids`
            - `extracted_data`: contains only the entries whose IDs were listed in `patient_ids`
    
    Raises:
        ValueError: If the `extracted_data` is empty
    """
    patient_data= patient_data.copy()
    set_data= defaultdict(list)
    
    for patient_id in patient_ids:
        values= patient_data.pop(patient_id, None)
        if values is not None:
            set_data[patient_id].extend(values)
    
    if len(set_data)==0:
        raise ValueError("No values in patient_ids are not found in patient_data. Result set is empty.\nPatient_ids : {}".format(", ".join(patient_ids)))

    return patient_data, set_data

def subsets_from_patient_splits(dataset:Dataset, patient_to_indices:dict[str,list[int]], set_splitted:dict[str,list[int]]) -> Subset:
    """
    Generate a Subset given the original dataset and the dictionary

    Args:
        dataset (Dataset):                              Original dataset
        patient_to_indices (dict[str, list[int]])):     Dictionary with patient_id as key and list of indeces as value (from original dataset)
        set_splitted (dict[str,list[int]]):             Dictionary with patient_id as key and list of labels of integers as value

    Returns:
        Subset: Subset from the split
    """
    indeces = []
    for patient in set_splitted.keys():
        indeces.extend(patient_to_indices[patient])

    subset = Subset(dataset, indeces)

    return subset
