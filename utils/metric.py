import matplotlib.pyplot as plt
import numpy as np
import warnings
import os

class Metrics():
    """Static class to save, load and plot the metrics"""
    @staticmethod
    def save(filename:str, train_metric:np.ndarray=None, val_metric:np.ndarray=None, test_metric:np.ndarray=None, overwrite:bool=True) -> None:
        """
        Save the metric at in a specific file. If the file does not exists the function will create the full path to the file

        Args:
            filename (str):             File name where the data will be saved. The `.npz` extension will be appended to the filename if it is not already there
            train_metric (np.ndarray):  Array with the datas about training metrics. If None an empty array will be saved
            val_metric (np.ndarray):    Array with the datas about validation metrics. If None an empty array will be saved
            test_metric (np.ndarray):   Array with the datas about testing metrics. If None an empty array will be saved
            overwrite (bool):           Overwrite the file if already exists
        """
        train_metric=   train_metric    if (train_metric is not None)   else np.full(0, np.nan, dtype=np.int8)
        val_metric=     val_metric      if (val_metric is not None)     else np.full(0, np.nan, dtype=np.int8)
        test_metric=    test_metric     if (test_metric is not None)    else np.full(0, np.nan, dtype=np.int8)
        
        filename= filename if filename.endswith(".npz") else f"{filename}.npz"
        if os.path.exists(filename) and (not overwrite):
            warnings.warn(f"File '{filename}' already exists, it has not been overwritten")
            return None
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        np.savez(
            filename, 
            train_metric=train_metric, 
            val_metric=val_metric, 
            test_metric=test_metric
        )

    @staticmethod
    def load(filename:str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the metric at in a specific file. If the file does not exists raise an Exception
            :param filename (str): File name where the data will be loaded. The `.npz` extension will be appended to the filename if it is not already there
            :return tuple(np.ndarray, np.ndarray, np.ndarray):  The array will be: train_metric, val_metric, test_metric
        """
        filename= filename if filename.endswith(".npz") else f"{filename}.npz"
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No such file or directory: {filename}")
        
        arrays= np.load(filename)
        return arrays["train_metric"], arrays["val_metric"], arrays["test_metric"]
    
    @staticmethod
    def fusion(filename:str, train_metric:np.ndarray=None, val_metric:np.ndarray=None, test_metric:np.ndarray=None, filename_before:bool=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the metric at in a specific file. If the file does not exists raise an Exception. Extend the data loaded with the `np.ndarray` parameters
        
        Args:
            filename (str):                             File name where the data will be loaded. The `.npz` extension will be appended to the filename if it is not already there
            train_metric (np.ndarray):                  Array with the datas about training metrics. If None an empty array will be saved
            val_metric (np.ndarray):                    Array with the datas about validation metrics. If None an empty array will be saved
            test_metric (np.ndarray):                   Array with the datas about testing metrics. If None an empty array will be saved
            filename_before (bool):                     Decide if the data loaded from the `filename` must be added before the parameter of after
        
        Returns:
            tuple(np.ndarray, np.ndarray, np.ndarray):  The array will be: train_metric, val_metric, test_metric
        """
        train, val, test = Metrics.load(filename)
        
        if filename_before:
            train_metric = np.append(train, train_metric.copy())
            val_metric = np.append(val, val_metric.copy())
            test_metric = np.append(test, test_metric.copy())
        else:
            train_metric = np.append(train_metric.copy(), train)
            val_metric = np.append(val_metric.copy(), val)
            test_metric = np.append(test_metric.copy(), test)
        
        return train_metric, val_metric, test_metric
    
    @staticmethod
    def plot(train_metric:np.ndarray=None, val_metric:np.ndarray=None, test_metric:np.ndarray=None, metric_name:str="Metric", marker:bool=False, show:str='none', start_check:int|float=0.2, best_k:int=3, higher_is_better:bool=False):
        """
        Plot the metrics on the same figure

        Args:
            train_metric (np.ndarray):  Array with the datas about training metrics. If None an empty array will be saved
            val_metric (np.ndarray):    Array with the datas about validation metrics. If None an empty array will be saved
            test_metric (np.ndarray):   Array with the datas about testing metrics. If None an empty array will be saved
            metric_name (str):          Title of the plot
            marker (bool):              Use a marker to highlight the points
            show (str):                 There are different choises:
                                        - if 'none' the train will not be show
                                        - if 'full' show full train
                                        - if 'points' show test metric for best K min/max validation values
                                        - if 'both' show both 'full' and 'points'
            start_check (int|float):    If `show` is 'full' or 'both' start checking for the best value from a defined epoch. If it is int then it is the epoch num, if it is float then it is the percentage of the total
            best_k (int):               Show K train point
            higher_is_better (bool):    If True, higher metric values are better (e.g., accuracy). If False, lower metric values are better (e.g., loss)
        """
        train_metric=   train_metric    if (train_metric is not None)   else np.full(0, np.nan, dtype=np.int8)
        val_metric=     val_metric      if (val_metric is not None)     else np.full(0, np.nan, dtype=np.int8)
        test_metric=    test_metric     if (test_metric is not None)    else np.full(0, np.nan, dtype=np.int8)
        
        possibilities= ['full', 'points', 'both', 'none']
        if show not in possibilities:
            raise ValueError("show '{}' not exists, choose between '{}'".format(show, "', '".join(possibilities)))
        
        if isinstance(start_check, float):
            start_check= round(start_check*len(train_metric))
        
        figure= plt.figure(figsize=(11,5))
        figure.suptitle(metric_name)
        
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        
        marker= "." if marker else None
        
        val_ascending_indeces= np.argsort(val_metric[start_check:]) + start_check
        best_k_indeces= val_ascending_indeces[-best_k:] if higher_is_better else val_ascending_indeces[:best_k]
        
        _= plt.plot(train_metric, label="train", marker=marker)
        _= plt.plot(val_metric, label="val", marker=marker)
        
        color= None
        if (show in ['full', 'both']):
            plot_object= plt.plot(test_metric, label="test", marker=marker)
            color = plot_object[0].get_color()
        if (show in ['points', 'both']):
            _= plt.plot(best_k_indeces, test_metric[best_k_indeces], label="test values", marker='o', linestyle="None", color=color)
            
            ymin, _ = plt.get_ylim()
            plt.vlines(x=best_k_indeces, ymin=ymin, ymax=test_metric[best_k_indeces], colors='gray', linestyles='--')

            vertical_offset= 10 if higher_is_better else -15
            for idx in best_k_indeces:
                test_value = test_metric[idx]                
                plt.annotate(
                    text        = f"{test_value:.4f}",
                    xy            = (idx, test_value),
                    textcoords  = "offset points",
                    xytext      = (0,vertical_offset),
                    ha          = "center",
                    fontsize    = 9
                )
                
                plt.annotate(
                    text        = idx,
                    xy          = (idx, 0),
                    xycoords    = plt.gca().get_xaxis_transform(),
                    xytext      = (0, 0),
                    textcoords = "offset points",
                    ha         = "center",
                    va         = "top",
                    fontsize   = 9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="gray", alpha=0.8, edgecolor="gray")
                )
        
        plt.grid(True)
        plt.legend()
        plt.show()
