from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import numpy as np
import warnings
import os

class EmbeddingVisualizer:
    """Handles saving, loading, and plotting of embedding projections using matplotlib"""
    @staticmethod
    def save(projected:np.ndarray, labels:np.ndarray, save_path:str, overwrite:bool=True) -> None:
        """
        Save projected data and labels to disk
        
        Args:
            projected (np.ndarray): Projected features array of shape (n_samples, n_components)
            labels (np.ndarray):    Labels array of shape (n_samples,)
            save_path (str):        File path where the data will be saved. The `.npz` extension will be appended to the filename if it is not already there
            overwrite (bool):       Overwrite the file if already exists
        """
        save_path= save_path if save_path.endswith(".npz") else f"{save_path}.npz"
        if os.path.exists(save_path) and (not overwrite):
            warnings.warn(f"File '{save_path}' already exists, it has not been overwritten")
            return None
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        save_dict = {
            'projected': projected,
            'labels': labels
        }
        
        np.savez(save_path, **save_dict)
    
    @staticmethod
    def load(load_path:str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load projected data and labels from disk
            :param load_path (str): File name where the data will be loaded. The `.npz` extension will be appended to the filename if it is not already there
            :return tuple(np.ndarray, np.ndarray):  The array will be: projected_data, labels
            :raise ValueError: If the file does not exist
        """
        load_path= load_path if load_path.endswith(".npz") else f"{load_path}.npz"
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No such file or directory: {load_path}")
        
        arrays= np.load(load_path)
        return arrays['projected'],  arrays['labels']
    
    @staticmethod
    def plot(projected:np.ndarray, labels:np.ndarray, title:str, show:bool=True, save_path:str=None, dpi:int=150):
        """
        Create matplotlib visualization of embeddings. For 2D embeddings a single scatter plot is shown.
        For 3D or higher, a grid of pairwise 2D scatter plots is produced (upper-triangle layout, correlation-matrix style)
        
        Args:
            projected (np.ndarray): Projected features of shape (num_samples, 2) or (num_samples, 3)
            labels (np.ndarray):    Labels of shape (num_samples,)
            title (str):            Title of the plot
            show (bool):            If True, display the plot
            save_path (str):        If provided, save the figure to this path
            dpi (int):              Resolution for saved figure
        """
        ALPHA = 0.7
        MARKER_SIZE = 20

        _, n_components = projected.shape

        if n_components < 2:
            raise ValueError(f"Need at least 2 components to plot, got {n_components}")

        unique_labels = np.unique(labels)

        # ---- Case: 2D ----
        if (n_components == 2):
            fig = plt.figure(figsize=(11, 11))
            ax = fig.add_subplot(111)

            for label in unique_labels:
                mask = (labels == label)
                ax.scatter(
                    projected[mask, 0],
                    projected[mask, 1],
                    alpha=ALPHA,
                    s=MARKER_SIZE,
                    label=f'Class {label}',
                    edgecolors='white',
                    linewidth=0.5
                )

            ax.set_xlabel('Component 1', fontsize=12)
            ax.set_ylabel('Component 2', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_title(title, fontsize=14, pad=20)

            handles, legend_labels = ax.get_legend_handles_labels()

        # --- Case: 3D ----
        elif (n_components == 3):
            fig = plt.figure(figsize=(11, 11))
            ax = fig.add_subplot(111, projection='3d')
            
            for label in unique_labels:
                mask = (labels == label)
                ax.scatter(
                    projected[mask, 0],
                    projected[mask, 1],
                    projected[mask, 2],
                    
                    alpha=ALPHA,
                    s=MARKER_SIZE,
                    
                    label=f'Class {label}',
                    edgecolors='white',
                    linewidth=0.5
                )
            
            ax.set_xlabel('Component 1', fontsize=12)
            ax.set_ylabel('Component 2', fontsize=12)
            ax.set_zlabel('Component 3', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_title(title, fontsize=14, pad=20)
        
            ax.set_title(title, fontsize=14, pad=20)
            ax.legend(loc='best', framealpha=0.9)
            
            handles, legend_labels = ax.get_legend_handles_labels()
            
        # ---- Case: 4D or higher ----
        else:
            fig, axes = plt.subplots(
                nrows   = n_components,
                ncols   = n_components,
                figsize = (4*n_components, 4*n_components),
                squeeze = False
            )

            handles, legend_labels = None, None

            for i in range(n_components):
                for j in range(n_components):

                    ax:Axes = axes[i, j]

                    # Only plot upper triangle pairs (i < j)
                    if i < j:
                        for label in unique_labels:
                            mask = (labels == label)
                            ax.scatter(
                                projected[mask, j],
                                projected[mask, i],
                                alpha=ALPHA,
                                s=MARKER_SIZE,
                                label=f'Class {label}',
                                edgecolors='white',
                                linewidth=0.5
                            )

                            # Capture legend once
                            if handles is None:
                                handles, legend_labels = ax.get_legend_handles_labels()

                        ax.set_xlabel(f'Component {j+1}')
                        ax.set_ylabel(f'Component {i+1}')
                        ax.grid(True, alpha=0.3, linestyle='--')

                    else:
                        # Disable unused cells (diagonal + lower triangle)
                        ax.axis('off')

            fig.suptitle(title, fontsize=16, y=0.99)

        # ---- Shared legend ----
        if handles:
            fig.legend(handles, legend_labels, loc='upper right', framealpha=0.9)

        plt.tight_layout(rect=[0, 0, 0.96, 0.97])

        if show:
            plt.show()

        if (save_path is not None):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if not(save_path.endswith(".png")):
                save_path = save_path + ".png"
            fig.savefig(save_path, dpi=dpi, format="png", bbox_inches="tight")

        if not(show):
            plt.close(fig)

    @staticmethod
    def plot_from_file(load_path:str, title:str, show:bool=True, save_path:str=None, dpi:int=150):
        """
        Load projected data and labels from disk, then create matplotlib visualization of embeddings
        
        Args:
            load_path (str):    File name where the data will be loaded. The `.npz` extension will be appended to the filename if it is not already there
            title (str):        Title of the plot
            show (bool):        If True, display the plot
            save_path (str):    If provided, save the figure to this path
            dpi (int):          Resolution for saved figure
        """
        projected, labels = EmbeddingVisualizer.load(load_path)
        EmbeddingVisualizer.plot(projected, labels, title, show, save_path, dpi)
