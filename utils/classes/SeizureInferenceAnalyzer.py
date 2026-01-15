from torch import Tensor
import torch

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

import pandas as pd
import numpy as np
import warnings
import os

class SeizureInferenceAnalyzer:
    """Collects, stores, and analyzes inference-time predictions"""
    def __init__(self):
        """Initialize an empty analyzer"""
        self._logit_0 = []
        self._logit_1 = []
        self._label = []
        self._time_to_seizure = []
        self._patient_id = []

    def append(self, logits:Tensor, label:Tensor, time_to_seizure_seconds:Tensor, patient_id:tuple[str]):
        """
        Append inference information for a batch of samples

        Args:
            logits (Tensor):                    Raw model logits with shape (batch, 2)
            label (Tensor):                     Ground-truth labels (0 or 1) with shape (batch,)
            time_to_seizure_seconds (Tensor):   Signed time (seconds) to nearest seizure event with shape (batch,). If the sample has alredy a seizure indicates the nearest end of the current seizure
            patient_id (tuple[str]):            Patient identifier for each sample with length (batch)
        """
        # Move tensors to CPU and detach from graph if necessary
        logits = logits.detach().cpu()
        label = label.detach().cpu()
        time_to_seizure_seconds = time_to_seizure_seconds.detach().cpu()

        batch_size = logits.shape[0]

        for i in range(batch_size):
            self._logit_0.append(float(logits[i, 0]))
            self._logit_1.append(float(logits[i, 1]))
            self._label.append(int(label[i]))
            self._time_to_seizure.append(float(time_to_seizure_seconds[i]))
            self._patient_id.append(str(patient_id[i]))

    def save(self, filepath:str, overwrite:bool=True):
        """
        Save collected inference data to disk.
        
        Args:
            filepath (str):     Path where the data will be saved
            overwrite (bool):   Overwrite the file if already exists
        """
        filepath= filepath if filepath.endswith(".csv") else f"{filepath}.csv"
        if os.path.exists(filepath) and (not overwrite):
            warnings.warn(f"File '{filepath}' already exists, it has not been overwritten")
            return None

        df = pd.DataFrame({
            "logit_0": self._logit_0,
            "logit_1": self._logit_1,
            "label": self._label,
            "time_to_seizure_seconds": self._time_to_seizure,
            "patient_id": self._patient_id,
        })

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)

    @staticmethod
    def load(filepath:str):
        """
        Load previously saved inference data.
            :param filepath (str): Path to a file produced by :func:`save`
            :return analyzer (SeizureInferenceAnalyze): Loaded analyzer instance
        """
        filepath= filepath if filepath.endswith(".csv") else f"{filepath}.csv"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No such file or directory: {filepath}")
        
        df = pd.read_csv(filepath)

        analyzer = SeizureInferenceAnalyzer()
        analyzer._logit_0 = df["logit_0"].tolist()
        analyzer._logit_1 = df["logit_1"].tolist()
        analyzer._label = df["label"].tolist()
        analyzer._time_to_seizure = df["time_to_seizure_seconds"].tolist()
        analyzer._patient_id = df["patient_id"].tolist()

        return analyzer

    def plot_error_histograms(self, tau:float, patient_ids:list[str]=None, max_abs_time:float=None, bins:int=30, show:bool=True, save_path:str=None):
        """
        Plot error histograms for both classes and report detailed statistics.

        This function analyzes:
        - Class 0 (non-seizure): false positives and their temporal distribution
        - Class 1 (seizure): false negatives and their temporal distribution
        - Confidence distributions for both classes

        Args:
            tau (float):            Decision threshold applied to seizure probability
            patient_ids (list):     If provided, only samples from these patients are considered
            max_abs_time (float):   If provided, only samples with |time_to_seizure| <= max_abs_time are considered
            bins (int):             Number of bins for histograms
            show (bool):            Show the figure
            save_path (str):        If provided, save the figure at the specified path
        """
        # Build DataFrame from internal storage
        df = pd.DataFrame({
            "logit_0": self._logit_0,
            "logit_1": self._logit_1,
            "label": self._label,
            "time": self._time_to_seizure,
            "patient_id": self._patient_id,
        })

        # Optional filtering
        if (patient_ids is not None):
            df = df[df["patient_id"].isin(patient_ids)]

        if (max_abs_time is not None):
            df = df[df["time"].abs() <= max_abs_time]

        # Convert logits → probabilities
        logits = df[["logit_0", "logit_1"]].values
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

        df["prob_seizure"] = probs[:, 1]
        df["confidence"] = probs.max(axis=1)

        # Apply threshold to get predictions
        df["predicted"] = (df["prob_seizure"] >= tau).astype(int)

        # ========== CLASS 0 ANALYSIS (non-seizure) ==========
        neg_df = df[df["label"] == 0]
        n_negative = len(neg_df)
        n_correct_neg = (neg_df["predicted"] == 0).sum()
        n_false_positive = (neg_df["predicted"] == 1).sum()
        
        # ========== CLASS 1 ANALYSIS (seizure) ==========
        pos_df = df[df["label"] == 1]
        n_positive = len(pos_df)
        n_correct_pos = (pos_df["predicted"] == 1).sum()
        n_false_negative = (pos_df["predicted"] == 0).sum()

        # Print statistics in organized format
        statistics_class_0  = (
            "Total samples:\n"
            "   {}\n"
            "Correctly classified (TN):\n"
            "   {} ({:5.2f}%)\n"
            "Misclassified (FP):\n"
            "   {} ({:5.2f}%)"
        ).format(
            n_negative,
            n_correct_neg, (100 * n_correct_neg / n_negative),
            n_false_positive, (100 * n_false_positive / n_negative)
        )
        
        statistics_class_1  = (
            "Total samples:\n"
            "   {}\n"
            "Correctly classified (TP):\n"
            "   {} ({:5.2f}%)\n"
            "Misclassified (FN):\n"
            "   {} ({:5.2f}%)\n"
        ).format(
            n_positive,
            n_correct_pos, (100 * n_correct_pos / n_positive),
            n_false_negative, (100 * n_false_negative / n_positive)
        )

        # ========== PLOTTING ==========
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ---------- ROW 1: CLASS 0 (Non-seizure) errors ----------
        
        # Subplot 1: Time-to-seizure for false positives
        fp_df = neg_df[neg_df["predicted"] == 1]
        pre_ictal_fp = fp_df[fp_df["time"] > 0]["time"]
        post_ictal_fp = fp_df[fp_df["time"] < 0]["time"]

        ax:Axes = axes[0, 0]
        ax.hist(
            pre_ictal_fp,
            bins=bins,
            alpha=0.6,
            label="Pre-ictal FP (seizure ahead)",
        )
        ax.hist(
            post_ictal_fp,
            bins=bins,
            alpha=0.6,
            label="Post-ictal FP (seizure behind)",
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title("Class 0: False Positives vs Time to Seizure")
        ax.set_xlabel("Time to nearest seizure (seconds)")
        ax.set_ylabel("Number of samples")
        ax.legend(loc="best")

        # Subplot 2: Confidence distribution for class 0
        ax:Axes = axes[0, 1]
        ax.hist(
            neg_df[neg_df["predicted"] == 0]["confidence"],
            bins=bins,
            alpha=0.7,
            label="Correct (TN)",
        )
        ax.hist(
            neg_df[neg_df["predicted"] == 1]["confidence"],
            bins=bins,
            alpha=0.7,
            label="False Positive",
        )
        ax.text(
            x= ax.get_xlim()[1] * 1.05,
            y= ax.get_ylim()[1]//2,
            s= statistics_class_0
        )
        ax.set_title("Class 0: Confidence Distribution")
        ax.set_xlabel("Model confidence")
        ax.set_ylabel("Number of samples")
        ax.legend()

        # ---------- ROW 2: CLASS 1 (Seizure) errors ----------
        
        # Subplot 3: Time-to-seizure for false negatives
        fn_df = pos_df[pos_df["predicted"] == 0]
        pre_ictal_fn = fn_df[fn_df["time"] > 0]["time"]
        post_ictal_fn = fn_df[fn_df["time"] < 0]["time"]
        ictal_fn = fn_df[fn_df["time"] == 0]["time"]

        ax:Axes = axes[1, 0]
        ax.hist(
            pre_ictal_fn,
            bins=bins,
            alpha=0.7,
            label="Pre-ictal FN (seizure ahead)",
        )
        ax.hist(
            post_ictal_fn,
            bins=bins,
            alpha=0.7,
            label="Post-ictal FN (seizure behind)",
        )
        ax.hist(
            ictal_fn,
            bins=bins,
            alpha=0.7,
            label="Ictal FN (seizure edge)",
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title("Class 1: False Negatives vs Time to Seizure")
        ax.set_xlabel("Time to nearest seizure (seconds)")
        ax.set_ylabel("Number of samples")
        ax.legend()

        # Subplot 4: Confidence distribution for class 1
        ax:Axes = axes[1, 1]
        ax.hist(
            pos_df[pos_df["predicted"] == 1]["confidence"],
            bins=bins,
            alpha=0.7,
            label="Correct (TP)",
        )
        ax.hist(
            pos_df[pos_df["predicted"] == 0]["confidence"],
            bins=bins,
            alpha=0.7,
            label="False Negative",
        )
        ax.text(
            x= ax.get_xlim()[1] * 1.05,
            y= ax.get_ylim()[1]//2,
            s= statistics_class_1
        )
        ax.set_title("Class 1: Confidence Distribution")
        ax.set_xlabel("Model confidence")
        ax.set_ylabel("Number of samples")
        ax.legend(loc="best")

        # Overall title and layout
        plt.suptitle(f"Classification Error Analysis (τ = {tau})", fontsize=14, y=0.995)
        
        # Save and/or show
        if (save_path is not None):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if not(save_path.endswith(".png")):
                save_path = save_path + ".png"
            fig.savefig(save_path, dpi=200, format="png", bbox_inches="tight")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def plot_roc_curve(self, tau:float=None, patient_ids:list[str]=None, max_abs_time:float=None, show:bool=True, save_path:str=None):
        """
        Plot ROC curve for the seizure detection model.
        
        Args:
            tau (float):            If provided, mark this threshold on the ROC curve with lines
            patient_ids (list):     If provided, only samples from these patients are considered
            max_abs_time (float):   If provided, only samples with |time_to_seizure| <= max_abs_time are considered
            show (bool):            Show the figure
            save_path (str):        If provided, save the figure at the specified path
        """
        # Build DataFrame from internal storage
        df = pd.DataFrame({
            "logit_0": self._logit_0,
            "logit_1": self._logit_1,
            "label": self._label,
            "time": self._time_to_seizure,
            "patient_id": self._patient_id,
        })

        # Optional filtering
        if (patient_ids is not None):
            df = df[df["patient_id"].isin(patient_ids)]

        if (max_abs_time is not None):
            df = df[df["time"].abs() <= max_abs_time]

        # Convert logits → probabilities
        logits = df[["logit_0", "logit_1"]].values
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        
        prob_seizure = probs[:, 1]
        true_labels = df["label"].values

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(true_labels, prob_seizure)
        roc_auc = auc(fpr, tpr)

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot ROC curve
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        
        # Plot diagonal reference line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')

        # If tau is specified, mark it on the curve
        if (tau is not None):
            # Find the point on ROC curve closest to this threshold
            # Find index where threshold is closest to tau
            idx = np.argmin(np.abs(thresholds - tau))
            fpr_at_tau = fpr[idx]
            tpr_at_tau = tpr[idx]
            
            # Plot the point
            ax.plot(fpr_at_tau, tpr_at_tau, 'ro', markersize=8, label=f'τ = {tau:.2f}')
            
            # Draw vertical line from x-axis to the point
            ax.plot([fpr_at_tau, fpr_at_tau], [0, tpr_at_tau], 'r--', linewidth=1.5, alpha=0.7)
            
            # Draw horizontal line from y-axis to the point
            ax.plot([0, fpr_at_tau], [tpr_at_tau, tpr_at_tau], 'r--', linewidth=1.5, alpha=0.7)
            
            # Add text annotations for FPR and TPR values
            ax.text(fpr_at_tau, -0.05, f'{fpr_at_tau:.3f}', ha='center', va='top', fontsize=9, color='red')
            ax.text(-0.05, tpr_at_tau, f'{tpr_at_tau:.3f}', ha='right', va='center', fontsize=9, color='red')

        # Labels and formatting
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
        ax.set_title('ROC Curve - Seizure Detection', fontsize=14)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_aspect('equal')
        
        plt.tight_layout()

        # Save and/or show
        if (save_path is not None):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if not(save_path.endswith(".png")):
                save_path = save_path + ".png"
            fig.savefig(save_path, dpi=200, format="png")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
