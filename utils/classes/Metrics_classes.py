import torch
from torch import Tensor
from sklearn.metrics import confusion_matrix

class Loss_Meter:
    """Keep track of average losses over time"""
    def __init__(self, name:str=""):
        """
        Keep track of average losses over time
            :param name (str): Name to add in the string value of `get_metric`
        """
        self.name= name
        self.sum =   0.0
        self.count = 0.0

    def update(self, losses:Tensor):
        """Update the meter with losses values"""
        self.sum+= losses.sum().item()
        self.count+= len(losses)
    
    def get_metric(self) -> tuple[str, float]:
        """Returns the name of the loss and the value of the average loss"""
        name= "loss" if len(self.name)==0 else f"loss_{self.name}"
        if self.count == 0:
            return  name, 0.0
        return name, self.sum/self.count

class Accuracy_Meter:
    """Keep track of weighted accuracy values and target class probabilities over time."""
    def __init__(self, class_weight:list[int]=None, num_classes:int=2, tau:float=0.5):
        """
        Initialize the meter with class weights.
            :param class_samples (list[int]):   A list of weight per class
            :param num_classes (int):           Number of classes
            :param tau (float):                 Threshold for binary classification
        """
        self.tau = tau
        self.num_classes = num_classes
        self.class_weight = torch.tensor(class_weight, dtype=torch.float) if (class_weight is not None) else None
        
        self.weighted_correct = 0.0
        self.total_weight = 0.0
        
        self.class_correct = torch.zeros(num_classes, dtype=torch.float)
        self.class_total = torch.zeros(num_classes, dtype=torch.float)
        
        self.class_prob_sum = torch.zeros(num_classes, dtype=torch.float)

    def update(self, output:Tensor, target:Tensor) -> None:
        """
        Update the meter with model output and target values.
            :param output (Tensor): Model outputs (logits) (batch_size, num_classes)
            :param target (Tensor): Ground-truth labels (batch_size,)
        """
        output = output.cpu()
        target = target.cpu()
        
        preds = (torch.softmax(output, dim=1)[:, 1] >= self.tau).long() if (self.num_classes==2) else output.argmax(dim=1)
        correct_mask = (preds == target).float()

        # Assign each sample a weight based on its true class
        target = target.to(dtype=torch.int64)
        sample_weights = self.class_weight[target] if (self.class_weight is not None) else torch.ones_like(target, dtype=torch.float)

        weighted_correct = (correct_mask * sample_weights).sum().item()
        total_weight = sample_weights.sum().item()

        self.weighted_correct += weighted_correct
        self.total_weight += total_weight
        
        # probs.gather selects the probability from the dim=1 (class dim) using the index specified by target.unsqueeze(1)
        probs = torch.softmax(output, dim=1)
        target_probs = probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
        
        # Update class-wise statistics
        for class_idx in range(self.num_classes):
            class_mask = (target == class_idx)
            if class_mask.any():
                self.class_correct[class_idx] += correct_mask[class_mask].sum().item()
                self.class_total[class_idx] += class_mask.sum().item()
                
                self.class_prob_sum[class_idx] += target_probs[class_mask].sum().item()

    def get_metric(self) -> tuple[str, float]:
        """Return ('weighted_accuracy', value)."""
        if self.total_weight == 0:
            return "weighted_accuracy", 0.0
        return "weighted_accuracy", self.weighted_correct / self.total_weight
    
    def get_class_accuracy(self) -> list[tuple[str, float]]:
        """
        Return class-wise accuracy for each class.
            :return list(tuple): [('class_0_acc', value), ('class_1_acc', value), ...]
        """
        class_accuracies = []
        for class_idx in range(self.num_classes):
            if self.class_total[class_idx] > 0:
                class_accuracies.append((
                    f"class_{class_idx}_acc",
                    (self.class_correct[class_idx] / self.class_total[class_idx]).item()
                ))
            else:
                class_accuracies.append((f"class_{class_idx}_acc", 0.0))
        
        return class_accuracies

    def get_avg_target_prob(self) -> list[tuple[str, float]]:
        """
        Return the average softmax probability for samples belonging to each class. This measures the model's average confidence in the *correct* class.
            :return list(tuple): [('avg_prob_class_0', value), ('avg_prob_class_1', value), ...]
        """
        avg_probs = []
        for class_idx in range(self.num_classes):
            if self.class_total[class_idx] > 0:
                avg_probs.append((
                    f"avg_prob_class_{class_idx}",
                    (self.class_prob_sum[class_idx] / self.class_total[class_idx]).item()
                ))
            else:
                avg_probs.append((f"avg_prob_class_{class_idx}", 0.0))
        
        return avg_probs
    
    def get_macro_average(self) -> tuple[str, float]:
        """Return ('macro_average', value)."""
        total = 0
        count = 0
        for _,value in self.get_class_accuracy():
            total += value
            count += 1
        
        if total == 0:
            return "macro_average", 0.0
        return "macro_average", total / count

class ConfusionMatrix_Meter():
    """Keep track of confusion matrix over time"""
    def __init__(self, num_classes:int, tau:float=0.5):
        """
        Keep track of confusion matrix over time
            :param num_classes (int):           Number of classes
            :param tau (float):                 Threshold for binary classification
        """
        if (num_classes != 2):
            raise NotImplementedError(f"For now the class can manage only when the input has 2 classes")
        
        self.tau = tau
        self.num_classes = num_classes
        self.labels= range(num_classes)
        self.num_samples_per_class = [0] * num_classes
        self.tp = 0.0  # True Positives
        self.tn = 0.0  # True Negatives
        self.fp = 0.0  # False Positives
        self.fn = 0.0  # False Negatives

    def update(self, output:Tensor, target:Tensor) -> None:
        """
        Update the meter with output and target values
            :param output (Tensor): Output of the model with size (batch_size, num_classes)
            :param target (Tensor): Target value for the output with size (batch_size)
        """
        preds = (torch.softmax(output, dim=1)[:, 1] >= self.tau).long() if (self.num_classes==2) else output.argmax(dim=1)
        
        cm = confusion_matrix(target.cpu(), preds.cpu(), labels=self.labels)
        
        # Extract values: cm[row, col] where row=actual, col=predicted
        # [[TN, FP],
        #  [FN, TP]]
        self.tn += cm[0, 0]
        self.fp += cm[0, 1]
        self.fn += cm[1, 0]
        self.tp += cm[1, 1]
        
        num_pos_samples = (target == 1).sum().item()
        num_neg_samples = (target == 0).sum().item()
        
        self.num_samples_per_class[0] += num_neg_samples
        self.num_samples_per_class[1] += num_pos_samples

    def get_specificity(self, label:int=None) -> tuple[str, float]:
        """Returns the name of the specificity and the value of specificity"""
        if (label is None) or (label==1):
            tn, fp = self.tn, self.fp
        elif (label==0):
            tn, fp = self.tp, self.fn
        else:
            raise ValueError(f"Label {label} does not exists")
        
        name = "specificity" if (label is None) else f"specificity_class_{label}"
        if tn + fp == 0:
            return name, 0.0
        return name, tn / (tn + fp)
    
    def get_precision(self, label:int=None) -> tuple[str, float]:
        """Returns the name of the precision and the value of precision"""
        if (label is None) or (label==1):
            tp, fp = self.tp, self.fp
        elif (label==0):
            tp, fp = self.tn, self.fn
        else:
            raise ValueError(f"Label {label} does not exists")
        
        name = "precision" if (label is None) else f"precision_class_{label}"
        if tp + fp == 0:
            return name, 0.0
        return name, tp / (tp + fp)

    def get_recall(self, label:int=None) -> tuple[str, float]:
        """Returns the name of the recall and the value of recall"""
        if (label is None) or (label==1):
            tp, fn = self.tp, self.fn
        elif (label==0):
            tp, fn = self.tn, self.fp
        else:
            raise ValueError(f"Label {label} does not exists")
        
        name = "recall" if (label is None) else f"recall_class_{label}"
        if tp + fn == 0:
            return name, 0.0
        return name, tp / (tp + fn)
    
    def get_f1_score(self, label:int=None) -> tuple[str, float]:
        """Returns the name of the f1-score and the value of f1-score"""
        precision = self.get_precision(label)[1]
        recall = self.get_recall(label)[1]
        
        name = "f1-score" if (label is None) else f"f1-score_class_{label}"
        if precision + recall == 0:
            return name, 0.0
        return name, 2 * (precision * recall) / (precision + recall)
    
    def get_f1_scores(self) -> list[tuple[str, float]]:
        """Returns the name of the f1-score and the value of the f1-score for each class"""
        f1_scores = []
        for label in self.labels:
            f1_scores.append(self.get_f1_score(label))
        
        return f1_scores

    def get_balanced_accuracy(self):
        """Returns the name of the balanced accuracy and the value of balanced accuracy"""
        sensitivity = self.get_precision()[1]
        specificity = self.get_specificity()[1]
        
        return "real_balanced_accuracy", (sensitivity + specificity) / 2
        
    
    def get_weighted_f1_score(self) -> tuple[str, float]:
        """Returns the name of the weighted f1-score and the value of weighted f1-score"""
        f1_score_pos = self.get_f1_score(1)[1]
        f1_score_neg = self.get_f1_score(0)[1]
        
        if (f1_score_pos == 0.0) and (f1_score_neg == 0.0):
            return "weighted_f1_score", 0.0
        
        weight_pos_class = self.num_samples_per_class[1] / (self.num_samples_per_class[0] + self.num_samples_per_class[1])
        weight_neg_class = self.num_samples_per_class[0] / (self.num_samples_per_class[0] + self.num_samples_per_class[1])
        
        weighted_f1_score_pos = weight_pos_class * f1_score_pos
        weighted_f1_score_neg = weight_neg_class * f1_score_neg
        
        return "weighted_f1_score", (weighted_f1_score_pos + weighted_f1_score_neg)
