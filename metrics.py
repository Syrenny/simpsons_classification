import numpy as np
import torch
from torchmetrics.functional.classification import multiclass_confusion_matrix, multiclass_stat_scores


class Metrics:
    def __init__(self, config):
        self.stat_scores = torch.DoubleTensor(4, config["num_classes"]).zero_().to(config["device"])
        self.confusion_matrix = torch.DoubleTensor(config["num_classes"], config["num_classes"]).zero_().to(config["device"])
        self.num_classes = config["num_classes"]

    def update(self, preds, target):
        cm = multiclass_confusion_matrix(preds, target, num_classes=self.num_classes)
        self.confusion_matrix = self.confusion_matrix.add(cm)
        ss = multiclass_stat_scores(preds, target, self.num_classes, average=None).transpose(0, 1)  # every tp, tn,..., sup in different columns
        self.stat_scores = self.stat_scores.add(ss[:4]) # except support column

    def accuracy(self):
        # (tp + tn) / (tp + tn + fp + fn)
        numer = self.stat_scores[0].add(self.stat_scores[2])
        denom = self.stat_scores.sum(0) # tp + tn + fp + fn
        result = numer.div(denom)
        return np.nan_to_num(result.cpu().numpy())

    def recall(self):
        # tp / (tp + fn)
        result = self.stat_scores[0].div(self.stat_scores[0].add(self.stat_scores[3]))
        return np.nan_to_num(result.cpu().numpy())

    def precision(self):
        # tp / (tp + fp)
        result = self.stat_scores[0].div(self.stat_scores[0].add(self.stat_scores[1]))
        return np.nan_to_num(result.cpu().numpy())

    def get_confusion_matrix(self):
        return self.confusion_matrix.cpu().numpy()

    def reset_metrics(self):
        self.stat_scores = self.stat_scores.fill_(0)
        self.confusion_matrix = self.confusion_matrix.fill_(0)
