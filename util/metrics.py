""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn.functional as F
import numpy as np

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    output = F.softmax(output.float(), dim=1)  # F.softmax(torch.tensor(output).float(), dim=1)
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.shape[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
import torch

from sklearn.metrics import confusion_matrix, precision_score

import csv
from util.tax_entry import new_genus_dict
def macro_average_precision(y_pred, y_true):
    y_pred = F.softmax(y_pred.float(), dim=1).numpy() 
    y_pred = np.argmax(y_pred, axis=-1)
    classes = set(list(y_true.cpu().numpy()))  
    macro_avg_precision = 0.0
    macro_avg_recall = 0.0
    macro_avg_f1 = 0.0
    for cls in classes:
        y_true_cls = [1 if label == cls else 0 for label in y_true.cpu()]
        y_pred_cls = [1 if label == cls else 0 for label in y_pred]

        if confusion_matrix(y_true_cls, y_pred_cls).ravel().shape[0] == 1:
            fp = tn = fn = 0
            tp = len(y_true)
        else:
            tn, fp, fn, tp = confusion_matrix(y_true_cls, y_pred_cls).ravel()

        # 计算精度
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        # 累加精度
        macro_avg_precision += precision
        macro_avg_recall += recall
        macro_avg_f1 += f1

    # 计算宏平均精度
    macro_avg_precision /= len(classes)  #ttp / (ttp + tfp)
    macro_avg_recall /= len(classes)
    macro_avg_f1 /= len(classes)
    return macro_avg_precision * 100.0, macro_avg_recall * 100, macro_avg_f1 * 100

def weighted_macro_average_precision(y_pred, y_true):
    y_pred = F.softmax(y_pred.float(), dim=1).numpy() 
    y_pred = np.argmax(y_pred, axis=-1)
    classes = set(list(y_true.cpu().numpy())) 
    weighted_macro_avg_precision = 0.0
    for cls in classes:
        y_true_cls = [1 if label == cls else 0 for label in y_true.cpu()]
        y_pred_cls = [1 if label == cls else 0 for label in y_pred]
        if confusion_matrix(y_true_cls, y_pred_cls).ravel().shape[0] == 1:
            fp = tn = fn = 0
            tp = len(y_true)
        else:
            tn, fp, fn, tp = confusion_matrix(y_true_cls, y_pred_cls).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        weighted_macro_avg_precision += (precision * sum(y_true_cls) / len(y_true))

    return weighted_macro_avg_precision * 100.0


def micro_average_precision(y_pred, y_true):
    y_pred = F.softmax(y_pred.float(), dim=1).numpy() 
    y_pred = np.argmax(y_pred, axis=-1)
    classes = set(list(y_true.cpu().numpy())) 
    micro_avg_precision = 0.0
    ttp = 0.0
    ttn = 0.0
    tfp = 0.0
    tnp = 0.0
    for cls in classes:
        y_true_cls = [1 if label == cls else 0 for label in y_true.cpu()]
        y_pred_cls = [1 if label == cls else 0 for label in y_pred]

        if confusion_matrix(y_true_cls, y_pred_cls).ravel().shape[0] == 1:
            fp = tn = fn = 0
            tp = len(y_true)
        else:
            tn, fp, fn, tp = confusion_matrix(y_true_cls, y_pred_cls).ravel()
        ttp += tp
        tfp += fp
    micro_avg_precision = ttp / (ttp + tfp)
    return micro_avg_precision * 100.0




def macro_average_precision_for_retrieval(y_pred, y_true):
    classes = set(list(y_true.cpu().numpy())) 
    macro_avg_precision = 0.0
    macro_avg_recall = 0.0
    macro_avg_f1 = 0.0
    for cls in classes:
        y_true_cls = [1 if label == cls else 0 for label in y_true.cpu()]
        y_pred_cls = [1 if label == cls else 0 for label in y_pred]
        if confusion_matrix(y_true_cls, y_pred_cls).ravel().shape[0] == 1:
            fp = tn = fn = 0
            tp = len(y_true)
        else:
            tn, fp, fn, tp = confusion_matrix(y_true_cls, y_pred_cls).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        macro_avg_precision += precision
        macro_avg_recall += recall
        macro_avg_f1 += f1
    macro_avg_precision /= len(classes) 
    macro_avg_recall /= len(classes)
    macro_avg_f1 /= len(classes)
    return macro_avg_precision * 100.0, macro_avg_recall * 100, macro_avg_f1 * 100


