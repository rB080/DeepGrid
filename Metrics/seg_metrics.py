import torch
from sklearn.metrics import roc_curve as rc
from sklearn.metrics import auc as auc
import numpy as np

def convert(x):
    return x.detach().cpu().item()

def confmat_params(pred, mask):
    pred = torch.round(pred)
    TP = (mask * pred).sum()
    TN = ((1 - mask) * (1 - pred)).sum()
    FP = pred.sum() - TP
    FN = mask.sum() - TP
    return TP,TN,FP,FN

def metrics(pred, mask):
    TP, TN, FP, FN = confmat_params(pred, mask)
    acc = (TP + TN)/ (TP + TN + FP + FN)
    acc = convert(torch.sum(acc))
    iou = (TP)/(TP + FN + FP)
    iou = convert(torch.sum(iou))
    sen = TP / (TP + FN)
    sen = convert(torch.sum(sen))
    prec = (TP)/ (TP + FP)
    prec = convert(torch.sum(prec))
    recc = TP / (TP + FN)
    recc = convert(torch.sum(recc))
    dice = (2*TP)/(2*TP+FP+FN)
    dice = convert(torch.sum(dice))
    return acc,sen,prec,recc,dice,iou

def giveAUC(gt_list, pred_list):
    G, P = np.array(gt_list).ravel(), np.array(pred_list).ravel()
    fpr, tpr, threshold = rc(G, P)
    auroc = auc(fpr, tpr)
    return auroc, fpr, tpr