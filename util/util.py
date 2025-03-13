# By Yuxiang Sun, Dec. 4, 2020
# Email: sun.yuxiang@outlook.com

import numpy as np 
from PIL import Image 
import torch
import torch.nn as nn
import torch.nn.functional as F


 
# 0:background, 1:negative, 2:positive
def get_palette():
    background = [0,0,0]
    negative   = [0, 163, 226]
    positive   = [173, 69, 31]

    palette    = np.array([background,negative, positive])
    return palette

def visualize(image_name, predictions, data_name):
    palette = get_palette()
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)): 
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        img.save('runs_demo/Pred_' + data_name + '_' + image_name[i] + '.png')

def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    consider_unlabeled = True  # must consider the background, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    F1_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN
        if (recall_per_class[cid] == np.nan) | (precision_per_class[cid] == np.nan) |(precision_per_class[cid]==0)|(recall_per_class[cid]==0):
            F1_per_class[cid] = np.nan
        else :
            F1_per_class[cid] = 2 / (1/precision_per_class[cid] +1/recall_per_class[cid])

    return precision_per_class, recall_per_class, iou_per_class,F1_per_class


def lovasz_softmax(probs, labels, classes='present'):
    """
    Computes the Lovasz-Softmax loss.
    :param probs: [B, C, H, W] Softmax probabilities (model outputs after softmax).
    :param labels: [B, H, W] Ground truth labels (0 to C-1, not one-hot).
    :param classes: Classes to compute the loss on. Default is 'present'.
    :return: Lovasz-Softmax loss.
    """
    B, C, H, W = probs.shape  # Batch size, number of classes, height, width

    # Flatten predictions and ground truth
    probs = probs.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
    labels = labels.view(-1)  # [B*H*W]

    # Filter out ignore labels (-1)
    valid_mask = labels != -1  # Ignore index is -1
    probs = probs[valid_mask]  # Keep only valid pixels
    labels = labels[valid_mask]

    # Compute per-class Lovasz hinge loss
    losses = []
    for c in range(C):
        if classes == 'present' and not (labels == c).any():
            # Skip class if it is not present
            continue
        fg = (labels == c).float()  # Foreground for class c
        errors = (fg - probs[:, c]).abs()  # Errors between prediction and ground truth
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]  # Sort foreground by prediction error
        losses.append(lovasz_grad(fg_sorted) @ errors_sorted)  # Compute Lovasz loss

    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, requires_grad=True).to(probs.device)


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors.
    :param gt_sorted: [N] Sorted ground truth labels (binary labels 0/1).
    :return: Gradient of the Lovasz extension.
    """
    gts = gt_sorted.sum()  # Total number of foreground pixels
    intersection = gts - gt_sorted.float().cumsum(0)  # Foreground pixels left
    union = gts + (1 - gt_sorted).float().cumsum(0)  # Foreground pixels + background pixels
    jaccard = 1.0 - intersection / union
    jaccard[1:] = jaccard[1:] - jaccard[:-1]  # Gradients
    return jaccard


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovasz-Softmax loss module for multi-class semantic segmentation.
    """
    def __init__(self):
        super(LovaszSoftmaxLoss, self).__init__()

    def forward(self, logits, labels):
        """
        Forward pass for Lovasz-Softmax loss.
        :param logits: [B, C, H, W] Logits output from the model (before softmax).
        :param labels: [B, H, W] Ground truth labels (0 to C-1, not one-hot encoded).
        :return: Lovasz-Softmax loss.
        """
        probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
        return lovasz_softmax(probs, labels)


    