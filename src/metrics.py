import torch

def dice_coeff(logits, target, eps=1e-7):
    pred = (torch.sigmoid(logits) > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=[1,2,3])
    union = pred.sum(dim=[1,2,3]) + target.sum(dim=[1,2,3])
    dice = (2*inter + eps) / (union + eps)
    return dice.mean().item()

def iou_score(logits, target, eps=1e-7):
    pred = (torch.sigmoid(logits) > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=[1,2,3])
    union = (pred + target - pred*target).sum(dim=[1,2,3])
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()

def precision_recall(logits, target, eps=1e-7):
    pred = (torch.sigmoid(logits) > 0.5).float()
    target = (target > 0.5).float()
    tp = (pred * target).sum(dim=[1,2,3])
    fp = ((pred == 1) & (target == 0)).sum(dim=[1,2,3]).float()
    fn = ((pred == 0) & (target == 1)).sum(dim=[1,2,3]).float()
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return precision.mean().item(), recall.mean().item()
