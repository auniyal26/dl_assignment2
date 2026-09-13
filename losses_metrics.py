import torch
import torch.nn as nn

from config import CLS_WEIGHT, SEG_WEIGHT

classification_loss_fn = nn.CrossEntropyLoss()
bce_loss_fn = nn.BCEWithLogitsLoss()

def dice_loss_from_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)

    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def segmentation_loss_fn(logits, targets):
    bce = bce_loss_fn(logits, targets)
    dice = dice_loss_from_logits(logits, targets)

    return 0.5 * bce + 0.5 * dice


def multitask_loss(cls_logits, labels, seg_logits, masks):
    cls_loss = classification_loss_fn(cls_logits, labels)
    seg_loss = segmentation_loss_fn(seg_logits, masks)

    total = (
        CLS_WEIGHT * cls_loss
        + SEG_WEIGHT * seg_loss
    )

    return total, cls_loss, seg_loss


@torch.no_grad()
def segmentation_metrics_from_logits(
    logits,
    targets,
    threshold=0.5,
    eps=1e-7
):
    pred = (torch.sigmoid(logits) >= threshold).float()

    dims = (1, 2, 3)

    intersection = (pred * targets).sum(dim=dims)
    pred_sum = pred.sum(dim=dims)
    target_sum = targets.sum(dim=dims)

    union = pred_sum + target_sum - intersection

    iou = (intersection + eps) / (union + eps)
    dice = (
        (2 * intersection + eps)
        / (pred_sum + target_sum + eps)
    )

    return iou, dice
