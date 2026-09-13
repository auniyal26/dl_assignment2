import copy
import time

import torch
import pandas as pd

from config import (
    DEVICE,
    LR,
    EPOCHS,
    LR_FACTOR,
    LR_PATIENCE,
    LR_MIN,
    LR_THRESHOLD,
)
from model import MultiTaskPetCNN
from losses_metrics import multitask_loss, segmentation_metrics_from_logits
from data import make_loaders
from utils import seed_everything


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)

    totals = {
        "loss": 0.0,
        "cls_loss": 0.0,
        "seg_loss": 0.0,
        "correct": 0,
        "n": 0,
        "iou_sum": 0.0,
        "dice_sum": 0.0,
    }

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels, masks in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            if training:
                optimizer.zero_grad()

            cls_logits, seg_logits = model(images)

            loss, cls_loss, seg_loss = multitask_loss(
                cls_logits,
                labels,
                seg_logits,
                masks,
            )

            if training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)

            totals["loss"] += loss.item() * batch_size
            totals["cls_loss"] += cls_loss.item() * batch_size
            totals["seg_loss"] += seg_loss.item() * batch_size
            totals["correct"] += (
                cls_logits.argmax(dim=1) == labels
            ).sum().item()
            totals["n"] += batch_size

            iou, dice = segmentation_metrics_from_logits(
                seg_logits,
                masks,
            )

            totals["iou_sum"] += iou.sum().item()
            totals["dice_sum"] += dice.sum().item()

    n = totals["n"]

    return {
        "loss": totals["loss"] / n,
        "cls_loss": totals["cls_loss"] / n,
        "seg_loss": totals["seg_loss"] / n,
        "accuracy": totals["correct"] / n,
        "iou": totals["iou_sum"] / n,
        "dice": totals["dice_sum"] / n,
    }


def train_experiment(
    base_trainval,
    base_test,
    train_mode,
    epochs=EPOCHS,
):
    seed_everything()

    train_loader, val_loader, test_loader = make_loaders(
        base_trainval,
        base_test,
        train_mode,
    )

    model = MultiTaskPetCNN().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    # Adaptive learning-rate schedule:
    # monitor the joint validation objective and reduce LR
    # when it stops improving.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        threshold=LR_THRESHOLD,
        threshold_mode="rel",
        min_lr=LR_MIN,
    )

    history = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        start = time.time()

        lr_used = optimizer.param_groups[0]["lr"]

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
        )

        # Scheduler decision is based only on validation loss.
        scheduler.step(val_metrics["loss"])
        next_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": lr_used,
            "next_lr": next_lr,
        }

        row.update({
            f"train_{k}": v
            for k, v in train_metrics.items()
        })

        row.update({
            f"val_{k}": v
            for k, v in val_metrics.items()
        })

        history.append(row)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())

        elapsed = time.time() - start

        if next_lr < lr_used:
            lr_text = f"lr {lr_used:.2e} -> {next_lr:.2e}"
        else:
            lr_text = f"lr {lr_used:.2e}"

        print(
            f"{train_mode:>14s} | "
            f"epoch {epoch:02d}/{epochs} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val acc {val_metrics['accuracy']:.4f} | "
            f"val IoU {val_metrics['iou']:.4f} | "
            f"val Dice {val_metrics['dice']:.4f} | "
            f"{lr_text} | "
            f"{elapsed:.1f}s"
        )

    # Evaluate the checkpoint with the best joint validation loss,
    # not automatically the final epoch.
    model.load_state_dict(best_state)

    test_metrics = run_epoch(
        model,
        test_loader,
        optimizer=None,
    )

    return (
        model,
        pd.DataFrame(history),
        test_metrics,
        test_loader,
    )
