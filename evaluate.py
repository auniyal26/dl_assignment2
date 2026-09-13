from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config import DEVICE, BATCH_SIZE, OUTPUT_ROOT
from data import make_datasets
from losses_metrics import segmentation_metrics_from_logits

def denormalize_image(x):
    x = x.detach().cpu() * 0.5 + 0.5
    return x.clamp(0, 1)


@torch.no_grad()
def classification_predictions(model, loader):
    model = model.to(DEVICE)
    model.eval()

    y_true, y_pred = [], []

    for images, labels, masks in loader:
        images = images.to(DEVICE)

        logits, _ = model(images)
        pred = logits.argmax(dim=1).cpu().numpy()

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(pred.tolist())

    model.cpu()

    return np.array(y_true), np.array(y_pred)


@torch.no_grad()
def per_sample_segmentation_scores(model, dataset):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = model.to(DEVICE)
    model.eval()

    ious, dices = [], []

    for images, labels, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        _, seg_logits = model(images)

        iou, dice = segmentation_metrics_from_logits(
            seg_logits,
            masks
        )

        ious.extend(iou.cpu().numpy().tolist())
        dices.extend(dice.cpu().numpy().tolist())

    model.cpu()

    return np.array(ious), np.array(dices)


def save_confusion_matrix(
    model,
    loader,
    class_names,
    experiment_name
):
    y_true, y_pred = classification_predictions(
        model,
        loader
    )

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(18, 18))
    disp = ConfusionMatrixDisplay(
        cm,
        display_labels=class_names
    )
    disp.plot(
        ax=ax,
        xticks_rotation=90,
        colorbar=False
    )
    ax.set_title(
        f"Confusion Matrix — {experiment_name}"
    )

    plt.tight_layout()

    out = OUTPUT_ROOT / (
        experiment_name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        + "_confusion_matrix.png"
    )

    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    return (y_true == y_pred).mean()


@torch.no_grad()
def save_segmentation_examples(
    model,
    dataset,
    indices,
    experiment_name
):
    model = model.to(DEVICE)
    model.eval()

    fig, axes = plt.subplots(
        len(indices),
        3,
        figsize=(10, 3 * len(indices))
    )

    for r, idx in enumerate(indices):
        image, label, gt_mask = dataset[idx]

        _, seg_logits = model(
            image.unsqueeze(0).to(DEVICE)
        )

        pred_mask = (
            torch.sigmoid(seg_logits)[0, 0] >= 0.5
        ).float().cpu().numpy()

        img_np = (
            denormalize_image(image)
            .permute(1, 2, 0)
            .numpy()
        )
        gt_np = gt_mask.squeeze(0).numpy()

        axes[r, 0].imshow(img_np)
        axes[r, 0].set_title("Input Image")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(gt_np, cmap="gray")
        axes[r, 1].set_title("Ground-Truth Mask")
        axes[r, 1].axis("off")

        axes[r, 2].imshow(pred_mask, cmap="gray")
        axes[r, 2].set_title("Predicted Mask")
        axes[r, 2].axis("off")

    plt.suptitle(
        f"{experiment_name}: Successful and Unsuccessful Segmentation Examples"
    )
    plt.tight_layout()

    out = OUTPUT_ROOT / (
        experiment_name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        + "_segmentation_examples.png"
    )

    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    model.cpu()


def save_history_plots(histories):
    metrics = [
        ("loss", "Total loss"),
        ("accuracy", "Classification accuracy"),
        ("iou", "IoU"),
        ("dice", "Dice"),
    ]

    for metric, title in metrics:
        plt.figure(figsize=(10, 6))

        for name, h in histories.items():
            plt.plot(
                h["epoch"],
                h[f"train_{metric}"],
                label=f"{name} — train"
            )
            plt.plot(
                h["epoch"],
                h[f"val_{metric}"],
                linestyle="--",
                label=f"{name} — val"
            )

        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(f"Training and Validation {title}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_ROOT / f"{metric}_curves.png",
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()


def valid_mask_indices(dataset):
    return np.array([
        i
        for i in range(len(dataset))
        if dataset[i][2].sum().item() > 0
    ])
