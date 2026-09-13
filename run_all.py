import copy
from pathlib import Path

import pandas as pd
import torch

from config import DEVICE, OUTPUT_ROOT, EPOCHS
from data import (
    load_base_datasets,
    make_datasets,
    find_empty_raw_masks
)
from train import train_experiment
from evaluate import (
    save_confusion_matrix,
    save_segmentation_examples,
    save_history_plots,
    per_sample_segmentation_scores,
    valid_mask_indices
)

def main():
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    base_trainval, base_test = load_base_datasets()

    print("Trainval size:", len(base_trainval))
    print("Test size:", len(base_test))
    print("Classes:", len(base_trainval.classes))

    empty_trainval = find_empty_raw_masks(base_trainval)
    empty_test = find_empty_raw_masks(base_test)

    print(
        "Empty trainval masks:",
        len(empty_trainval),
        empty_trainval
    )
    print(
        "Empty test masks:",
        len(empty_test),
        empty_test
    )

    experiments = {
        "No Augmentation": "baseline",
        "Geometric Augmentation": "geometric",
        "Geometric + Appearance": "geo_appearance",
    }

    models = {}
    histories = {}
    results = {}

    for name, mode in experiments.items():
        print("\n" + "=" * 80)
        print(name)
        print("=" * 80)

        model, history, test_metrics, test_loader = train_experiment(
            base_trainval,
            base_test,
            mode,
            epochs=EPOCHS
        )

        models[name] = copy.deepcopy(model).cpu()
        histories[name] = history

        history.to_csv(
            OUTPUT_ROOT / (
                name.lower()
                .replace(" ", "_")
                .replace("+", "plus")
                + "_history.csv"
            ),
            index=False
        )

        checkpoint_path = OUTPUT_ROOT / (
            name.lower()
            .replace(" ", "_")
            .replace("+", "plus")
            + "_best.pt"
        )
        torch.save(model.state_dict(), checkpoint_path)

        accuracy = save_confusion_matrix(
            models[name],
            test_loader,
            base_trainval.classes,
            name
        )

        _, _, test_ds = make_datasets(
            base_trainval,
            base_test,
            "baseline"
        )

        ious, dices = per_sample_segmentation_scores(
            models[name],
            test_ds
        )

        valid_ids = valid_mask_indices(test_ds)
        valid_ious = ious[valid_ids]
        valid_dices = dices[valid_ids]

        best_ids = valid_ids[
            valid_ious.argsort()[::-1][:3]
        ]
        worst_ids = valid_ids[
            valid_ious.argsort()[:2]
        ]
        selected = list(best_ids) + list(worst_ids)

        save_segmentation_examples(
            models[name],
            test_ds,
            selected,
            name
        )

        results[name] = {
            "Classification Accuracy": accuracy,
            "Mean IoU": test_metrics["iou"],
            "Mean Dice": test_metrics["dice"],
            "Valid-mask Mean IoU": valid_ious.mean(),
            "Valid-mask Mean Dice": valid_dices.mean(),
        }

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison_df = pd.DataFrame.from_dict(
        results,
        orient="index"
    ).reset_index(names="Experiment")

    comparison_df.to_csv(
        OUTPUT_ROOT / "comparison.csv",
        index=False
    )

    print("\nFinal comparison:")
    print(comparison_df.to_string(index=False))

    save_history_plots(histories)

    print("\nSaved all outputs to:")
    print(OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
