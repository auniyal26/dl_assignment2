import random
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from config import (
    DATA_ROOT, IMG_SIZE, BATCH_SIZE, VAL_FRACTION, SEED
)

class PetMultiTaskDataset(Dataset):
    def __init__(self, base_dataset, indices, mode="baseline", img_size=IMG_SIZE):
        assert mode in {"baseline", "geometric", "geo_appearance"}
        self.base = base_dataset
        self.indices = list(indices)
        self.mode = mode
        self.img_size = img_size

        self.color_jitter = T.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25
        )

    def __len__(self):
        return len(self.indices)

    def _paired_geometry(self, image, mask):
        i, j, h, w = T.RandomResizedCrop.get_params(
            image,
            scale=(0.80, 1.00),
            ratio=(0.90, 1.10)
        )

        image = TF.resized_crop(
            image, i, j, h, w,
            [self.img_size, self.img_size],
            InterpolationMode.BILINEAR
        )
        mask = TF.resized_crop(
            mask, i, j, h, w,
            [self.img_size, self.img_size],
            InterpolationMode.NEAREST
        )

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        angle = random.uniform(-15.0, 15.0)
        image = TF.rotate(
            image,
            angle=angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=0
        )
        mask = TF.rotate(
            mask,
            angle=angle,
            interpolation=InterpolationMode.NEAREST,
            fill=2
        )

        return image, mask

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        image, target = self.base[base_idx]
        label, mask = target

        if self.mode == "baseline":
            image = TF.resize(
                image,
                [self.img_size, self.img_size],
                interpolation=InterpolationMode.BILINEAR
            )
            mask = TF.resize(
                mask,
                [self.img_size, self.img_size],
                interpolation=InterpolationMode.NEAREST
            )
        else:
            image, mask = self._paired_geometry(image, mask)

        if self.mode == "geo_appearance":
            image = self.color_jitter(image)

        image = TF.to_tensor(image)
        image = TF.normalize(
            image,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

        # Trimap values:
        # 1 = pet, 2 = background, 3 = boundary.
        # Binary target: pet + boundary = foreground.
        mask_np = np.array(mask)
        mask_bin = (mask_np != 2).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0)

        return image, int(label), mask_tensor


def load_base_datasets():
    base_trainval = OxfordIIITPet(
        root=DATA_ROOT,
        split="trainval",
        target_types=["category", "segmentation"],
        download=True
    )

    base_test = OxfordIIITPet(
        root=DATA_ROOT,
        split="test",
        target_types=["category", "segmentation"],
        download=True
    )

    return base_trainval, base_test


def fixed_split_indices(base_trainval, base_test):
    n = len(base_trainval)

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g).tolist()

    n_val = int(round(VAL_FRACTION * n))
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]
    test_indices = list(range(len(base_test)))

    return train_indices, val_indices, test_indices


def make_datasets(base_trainval, base_test, train_mode):
    train_idx, val_idx, test_idx = fixed_split_indices(base_trainval, base_test)

    train_ds = PetMultiTaskDataset(base_trainval, train_idx, train_mode)
    val_ds = PetMultiTaskDataset(base_trainval, val_idx, "baseline")
    test_ds = PetMultiTaskDataset(base_test, test_idx, "baseline")

    return train_ds, val_ds, test_ds


def make_loaders(base_trainval, base_test, train_mode):
    train_ds, val_ds, test_ds = make_datasets(
        base_trainval,
        base_test,
        train_mode
    )

    g = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


def find_empty_raw_masks(base_dataset):
    empty = []

    for i in range(len(base_dataset)):
        _, target = base_dataset[i]
        _, raw_mask = target
        raw = np.array(raw_mask)

        if np.all(raw == 2):
            empty.append(i)

    return empty
