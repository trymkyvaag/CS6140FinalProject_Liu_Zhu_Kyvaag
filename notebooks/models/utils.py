# notebooks/models/utils.py

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# -----------------------------
# Arg parsing
# -----------------------------
def parse_args(
    description: str = "CNN classifier",
    default_train_root: str = "/content/data/Training",
    default_test_root: str = "/content/data/Testing",
    default_image_size=(224, 224),
    default_batch_size: int = 32,
    default_epochs: int = 30,
    default_lr: float = 1e-3,
):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--train_root",
        type=str,
        default=default_train_root,
        help="Path to training data (ImageFolder root)",
    )
    parser.add_argument(
        "--test_root",
        type=str,
        default=default_test_root,
        help="Path to testing data (ImageFolder root)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=list(default_image_size),
        help="Input image size (H W)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=default_batch_size,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=default_epochs,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=default_lr,
        help="Learning rate",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Fraction of training data to use as validation",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="If set, do not produce comparison plots.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="If set, ignore existing checkpoints and retrain all models.",
    )

    return parser.parse_args()


# -----------------------------
# Device
# -----------------------------
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


# -----------------------------
# Transforms (ImageNet style)
# -----------------------------
def build_transforms(image_size):
    """
    Use ImageNet mean/std so ResNet18 transfer learning is happy.
    This is also fine for your smaller CNNs.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, test_transform


# -----------------------------
# Dataloaders with train/val/test split
# -----------------------------
def get_dataloaders_with_val(args, device):
    """
    - Uses args.train_root as full TRAIN set → split into train + val.
    - Uses args.test_root as TEST set only for final evaluation.
    """
    image_size = tuple(args.image_size)
    train_transform, test_transform = build_transforms(image_size)

    # Full train set (to be split into train+val)
    full_train = datasets.ImageFolder(root=args.train_root,
                                      transform=train_transform)

    n_total = len(full_train)
    n_val = int(args.val_fraction * n_total)
    n_train = n_total - n_val

    train_data, val_data = random_split(full_train, [n_train, n_val])

    # Separate test set
    test_data = datasets.ImageFolder(root=args.test_root,
                                     transform=test_transform)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    print("Classes:", full_train.dataset.classes)
    print("Num total train images:", n_total)
    print("Num train images:", n_train)
    print("Num val images:", n_val)
    print("Num test images:", len(test_data))

    num_classes = len(full_train.dataset.classes)

    return train_loader, val_loader, test_loader, num_classes


# -----------------------------
# Train / eval utilities
# -----------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, device, split_name="Val"):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        logits = model(inputs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)

    loss = running_loss / running_total
    acc = running_correct / running_total

    print(f"{split_name} loss: {loss:.4f}  {split_name} acc: {acc*100:.2f}%")
    return loss, acc
