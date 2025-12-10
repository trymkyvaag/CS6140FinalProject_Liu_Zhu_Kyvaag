# notebooks/models/utils.py
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np


def parse_args(
    description: str = "CNN classifier",
    default_train_root: str = "/content/data/Training",
    default_test_root: str = "/content/data/Testing",
    default_image_size=(224, 224),
    default_batch_size: int = 32,
    default_epochs: int = 30,
    default_lr: float = 1e-3,
):
    """Common argparse for image classification experiments."""
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
        "--ckpt_path",
        type=str,
        default=None,
        help="Optional checkpoint to load (.pth)",
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


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def build_transforms(image_size):
    """Common transforms for all models."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])


def get_dataloaders(args, device):
    """Create train & test loaders + return num_classes."""
    image_size = tuple(args.image_size)
    transform = build_transforms(image_size)

    train_data = datasets.ImageFolder(
        root=args.train_root, transform=transform)
    test_data = datasets.ImageFolder(root=args.test_root, transform=transform)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
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

    print("Classes:", train_data.classes)
    print("Num training images:", len(train_data))
    print("Num testing images:", len(test_data))

    num_classes = len(train_data.classes)
    return train_loader, test_loader, num_classes


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
def evaluate(model, loader, device, split_name="Test"):
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


def maybe_load_checkpoint(model, ckpt_path, device):
    """Load a checkpoint if provided (handles raw state_dict or dict['state_dict'])."""
    if ckpt_path is None:
        return model

    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)
    return model


def train_loop(model, train_loader, test_loader, optimizer, device, epochs):
    """Common training loop with logging."""
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(
            model, test_loader, device, split_name="Val/Test")

        elapsed = time.time() - start
        print(
            f"Epoch {epoch}/{epochs} - "
            f"{elapsed:.1f}s - "
            f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
            f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}"
        )


def train_with_history_and_checkpoint(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
    ckpt_path,
):
    best_val_acc = -1.0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, device, split_name="Val"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path}")

    return history


def load_if_exists(model, ckpt_path, device):
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Loading existing checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return True
    return False
