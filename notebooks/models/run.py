# notebooks/models/compare_models.py

import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import (
    parse_args,
    get_device,
    get_dataloaders,
    train_one_epoch,
    evaluate,
)

from notebooks.models.cnn import CNNBaseline
from notebooks.models.shallow_cnn import ShallowCNN


def train_with_history(model, train_loader, val_loader, device, epochs, lr):
    """Train a model and record per-epoch metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(
            model, val_loader, device, split_name="Val")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    return history


def plot_comparison(hist_shallow, hist_deep, save_path=None):
    epochs = np.arange(1, len(hist_shallow["val_acc"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, np.array(
        hist_shallow["val_acc"]) * 100.0, label="ShallowCNN")
    plt.plot(epochs, np.array(
        hist_deep["val_acc"]) * 100.0, label="CNNBaseline")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("ShallowCNN vs CNNBaseline – Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"Saved comparison plot to {save_path}")

    plt.show()


def main():
    args = parse_args(
        description="Compare ShallowCNN vs CNNBaseline on brain tumor dataset",
        default_train_root="/content/data/Training",
        default_test_root="/content/data/Testing",
        default_image_size=(224, 224),
        default_batch_size=32,
        default_epochs=10,   # can override with --epochs
        default_lr=1e-3,
    )

    device = get_device()
    train_loader, test_loader, num_classes = get_dataloaders(args, device)
    image_size = tuple(args.image_size)

    # ---- Shallow model ----
    print("\n=== Training ShallowCNN ===")
    shallow_model = ShallowCNN(
        num_classes=num_classes,
        image_size=image_size,
    ).to(device)

    hist_shallow = train_with_history(
        shallow_model,
        train_loader,
        test_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
    )

    print("\nFinal ShallowCNN test performance:")
    evaluate(shallow_model, test_loader, device, split_name="Test")

    # ---- Deeper CNNBaseline ----
    print("\n=== Training CNNBaseline ===")
    deep_model = CNNBaseline(
        in_channels=3,
        num_classes=num_classes,
    ).to(device)

    hist_deep = train_with_history(
        deep_model,
        train_loader,
        test_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
    )

    print("\nFinal CNNBaseline test performance:")
    evaluate(deep_model, test_loader, device, split_name="Test")

    # ---- Plot comparison ----
    plot_comparison(hist_shallow, hist_deep,
                    save_path="model_comparison_val_acc.png")


if __name__ == "__main__":
    main()
