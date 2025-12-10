# notebooks/models/compare_models.py

import os
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

from shallow_cnn import ShallowCNN
from cnn import CNN
from advCNN import advCNN
from resnet18_tumor import TumorResNet18


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


def run_model(name, model_ctor, model_kwargs, args, device, train_loader, test_loader):
    """
    Train or load a model + its history.

    - checkpoint: checkpoints/{name}_best.pth
    - history:    checkpoints/{name}_history.pt
    """
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/{name}_best.pth"
    hist_path = f"checkpoints/{name}_history.pt"

    model = model_ctor(**model_kwargs).to(device)

    if (
        not args.retrain
        and os.path.exists(ckpt_path)
        and os.path.exists(hist_path)
    ):
        print(f"\n=== {name}: loading from checkpoint ===")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        history = torch.load(hist_path)
    else:
        print(f"\n=== {name}: training ===")
        history = train_with_history(
            model, train_loader, test_loader, device, epochs=args.epochs, lr=args.lr
        )

        torch.save(model.state_dict(), ckpt_path)
        torch.save(history, hist_path)
        print(f"Saved {name} checkpoint to {ckpt_path}")
        print(f"Saved {name} history to    {hist_path}")

    print(f"\nFinal {name} test performance:")
    evaluate(model, test_loader, device, split_name="Test")

    return history


def plot_comparison(hist_shallow, hist_deep, hist_better, hist_resnet, save_path=None):
    epochs = np.arange(1, len(hist_shallow["val_acc"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, np.array(
        hist_shallow["val_acc"]) * 100.0, label="ShallowCNN")
    plt.plot(epochs, np.array(
        hist_deep["val_acc"]) * 100.0, label="CNNBaseline")
    plt.plot(epochs, np.array(
        hist_better["val_acc"]) * 100.0, label="TooComplexCNN")
    plt.plot(epochs, np.array(
        hist_resnet["val_acc"]) * 100.0, label="ResNet18 (transfer learning)")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title(
        "ShallowCNN vs CNNBaseline vs BetterCNN vs ResNet18 – Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"Saved comparison plot to {save_path}")

    plt.show()


def main():
    args = parse_args(
        description="Compare ShallowCNN vs CNN vs BetterCNN on brain tumor dataset",
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

    hist_shallow = run_model(
        "shallow_cnn",
        ShallowCNN,
        {"num_classes": num_classes, "image_size": image_size},
        args,
        device,
        train_loader,
        test_loader,
    )

    hist_deep = run_model(
        "cnn",
        CNN,
        {"in_channels": 3, "num_classes": num_classes},
        args,
        device,
        train_loader,
        test_loader,
    )

    hist_better = run_model(
        "tooComplex",
        advCNN,
        {"num_classes": num_classes, "image_size": image_size},
        args,
        device,
        train_loader,
        test_loader,
    )

    hist_resnet = run_model(
        "resnet18_tumor",                # matches checkpoints/resnet18_tumor_*.pth
        TumorResNet18,
        {"num_classes": num_classes},
        args,
        device,
        train_loader,
        test_loader,
    )

    if not args.no_plot:
        plot_comparison(
            hist_shallow,
            hist_deep,
            hist_better,
            hist_resnet,
            save_path="model_comparison_val_acc_four_models.png",
        )



if __name__ == "__main__":
    main()
