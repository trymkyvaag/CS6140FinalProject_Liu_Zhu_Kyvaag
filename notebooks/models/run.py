# notebooks/models/compare_models.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils import (
    parse_args,
    get_device,
    get_dataloaders_with_val,
    train_one_epoch,
    evaluate,
)

from shallow_cnn import ShallowCNN
from cnn import CNN
from advCNN import advCNN


from torchvision.models import resnet18, ResNet18_Weights


# -----------------------------
# ResNet18 transfer model
# -----------------------------
class TumorResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)  # logits


# -----------------------------
# Training with history + ckpt
# -----------------------------
def train_with_history_and_checkpoint(
    name,
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    weight_decay=0.0,
    retrain=False,
):
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/{name}_best.pth"
    hist_path = f"checkpoints/{name}_history.pt"

    # If not retraining and both files exist → load & return
    if (not retrain) and os.path.exists(ckpt_path) and os.path.exists(hist_path):
        print(f"\n=== {name}: loading existing checkpoint & history ===")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        history = torch.load(hist_path)
        return model, history

    print(f"\n=== {name}: training from scratch ===")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

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
            model, val_loader, device, split_name=f"{name} Val"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[{name}] Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # Save best ckpt by val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ {name}: saved new best to {ckpt_path}")

    # Save history and reload best weights
    torch.save(history, hist_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"{name}: loaded best checkpoint for final test eval")

    return model, history


# -----------------------------
# Plotting
# -----------------------------
def plot_comparison(histories, labels, save_path=None):
    plt.figure(figsize=(8, 5))

    for history, label in zip(histories, labels):
        epochs = np.arange(1, len(history["val_acc"]) + 1)
        plt.plot(epochs, np.array(history["val_acc"]) * 100.0, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Model Comparison – Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"Saved comparison plot to {save_path}")

    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args(
        description="Compare ShallowCNN, CNNBaseline, BetterCNN, ResNet18 on brain tumor dataset",
        default_train_root="/content/data/Training",
        default_test_root="/content/data/Testing",
        default_image_size=(224, 224),
        default_batch_size=32,
        default_epochs=10,   # can override with --epochs
        default_lr=1e-3,
    )

    device = get_device()
    train_loader, val_loader, test_loader, num_classes = get_dataloaders_with_val(
        args, device
    )
    image_size = tuple(args.image_size)

    histories = []
    labels = []

    # ---- ShallowCNN ----
    shallow = ShallowCNN(num_classes=num_classes,
                         image_size=image_size).to(device)
    shallow, hist_shallow = train_with_history_and_checkpoint(
        name="shallow_cnn",
        model=shallow,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=0.0,
        retrain=args.retrain,
    )
    print("\nFinal ShallowCNN Test performance:")
    evaluate(shallow, test_loader, device, split_name="ShallowCNN Test")
    histories.append(hist_shallow)
    labels.append("ShallowCNN")

    # ---- CNNBaseline ----
    cnn = CNN(in_channels=3, num_classes=num_classes).to(device)
    cnn, hist_cnn = train_with_history_and_checkpoint(
        name="cnn_baseline",
        model=cnn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=0.0,
        retrain=args.retrain,
    )
    print("\nFinal CNNBaseline Test performance:")
    evaluate(cnn, test_loader, device, split_name="CNNBaseline Test")
    histories.append(hist_cnn)
    labels.append("CNNBaseline")

    # ---- BetterCNN ----
    better = advCNN(num_classes=num_classes, image_size=image_size).to(device)
    better, hist_better = train_with_history_and_checkpoint(
        name="advCNN",
        model=better,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=1e-4,
        retrain=args.retrain,
    )
    print("\nFinal advCNN Test performance:")
    evaluate(better, test_loader, device, split_name="advCNN Test")
    histories.append(hist_better)
    labels.append("advCNN")

    # ---- ResNet18 ----
    resnet = TumorResNet18(num_classes=num_classes).to(device)
    resnet, hist_resnet = train_with_history_and_checkpoint(
        name="resnet18",
        model=resnet,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=1e-4,           # smaller LR for pretrained backbone
        weight_decay=1e-4,
        retrain=args.retrain,
    )
    print("\nFinal ResNet18 Test performance:")
    evaluate(resnet, test_loader, device, split_name="ResNet18 Test")
    histories.append(hist_resnet)
    labels.append("ResNet18")

    # ---- Plot comparison ----
    if not args.no_plot:
        plot_comparison(
            histories,
            labels,
            save_path="model_comparison_val_acc_four_models.png",
        )


if __name__ == "__main__":
    main()
