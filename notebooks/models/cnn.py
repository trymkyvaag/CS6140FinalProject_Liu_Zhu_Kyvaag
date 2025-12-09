import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import argparse


class CNNBaseline(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 4):
        """
        Default is set for brain tumor:
        - in_channels = 3 (RGB images)
        - num_classes = 4 (glioma, meningioma, notumor, pituitary)
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)

        # Make it independent of input size by pooling down to 2×2
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        # 64 channels * 2 * 2 = 256
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, num_classes)

        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.pool(x)                      # (B, 64, 2, 2)
        x = x.view(x.size(0), -1)             # (B, 256)

        x = self.act(self.linear1(x))
        x = F.log_softmax(self.linear2(x), dim=-1)
        return x


def train(train_loader, model, optimizer, epochs):
    device = next(model.parameters()).device
    model.train()
    losses = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_losses = []

        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = F.nll_loss(outputs, labels)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            epoch_losses.append(loss.item())

            if step % 50 == 0 and step > 0:
                print(
                    f"[Step {step}] Mean last 50 losses: {np.mean(losses[-50:]):.4f}"
                )

        print(f"Epoch {epoch+1} mean loss: {np.mean(epoch_losses):.4f}")

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def evaluate(eval_loader, model):
    device = next(model.parameters()).device
    model.eval()
    accuracies = []

    for inputs, labels in eval_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        preds = outputs.argmax(dim=1)

        batch_acc = (preds == labels).float().mean().item()
        accuracies.append(batch_acc)

    accuracy = np.mean(accuracies)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description="CNN baseline for brain tumor classification"
    )

    parser.add_argument(
        "--train_root",
        type=str,
        default="/content/data/Training",
        help="Path to training data (ImageFolder root)"
    )

    parser.add_argument(
        "--test_root",
        type=str,
        default="/content/data/Testing",
        help="Path to testing data (ImageFolder root)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Train root:", args.train_root)
    print("Test root:", args.test_root)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    train_data = datasets.ImageFolder(
        root=args.train_root, transform=transform
    )
    test_data = datasets.ImageFolder(
        root=args.test_root, transform=transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    model = CNNBaseline(
        in_channels=3,
        num_classes=len(train_data.classes),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(train_loader, model, optimizer, epochs=args.epochs)
    evaluate(test_loader, model)
