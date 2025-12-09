import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim


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


if __name__ == "__main__":

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # (C,H,W), values in [0,1]
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),  # for 3-channel RGB
            std=(0.5, 0.5, 0.5)
        )
    ])

    # ---------- Datasets ----------
    train_root = "/Users/trymkyvag/Desktop/Northeastern/Fall 25/CS 6140/Final Project/CS6140FinalProject_Liu_Zhu_Kyvaag/data/Training"
    test_root = "/Users/trymkyvag/Desktop/Northeastern/Fall 25/CS 6140/Final Project/CS6140FinalProject_Liu_Zhu_Kyvaag/data/Testing"

    train_data = datasets.ImageFolder(root=train_root, transform=transform)
    test_data = datasets.ImageFolder(root=test_root, transform=transform)

    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    print("Classes:", train_data.classes)
    print("Number of training images:", len(train_data))
    print("Number of testing images:", len(test_data))


    model = CNNBaseline(
        in_channels=3,                     # RGB images
        num_classes=len(train_data.classes)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ---------- Train & eval ----------
    train(train_loader, model, optimizer, epochs=10)
    evaluate(test_loader, model)
