import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

from utils import (
    parse_args,
    get_device,
    train_one_epoch,
    evaluate,
)


class TumorResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1
        self.backbone = resnet18(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def train_with_history(model, train_loader, val_loader, device, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

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


def main():
    args = parse_args(
        description="ResNet18 transfer learning for brain tumor classification",
        default_train_root="/content/data/Training",
        default_test_root="/content/data/Testing",
        default_image_size=(224, 224),
        default_batch_size=32,
        default_epochs=15,
        default_lr=1e-4,
    )

    device = get_device()
    image_size = tuple(args.image_size)
    weights = ResNet18_Weights.IMAGENET1K_V1
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

    train_data = datasets.ImageFolder(
        root=args.train_root, transform=train_transform)
    test_data = datasets.ImageFolder(
        root=args.test_root, transform=test_transform)

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
    model = TumorResNet18(num_classes=num_classes).to(device)

    history = train_with_history(
        model,
        train_loader,
        test_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
    )

    print("\nFinal ResNet18 test performance:")
    evaluate(model, test_loader, device, split_name="Test")

    torch.save(model.state_dict(), "checkpoints/resnet18_tumor_best.pth")
    torch.save(history, "checkpoints/resnet18_tumor_history.pt")


if __name__ == "__main__":
    main()
