import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#AAA test push -boxun
class ShallowCNN(nn.Module):
    def __init__(self, num_classes, image_size=(224, 224)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        h = image_size[0] // 4
        w = image_size[1] // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * h * w, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total > 0 else 0.0
    return acc, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-dir",
        type=str,
        default=r"C:\\Users\\Rate\\.cache\\kagglehub\\datasets\\masoudnickparvar\\brain-tumor-mri-dataset\\versions\\1\\Testing",
        help="Path to test dataset root (ImageFolder layout)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="notebooks/models/shallow_cnn_best.pth",
        help="Path to saved model (state_dict)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224, help="square image size to resize to")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    model_path = Path(args.model_path)

    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        return 2
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_data = datasets.ImageFolder(root=str(test_dir), transform=transform)
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    num_classes = len(test_data.classes)
    print("Num classes:", num_classes, "Num images:", len(test_data))

    model = ShallowCNN(num_classes=num_classes, image_size=(args.image_size, args.image_size))

    state = torch.load(str(model_path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    try:
        model.load_state_dict(state)
    except Exception:
        # try stripping potential 'module.' prefixes from keys
        new_state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(new_state)

    model.to(device)

    acc, correct, total = evaluate(model, test_loader, device)
    print(f"Accuracy: {acc*100:.2f}% ({correct}/{total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
