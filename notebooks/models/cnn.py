# notebooks/models/cnn_baseline.py

import torch
import torch.nn as nn

from utils import (
    parse_args,
    get_device,
    get_dataloaders,
    maybe_load_checkpoint,
    train_loop,
    evaluate,
)


class CNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 4):
        """
        Default is set for brain tumor:
        - in_channels = 3 (RGB images)
        - num_classes = 4 (glioma, meningioma, notumor, pituitary)
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=5, stride=2, padding=2
        )
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)

        # Make it independent of input size by pooling down to 2×2
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        # 64 channels * 2 * 2 = 256
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, num_classes)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.pool(x)                      # (B, 64, 2, 2)
        x = x.view(x.size(0), -1)             # (B, 256)

        x = self.act(self.linear1(x))
        # return logits (CrossEntropy in utils)
        x = self.linear2(x)
        return x


def main():
    args = parse_args(
        description="CNN baseline for brain tumor classification",
        default_train_root="/content/data/Training",
        default_test_root="/content/data/Testing",
        default_image_size=(224, 224),
        default_batch_size=32,
        default_epochs=10,
        default_lr=1e-3,
    )

    device = get_device()
    train_loader, test_loader, num_classes = get_dataloaders(args, device)

    model = CNN(
        in_channels=3,
        num_classes=num_classes,
    ).to(device)

    model = maybe_load_checkpoint(model, args.ckpt_path, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Common training loop + logging
    train_loop(model, train_loader, test_loader,
               optimizer, device, args.epochs)

    print("Final evaluation on test set:")
    evaluate(model, test_loader, device, split_name="Test")


if __name__ == "__main__":
    main()
