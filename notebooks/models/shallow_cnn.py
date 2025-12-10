# notebooks/models/shallow_cnn.py
import torch
import torch.nn as nn

from utils import (
    parse_args,
    get_device,
    get_dataloaders,
    maybe_load_checkpoint,
    train_loop,
)


class ShallowCNN(nn.Module):
    """
    Shallow CNN:
    - 2 conv blocks with max-pooling
    - 1 fully-connected (linear) layer as classifier
    """

    def __init__(self, num_classes: int, image_size=(224, 224)):
        super().__init__()
        self.image_size = image_size

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # H, W -> H/2, W/2

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # H, W -> H/4, W/4
        )

        h, w = image_size
        flattened_dim = 32 * (h // 4) * (w // 4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)   # logits
        return x


def main():
    args = parse_args(
        description="Shallow CNN baseline for brain tumor classification",
        default_train_root="/content/data/Training",
        default_test_root="/content/data/Testing",
        default_image_size=(224, 224),
        default_epochs=30,
        default_batch_size=32,
        default_lr=1e-3,
    )

    device = get_device()
    train_loader, test_loader, num_classes = get_dataloaders(args, device)

    image_size = tuple(args.image_size)
    model = ShallowCNN(num_classes=num_classes,
                       image_size=image_size).to(device)

    model = maybe_load_checkpoint(model, args.ckpt_path, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loop(model, train_loader, test_loader,
               optimizer, device, args.epochs)

    print("Final evaluation on test set:")
    from utils import evaluate  
    evaluate(model, test_loader, device, split_name="Test")


if __name__ == "__main__":
    main()
