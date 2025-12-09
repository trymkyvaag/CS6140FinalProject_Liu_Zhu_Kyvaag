# notebooks/models/shallow_cnn.py
import torch
import torch.nn as nn



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

