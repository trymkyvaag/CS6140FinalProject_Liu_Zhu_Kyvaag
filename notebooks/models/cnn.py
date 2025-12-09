# notebooks/models/cnn_baseline.py

import torch
import torch.nn as nn



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


