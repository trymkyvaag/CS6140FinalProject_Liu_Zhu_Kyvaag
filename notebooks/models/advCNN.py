import torch
import torch.nn as nn


class advCNN(nn.Module):
    """
    This thing is so ass:
    """

    def __init__(self, num_classes: int, image_size=(224, 224)):
        super().__init__()
        self.image_size = image_size

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)   # H,W -> H/2, W/2
            )

        self.features = nn.Sequential(
            conv_block(3, 32),    # 224 -> 112
            conv_block(32, 64),   # 112 -> 56
            conv_block(64, 128),  # 56 -> 28
        )


        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),               
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)  
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
