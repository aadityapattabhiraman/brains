#!/home/akugyo/Programs/Python/torch/bin/python

import torch
from torch import nn


class YoloV2(nn.Module):

    def __init__(self):

        super().__init__()
        self.yolo = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=1000,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            nn.Flatten(),
            nn.Linear(
                in_features=1000 * 14 * 14,
                out_features=1000,
            ),
        )

    def forward(self, x):
        x = self.yolo(x)
        return self.classifier(x)


if __name__ == "__main__":
    img = torch.randn(1, 3, 448,448, dtype=torch.float)
    model = YoloV2()
    model(img)
