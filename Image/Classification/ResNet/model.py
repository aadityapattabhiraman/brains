#!/home/akugyo/Programs/Python/torch/bin/python

import torch
from torch import nn


class ResidualNetwork(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1
    ) -> None:

        super().__init__()
        self.residual = nn.Sequential(
                nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1
            ),
            nn.MaxPool2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(out_channels),
        )

        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                ),
                nn.MaxPool2d(out_channels),
            )

    def forward(self, x):
        x = self.residual(x)
        x += self.shortcut(x)
        return self.relu(x)


class ResNet(nn.Module):

    def __init__(
            self,
            num_classes: int = 1000
    ) -> None:

        super().__init__()
        self.in_channels = 3

        self.resnet = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                padding=3,
                stride=2,
            ),
            nn.MaxPool2d(64),
            self._make_layer(64, 2, stride=1),
            self._make_layer(128, 2, stride=2),
            self._make_layer(256, 2, stride=2),
            self._make_layer(512, 2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def _make_layer(
            self,
            out_channels,
            num_blocks,
            stride,
    ):

        layers = []
        layers.append(ResidualNetwork(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        for _ in range(num_blocks):
            layers.append(ResidualNetwork(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):

        self.resnet(x)


model = ResNet()
print(model)
