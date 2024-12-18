#!/home/akugyo/Programs/Python/torch/bin/python


import torch
from torch import nn


class YoloV1(nn.Module):

    def __init__(self):

        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                padding=3,
                stride=2,
            ),
            nn.MaxPool2d(
                kernel_size=2,
                padding=0,
                stride=2,
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=192,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.MaxPool2d(
                kernel_size=2,
                padding=0,
                stride=2,
            ),
            nn.Conv2d(
                in_channels=192,
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
            nn.Conv2d(
                in_channels=256,
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
                padding=0,
                stride=2,
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
            nn.Conv2d(
                in_channels=512,
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
            nn.MaxPool2d(
                kernel_size=2,
                padding=0,
                stride=2,
            ),
        )

        self.block_2 = nn.Sequential(
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
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                padding=1,
                stride=2,
            ),
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=1024 * 7 * 7,
                out_features=4096,
            ),
            nn.Dropout(p=0.5),
            nn.Linear(
                in_features=4096,
                out_features=30,
            ),
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        return self.linear(x)


if __name__ == "__main__":
    img = torch.randn(1, 3, 448, 448, dtype=torch.float32)
    model = YoloV1()
    model(img)
