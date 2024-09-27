import torch
from torch import nn 


class AlexNet(nn.Module):
	"""
	Model Architecture for AlexNet
	"""

	def __init__(self):
		super().__init__()

		self.block1 = nn.Sequential(
			nn.Conv2d(
				in_channels=3,
				out_channels=96,
				kernel_size=11,
				padding=0,
				stride=4,
			),
			nn.ReLU(),
			nn.LocalResponseNorm(2, alpha=1e-4, beta=0.75, k=1),
			nn.MaxPool2d(
				kernel_size=3,
				padding=0,
				stride=2,
			),
			nn.Conv2d(
				in_channels=96,
				out_channels=256,
				kernel_size=5,
				padding="same",
				stride=1
			),
			nn.ReLU(),
			nn.LocalResponseNorm(2, alpha=1e-4, beta=0.75, k=1),
			nn.MaxPool2d(
				kernel_size=3,
				padding=0,
				stride=2,
			),
		)

		self.block2 = nn.Sequential(
			nn.Conv2d(
				in_channels=256,
				out_channels=384,
				kernel_size=3,
				padding="same",
				stride=1,
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=384,
				out_channels=384,
				kernel_size=3,
				padding="same",
				stride=1,
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=384,
				out_channels=256,
				kernel_size=3,
				padding="same",
				stride=1,
			),
			nn.ReLU(),
			nn.MaxPool2d(
				kernel_size=3,
				padding=0,
				stride=2,
			),
		)

		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(
				in_features=256 * 6 * 6,
				out_features=4096,
			),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(
				in_features=4096,
				out_features=4096,
			),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(
				in_features=4096,
				out_features=1000,
			),
		)

	def forward(self, x: torch.Tensor):
		x = self.block1(x)
		x = self.block2(x)
		x = self.classifier(x)
		return x