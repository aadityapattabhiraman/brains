import torch
from torch import nn  


class VGGNet(nn.Module):
	"""
	Model Architecture for VGG-16
	"""

	def __init__(self):
		self.block1 = nn.Sequential(
			nn.Conv2d(
				in_channels=3,
				out_channels=64,
				kernel_size=3,
				padding=1,
				stride=1,
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=64,
				out_channels=64,
				kernel_size=3,
				padding=1,
				stride=1,
			),
			nn.ReLU(),
			nn.MaxPool2d(
				kernel_size=2,
				stride=2
			),
		)

		self.block2 = nn.Sequential(
			nn.Conv2d(
				in_channels=64,
				out_channels=128,
				kernel_size=3,
				padding=1,
				stride=1
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=128,
				out_channels=128,
				kernel_size=3,
				padding=1,
				stride=1
			),
			nn.ReLU(),
			nn.MaxPool2d(
				kernel_size=2,
				stride=2
			),
		)

		self.block3 = nn.Sequential(
			nn.Conv2d(
				in_channels=128,
				out_channels=256,
				kernel_size=3,
				padding=1,
				stride=1
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=256,
				out_channels=256,
				kernel_size=3,
				padding=1,
				stride=1
			),
			nn.ReLU(),
			nn.MaxPool2d(
				kernel_size=2,
				stride=2
			),
		)

		self.block4 = nn.Sequential(
			nn.Conv2d(
				in_channels=256,
				out_channels=512,
				kernel_size=3,
				padding=1,
				stride=1
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=512,
				out_channels=512,
				kernel_size=3,
				padding=1,
				stride=1
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=512,
				out_channels=512,
				kernel_size=3,
				padding=1,
				stride=1
			),
			nn.ReLU(),
			nn.MaxPool2d(
				kernel_size=2,
				stride=2
			),
		)

		self.block5 = nn.Sequential(
			nn.Conv2d(
				in_channels=512,
				out_channels=512,
				kernel_size=3,
				padding=1,
				stride=1
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=512,
				out_channels=512,
				kernel_size=3,
				padding=1,
				stride=1
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=512,
				out_channels=512,
				kernel_size=3,
				padding=1,
				stride=1
			),
			nn.ReLU(),
			nn.MaxPool2d(
				kernel_size=2,
				stride=2
			),
		)

		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(
				in_features=512,
				out_features=4096,
			),
			nn.ReLU(),
			nn.Linear(
				in_features=4096,
				out_features=4096,
			),
			nn.ReLU(),
			nn.Linear(
				in_features=4096,
				out_features=1000,
			),
		)

	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.block5(x)
		x = self.classifier(x)