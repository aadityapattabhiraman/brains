#!/home/akugyo/Programs/Python/PyTorch/bin/python

import torch
from torch import nn


class Inception(nn.Module):
	"""
	Model architecture for Inception block
	"""

	def __init__(
		self, 
		in_channels: int, 
		size_1x1: int, 
		reduce_3x3: int, 
		size_3x3: int, 
		reduce_5x5: int, 
		size_5x5: int, 
		proj_size: int) -> None:

		super().__init__()

		self.branch_1 = nn.Sequential(
			nn.Conv2d(
				in_channels=in_channels,
				out_channels=size_1x1,
				kernel_size=1,
			),
			nn.ReLU(),
		)

		self.branch_2 = nn.Sequential(
			nn.Conv2d(
				in_channels=in_channels,
				out_channels=reduce_3x3,
				kernel_size=1,
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=reduce_3x3,
				out_channels=size_3x3,
				kernel_size=3,
				padding=1,
			),
			nn.ReLU(),
		)

		self.branch_3 = nn.Sequential(
			nn.Conv2d(
				in_channels=in_channels,
				out_channels=reduce_5x5,
				kernel_size=1,
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=reduce_5x5,
				out_channels=size_5x5,
				kernel_size=5,
				padding=2,
			),
			nn.ReLU(),
		)

		self.branch_4 = nn.Sequential(
			nn.MaxPool2d(
				kernel_size=3,
				stride=1,
				padding=1,
			),
			nn.Conv2d(
				in_channels=in_channels,
				out_channels=proj_size,
				kernel_size=1,
			),
			nn.ReLU(),
		)

	def forward(self, x):
		x_1 = self.branch_1(x)
		x_2 = self.branch_2(x)
		x_3 = self.branch_3(x)
		x_4 = self.branch_4(x)

		return torch.cat([x_1, x_2, x_3, x_4], 1)


class GoogLeNet(nn.Module):
	"""
	Model architecture for GoogLeNet
	"""

	def __init__(self):
		super().__init__()

		self.initial = nn.Sequential(
			nn.Conv2d(
				in_channels=3,
				out_channels=64,
				kernel_size=7,
				stride=2,
				padding=3,
			),
			nn.ReLU(),
			nn.MaxPool2d(
				kernel_size=3,
				stride=2,
				ceil_mode=True,
			),
			nn.Conv2d(
				in_channels=64,
				out_channels=64,
				kernel_size=1,
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=64,
				out_channels=192,
				kernel_size=3,
				padding=1,
			),
			nn.ReLU(),
			nn.MaxPool2d(
				kernel_size=3,
				stride=2,
				ceil_mode=True,
			),
		)

		self.inception_3a = Inception(192, 64, 96, 128, 16, 32, 32)

		self.inception_3b = nn.Sequential(
			Inception(256, 128, 128, 192, 32, 96, 64),
			nn.MaxPool2d(
				kernel_size=3,
				stride=2,
				ceil_mode=True,
			),
		)

		self.inception_4a = Inception(480, 192, 96, 208, 16, 48, 64)

		self.inception_4b = Inception(512, 160, 112, 224, 24, 64, 64)

		self.inception_4c = Inception(512, 128, 128, 256, 24, 64, 64)

		self.inception_4d = Inception(512, 112, 144, 288, 32, 64, 64)

		self.inception_4e = nn.Sequential(
			Inception(528, 256, 160, 320, 32, 128, 128),
			nn.MaxPool2d(
				kernel_size=3,
				stride=2,
				ceil_mode=True,
			),
		)

		self.inception_5a = Inception(832, 256, 160, 320, 32, 128, 128)

		self.inception_5b = Inception(832, 384, 192, 384, 48, 128, 128)

		self.done = nn.Sequential(
			nn.AvgPool2d(
				kernel_size=7,
				stride=1,
			),
			nn.Flatten(),
			nn.Dropout(p=0.4),
			nn.Linear(
				in_features=1024,
				out_features=1000,
			),
			nn.Softmax(dim=1),
		)
		
	def forward(self, x):
		x = self.initial(x)
		x = self.inception_3a(x)
		x = self.inception_3b(x)
		x = self.inception_4a(x)
		x = self.inception_4b(x)
		x = self.inception_4c(x)
		x = self.inception_4d(x)
		x = self.inception_4e(x)
		x = self.inception_5a(x)
		x = self.inception_5b(x)
		x = self.done(x)

		return x

if __name__ == "__main__":
	img = torch.randn(8, 3, 224, 224).to("cuda")
	model = GoogLeNet()
	model.to("cuda")
	model(img)
	print(img.shape)