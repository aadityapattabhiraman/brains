#!/home/akugyo/Programs/Python/PyTorch/bin/python

import torch
from torch import nn


class Inception(nn.Module):
	"""
	Model architecture for Inception block
	"""

	def __init__(self, in_channels):
		super().__init__()

		self.branch_1 = nn.Sequential(
			nn.Conv2d(
				in_channels,
				out_channels=64,
				kernel_size=1,
			),
			nn.ReLU(),
		)

		self.branch_2 = nn.Sequential(
			nn.Conv2d(
				in_channels,
				out_channels=96,
				kernel_size=1,
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=96,
				out_channels=128,
				kernel_size=3,
				padding=1,
			),
			nn.ReLU(),
		)

		self.branch_3 = nn.Sequential(
			nn.Conv2d(
				in_channels,
				out_channels=16,
				kernel_size=1,
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=16,
				out_channels=32,
				kernel_size=5,
				padding=2
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
				in_channels,
				out_channels=32,
				kernel_size=1,
			),
			nn.ReLU(),
		)

	def forward(self, x):
		x_1 = self.branch_1(x)
		print(x_1.shape)
		x_2 = self.branch_2(x)
		print(x_2.shape)
		x_3 = self.branch_3(x)
		print(x_3.shape)
		x_4 = self.branch_4(x)
		print(x_4.shape)

		output = torch.cat([x_1, x_2, x_3, x_4], 1)
		print(output.shape)
		return output


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
				padding=1,
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
			),
			nn.ReLU(),
			nn.MaxPool2d(
				kernel_size=3,
				stride=2,
				padding=1,
			),
		)

		self.inception_1 = Inception(192)
		self.inception_2 = Inception(256)

		self.inception_3 = nn.Sequential(
			nn.MaxPool2d(
				kernel_size=3,
				stride=2,
				padding=1,
			),
			Inception(480),
		)

		self.inception_4 = Inception(512)
		self.inception_5 = Inception(512)
		self.inception_6 = Inception(512)
		self.inception_7 = Inception(528)

		self.inception_8 = nn.Sequential(
			nn.MaxPool2d(
				kernel_size=3,
				stride=2,
				padding=1,
			),
			Inception(832),
		)

		self.inception_9 = Inception(832)

		self.done = nn.Sequential(
			nn.AdaptiveAvgPool2d(
				(1, 1)
			),
			nn.Flatten(),
			nn.Dropout(p=0.5),
			nn.Linear(
				1024,
				1000))

		
	def forward(self, x):
		x = self.initial(x)
		x = self.inception_1(x)
		x = self.inception_2(x)
		x = self.inception_3(x)
		x = self.inception_4(x)
		x = self.inception_5(x)
		x = self.inception_6(x)
		x = self.inception_7(x)
		x = self.inception_8(x)
		x = self.inception_9(x)
		x = self.done(x)
		return x


if __name__ == "__main__":
	img = torch.randn(8, 3, 224, 224).to("cuda")
	model = GoogLeNet()
	model.to("cuda")
	model(img)
	print(img.shape)