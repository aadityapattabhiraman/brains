import torch
from torch import nn


class GoogLeNet(nn.Module):
	"""
	Model architecture for GoogLeNet
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		pass


if __name__ == "__main__":
	img = torch.randn(3, 224, 224)
	print(img.shape)