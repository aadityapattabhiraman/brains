import os
from torch import nn


config = {
	"batch_size": 4096,
	"num_workers": os.cpu_count(),
	"criterion": nn.CrossEntropyLoss(),
	"epochs": 15
}

def get_config() -> dict:
	return config