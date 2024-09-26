#!/home/akugyo/Programs/Python/PyTorch/bin/python

import torch
import warnings
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from config import get_config
from model import AlexNet


def get_data(config):
	transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(256),
		transforms.Resize(size=(227, 227)),
	    transforms.ToTensor(),
    ])

	train = datasets.ImageFolder(
		root=f"{config["root"]}/Train/",
		transform=transform,
	)

	test = datasets.ImageFolder(
		root=f"{config["root"]}/Test/",
		transform=transform,
	)

	train_dataloader = DataLoader(
		dataset=train, 
		batch_size=config["batch_size"], 
		shuffle=True,
		num_workers=config["num_workers"],
		pin_memory=True
	)

	test_dataloader = DataLoader(
		dataset=test,
		batch_size=1,
		shuffle=False,
		num_workers=config["num_workers"],
		pin_memory=True
	)
	img, label = train[0][0], train[0][1]

	return train_dataloader, test_dataloader, img


def train(config):
	device = "cuda" if torch.cuda.is_available() else "cpu"

	if device == "cuda":
		print(f"Device name: {torch.cuda.get_device_name(device.index)}")
		print(f"Device memory: {torch.cuda.get_device_properties(device.index).
			total_memory / 1024 ** 3} GB")
	else:
		print("USING CPU")
		print("NOTE: If you have a GPU, consider using it for training.")

	Path(f"{config["model_folder"]}").mkdir(parents=True, exist_ok=True)

	train_dataloader, test_dataloader, img = get_data(config)
	
	model = AlexNet()
	model.to(device)
	img = img.to(device)
	model(img.unsqueeze(0))



if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	config = get_config()
	train(config)