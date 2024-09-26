import os
from pathlib import Path


def get_config() -> dict:
	return {
		"batch_size": 4096,
		"num_workers": os.cpu_count(),
		"epochs": 15,
		"lr": 10**-4,
		"model_folder": "weights",
		"model_basename": "tmodel_",
		"preload": "latest",
		"root": "../../Datasets/Imagenet-1k/"
	}

def get_weights_file_path(config, epoch: str):
	model_folder = f"{config["model_folder"]}"
	model_filename = f"{config["model_basename"]}{epoch}.pth"
	return str(Path(".")/model_folder/model_filename)

def latest_weights_file_path(config):
	model_folder = f"{config["model_folder"]}"
	model_filename = f"{config["model_basename"]}"
	weight_files = list(Path(model_folder).glob(model_filename))

	if len(weight_files) == 0:
		return None
	weight_files.sort()
	return str(weight_files[-1])