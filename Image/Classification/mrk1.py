#!/home/akugyo/Programs/Python/PyTorch/bin/python

from datasets import load_dataset
import matplotlib.pyplot as plt


cache_dir = "../Datasets/"
dataset = load_dataset("frgfm/imagewoof", "full_size", cache_dir=cache_dir)

first_image = dataset["train"][0]["image"]

plt.imshow(first_image)
plt.show()