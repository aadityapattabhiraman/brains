from datasets import load_dataset

# Specify the dataset name and cache directory
dataset_name = "glue/mnli"
cache_dir = "../Datasets/"

# Load the dataset
dataset = load_dataset("frgfm/imagewoof", "full_size", cache_dir=cache_dir)