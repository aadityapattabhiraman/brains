from datasets import load_dataset


cache_dir = "../Datasets/"
dataset = load_dataset("frgfm/imagewoof", "full_size", cache_dir=cache_dir)