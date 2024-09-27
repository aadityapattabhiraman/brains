#!/home/akugyo/Programs/Python/PyTorch/bin/python

import torch
import warnings
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path
from config import get_config, latest_weights_file_path, get_weights_file_path
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

    return train_dataloader, test_dataloader


def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).
            total_memory / 1024 ** 3} GB")
    else:
        print("USING CPU")
        print("NOTE: If you have a GPU, consider using it for training.")

    Path(f"{config["model_folder"]}").mkdir(parents=True, exist_ok=True)

    train_dataloader, test_dataloader = get_data(config)
    model = AlexNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config["preload"]

    if preload == "latest":
        model_filename = latest_weights_file_path(config)
    else:
        if preload:
            model_filename = get_weights_file_path(config, preload)
        else:
            model_filename = None

    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_Step = state["global_step"]

    else:
        print("Starting training from scratch")

    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(initial_epoch, config["epochs"]):
        torch.cuda.empty_cache()
        model.train()
        train_loss, train_acc = 0, 0
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        for X, y in batch_iterator:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            train_loss += loss.item()
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataloader)

        model.eval()
        test_loss, test_acc = 0, 0

        with torch.inference_mode():
            for X, y in tqdm(test_dataloader, desc="Testing Batches"):
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                test_pred_logits = model(X)
                loss = criterion(test_pred_logits, y)
                test_loss += loss.item()
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

        test_loss = test_loss / len(test_dataloader)
        test_acc = test_acc / len(test_dataloader)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)