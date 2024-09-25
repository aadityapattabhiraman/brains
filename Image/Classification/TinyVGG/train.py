#!/home/akugyo/Programs/Python/PyTorch/bin/python

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from model import TinyVGG
from config import get_config


device = "cuda" if torch.cuda.is_available() else "cpu"
config = get_config()

transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

train = datasets.ImageFolder(
    root="../../Datasets/Imagenet-1k/Train/",
    transform=transform
)

test = datasets.ImageFolder(
    root="../../Datasets/Imagenet-1k/Test/",
    transform=transform
)

class_names = train.classes 
model = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train_dataloader = DataLoader(
    dataset=train,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
    pin_memory=True
)

test_dataloader = DataLoader(
    dataset=test,
    shuffle=False,
    num_workers=config["num_workers"],
    pin_memory=True
)


results = {
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": []
}

model.to(device)

for epoch in tqdm(range(config["epochs"]), desc="Training Epochs"):
    model.train()
    train_loss, train_acc = 0, 0

    for X, y in tqdm(train_dataloader, desc="Training Batches"):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    results["train_loss"].append(train_loss.item() 
        if isinstance(train_loss, torch.Tensor) else train_loss)
    results["train_acc"].append(train_acc.item() 
        if isinstance(train_acc, torch.Tensor) else train_acc)
    results["test_loss"].append(test_loss.item() 
        if isinstance(test_loss, torch.Tensor) else test_loss)
    results["test_acc"].append(test_acc.item() 
        if isinstance(test_acc, torch.Tensor) else test_acc)


torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    }, "checkpoint.pth")