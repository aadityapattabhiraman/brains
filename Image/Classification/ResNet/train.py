#!/home/akugyo/Programs/Python/torch/bin/python

import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from model import ResNet


class ImageNet(Dataset):

    def __init__(
            self,
            root: str,
            json_path: str,
            transform=None,
            target_transform=None
    ) -> None:
        """
        Takes 4 arguments:
        root: Root Directory,
        json_path: Path of the json file.
        transform: Transformation for data
        target_transform: Transformation for target data
        """

        self.data = datasets.ImageFolder(root=root)
        self.transform = transform
        self.target_transform = target_transform

        with open(json_path, "r") as f:
            self.class_mapping = json.load(f)

        self.custom_labels = {class_name: self.class_mapping[class_name]
                              for class_name in self.data.classes}

    def __len__(self):
        """
        Returns the total number of samples.
        """

        return len(self.data)

    def __getitem__(self, index):
        """
        Returns an image given its index.
        """

        image, label = self.data[index]
        class_name = self.data.classes[label]
        del label

        custom_label = self.custom_labels[class_name]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            image = self.target_transform(custom_label)

        return image, custom_label


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train = ImageNet(
    root="../../../Dataset/ImageNet/Train/",
    json_path="../../../Dataset/ImageNet/classes.json",
    transform=transform)
train_dataloader = DataLoader(
    train,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

for images, labels in train_dataloader:
    print(images.shape)
    print(labels)
    break


def train_loop(num_epochs: int):
    device = "cuda" if torch.cuda.is_available else "cpu"

    if device == "cuda":
        print("Using CUDA")
    else:
        print("Using CPU")
        print("NOTE: If you have a GPU, consider using one.")

    model = ResNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-9)
    loss_fn = nn.CrossEntropyLoss().to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0, 0
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch: {epoch:02d}")

        for X, y in batch_iterator:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataloader)

        print(f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f}")

    torch.save({"epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()}, "model.pth")
