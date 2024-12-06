#!/home/akugyo/Programs/Python/torch/bin/python

import json
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class ImageNet(Dataset):

    def __init__(
            self,
            root,
            json_path,
            transform=None,
            target_transform=None
    ) -> None:

        super().__init__()
        self.data = datasets.ImageFolder(root)
        self.transform = transform
        self.target_transform = target_transform

        with open(json_path, "r") as f:
            self.data_mapping = json.load(f)

        self.data_classes = {class_name: self.data_mapping[class_name]
                             for class_name in self.data.classes}

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        image, label = self.data[index]
        class_name = self.data.classes[label]
        del label
        custom_label = self.data_classes[class_name]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            custom_label = self.target_transform(custom_label)

        return image, custom_label


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.485, .456, .406],
                         std=[.229, .224, .225])
])

train = ImageNet(
    root="../../../Dataset/ImageNet/Train/",
    json_path="../../../Dataset/ImageNet/classes.json",
    transform=transform,
)

train_dataloader = DataLoader(
    train,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)

for image, labels in train_dataloader:
    print(image.shape)
    print(labels)
    break
