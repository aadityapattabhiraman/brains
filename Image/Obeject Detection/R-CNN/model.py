import torch.nn.functional as F

class RoIPoolLayer(nn.Module):
    def __init__(self, output_size):
        super(RoIPoolLayer, self).__init__()
        self.output_size = output_size

    def forward(self, feature_map, rois):
        # RoIs is expected to be in the format [batch, x1, y1, x2, y2]
        # Extract regions from the feature map
        pooled_regions = []
        for roi in rois:
            x1, y1, x2, y2 = roi
            region = feature_map[:, y1:y2, x1:x2]
            pooled = F.adaptive_max_pool2d(region, self.output_size)
            pooled_regions.append(pooled)
        
        return torch.stack(pooled_regions)
      
class R_CNN(nn.Module):
    def __init__(self, num_classes=21, roi_pool_size=(7, 7)):
        super(R_CNN, self).__init__()
        self.features = vgg16
        self.roi_pool = RoIPoolLayer(output_size=roi_pool_size)
        self.fc1 = nn.Linear(512 * roi_pool_size[0] * roi_pool_size[1], 4096)
        self.fc2 = nn.Linear(4096, num_classes)
        self.bbox_regressor = nn.Linear(4096, 4)  # To predict bounding box offsets

    def forward(self, images, rois):
        feature_maps = self.features(images)  # Extract features
        pooled_rois = self.roi_pool(feature_maps, rois)  # RoI Pooling
        x = pooled_rois.view(pooled_rois.size(0), -1)  # Flatten for FC layers
        x = F.relu(self.fc1(x))  # First FC layer
        cls_scores = self.fc2(x)  # Classifier output
        bbox_preds = self.bbox_regressor(x)  # Bounding box prediction
        
        return cls_scores, bbox_preds

class ObjectDetectionDataset(Dataset):
    def __init__(self, image_paths, rois, labels):
        self.image_paths = image_paths
        self.rois = rois  # List of region proposals (x1, y1, x2, y2)
        self.labels = labels  # Corresponding labels for each region
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        rois = self.rois[idx]
        labels = self.labels[idx]
        return img, rois, labels
