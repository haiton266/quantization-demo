"""
data.py — PyTorch Data Pipeline for Traffic Sign Recognition

What is this?
------------
This script handles the loading, preprocessing, and augmentation of the 
traffic sign dataset using PyTorch's Dataset and DataLoader. It supports
both the training set (with labels) and the test set (for submissions).

Workflow:
  1. Define TestDataset for unlabeled test images
  2. Define base transforms (Resize 32x32, ToTensor)
  3. Load ImageFolder for training and split into 80/20 train/val
  4. Return DataLoaders for training, validation, and testing
"""

import os
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
import torchvision.transforms.v2 as transforms
from torchvision.datasets import ImageFolder
from PIL import Image

class TestDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        if os.path.exists(directory):
            self.image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        else:
            self.image_files = []

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # dummy label

def get_dataloaders(base_dir="kaggle_testing", batch_size=32, num_classes=10):
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    
    # Base transforms
    base_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((32, 32))
    ])

    if not os.path.exists(train_dir):
        print(f"Warning: {train_dir} not found. Creating dummy data.")
        full_dataset = TensorDataset(torch.rand(1000, 3, 32, 32), torch.randint(0, num_classes, (1000,)))
    else:
        full_dataset = ImageFolder(root=train_dir, transform=base_transform)
        
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    test_dataset = TestDataset(directory=test_dir, transform=base_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers) if len(test_dataset) > 0 else None
    
    return train_loader, val_loader, test_loader
