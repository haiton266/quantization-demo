"""
module.py — PyTorch Lightning System for Traffic Sign Training

What is this?
------------
Wraps the core CNN in a LightningModule to automate training loops, 
validation benchmarks, and optimizer configuration. It also defines the
augmentation strategy used during backpropagation.

Key Components:
  - TrafficSignLightningModel: The main PL system
  - Augmentation: Random rotations, affine transforms, and color jitter
  - Loss: CrossEntropyLoss
  - Optimization: Adam optimizer with configurable learning rate
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.transforms.v2 as transforms
from model import EdgeTrafficSignCNN

class TrafficSignLightningModel(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = EdgeTrafficSignCNN(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        
        # Data Augmentation pipeline for small 32x32 images
        self.train_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=36),
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1),
        ])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.train_transforms(x)
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
