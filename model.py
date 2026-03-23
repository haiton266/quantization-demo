"""
model.py — Lightweight Separable CNN Architecture (PyTorch)

What is this?
------------
Defines the core neural network architecture. It uses Depthwise Separable
Convolutions and Global Average Pooling to minimize the parameter count 
(<200k) making it ideal for resource-constrained hardware like ESP32-S3.

Architecture:
  - Input: 3x32x32 RGB images
  - Conv1: Standard convolution for initial feature extraction
  - Block 1 & 2: SeparableConv2d + BatchNorm + MaxPool
  - Global Average Pooling (GAP) instead of large Dense layers
  - Final Linear classifier for traffic sign categories
"""

import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class EdgeTrafficSignCNN(nn.Module):
    """
    Builds a lightweight CNN utilizing Depthwise Separable Convolutions 
    to heavily reduce parameter count and multiply-accumulate operations (MACs).
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        # Block 2
        self.sepconv1 = SeparableConv2d(16, 32)
        self.relu2 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        
        # Block 3
        self.sepconv2 = SeparableConv2d(32, 64)
        self.relu3 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)
        
        # Global Average Pooling replaces dense flattening + dense layer
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier Layer
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.bn1(self.relu2(self.sepconv1(x))))
        x = self.pool3(self.bn2(self.relu3(self.sepconv2(x))))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
