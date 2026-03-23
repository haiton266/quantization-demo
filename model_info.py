"""
model_info.py — Architecture Analysis and Footprint Estimation

What is this?
------------
A utility script to inspect the model's structure, calculate parameter
counts, estimate file sizes (FP32 vs INT8), and verify tensor shapes
via a dummy forward pass.

Feature:
  1. Prints full layer-by-layer architecture
  2. Calculates total vs trainable parameters
  3. Estimates binary size on disk
  4. Validates input/output shape consistency
"""

import os
import torch
from model import EdgeTrafficSignCNN

def get_model_info():
    # Instantiate the model
    # Note: adjust num_classes if your dataset differs
    model = EdgeTrafficSignCNN(num_classes=10)
    
    print("="*40)
    print(" MODEL ARCHITECTURE")
    print("="*40)
    print(model)
    print("\n")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("="*40)
    print(" PARAMETER INFO")
    print("="*40)
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"Non-trainable Params:  {total_params - trainable_params:,}")
    print("\n")
    
    # Size estimation
    # float32 = 4 bytes per parameter
    param_size_mb = (total_params * 4) / (1024 ** 2)
    # INT8 = 1 byte per parameter
    param_size_kb_int8 = total_params / 1024
    
    print("="*40)
    print(" ESTIMATED MODEL SIZE (WEIGHTS ONLY)")
    print("="*40)
    print(f"FP32 Size:  {param_size_mb:.4f} MB")
    print(f"INT8 Size:  {param_size_kb_int8:.2f} KB")
    print("\n")
    
    # Dummy forward pass to check I/O shapes
    # Assuming input size of 32x32 RGB images (common for traffic signs e.g., GTSRB, CIFAR)
    try:
        dummy_input = torch.randn(1, 3, 32, 32)
        output = model(dummy_input)
        
        print("="*40)
        print(" TENSOR SHAPES (Using 1x3x32x32 input)")
        print("="*40)
        print(f"Input Shape:  {tuple(dummy_input.shape)}")
        print(f"Output Shape: {tuple(output.shape)}")
    except Exception as e:
        print(f"Failed to run dummy inference: {e}")

if __name__ == '__main__':
    get_model_info()
