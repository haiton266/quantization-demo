"""
export_onnx.py — Export PyTorch Models to ONNX Format

What is this?
------------
Exports a trained PyTorch model to the ONNX (Open Neural Network
Exchange) format. This is an intermediate step for deploying PyTorch
models to hardware-specific runtimes or converting to other formats.

Workflow:
  1. Load the PyTorch Lightning checkpoint
  2. Define dummy input matching the expected shape (1, 3, 32, 32)
  3. Export to ONNX with constant folding and dynamic axes
  4. Report the model footprint (parameters and payload size)
"""

import torch
import os
from module import TrafficSignLightningModel

def export_to_onnx(model, input_shape, export_path="edge_ai_traffic_sign.onnx"):
    """
    Exports the PyTorch model to ONNX.
    For ESP32-S3, one can then convert ONNX -> TensorFlow -> TFLite (INT8).
    """
    model.eval()
    dummy_input = torch.randn(1, *input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX model successfully saved to: {export_path}")
    return export_path

def main():
    INPUT_SHAPE = (3, 32, 32)
    checkpoint_path = 'checkpoints/best_edge_model.ckpt'
    
    print("\nStarting Inference Model Export...")
    if os.path.exists(checkpoint_path):
        best_model = TrafficSignLightningModel.load_from_checkpoint(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}, using uninitialized model.")
        best_model = TrafficSignLightningModel(num_classes=10)
        
    onnx_path = export_to_onnx(best_model.model, INPUT_SHAPE, export_path="edge_ai_traffic_sign.onnx")
    
    param_count = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    
    # --- Resource Footprint Report ---
    onnx_size_bytes = os.path.getsize(onnx_path)
    onnx_size_kb = onnx_size_bytes / 1024
    
    print("\n" + "="*50)
    print("🚀 EDGE AI FOOTPRINT SUMMARY")
    print("="*50)
    print(f"Target Hardware:       ESP32-S3")
    print(f"Model Architecture:    Separable CNN + Global Average Pooling")
    print(f"Parameter Count:       {param_count:,} out of 200,000 max ({param_count/200000*100:.2f}%)")
    print(f"ONNX Payload Size:     {onnx_size_kb:.2f} KB (Before INT8 Quantization)")
    print("="*50)
    
    print("\nDEPLOYMENT NOTES FOR ESP32-S3:")
    print("- To deploy, convert this ONNX model to TensorFlow Edge/TFLite")
    print("- TFLite Micro will run this network within the internal 512KB SRAM.")

if __name__ == "__main__":
    main()
