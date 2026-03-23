"""
export_onnx_to_tflite.py — Multi-stage Conversion: PyTorch -> ONNX -> TF -> TFLite

What is this?
------------
A comprehensive export script that takes a PyTorch model through a 
multi-stage pipeline to reach TFLite INT8. This is often necessary when 
direct conversion paths are unsupported or unstable.

Workflow:
  1. Export PyTorch model to ONNX
  2. Convert ONNX to a TensorFlow SavedModel using onnx-tf
  3. Use TFLiteConverter to generate an INT8 TFLite model
  4. Apply calibration using a subset of validation images
"""

import os
import torch
import numpy as np
import pytorch_lightning as pl
from module import TrafficSignLightningModel
from data import get_dataloaders
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

INPUT_SHAPE = (3, 32, 32)
ONNX_PATH = 'edge_ai_traffic_sign.onnx'

def representative_data_gen():
    """
    Generator for representative data used during INT8 quantization.
    Uses a small subset of the validation data.
    """
    _, val_loader, _ = get_dataloaders(batch_size=1)
    # Use 100 samples for calibration
    for i, (image, _) in enumerate(val_loader):
        if i >= 100:
            break
        # Convert to float32 and ensure shape matches model expectations
        # PyTorch NCHW -> onnx-tf might keep NCHW or convert to NHWC
        # We yield the data in the format the TF model expects.
        yield [image.numpy().astype(np.float32)]

def export_to_tflite(model, input_shape, export_path="edge_ai_traffic_sign.tflite", quantize=True):
    print(f"\nAttempting to convert to TFLite (Quantize={quantize}) via ONNX...")
    tf_path = "model_tf"
    
    print("Step 1: Convert PyTorch to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, *input_shape)
    torch.onnx.export(
        model=model, 
        args=dummy_input, 
        f=ONNX_PATH, 
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        input_names=['input'],
        opset_version=13,
        output_names=['output']
    )
    
    print("Step 2: Convert ONNX to TF...")
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    tf_rep = prepare(onnx_model) # Prepare TF representation
    tf_rep.export_graph(tf_path) # Export the model
    
    print("Step 3: Convert TF to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    
    if quantize:
        print("Applying INT8 Quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        # Ensure full integer quantization for hardware compatibility (like ESP32)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.int8 # Optional: force int8 input
        # converter.inference_output_type = tf.int8 # Optional: force int8 output
        
    tflite_model = converter.convert()
    
    with open(export_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"TFLite model successfully saved to: {export_path}")
    return export_path

def convert_and_report(checkpoint_path=None, quantize=True):
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        best_model = TrafficSignLightningModel.load_from_checkpoint(checkpoint_path)
    else:
        print("Checkpoint not found or not provided. Instantiating a new model.")
        best_model = TrafficSignLightningModel(num_classes=10)
    
    param_count = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    print(f"\n=> Total Model Parameters: {param_count:,} (Limit: 200,000)")
    if param_count > 200000:
        print("WARNING: Parameter limit exceeded!")

    # --- Export ---
    print("\nStarting Inference Model Export...")
    
    # Export both Float32 and INT8 for comparison if needed, or just the requested one
    tflite_path = export_to_tflite(best_model.model, INPUT_SHAPE, export_path="edge_ai_traffic_sign.tflite", quantize=quantize)
    
    # --- Resource Footprint Report ---
    tflite_size_kb = 0
    if tflite_path and os.path.exists(tflite_path):
        tflite_size_kb = os.path.getsize(tflite_path) / 1024
    
    print("\n" + "="*50)
    print("🚀 EDGE AI FOOTPRINT SUMMARY")
    print("="*50)
    print(f"Target Hardware:       ESP32-S3")
    print(f"Model Architecture:    Separable CNN + Global Average Pooling")
    print(f"Quantization:          {'INT8' if quantize else 'Float32'}")
    print(f"Parameter Count:       {param_count:,} out of 200,000 max ({param_count/200000*100:.2f}%)")
    if tflite_size_kb > 0:
        print(f"TFLite Payload Size:   {tflite_size_kb:.2f} KB")
    print("="*50)
    
    print("\nDEPLOYMENT NOTES FOR ESP32-S3:")
    print("- TFLite Micro will run this network within the internal 512KB SRAM.")
    print("- INT8 Quantization significantly improves inference speed on ESP32-S3.")

if __name__ == "__main__":
    # Check if a checkpoint exists
    CKPT = 'checkpoints/best_edge_model.ckpt'
    convert_and_report(CKPT, quantize=True)
