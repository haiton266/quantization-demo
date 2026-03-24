"""
test_tflite_int8.py — TFLite INT8 Quantized Inference and Benchmark

What is this?
------------
Runs inference using the TFLite Interpreter on the fully quantized 
INT8 model. It expects UINT8 inputs, simulating the exact behavior 
on edge microcontrollers.

Workflow:
  1. Load lenet5_int8_quant.tflite
  2. Verify UINT8 input requirement
  3. Load test images without float normalization
  4. Execute inference and generate submission2.csv
"""

import os
# https://github.com/tensorflow/model-optimization/issues/1140
os.environ['TF_USE_LEGACY_KERAS'] = "1"
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd

def test_tflite():
    # Configuration
    # model_path = "models/exports/tflite_models/lenet5_int8_quant_without_qat.tflite"
    # output_csv = "results/submission_tflite_int8_without_qat.csv"
    model_path = "models/checkpoints_tf/lenet5_int8.tflite"
    output_csv = "results/submission_tflite_int8_qat.csv"
    test_dir = "data/test"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    # Load TFLite model and allocate tensors
    print(f"Loading TFLite model from {model_path}...")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    print(f"Input Shape: {input_shape}")
    print(f"Input Dtype: {input_dtype}")
    
    # Robust NCHW detection: channel dim is 1st (after batch) only if last dim != 3
    is_nchw = (input_shape[1] == 3 and input_shape[3] != 3)
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} not found.")
        return

    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    
    if not image_files:
        print(f"No images found in {test_dir}")
        return

    ids = []
    preds = []

    print(f"Running inference on {len(image_files)} images...")
    
    for f in image_files:
        img_path = os.path.join(test_dir, f)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((32, 32))
        
        # For UINT8 input: keep values in [0, 255], no float normalization
        img_data = np.array(img).astype(np.uint8)  # shape: (32, 32, 3)
        
        if is_nchw:
            # Transpose from (H, W, C) to (C, H, W)
            img_data = np.transpose(img_data, (2, 0, 1))
        
        # Add batch dimension -> (1, 32, 32, 3), dtype uint8
        img_data = np.expand_dims(img_data, axis=0)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_data)
        
        # Run inference
        interpreter.invoke()

        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)
        
        ids.append(os.path.splitext(f)[0])
        preds.append(prediction)

    # Prepare submission
    submission_df = pd.DataFrame({'Id': ids, 'Prediction': preds})
    
    # Read the sample submission to maintain order and format
    sample_sub_path = os.path.join('data', 'sample_submission.csv')
    if os.path.exists(sample_sub_path):
        print("Formatting submission based on sample_submission.csv...")
        sample_sub = pd.read_csv(sample_sub_path, dtype={'Id': str})
        submission_df['Id'] = submission_df['Id'].astype(str)
        
        # Merge our predictions into the sample submission
        final_sub = sample_sub.drop(columns=['Label'], errors='ignore').merge(
            submission_df, on='Id', how='left'
        )
        # Fill missing with 0 and rename column to Label
        final_sub['Label'] = final_sub['Prediction'].fillna(0).astype(int)
        final_sub = final_sub[['Id', 'Label']]
        
        final_sub.to_csv(output_csv, index=False)
        print(f"Generated predictions for {len(preds)} test images.")
        print(f"Saved submission to {output_csv}")
    else:
        submission_df = submission_df.rename(columns={'Prediction': 'Label'})
        submission_df.to_csv(output_csv, index=False)
        print(f"Saved basic submission to {output_csv}")

if __name__ == "__main__":
    test_tflite()