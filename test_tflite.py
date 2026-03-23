"""
test_tflite.py — TFLite Float32 Inference and Benchmark

What is this?
------------
Runs inference using the TensorFlow Lite Interpreter on the float32 
version of the model. This is used to verify the conversion quality 
from Keras to TFLite before quantization.

Workflow:
  1. Load lenet5_float32.tflite
  2. Allocate tensors and identify I/O details
  3. Preprocess test images to [0, 1] range
  4. Run single-image inference in a loop
  5. Generate submission2.csv
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd

def test_tflite():
    # Configuration
    model_path = "tflite_models/lenet5_float32.tflite"
    test_dir = "kaggle_testing/test"
    output_csv = "submission2.csv"
    
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
    
    # Identify if NCHW or NHWC
    # input_shape is usually [1, C, H, W] or [1, H, W, C]
    is_nchw = (input_shape[1] == 3)
    
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
        
        # Preprocessing: Scale to [0, 1] as per data.py
        img_data = np.array(img).astype(np.float32) / 255.0
        
        if is_nchw:
            # Transpose from (H, W, C) to (C, H, W)
            img_data = np.transpose(img_data, (2, 0, 1))
        
        # Add batch dimension
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
    sample_sub_path = os.path.join('kaggle_testing', 'sample_submission.csv')
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
