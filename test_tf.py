"""
test_tf.py — Evaluation and Submission Generation (TensorFlow/Keras)

What is this?
------------
Evaluates a trained Keras model (.keras) on the test dataset and exports
predictions to a submission CSV. Use this for models trained via train_tf.py.

Workflow:
  1. Load and preprocess test images
  2. Load the best Keras model from checkpoints_tf/
  3. Run model.predict() on the entire test batch
  4. Generate and save submission_tf.csv
"""

import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import models, layers
from train_tf import get_lenet5, NUM_CLASSES, BATCH_SIZE

def load_test_images(test_dir="kaggle_testing/test", image_size=(32, 32)):
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = []
    ids = []
    
    # Load and preprocess images
    rescale = layers.Rescaling(1./255)
    
    for filename in image_files:
        img_path = os.path.join(test_dir, filename)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
        ids.append(os.path.splitext(filename)[0])
    
    # Convert list to numpy array and rescale (normalize)
    images = np.array(images)
    images = rescale(images)
    
    return images, ids

def test():
    print("Loading test dataset from kaggle_testing/test...")
    test_dir = "kaggle_testing/test"
    
    if not os.path.exists(test_dir):
        print(f"Error: {test_dir} not found. Check if kaggle_testing/test contains images.")
        return

    images, ids = load_test_images(test_dir=test_dir)
    
    if len(images) == 0:
        print("No test images found.")
        return

    checkpoint_path = 'checkpoints_tf/best_lenet5_model.keras'
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found. Please train the model first.")
        return
        
    print(f"Loading best model from {checkpoint_path}")
    model = models.load_model(checkpoint_path)
    
    print("\nRunning TensorFlow Inference on Test Set...")
    predictions = model.predict(images, batch_size=BATCH_SIZE)
    preds = np.argmax(predictions, axis=1)
    
    # Generate submission CSV logic to match sample submission order
    submission_df = pd.DataFrame({'Id': ids, 'Prediction': preds})
    
    sample_sub_path = os.path.join('kaggle_testing', 'sample_submission.csv')
    output_csv = "submission_tf.csv"
    
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path, dtype={'Id': str})
        submission_df['Id'] = submission_df['Id'].astype(str)
        
        final_sub = sample_sub.drop(columns=['Label']).merge(
            submission_df, on='Id', how='left'
        )
        final_sub['Label'] = final_sub['Prediction'].fillna(0).astype(int)
        final_sub = final_sub.drop(columns=['Prediction'])
        
        final_sub.to_csv(output_csv, index=False)
        print(f"\nGenerated predictions for {len(preds)} test images.")
        print(f"Saved submission formatted based on sample_submission.csv to {output_csv}")
    else:
        print(f"Warning: {sample_sub_path} not found. Generating a basic submission CSV.")
        submission_df = submission_df.rename(columns={'Prediction': 'Label'})
        submission_df.to_csv(output_csv, index=False)
        print(f"\nSaved basic submission to {output_csv}")

if __name__ == "__main__":
    test()
