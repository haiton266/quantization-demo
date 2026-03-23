"""
val_tf.py — Quick Validation Benchmark (TensorFlow/Keras)

What is this?
------------
Evaluates the best trained Keras model on the reserved validation 
split to report accuracy and loss metrics.

Workflow:
  1. Load data via load_data() from train_tf.py
  2. Load best_lenet5_model.keras
  3. Run model.evaluate()
  4. Print results
"""

import tensorflow as tf
from tensorflow.keras import models, layers
import os
from train_tf import get_lenet5, load_data, BATCH_SIZE, NUM_CLASSES

def val():
    print("Loading data for TensorFlow Validation...")
    _, val_ds = load_data(batch_size=BATCH_SIZE)
    
    if val_ds is None:
        print("Dataset not found. Exiting.")
        return

    checkpoint_path = 'checkpoints_tf/best_lenet5_model.keras'
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found. Please train the model first.")
        return

    print(f"Loading best model from {checkpoint_path}")
    model = models.load_model(checkpoint_path)
    
    print("\nStarting TensorFlow Evaluation...")
    results = model.evaluate(val_ds)
    
    print(f"\nValidation Loss: {results[0]:.4f}")
    print(f"Validation Accuracy: {results[1]:.4f}")

if __name__ == "__main__":
    val()
