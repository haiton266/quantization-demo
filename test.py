"""
test.py — Inference and Kaggle Submission Generation (PyTorch)

What is this?
------------
Runs inference on the test set using a trained PyTorch checkpoint and 
generates a CSV file formatted for Kaggle competition submissions.

Workflow:
  1. Load test images from kaggle_testing/test
  2. Load weights from best_edge_model.ckpt
  3. Run batch inference using PyTorch Lightning Trainer
  4. Map predictions to image IDs and save to submission.csv
"""

import os
import torch
import pandas as pd
import pytorch_lightning as pl
from module import TrafficSignLightningModel
from data import get_dataloaders

NUM_CLASSES = 10
BATCH_SIZE = 32

def test():
    print("Loading test dataset from kaggle_testing/...")
    _, _, test_loader = get_dataloaders(batch_size=BATCH_SIZE, num_classes=NUM_CLASSES)
    
    if test_loader is None or len(test_loader) == 0:
        print("Error: Test loader is empty. Check if kaggle_testing/test contains images.")
        return

    checkpoint_path = 'checkpoints/best_edge_model.ckpt'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found. Please train the model first.")
        return
        
    print(f"Loading checkpoint from {checkpoint_path}")
    model = TrafficSignLightningModel.load_from_checkpoint(checkpoint_path)
    
    trainer = pl.Trainer(logger=False, enable_progress_bar=True)
    
    print("\nRunning Inference on Test Set...")
    predictions = trainer.predict(model, test_loader)
    
    # Flatten the list of prediction batches into a single list
    preds = torch.cat(predictions).tolist()
    
    # The order of predictions matches the dataset's file list because shuffle=False in test_loader
    image_files = test_loader.dataset.image_files
    
    # Match the image files to their corresponding IDs (filename without extension)
    # The IDs in sample submission might be zero-padded like '01659'
    ids = [os.path.splitext(f)[0] for f in image_files]
    
    submission_df = pd.DataFrame({'Id': ids, 'Prediction': preds})
    
    # Read the sample submission to maintain order and format
    sample_sub_path = os.path.join('kaggle_testing', 'sample_submission.csv')
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path, dtype={'Id': str})
        submission_df['Id'] = submission_df['Id'].astype(str)
        
        # Merge our predictions into the sample submission
        final_sub = sample_sub.drop(columns=['Label']).merge(
            submission_df, on='Id', how='left'
        )
        # Fill missing with 0 and rename column to Label
        final_sub['Label'] = final_sub['Prediction'].fillna(0).astype(int)
        final_sub = final_sub.drop(columns=['Prediction'])
        
        output_csv = "submission.csv"
        final_sub.to_csv(output_csv, index=False)
        print(f"\nGenerated predictions for {len(preds)} test images.")
        print(f"Saved submission formatted based on sample_submission.csv to {output_csv}")
    else:
        # Fallback if sample_submission.csv doesn't exist
        print(f"Warning: {sample_sub_path} not found. Generating a basic submission CSV.")
        submission_df = submission_df.rename(columns={'Prediction': 'Label'})
        output_csv = "submission.csv"
        submission_df.to_csv(output_csv, index=False)
        print(f"\nGenerated predictions for {len(preds)} test images.")
        print(f"Saved submission to {output_csv}")

if __name__ == "__main__":
    test()
