"""
train.py — Main Training Entry Point (PyTorch Lightning)

What is this?
------------
The primary script for training the traffic sign classifier using PyTorch 
Lightning. It handles dataset loading, model initialization, and 
automated checkpointing based on validation accuracy.

Workflow:
  1. Load dataloaders from data.py
  2. Initialize TrafficSignLightningModel
  3. Set up EarlyStopping and ModelCheckpoint callbacks
  4. Run trainer.fit() for up to 50 epochs
"""

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from module import TrafficSignLightningModel
from data import get_dataloaders

INPUT_SHAPE = (3, 32, 32)
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 50

def train():
    print("Loading dataset from kaggle_testing/...")
    train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE, num_classes=NUM_CLASSES)
    
    # --- Build Lightning Model ---
    model = TrafficSignLightningModel(num_classes=NUM_CLASSES)
    
    # --- Callbacks ---
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=8,
        mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best_edge_model',
        monitor='val_acc',
        mode='max',
        save_top_k=1
    )
    
    # --- Train Model ---
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=False,
        enable_progress_bar=True
    )
    
    print("\nStarting Training...")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    train()
