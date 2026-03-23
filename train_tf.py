"""
train_tf.py — LeNet-5 Training Pipeline (TensorFlow/Keras)

What is this?
------------
Trains or fine-tunes a LeNet-5 architecture on the traffic sign dataset
using Keras. This script is used as the baseline for TFLite and QAT 
experiments.

Workflow:
  1. Load and split dataset with tf.keras.utils
  2. Build LeNet-5 Sequential model
  3. Apply Adam optimizer and Categorical CrossEntropy
  4. Train with EarlyStopping and save the best .keras model
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os

# --- Model Parameters ---
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 5e-4

def get_lenet5(num_classes=10):
    """
    Classic LeNet-5 architecture adapted for RGB images.
    """
    model = models.Sequential([
        # Explicit Input layer (avoids input_shape deprecation warning)
        layers.Input(shape=INPUT_SHAPE),

        # Layer 1: Convolution
        layers.Conv2D(6, (5, 5), activation='relu', padding='valid'),
        # Layer 2: Pooling
        layers.AveragePooling2D(pool_size=(2, 2)),

        # Layer 3: Convolution
        layers.Conv2D(16, (5, 5), activation='relu', padding='valid'),
        # Layer 4: Pooling
        layers.AveragePooling2D(pool_size=(2, 2)),

        layers.Flatten(),

        # Layer 5: Fully Connected
        layers.Dense(120, activation='relu'),
        # Layer 6: Fully Connected
        layers.Dense(84, activation='relu'),
        # Layer 7: Output
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def load_data(base_dir="kaggle_testing", batch_size=32):
    train_dir = os.path.join(base_dir, "train")

    if not os.path.exists(train_dir):
        print(f"Warning: {train_dir} not found. Training logic might fail or use dummy data.")
        return None, None

    # Load and split dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(32, 32),
        batch_size=batch_size,
        label_mode='categorical'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(32, 32),
        batch_size=batch_size,
        label_mode='categorical'
    )

    # Normalization and Augmentation
    rescale = layers.Rescaling(1./255)

    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1),
    ])

    train_ds = train_ds.map(lambda x, y: (data_augmentation(rescale(x), training=True), y))
    val_ds = val_ds.map(lambda x, y: (rescale(x), y))

    # Prefetch for performance
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds

def train():
    print("Loading data for TensorFlow...")
    train_ds, val_ds = load_data(batch_size=BATCH_SIZE)

    if train_ds is None:
        print("Dataset not found. Exiting.")
        return

    # --- Build and Compile Model ---
    model = get_lenet5(num_classes=NUM_CLASSES)

    # Load from pretrained if requested (Uncomment and adjust path if you have weights)
    # weights_path = 'pretrained_lenet5.weights.h5'
    # if os.path.exists(weights_path):
    #     print(f"Loading pretrained weights from {weights_path}")
    #     model.load_weights(weights_path)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # --- Callbacks ---
    checkpoint_dir = 'checkpoints_tf'
    os.makedirs(checkpoint_dir, exist_ok=True)

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint(
        os.path.join(checkpoint_dir, 'best_lenet5_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    print("\nStarting TensorFlow Training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint]
    )

    print(f"\nTraining complete. Best model saved in {checkpoint_dir}")

if __name__ == "__main__":
    train()