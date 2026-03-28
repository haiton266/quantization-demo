"""
train_tf_qat.py — LeNet-5 Training Pipeline with QAT (TensorFlow/Keras)

What is this?
-------------
Phase 1: Trains LeNet-5 on the traffic sign dataset normally (float32).
Phase 2: Fine-tunes the saved model for 1 epoch using Quantization Aware
         Training (QAT) via tensorflow-model-optimization, then exports a
         fully-quantized INT8 TFLite model.

Workflow:
  1. Load and split dataset with tf.keras.utils
  2. Build LeNet-5 Sequential model
  3. Train with Adam + CategoricalCrossEntropy, EarlyStopping, ModelCheckpoint
  4. Reload best checkpoint → apply tfmot.quantization.keras.quantize_model
  5. Fine-tune QAT model for 1 epoch
  6. Convert to TFLite with full INT8 quantization and save

Dependencies:
  pip install tensorflow tensorflow-model-optimization
"""
import os
# https://github.com/tensorflow/model-optimization/issues/1140
os.environ['TF_USE_LEGACY_KERAS'] = "1"

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers, models, callbacks

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
INPUT_SHAPE   = (32, 32, 3)
NUM_CLASSES   = 10
BATCH_SIZE    = 32
EPOCHS        = 140          # Phase 1 — hard cap (EarlyStopping will fire first)
QAT_EPOCHS    = 45           # Phase 2 — single fine-tuning epoch
LEARNING_RATE = 1e-3
QAT_LR        = 1e-4         # Lower LR for QAT fine-tune pass

# Output paths
CHECKPOINT_DIR   = "models/checkpoints_tf"
BEST_FLOAT_MODEL = os.path.join(CHECKPOINT_DIR, "best_lenet5_model.keras")
QAT_SAVED_DIR    = os.path.join(CHECKPOINT_DIR, "qat_finetuned.keras")
TFLITE_PATH      = os.path.join(CHECKPOINT_DIR, "lenet5_int8.tflite")


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
def get_lenet5(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """Classic LeNet-5 adapted for 32×32 RGB input."""
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),

        # Block 1
        layers.Conv2D(6, (5, 5), activation="relu", padding="valid"),
        layers.AveragePooling2D(pool_size=(2, 2)),

        # Block 2
        layers.Conv2D(16, (5, 5), activation="relu", padding="valid"),
        layers.AveragePooling2D(pool_size=(2, 2)),

        layers.Flatten(),

        # Fully-connected head
        layers.Dense(120, activation="relu"),
        layers.Dense(84,  activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="lenet5")
    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(base_dir: str = "data", batch_size: int = BATCH_SIZE):
    """
    Loads train/val splits from  <base_dir>/train/  using directory structure.
    Returns (train_ds, val_ds) or (None, None) if the directory is missing.
    """
    train_dir = os.path.join(base_dir, "train")
    if not os.path.exists(train_dir):
        print(f"[WARNING] '{train_dir}' not found — dataset missing.")
        return None, None

    common_kwargs = dict(
        directory=train_dir,
        validation_split=0.2,
        seed=123,
        image_size=(32, 32),
        batch_size=batch_size,
        label_mode="categorical",
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        subset="training", **common_kwargs
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        subset="validation", **common_kwargs
    )

    rescale = layers.Rescaling(1.0 / 255)

    augment = tf.keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1),
    ])

    train_ds = train_ds.map(
        lambda x, y: (augment(rescale(x), training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda x, y: (rescale(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Phase 1 — Standard float32 training
# ---------------------------------------------------------------------------
def phase1_train(train_ds, val_ds) -> tf.keras.Model:
    """Trains LeNet-5 from scratch and saves the best checkpoint."""
    model = get_lenet5(num_classes=NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
        ),
        callbacks.ModelCheckpoint(
            BEST_FLOAT_MODEL,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    print("\n=== Phase 1: Standard Training ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cb_list,
    )
    print(f"[Phase 1] Best model saved → {BEST_FLOAT_MODEL}\n")
    return model


# ---------------------------------------------------------------------------
# Phase 2 — Quantization Aware Training fine-tune (1 epoch)
# ---------------------------------------------------------------------------
def phase2_qat(train_ds, val_ds) -> tf.keras.Model:
    """
    Loads the best float checkpoint, wraps it with QAT fake-quant nodes,
    fine-tunes for 1 epoch, then converts to a fully-quantised TFLite model.
    """
    print("\n=== Phase 2: QAT Fine-tuning ===")

    # ---- 2a. Reload best float model ----
    # tfmot.quantize_model requires a true Sequential or Functional instance.
    # Loading a .keras file can deserialise as a generic Model subclass, so we
    # rebuild the architecture and transfer weights via load_weights() instead.
    print(f"[QAT] Loading weights from {BEST_FLOAT_MODEL} ...")
    float_model = get_lenet5(num_classes=NUM_CLASSES)
    float_model.load_weights(BEST_FLOAT_MODEL)

    # ---- 2b. Wrap with QAT (inserts fake-quantisation ops) ----
    # quantize_model requires a Sequential or Functional model — which
    # get_lenet5() returns — and inserts "quant_" prefixed fake-quant wrappers.
    qat_model = tfmot.quantization.keras.quantize_model(float_model)

    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=QAT_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    qat_model.summary()

    # ---- 2c. Fine-tune for exactly 1 epoch ----
    print(f"[QAT] Fine-tuning for {QAT_EPOCHS} epoch(s)...")
    qat_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=QAT_EPOCHS,
    )

    # ---- 2d. Save the QAT Keras model (optional, useful for inspection) ----
    qat_model.save(QAT_SAVED_DIR)
    print(f"[QAT] QAT Keras model saved → {QAT_SAVED_DIR}")

    return qat_model


# ---------------------------------------------------------------------------
# Phase 3 — TFLite INT8 conversion
# ---------------------------------------------------------------------------
def phase3_tflite(qat_model: tf.keras.Model, val_ds) -> None:
    """
    Converts the QAT model to a fully-quantised INT8 TFLite flatbuffer.
    A small representative dataset is used so activations are also quantised.
    """
    print("\n=== Phase 3: TFLite INT8 Conversion ===")

    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)

    # INT8 optimisation flags
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset — used to calibrate activation ranges
    def representative_dataset():
        # Take a few batches from val_ds (images only, no labels)
        for images, _ in val_ds.take(10):
            for i in range(images.shape[0]):
                yield [tf.expand_dims(images[i], axis=0)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(TFLITE_PATH) / 1024
    print(f"[TFLite] INT8 model saved → {TFLITE_PATH}  ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    train_ds, val_ds = load_data(batch_size=BATCH_SIZE)
    if train_ds is None:
        print("[ERROR] Dataset not found. Aborting.")
        return

    # Phase 1: normal training
    phase1_train(train_ds, val_ds)

    # Phase 2: QAT fine-tune
    qat_model = phase2_qat(train_ds, val_ds)

    # Phase 3: export quantised TFLite model
    phase3_tflite(qat_model, val_ds)

    print("\nAll done.")
    print(f"  Float model  : {BEST_FLOAT_MODEL}")
    print(f"  QAT model    : {QAT_SAVED_DIR}")
    print(f"  TFLite INT8  : {TFLITE_PATH}")


if __name__ == "__main__":
    main()
