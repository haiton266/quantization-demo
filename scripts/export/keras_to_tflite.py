"""
export_keras_to_tflite.py — Convert Keras Models to TFLite (Float32 & INT8)

What is this?
------------
Converts a trained Keras model (.keras) to TensorFlow Lite format. It
generates both a full-precision float32 model and a post-training
quantized (PTQ) INT8 model for deployment on edge devices.

Workflow:
  1. Load the best Keras model from checkpoints_tf/
  2. Convert to standard Float32 TFLite
  3. Load calibration data and apply PTQ INT8 quantization
  4. Run a sanity-check inference on both models
  5. Compare model sizes
"""

import os
import tensorflow as tf
import numpy as np

# --- Paths ---
KERAS_MODEL_PATH = "models/checkpoints_tf/best_lenet5_model.keras"
TFLITE_OUTPUT_DIR = "models/exports/tflite_models"
TFLITE_FLOAT_PATH = os.path.join(TFLITE_OUTPUT_DIR, "lenet5_float32.tflite")
TFLITE_QUANT_PATH = os.path.join(TFLITE_OUTPUT_DIR, "lenet5_int8_quant_without_qat.tflite")

# --- Data config (used for quantization calibration) ---
DATA_DIR = "data/train"
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32
NUM_CALIB_BATCHES = 10  # Number of batches used for post-training quantization calibration


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'.\n"
            f"Please run train_tf.py first to generate the trained model."
        )
    print(f"Loading model from: {path}")
    model = tf.keras.models.load_model(path)
    model.summary()
    return model


def get_calibration_dataset():
    """
    Loads a small subset of the validation set for INT8 quantization calibration.
    Returns a generator that yields batches of preprocessed images.
    """
    if not os.path.exists(DATA_DIR):
        print(f"Warning: Calibration data not found at '{DATA_DIR}'. Skipping INT8 quantization.")
        return None

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    rescale = tf.keras.layers.Rescaling(1.0 / 255)
    val_ds = val_ds.map(lambda x, y: rescale(x))

    def representative_data_gen():
        for i, batch in enumerate(val_ds):
            if i >= NUM_CALIB_BATCHES:
                break
            # TFLite expects a list of numpy arrays, one per input tensor
            yield [batch.numpy().astype(np.float32)]

    return representative_data_gen


def convert_to_float32(model):
    """
    Standard float32 TFLite conversion — no quantization.
    Fastest to convert, largest model size, full precision.
    """
    print("\n--- Converting to Float32 TFLite ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    os.makedirs(TFLITE_OUTPUT_DIR, exist_ok=True)
    with open(TFLITE_FLOAT_PATH, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(TFLITE_FLOAT_PATH) / 1024
    print(f"Float32 model saved to: {TFLITE_FLOAT_PATH}  ({size_kb:.1f} KB)")
    return TFLITE_FLOAT_PATH


def convert_to_int8(model):
    """
    Post-training full integer quantization (INT8).
    Smaller model size (~4x), faster inference on edge devices.
    Requires a small calibration dataset to determine quantization ranges.
    """
    print("\n--- Converting to INT8 Quantized TFLite ---")
    representative_data_gen = get_calibration_dataset()

    if representative_data_gen is None:
        print("Skipping INT8 conversion (no calibration data).")
        return None

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Force full integer quantization (inputs and outputs included)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    os.makedirs(TFLITE_OUTPUT_DIR, exist_ok=True)
    with open(TFLITE_QUANT_PATH, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(TFLITE_QUANT_PATH) / 1024
    print(f"INT8 quantized model saved to: {TFLITE_QUANT_PATH}  ({size_kb:.1f} KB)")
    return TFLITE_QUANT_PATH


def run_inference_test(tflite_path, input_dtype=np.float32):
    """
    Runs a quick sanity-check inference using random input on the TFLite model.
    """
    print(f"\n--- Inference Test: {os.path.basename(tflite_path)} ---")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Input  : shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
    print(f"  Output : shape={output_details[0]['shape']}, dtype={output_details[0]['dtype']}")

    # Create a random test image in the expected dtype
    if input_dtype == np.uint8:
        dummy_input = np.random.randint(0, 255, size=input_details[0]['shape'], dtype=np.uint8)
    else:
        dummy_input = np.random.rand(*input_details[0]['shape']).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output[0])
    print(f"  Predicted class index (random input): {predicted_class}")
    print("  Inference test passed.")


def main():
    # 1. Load the trained Keras model
    model = load_model(KERAS_MODEL_PATH)

    # 2. Convert to Float32 TFLite
    float_path = convert_to_float32(model)

    # 3. Convert to INT8 quantized TFLite
    quant_path = convert_to_int8(model)

    # 4. Sanity-check inference on both models
    print("\n========== Inference Tests ==========")
    run_inference_test(float_path, input_dtype=np.float32)
    if quant_path:
        run_inference_test(quant_path, input_dtype=np.uint8)

    # 5. Size comparison
    print("\n========== Model Size Comparison ==========")
    float_kb = os.path.getsize(float_path) / 1024
    print(f"  Float32 : {float_kb:.1f} KB  →  {float_path}")
    if quant_path:
        quant_kb = os.path.getsize(quant_path) / 1024
        reduction = (1 - quant_kb / float_kb) * 100
        print(f"  INT8    : {quant_kb:.1f} KB  →  {quant_path}")
        print(f"  Size reduction: {reduction:.1f}%")

    print("\nDone. TFLite models are ready for deployment.")


if __name__ == "__main__":
    main()