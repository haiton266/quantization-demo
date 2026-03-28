# Edge AI Traffic Sign Classifier

This project contains an Edge AI traffic sign classification system designed for resource-constrained hardware like the ESP32-S3. It supports both **PyTorch Lightning** (Separable CNN) and **TensorFlow/Keras** (LeNet-5) implementations, with automated export pipelines to ONNX and TFLite.

## Setup and Installation

This project manages its dependencies and virtual environment using [uv](https://github.com/astral-sh/uv).

Since you have already created the `.venv` using `uv`, you can activate the environment and run the code with the following steps.

Note: my env include Python 3.12

### 1. Activate the Virtual Environment

On **Windows** (using Command Prompt or PowerShell):
```powershell
.venv\Scripts\activate
```

*(On macOS/Linux, it would be `source .venv/bin/activate`)*

### 2. Install Dependencies (If not already installed)

If you haven't yet synchronized your dependencies, install them into the active virtual environment:
```powershell
uv pip install -r requirements.txt
```

## Running the Code

With the virtual environment active, you can start the training process by running one of the training scripts in `scripts/training/`:

### PyTorch Training (Default)
```powershell
python scripts/training/train_pytorch.py
```

### TensorFlow Training
```powershell
python scripts/training/train_tensorflow.py
```

### TensorFlow QAT Training
```powershell
python scripts/training/train_tensorflow_qat_export.py
```

Alternatively, `uv` allows you to run scripts directly without explicitly activating the environment:
```powershell
uv run python scripts/training/train_pytorch.py
```

### Key Features:
1.  **Dual Framework Support:** Fully functional training pipelines for both PyTorch and TensorFlow.
2.  **Lightweight Architectures:** Optimized models (<200k parameters) suitable for ESP32-S3.
3.  **QAT (Quantization-Aware Training):** Native support for TensorFlow QAT to improve INT8 accuracy.
4.  **Automated Export:** Multi-stage conversion from PyTorch/TF to ONNX and TFLite.
5.  **Benchmarking:** Integrated scripts for TFLite INT8 inference benchmarking.

## Project Structure

```text
pioneer/
├── src/                    # Core source code
│   ├── models/             # Model architectures (traffic_sign_cnn.py)
│   ├── data/               # Data loading and preprocessing (dataset.py)
│   └── training/           # Training modules and logic (lightning_module.py)
├── scripts/                # Executable scripts
│   ├── training/           # Training scripts (train_pytorch, train_tensorflow, train_tensorflow_qat_export)
│   ├── evaluation/         # Evaluation, testing, and TFLite benchmarks
│   ├── export/             # Model export and conversion (ONNX, TFLite)
│   └── utils/              # Utility scripts (model_info.py)
├── models/                 # Storage for model checkpoints and exports
│   ├── checkpoints/        # PyTorch .ckpt files
│   ├── checkpoints_tf/     # TensorFlow .keras files
│   └── exports/            # Final ONNX and TFLite models
├── data/                   # Dataset directory (train/test/sample_submission.csv)
├── results/                # Output directory for submission CSVs
├── assets/                 # Project assets (images, documentation)
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
```

## Advanced Usage (Scripts)

You can run specialized scripts for individual tasks:

- **Training:**
  - `python scripts/training/train_pytorch.py`: Standalone PyTorch training.
  - `python scripts/training/train_tensorflow.py`: LeNet-5 training in TensorFlow.
  - `python scripts/training/train_tensorflow_qat_export.py`: Quantization-Aware Training and TFLite export.
- **Evaluation:**
  - `python scripts/evaluation/val_pytorch.py`: Evaluate PyTorch checkpoint on validation set.
  - `python scripts/evaluation/val_tensorflow.py`: Evaluate TensorFlow model on validation set.
  - `python scripts/evaluation/test_pytorch.py`: Generate Kaggle submission from PyTorch model.
  - `python scripts/evaluation/test_tensorflow.py`: Generate Kaggle submission from TensorFlow model.
  - `python scripts/evaluation/test_tflite_int8.py`: Benchmark INT8 quantized TFLite inference.
- **Export:**
  - `python scripts/export/export_onnx.py`: Export PyTorch model to ONNX.
  - `python scripts/export/onnx_to_tflite.py`: Multi-stage conversion (PT -> ONNX -> TF -> TFLite).
  - `python scripts/export/keras_to_tflite.py`: Convert Keras model to TFLite (Float32 & INT8).
- **Utility:**
  - `python scripts/utils/model_info.py`: Detailed architecture and parameter analysis.