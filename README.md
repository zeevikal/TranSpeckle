# TranSpeckle: Transformers Decode the Chaos in Speckle Pattern Signals
This repository contains the implementation of transformer-based architectures for speckle pattern reconstruction, as described in our paper "TranSpeckle: Transformers Decode the Chaos in Speckle Pattern Signals".

## Overview
This study evaluates whether Transformer-based models can provide measurable advantages over conventional CNN frameworks for speckle pattern analysis in reconstruction tasks. We compare four transformer architectures against a CNN baseline (U-Net with Dense blocks), demonstrating that TransUNet with Rotary Position Embeddings (RoPE) achieves superior performance in most experimental conditions.

## Models

### CNN Baseline
- U-Net architecture with Dense blocks
- Reproduced from Li et al. (2018)
- Implemented in Keras/TensorFlow

### Transformer Architectures

All implemented in PyTorch:
1. **SimpleViT + Linear Decoder**: Simplified Vision Transformer with direct linear projection
2. **SimpleViT + Attention Decoder**: SimpleViT with shallow self-attention decoder
3. **TransUNet**: Hybrid ViT-B/32 + U-Net decoder architecture
4. **TransUNet + RoPE**: TransUNet enhanced with Rotary Position Embeddings

## Key Features

- **Orientation-Aware Positional Embedding**: Explicit angular encoding relative to image center
- **Speckle-Specific Augmentation**: Border cropping, brightness/contrast variation, random rotations

## Dataset

The dataset consists of speckle patterns derived from:
- MNIST handwritten digits
- NIST handwritten letters  
- Google QuickDraw sketches

Original dataset from Li et al. (2018): [Deep speckle correlation](https://opg.optica.org/optica/fulltext.cfm?uri=optica-5-10-1181&id=397788)

### Data Organization

Place your data in Google Drive following this structure:
```
/content/drive/MyDrive/SpeckleML/
├── data sets/
│   ├── 2400/           # Full dataset (2400 training samples)
│   │   ├── x_train/
│   │   ├── y_train/
│   │   ├── x_valid/
│   │   └── y_valid/
│   ├── 1600_4/         # Train: 1600 samples, 4 diffusers
│   ├── 1600_2/         # Train: 1600 samples, 2 diffusers
│   ├── 800_4/          # Train: 800 samples, 4 diffusers
│   ├── 800_2/          # Train: 800 samples, 2 diffusers
│   ├── 800_1/          # Train: 800 samples, 1 diffuser
│   ├── group1/         # Test(2000 samples): seen objects, unseen diffusers
│   ├── group2/         # Test(200 samples): unseen objects (same type), unseen diffusers
│   ├── group3/         # Test(800 samples): unseen objects (new types), unseen diffusers
│   ├── group4_1/       # Test(28 samples): unseen objects, same diffuser
│   └── group4_2/       # Test(28 samples): unseen objects, unseen diffuser
├── cache/              # Preprocessed data cache
└── optuna_logs/        # Training logs and checkpoints
```

## Usage

### Data Preprocessing
```python
from data.preprocessing import load_processed_train_valid, load_processed_test

# Load training and validation data
x_train, y_train, x_val, y_val = load_processed_train_valid("2400")

# Load test data
x_test, y_test = load_processed_test("group1")
```

### Training

#### CNN Baseline
```python
python training/train_cnn.py --dataset 2400 --epochs 350 --batch_size 16
```

#### SimpleViT Models
```python
python training/train_simple_vit.py --model linear --dataset 2400 --epochs 500
python training/train_simple_vit.py --model attention --dataset 2400 --epochs 500
```

#### TransUNet Models
```python
python training/train_transunet.py --model base --dataset 2400 --epochs 500
python training/train_transunet.py --model rope --dataset 2400 --epochs 500
```

### Hyperparameter Optimization
```python
python training/optuna_tuning.py --model transunet --dataset 2400 --n_trials 20
```

### Evaluation
```python
from evaluation.evaluate import evaluate_model

# Evaluate on test groups
results = evaluate_model(
    model_path="/path/to/checkpoint.ckpt",
    test_group="group1",
    model_type="transunet"
)
```

### Evaluation Metrics

- **IoU (Intersection over Union)**: Measures pixel-level overlap of predicted and ground truth foreground regions
- **PCC (Pearson Correlation Coefficient)**: Measures linear correlation between predicted and ground truth pixel intensities


## Project Structure
```
TranSpeckle/
├── README.md
├── requirements.txt
├── data/
│   └── preprocessing.py          # Data loading and preprocessing
├── models/
│   ├── cnn_baseline.py          # CNN U-Net with Dense blocks
│   ├── simple_vit.py            # SimpleViT architectures
│   ├── transunet.py             # TransUNet and TransUNet+RoPE
│   └── modules/
│       ├── orientation_pe.py    # Orientation positional embedding
│       ├── augmentation.py      # Data augmentation
│       └── metrics.py           # Evaluation metrics
├── training/
│   ├── train_cnn.py            # CNN training script
│   ├── train_simple_vit.py     # SimpleViT training script
│   ├── train_transunet.py      # TransUNet training script
│   └── optuna_tuning.py        # Hyperparameter optimization
├── evaluation/
│   ├── evaluate.py             # Model evaluation
│   └── visualize.py            # Results visualization
└── notebooks/                   # Jupyter notebooks for each model
    ├── CNN_Baseline.ipynb
    ├── SimpleViT.ipynb
    ├── TransUNet.ipynb
    └── TransUNet_RoPE.ipynb
```


## Acknowledgments

This work builds upon the speckle pattern dataset and CNN baseline from Li et al. (2018). We thank the authors for making their data publicly available.
