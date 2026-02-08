# TranSpeckle: Transformers Decode the Chaos in Speckle Pattern Signals

This repository provides **Google Colab–ready Jupyter notebooks** implementing transformer-based architectures for speckle pattern reconstruction, as described in our paper  
**“TranSpeckle: Transformers Decode the Chaos in Speckle Pattern Signals.”**

All experiments are organized in a **one-notebook-per-model** format.  
Each notebook is fully self-contained and can be executed end-to-end in Google Colab.


## Overview

This study investigates whether Transformer-based architectures can provide measurable advantages over conventional CNN frameworks for speckle pattern reconstruction.  
We compare multiple transformer models against a CNN baseline (U-Net with Dense blocks), and show that **TransUNet augmented with Rotary Position Embeddings (RoPE)** achieves superior performance in most experimental conditions.


## Repository Design (Colab-First)

- Designed for **direct execution in Google Colab**
- **No standalone Python scripts** are required
- All code is embedded inside the notebooks
- Each notebook includes:
  - Environment setup
  - Dependency installation
  - Data loading
  - Training and evaluation
- Users may choose between:
  - **Loading pretrained weights**, or
  - **Training from scratch**


## Models

Each model is implemented in a dedicated notebook.

### CNN Baseline
- U-Net architecture with Dense blocks
- Reproduced from Li et al. (2018)
- Implemented in Keras / TensorFlow

### Transformer Architectures (PyTorch)

- **SimpleViT + Linear Decoder**
- **SimpleViT + Attention Decoder**
- **TransUNet ViT-B/32**
- **TransUNet ViT-B/32 + RoPE**
- **TransUNet R50-ViT-B/16 + RoPE**

## Colab Notebooks

Each model is provided as a standalone Google Colab notebook.
Click on any notebook below to open and run it directly in Colab.

| Order | Model | Notebook | Run |
|------|------|----------|-----|
| 01 | CNN U-Net (Baseline) | `01_CNN_U_Net.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zeevikal/TranSpeckle/blob/main/notebooks/01_CNN_U_Net.ipynb) |
| 02 | SimpleViT + Linear Decoder | `02_simpleViT_linear_decoder.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zeevikal/TranSpeckle/blob/main/notebooks/02_simpleViT_linear_decoder.ipynb) |
| 03 | SimpleViT + Attention Decoder | `03_simpleViT_attention_decoder.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zeevikal/TranSpeckle/blob/main/notebooks/03_simpleViT_attention_decoder.ipynb) |
| 04 | TransUNet ViT-B/32 | `04_TransUNet_B32.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zeevikal/TranSpeckle/blob/main/notebooks/04_TransUNet_B32.ipynb) |
| 05 | TransUNet ViT-B/32 + RoPE | `05_TransUNet_B32_with_RoPE.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zeevikal/TranSpeckle/blob/main/notebooks/05_TransUNet_B32_with_RoPE.ipynb) |
| 06 | TransUNet ResNet + RoPE | `06_TransUNet_RES_RoPE.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zeevikal/TranSpeckle/blob/main/notebooks/06_TransUNet_RES_RoPE.ipynb) |


## Key Features

- **Orientation-Aware Positional Embedding**  
  Explicit angular encoding relative to the image center

- **Rotary Position Embeddings (RoPE)**  
  Injected into transformer attention layers

- **Speckle-Specific Augmentation**  
  Border cropping, brightness/contrast variation, and random rotations


## Dataset

The dataset consists of speckle patterns derived from:
- MNIST handwritten digits
- NIST handwritten letters  
- Google QuickDraw sketches

Original dataset from Li et al. (2018): [Deep speckle correlation](https://opg.optica.org/optica/fulltext.cfm?uri=optica-5-10-1181&id=397788)

### Data Organization

Place your data in Google Drive following this structure:
```
/content/drive/MyDrive/TranSpeckle/
├── CNNuNET_checkpoints/     # Saved CNN-UNet model checkpoints
├── data_sets/               # Main datasets directory
│   ├── 2400/                # Full dataset (2400 training samples)
│   │   ├── x_train/
│   │   ├── y_train/
│   │   ├── x_valid/
│   │   └── y_valid/
│   ├── 1600_4/               # Train: 1600 samples, 4 diffusers
│   ├── 1600_2/               # Train: 1600 samples, 2 diffusers
│   ├── 800_4/                # Train: 800 samples, 4 diffusers
│   ├── 800_2/                # Train: 800 samples, 2 diffusers
│   ├── 800_1/                # Train: 800 samples, 1 diffuser
│   ├── group1/               # Test (2000 samples): seen objects, unseen diffusers
│   ├── group2/               # Test (200 samples): unseen objects (same type), unseen diffusers
│   ├── group3/               # Test (800 samples): unseen objects (new types), unseen diffusers
│   ├── group4_1/             # Test (28 samples): unseen objects, same diffuser
│   └── group4_2/             # Test (28 samples): unseen objects, unseen diffuser
├── cache/                    # Preprocessed data cache
└── optuna_logs/              # Optuna logs, checkpoints, and trial outputs
```

## Running the Notebooks

Each notebook is designed to run **end-to-end** in Google Colab.

By default, running all cells will:
- Mount Google Drive
- Install required dependencies
- Load the dataset from Google Drive
- Load the provided pretrained model weights (Pretrained checkpoints are provided separately and should be placed in the corresponding Google Drive directories).
- Run evaluation and visualization

In this default configuration, **no training is performed** and no additional user action is required beyond running the notebook.


## Training from Scratch (Optional)

Each notebook also supports training the model from scratch.

To enable training:
- Locate the training cells in the notebook (clearly marked)
- Remove the leading `#` from the training commands
- Run the training cells

Once enabled, the notebook will:
- Train the model end-to-end
- Save checkpoints and logs to Google Drive
- Optionally evaluate the trained model after training completes

No changes to the code logic are required.


## Project Names and Output Paths

To avoid overwriting existing experiments, each notebook uses configurable:
- Project names
- Output directories (checkpoints, logs, Optuna studies)

Users running multiple experiments are encouraged to:
- Modify the project name variables at the top of the notebook
- Use distinct output directories for different runs

This ensures that checkpoints, logs, and results from different experiments remain isolated.


## Notes on Execution

- All dependencies are installed directly within each notebook
- All paths assume the following Google Drive root:
  
/content/drive/MyDrive/TranSpeckle/

- Some notebooks may optionally clone external repositories (e.g., TransUNet) for reference implementations
- Training time and GPU memory usage depend on the selected model and configuration


## Expected Outputs

During execution, the notebooks may generate:
- Model checkpoints (`.ckpt`)
- Training logs and Optuna databases
- Evaluation metrics (IoU, PCC)
- Visualizations of reconstructed speckle patterns

All outputs are saved to Google Drive under the configured project directories.


## Acknowledgments

This work builds upon the speckle pattern dataset and CNN baseline from Li et al. (2018). We thank the authors for making their data publicly available.
