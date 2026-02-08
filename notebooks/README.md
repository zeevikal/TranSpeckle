# Notebooks

This directory contains the Google Colab notebooks used in the TranSpeckle project.

Each notebook implements a single model and is fully self-contained, including:
- environment setup
- dependency installation
- data loading
- optional training from scratch
- evaluation and visualization

## Notebook Order and Models

The notebooks are organized in execution order and correspond exactly
to the models listed in the main repository README:

- **01 – CNN U-Net (Baseline)**  
  `01_CNN_U_Net.ipynb`

- **02 – SimpleViT + Linear Decoder**  
  `02_simpleViT_linear_decoder.ipynb`

- **03 – SimpleViT + Attention Decoder**  
  `03_simpleViT_attention_decoder.ipynb`

- **04 – TransUNet ViT-B/32**  
  `04_TransUNet_B32.ipynb`

- **05 – TransUNet ViT-B/32 + RoPE**  
  `05_TransUNet_B32_with_RoPE.ipynb`

- **06 – TransUNet ResNet + RoPE**  
  `06_TransUNet_RES_RoPE.ipynb`

## Usage

To run a notebook, open it via the **Open in Colab** links provided
in the main repository README and execute all cells.

By default, notebooks load pretrained models and run evaluation.
Training from scratch can be enabled by uncommenting the relevant cells.
