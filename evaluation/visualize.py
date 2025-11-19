"""
Visualization Tools for Predictions and Training Curves

Provides functions to visualize model predictions, compare models,
and analyze training metrics.

Google Colab Setup:
    !pip install matplotlib numpy torch
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


def show_prediction_example(x_test, y_true, y_pred, index: int = 0, threshold: float = 0.5):
    """
    Display single prediction example with input, ground truth, and prediction
    
    Args:
        x_test: Test speckle patterns
        y_true: Ground truth masks
        y_pred: Predicted masks
        index: Example index to display
        threshold: Binarization threshold
    """
    def to_np(a):
        return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a

    x = to_np(x_test)
    yt = to_np(y_true)
    yp = to_np(y_pred)

    xi, yi, pi = x[index], yt[index], yp[index]

    speckle = xi.squeeze()
    if speckle.ndim == 3:
        speckle = speckle[0] if speckle.shape[0] <= 3 else speckle[..., 0]

    if yi.ndim == 3 and yi.shape[0] == 2:
        gt_mask = np.argmax(yi, axis=0)
    elif yi.ndim == 3 and yi.shape[-1] == 2:
        gt_mask = np.argmax(yi, axis=-1)
    else:
        gt_mask = yi.squeeze().astype(np.int64)

    if pi.ndim == 3 and pi.shape[0] == 2:
        pred_mask = np.argmax(pi, axis=0)
    elif pi.ndim == 3 and pi.shape[-1] == 2:
        pred_mask = np.argmax(pi, axis=-1)
    else:
        pred_mask = (pi.squeeze() >= threshold).astype(np.int64)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(speckle, cmap='gray')
    plt.title("Speckle Input")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask, cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap='gray')
    plt.title("Prediction")
    plt.axis('off')

    plt.suptitle(f"Example {index}", fontsize=14)
    plt.tight_layout()
    plt.show()


def show_prediction_batch(x_test, y_test, y_pred, start: int = 0, batch_size: int = 10):
    """
    Display multiple prediction examples
    
    Args:
        x_test: Test speckle patterns
        y_test: Ground truth masks
        y_pred: Predicted masks
        start: Starting index
        batch_size: Number of examples to show
    """
    total = len(x_test)
    end = min(start + batch_size, total)

    print(f" Showing examples {start} to {end-1} out of {total}")

    for idx in range(start, end):
        print(f"\n--- Example {idx} ---")
        show_prediction_example(x_test, y_test, y_pred, index=idx)


def compare_models_visual(
    x_test,
    y_test,
    predictions_dict: dict,
    index: int = 0,
    threshold: float = 0.5
):
    """
    Compare predictions from multiple models side by side
    
    Args:
        x_test: Test speckle patterns
        y_test: Ground truth masks
        predictions_dict: Dict mapping model names to predictions
        index: Example index
        threshold: Binarization threshold
    """
    def to_np(a):
        return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a

    x = to_np(x_test)
    yt = to_np(y_test)

    xi, yi = x[index], yt[index]

    speckle = xi.squeeze()
    if speckle.ndim == 3:
        speckle = speckle[0] if speckle.shape[0] <= 3 else speckle[..., 0]

    if yi.ndim == 3 and yi.shape[0] == 2:
        gt_mask = np.argmax(yi, axis=0)
    elif yi.ndim == 3 and yi.shape[-1] == 2:
        gt_mask = np.argmax(yi, axis=-1)
    else:
        gt_mask = yi.squeeze().astype(np.int64)

    num_models = len(predictions_dict)
    fig, axes = plt.subplots(1, num_models + 2, figsize=(4 * (num_models + 2), 4))

    axes[0].imshow(speckle, cmap='gray')
    axes[0].set_title("Speckle Input")
    axes[0].axis('off')

    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
        yp = to_np(y_pred)
        pi = yp[index]

        if pi.ndim == 3 and pi.shape[0] == 2:
            pred_mask = np.argmax(pi, axis=0)
        elif pi.ndim == 3 and pi.shape[-1] == 2:
            pred_mask = np.argmax(pi, axis=-1)
        else:
            pred_mask = (pi.squeeze() >= threshold).astype(np.int64)

        axes[i + 2].imshow(pred_mask, cmap='gray')
        axes[i + 2].set_title(model_name)
        axes[i + 2].axis('off')

    plt.suptitle(f"Example {index} - Model Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(results_dict: dict, metric: str = "iou_mean"):
    """
    Plot bar chart comparing metric across models and test groups
    
    Args:
        results_dict: Nested dict {model_name: {test_group: results}}
        metric: Metric to plot ("iou_mean" or "pcc_mean")
    """
    models = list(results_dict.keys())
    test_groups = list(results_dict[models[0]].keys())
    
    x = np.arange(len(test_groups))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, model in enumerate(models):
        values = [results_dict[model][group][metric] for group in test_groups]
        ax.bar(x + i * width, values, width, label=model)
    
    ax.set_xlabel('Test Groups')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{metric.upper()} Comparison Across Test Groups')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(test_groups)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_training_curves(csv_path: str, metrics: list = None):
    """
    Plot training curves from Lightning CSV logs
    
    Args:
        csv_path: Path to metrics.csv file
        metrics: List of metrics to plot
    """
    import pandas as pd
    
    if metrics is None:
        metrics = ['train_loss', 'val_loss', 'train_iou', 'val_iou']
    
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        if metric in df.columns:
            ax.plot(df['epoch'], df[metric], label=metric)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} over epochs')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_per_image_pcc(pcc_per_image: np.ndarray, title: str = "Per-Image PCC Distribution"):
    """
    Plot histogram of per-image PCC scores
    
    Args:
        pcc_per_image: Array of PCC scores per image
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.hist(pcc_per_image, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(pcc_per_image), color='red', linestyle='--', 
                label=f'Mean: {np.mean(pcc_per_image):.3f}')
    plt.xlabel('PCC Score')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
