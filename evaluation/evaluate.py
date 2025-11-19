"""
Model Evaluation with IoU and PCC Metrics

Evaluates model predictions using Intersection over Union (IoU) and
Pearson Correlation Coefficient (PCC).

Google Colab Setup:
    !pip install torch torchmetrics numpy
"""

import numpy as np
import torch
from pathlib import Path


def evaluate_model(
    checkpoint_path: str,
    model_class,
    test_group: str,
    batch_size: int = 100
):
    """
    Evaluate model on test set
    
    Args:
        checkpoint_path: Path to trained checkpoint
        model_class: Model class to load
        test_group: Test group name (e.g., "group1", "group2", "group3")
        batch_size: Batch size for inference
    
    Returns:
        Dictionary with IoU and PCC metrics
    """
    from modules.preprocessing import load_processed_test
    from modules.metrics import evaluate_predictions
    from prediction.predict import predict
    
    print(f"\n{'='*60}")
    print(f"Evaluating on {test_group}")
    print('='*60)
    
    x_test, y_test = load_processed_test(test_group)
    
    y_pred = predict(
        checkpoint_path=checkpoint_path,
        model_class=model_class,
        test_group=test_group,
        batch_size=batch_size
    )
    
    def to_numpy_gt(y):
        y_np = y.numpy() if isinstance(y, torch.Tensor) else y
        if y_np.ndim == 4 and y_np.shape[1] == 2:
            y_np = np.transpose(y_np, (0, 2, 3, 1))
        return y_np

    y_true_np = to_numpy_gt(y_test)
    
    print(f"\n Evaluating predictions...")
    results = evaluate_predictions(y_pred=y_pred, y_true=y_true_np, threshold=0.5)

    print("\n--- Summary ---")
    print(f"Mean IoU: {results['iou_mean']:.3f}")
    print(f"Mean PCC: {results['pcc_mean']:.3f}")
    
    return results


def evaluate_all_groups(
    checkpoint_path: str,
    model_class,
    test_groups: list = None,
    batch_size: int = 100
):
    """
    Evaluate model on all test groups
    
    Args:
        checkpoint_path: Path to trained checkpoint
        model_class: Model class to load
        test_groups: List of test groups to evaluate
        batch_size: Batch size for inference
    
    Returns:
        Dictionary with results for each test group
    """
    if test_groups is None:
        test_groups = ["group1", "group2", "group3", "group4_1", "group4_2"]
    
    all_results = {}
    for group in test_groups:
        print(f"\n{'='*60}")
        print(f"Evaluating on {group}")
        print('='*60)
        results = evaluate_model(
            checkpoint_path=checkpoint_path,
            model_class=model_class,
            test_group=group,
            batch_size=batch_size
        )
        all_results[group] = results
    
    print(f"\n{'='*60}")
    print("All Results Summary")
    print('='*60)
    for group, results in all_results.items():
        print(f"{group:10s} | IoU: {results['iou_mean']:.3f} | PCC: {results['pcc_mean']:.3f}")
    
    return all_results


def compare_models(
    checkpoints: dict,
    test_group: str,
    batch_size: int = 100
):
    """
    Compare multiple models on same test set
    
    Args:
        checkpoints: Dict mapping model names to (checkpoint_path, model_class) tuples
        test_group: Test group to evaluate on
        batch_size: Batch size for inference
    
    Returns:
        Dictionary with results for each model
    """
    from modules.preprocessing import load_processed_test
    from modules.metrics import evaluate_predictions
    from prediction.predict import predict
    
    print(f"\n Comparing {len(checkpoints)} models on {test_group}")
    
    x_test, y_test = load_processed_test(test_group)
    
    def to_numpy_gt(y):
        y_np = y.numpy() if isinstance(y, torch.Tensor) else y
        if y_np.ndim == 4 and y_np.shape[1] == 2:
            y_np = np.transpose(y_np, (0, 2, 3, 1))
        return y_np

    y_true_np = to_numpy_gt(y_test)
    
    results = {}
    for model_name, (checkpoint_path, model_class) in checkpoints.items():
        print(f"\n--- {model_name} ---")
        
        y_pred = predict(
            checkpoint_path=checkpoint_path,
            model_class=model_class,
            test_group=test_group,
            batch_size=batch_size
        )
        
        model_results = evaluate_predictions(y_pred=y_pred, y_true=y_true_np, threshold=0.5)
        results[model_name] = model_results
    
    print(f"\n{'='*60}")
    print(f"Comparison Summary - {test_group}")
    print('='*60)
    print(f"{'Model':<30} {'IoU':>8} {'PCC':>8}")
    print("-" * 60)
    for model_name, model_results in results.items():
        print(f"{model_name:<30} {model_results['iou_mean']:>8.3f} {model_results['pcc_mean']:>8.3f}")
    
    return results
