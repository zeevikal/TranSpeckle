"""
Generic Prediction Function for All Models

Generates predictions on test sets for any trained model.
Handles checkpoint loading and batch inference.

Google Colab Setup:
    !pip install torch pytorch-lightning
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


@torch.no_grad()
def predict(
    checkpoint_path: str,
    model_class,
    test_group: str,
    batch_size: int = 100,
    device: str = None
):
    """
    Generate predictions on test set
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        model_class: Model class to load (e.g., LitSimpleViTLinear)
        test_group: Test group name (e.g., "group1", "group2")
        batch_size: Batch size for inference
        device: Device to use (default: auto-detect)
    
    Returns:
        Predictions as numpy array (N, H, W, 2)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    from modules.preprocessing import load_processed_test
    
    print(f" Loading test data from: {test_group}")
    x_test, y_test = load_processed_test(test_group)
    
    x = x_test.float()
    if x.ndim == 3:
        x = x.unsqueeze(1)
    
    print(f" Loading model from checkpoint: {checkpoint_path}")
    model = model_class.load_from_checkpoint(checkpoint_path).to(device).eval()
    
    dl = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    
    print(f" Generating predictions...")
    preds = []
    for i, (xb,) in enumerate(dl):
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        
        if logits.shape[2:] != (256, 256):
            logits = F.interpolate(logits, size=(256, 256), mode="bilinear", align_corners=False)
        
        probs = F.softmax(logits, dim=1).permute(0, 2, 3, 1).cpu()
        preds.append(probs)
        
        if (i + 1) % 10 == 0:
            print(f"   Processed {(i + 1) * batch_size}/{len(x)} samples...")
    
    predictions = torch.cat(preds, dim=0).numpy()
    print(f" Predictions complete! Shape: {predictions.shape}")
    
    return predictions


@torch.no_grad()
def predict_from_data(
    checkpoint_path: str,
    model_class,
    x_test,
    batch_size: int = 100,
    device: str = None
):
    """
    Generate predictions from provided data tensors
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        model_class: Model class to load
        x_test: Test data tensor
        batch_size: Batch size for inference
        device: Device to use
    
    Returns:
        Predictions as numpy array (N, H, W, 2)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    x = torch.as_tensor(x_test).float()
    if x.ndim == 3:
        x = x.unsqueeze(1)
    
    print(f" Loading model from checkpoint: {checkpoint_path}")
    model = model_class.load_from_checkpoint(checkpoint_path).to(device).eval()
    
    dl = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    
    print(f" Generating predictions...")
    preds = []
    for xb, in dl:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        
        if logits.shape[2:] != (256, 256):
            logits = F.interpolate(logits, size=(256, 256), mode="bilinear", align_corners=False)
        
        probs = F.softmax(logits, dim=1).permute(0, 2, 3, 1).cpu()
        preds.append(probs)
    
    predictions = torch.cat(preds, dim=0).numpy()
    print(f" Predictions complete! Shape: {predictions.shape}")
    
    return predictions


def get_best_checkpoint(base_log_dir: str, model_name: str):
    """
    Find best checkpoint from Optuna study
    
    Args:
        base_log_dir: Base logging directory
        model_name: Model name for study lookup
    
    Returns:
        Tuple of (checkpoint_path, trial_number, best_val_iou)
    """
    import optuna
    
    db_file_path = Path(base_log_dir) / "study.db"
    storage_path = f"sqlite:////{db_file_path}"

    try:
        study = optuna.load_study(study_name=f"{model_name}_study", storage=storage_path)
        best = study.best_trial
        cand = best.user_attrs.get("best_model_path") or best.user_attrs.get("last_model_path")
        if cand and Path(cand).exists():
            print(f" Found best checkpoint from Optuna study")
            print(f"   Trial #{best.number} — val_iou: {best.value:.4f}")
            return cand, best.number, best.value
    except Exception as e:
        print(f" Could not load Optuna study: {e}")

    base = Path(base_log_dir)
    candidates = list(base.rglob("checkpoints/best.ckpt"))
    if candidates:
        best_file = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f" Found checkpoint: {best_file}")
        return str(best_file), None, None

    raise FileNotFoundError(f"❌ No checkpoint found under {base_log_dir}")
