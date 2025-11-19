"""
Generic Training Function for All Models

Provides unified training interface with Optuna hyperparameter optimization
for all transformer models.

Google Colab Setup:
    !pip install pytorch-lightning optuna

"""

import os
import random
import time
import numpy as np
import torch
import pytorch_lightning as pl
import optuna
from pathlib import Path


def set_seed(seed_value: int = 11):
    """Set all random seeds for reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed_value)


def train_model(
    model_class,
    objective_func,
    dataset_name: str,
    n_trials: int = #20,
    max_epochs: int = #500,
    batch_size: int = #16,
    base_log_dir: str = None
):
    """
    Train model with Optuna hyperparameter optimization
    
    Args:
        model_class: Model class (e.g., LitSimpleViTLinear, LitTransUNet)
        objective_func: Optuna objective function for this model
        dataset_name: Dataset name (e.g., "2400", "1600_4")
        n_trials: Number of Optuna trials
        max_epochs: Maximum training epochs per trial
        batch_size: Batch size for training
        base_log_dir: Base directory for logs and checkpoints
    
    Returns:
        Path to best checkpoint
    """
    set_seed(11)
    
    if base_log_dir is None:
        base_log_dir = f"/content/drive/MyDrive/SpeckleML/optuna_logs/{model_class.__name__}"
    
    Path(base_log_dir).mkdir(parents=True, exist_ok=True)
    
    from modules.preprocessing import load_cached_data
    from modules.datamodule import SpeckleDataModule
    
    print(f" Loading dataset: {dataset_name}")
    x_train, y_train, x_val, y_val = load_cached_data(dataset_name)
    
    dm = SpeckleDataModule(
        train_data=(x_train, y_train),
        val_data=(x_val, y_val),
        batch_size=batch_size,
        use_augmentation=True
    )
    
    db_file_path = Path(base_log_dir) / "study.db"
    storage_path = f"sqlite:////{db_file_path}"
    load_existing = db_file_path.exists()

    study = optuna.create_study(
        study_name=f"{model_class.__name__}_study",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=11),
        storage=storage_path,
        load_if_exists=load_existing
    )

    print(f"\n Starting Optuna optimization for {model_class.__name__}")
    print(f" Dataset: {dataset_name}")
    print(f" Trials: {n_trials}")
    print(f" Max epochs per trial: {max_epochs}")
    print(f" GPU: {torch.cuda.get_device_name(0)}")
    
    start_time = time.time()

    study.optimize(lambda trial: objective_func(trial, dm, base_log_dir), n_trials=n_trials)

    end_time = time.time()
    total_minutes = (end_time - start_time) / 60
    print(f"\n Total optimization time: {int(total_minutes // 60)}h {int((total_minutes % 60))}m {int((total_minutes * 60) % 60)}s")

    best = study.best_trial
    print("\n Best Trial")
    print(f"   Trial {best.number} — val_iou: {best.value:.4f}")
    print(f"   Params: {best.params}")
    
    best_checkpoint = best.user_attrs.get("best_model_path")
    if best_checkpoint:
        print(f"\n Best checkpoint: {best_checkpoint}")
    
    return best_checkpoint


def train_single_model(
    model_instance,
    dataset_name: str,
    max_epochs: int = 350,
    batch_size: int = 16,
    log_dir: str = None
):
    """
    Train a single model instance without Optuna optimization
    
    Args:
        model_instance: Instantiated Lightning model
        dataset_name: Dataset name
        max_epochs: Maximum training epochs
        batch_size: Batch size
        log_dir: Directory for logs and checkpoints
    
    Returns:
        Path to best checkpoint
    """
    set_seed(11)
    
    from modules.preprocessing import load_cached_data
    from modules.datamodule import SpeckleDataModule
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger
    
    if log_dir is None:
        log_dir = f"/content/drive/MyDrive/SpeckleML/logs/{model_instance.__class__.__name__}"
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    print(f" Loading dataset: {dataset_name}")
    x_train, y_train, x_val, y_val = load_cached_data(dataset_name)
    
    dm = SpeckleDataModule(
        train_data=(x_train, y_train),
        val_data=(x_val, y_val),
        batch_size=batch_size,
        use_augmentation=True
    )
    
    logger = CSVLogger(save_dir=log_dir, name=dataset_name, version=".")
    lrmon = LearningRateMonitor(logging_interval="epoch")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, dataset_name, "checkpoints"),
        filename="best",
        monitor="val_iou",
        mode="max",
        save_top_k=1,
        save_last=True
    )
    
    early_stop = EarlyStopping(
        monitor="val_iou",
        patience=30,
        mode="max",
        min_delta=1e-4
    )
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop, lrmon],
        logger=logger,
        accelerator='gpu',
        devices=1,
        enable_progress_bar=True
    )
    
    print(f"\n Training {model_instance.__class__.__name__} on {dataset_name}")
    print(f" GPU: {torch.cuda.get_device_name(0)}")
    
    trainer.fit(model_instance, dm)
    
    print(f"\n Training completed!")
    print(f" Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f" Best val_iou: {checkpoint_callback.best_model_score:.4f}")
    
    return checkpoint_callback.best_model_path
