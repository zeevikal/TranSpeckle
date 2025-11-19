"""
Data Preprocessing for Speckle Pattern Reconstruction

Loads, preprocesses, and caches speckle pattern data from Google Drive.
Handles 2x2 binning, normalization, and one-hot encoding.

Google Colab Setup:
    !pip install opencv-python torch

Data Structure:
    /content/drive/MyDrive/SpeckleML/data sets/
    ├── 2400/, 1600_4/, 1600_2/, 800_4/, 800_2/, 800_1/
    ├── group1/, group2/, group3/, group4_1/, group4_2/

"""

import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F


def binning_2x2(image: np.ndarray) -> np.ndarray:
    """Downsample image from 512x512 to 256x256 using 2x2 binning"""
    return image.reshape((image.shape[0] // 2, 2, image.shape[1] // 2, 2)).mean(axis=(1, 3))


def load_processed_train_valid(dataset_name: str, base_path: str = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and preprocess training and validation data
    
    Args:
        dataset_name: Dataset folder name (e.g., "2400", "1600_4")
        base_path: Base path to data (default: Google Drive)
    
    Returns:
        x_train, y_train, x_valid, y_valid as torch tensors
    """
    if base_path is None:
        base_path = Path('/content/drive/MyDrive/SpeckleML/data sets') / dataset_name
    else:
        base_path = Path(base_path) / dataset_name

    def process_pair(x_dir: Path, y_dir: Path, label: str) -> Tuple[torch.Tensor, torch.Tensor]:
        x_list, y_list = [], []
        x_files = sorted(x_dir.glob('*.tif'))
        print(f" Found {len(x_files)} files in '{label}'")

        t0 = time.time()
        for i, x_file in enumerate(x_files):
            fname = x_file.name
            y_file = y_dir / fname
            if not y_file.exists():
                print(f" Missing GT for: {fname}")
                continue

            x = cv2.imread(str(x_file), cv2.IMREAD_UNCHANGED)
            if len(x.shape) == 3:
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

            x = binning_2x2(x)
            x = (x / np.max(x)).astype(np.float32) if np.max(x) > 0 else x.astype(np.float32)
            x = np.expand_dims(x, axis=0)

            y = cv2.imread(str(y_file), cv2.IMREAD_UNCHANGED)
            if len(y.shape) == 3:
                y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)

            y = binning_2x2(y)
            y_binary = (y > 0).astype(np.uint8)
            y_tensor = torch.from_numpy(y_binary).long()
            y_onehot = F.one_hot(y_tensor, num_classes=2).permute(2, 0, 1).float()

            x_list.append(torch.from_numpy(x))
            y_list.append(y_onehot)

            if i % 100 == 0 and i > 0:
                print(f"⏳ Processed {i}/{len(x_files)} files in '{label}'")

        print(f" Done loading '{label}' in {time.time() - t0:.1f} seconds")
        return torch.stack(x_list), torch.stack(y_list)

    print(f" Starting to load and preprocess training and validation sets...")
    x_train, y_train = process_pair(base_path / 'x_train', base_path / 'y_train', 'train')
    x_valid, y_valid = process_pair(base_path / 'x_valid', base_path / 'y_valid', 'valid')

    print(f" Loaded all data from '{dataset_name}' — train: {x_train.shape[0]}, valid: {x_valid.shape[0]}")
    return x_train, y_train, x_valid, y_valid


def load_processed_test(dataset_name: str, base_path: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and preprocess test data
    
    Args:
        dataset_name: Test group name (e.g., "group1", "group2")
        base_path: Base path to data (default: Google Drive)
    
    Returns:
        x_test, y_test as torch tensors
    """
    if base_path is None:
        base_path = Path('/content/drive/MyDrive/SpeckleML/data sets') / dataset_name
    else:
        base_path = Path(base_path) / dataset_name
        
    x_path = base_path / 'x_test'
    y_path = base_path / 'y_test'

    x_list, y_list = [], []
    x_files = sorted(x_path.glob('*.tif'))
    print(f"\n Starting to load and preprocess test set from '{dataset_name}'")
    print(f" Found {len(x_files)} test files")

    t0 = time.time()
    for i, x_file in enumerate(x_files):
        fname = x_file.name
        y_file = y_path / fname

        if not y_file.exists():
            print(f" Missing GT for: {fname}")
            continue

        x = cv2.imread(str(x_file), cv2.IMREAD_UNCHANGED)
        if len(x.shape) == 3:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

        x = binning_2x2(x)
        x = (x / np.max(x)).astype(np.float32) if np.max(x) > 0 else x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        y = cv2.imread(str(y_file), cv2.IMREAD_UNCHANGED)
        if len(y.shape) == 3:
            y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)

        y = binning_2x2(y)
        y_binary = (y > 0).astype(np.uint8)
        y_tensor = torch.from_numpy(y_binary).long()
        y_onehot = F.one_hot(y_tensor, num_classes=2).permute(2, 0, 1).float()

        x_list.append(torch.from_numpy(x))
        y_list.append(y_onehot)

        if i % 50 == 0 and i > 0:
            print(f" Processed {i}/{len(x_files)} files")

    print(f" Done loading test set in {time.time() - t0:.1f} seconds")
    print(f" Total loaded: {len(x_list)} examples")
    return torch.stack(x_list), torch.stack(y_list)


def save_processed_data(x_train, y_train, x_valid, y_valid, dataset_name: str, cache_dir: str = None):
    """
    Save preprocessed data to cache for faster loading
    
    Args:
        x_train, y_train, x_valid, y_valid: Preprocessed tensors
        dataset_name: Dataset name
        cache_dir: Cache directory (default: Google Drive cache)
    
    Returns:
        Path to saved cache file
    """
    if cache_dir is None:
        cache_dir = f"/content/drive/MyDrive/SpeckleML/cache/{dataset_name}"
    
    save_path = os.path.join(cache_dir, "processed.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save({
        'x_train': x_train,
        'y_train': y_train,
        'x_valid': x_valid,
        'y_valid': y_valid
    }, save_path)

    print(f" Saved to: {save_path}")
    return save_path


def load_cached_data(dataset_name: str, cache_dir: str = None):
    """
    Load preprocessed data from cache
    
    Args:
        dataset_name: Dataset name
        cache_dir: Cache directory (default: Google Drive cache)
    
    Returns:
        x_train, y_train, x_valid, y_valid as torch tensors
    """
    if cache_dir is None:
        cache_dir = f"/content/drive/MyDrive/SpeckleML/cache/{dataset_name}"
    
    cache_path = os.path.join(cache_dir, "processed.pt")
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f" No cached data found at {cache_path}")
    
    print(f" Loading cached data from: {cache_path}")
    data = torch.load(cache_path)
    return data['x_train'], data['y_train'], data['x_valid'], data['y_valid']


def save_test_data(x_test, y_test, test_group: str, cache_dir: str = None):
    """
    Save preprocessed test data to cache for faster loading
    
    Args:
        x_test, y_test: Preprocessed tensors
        test_group: Test group name (e.g., "group1", "group2")
        cache_dir: Cache directory (default: Google Drive cache)
    
    Returns:
        Path to saved cache file
    """
    if cache_dir is None:
        cache_dir = f"/content/drive/MyDrive/SpeckleML/cache/{test_group}"
    
    save_path = os.path.join(cache_dir, "processed.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save({
        'x_test': x_test,
        'y_test': y_test
    }, save_path)

    print(f" Saved test set to: {save_path}")
    return save_path


def load_cached_test(test_group: str, cache_dir: str = None):
    """
    Load preprocessed test data from cache
    
    Args:
        test_group: Test group name (e.g., "group1", "group2")
        cache_dir: Cache directory (default: Google Drive cache)
    
    Returns:
        x_test, y_test as torch tensors
    """
    if cache_dir is None:
        cache_dir = f"/content/drive/MyDrive/SpeckleML/cache/{test_group}"
    
    cache_path = os.path.join(cache_dir, "processed.pt")
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f" No cached test data found at {cache_path}")
    
    print(f" Loading cached test data from: {cache_path}")
    data = torch.load(cache_path)
    return data['x_test'], data['y_test']
