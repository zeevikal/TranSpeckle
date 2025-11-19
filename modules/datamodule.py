"""
PyTorch Lightning DataModule for Speckle Patterns

Handles dataset creation, augmentation, and data loading with automatic
batching and worker management.

Google Colab Setup:
    !pip install pytorch-lightning albumentations
"""

from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as pl


class AugmentedSpeckleDataset(Dataset):
    """Dataset with on-the-fly augmentation"""
    
    def __init__(self, x_tensor, y_tensor, transform):
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor
        self.transform = transform

    def __len__(self):
        return self.x_tensor.size(0)

    def __getitem__(self, idx):
        x = self.x_tensor[idx]
        y = self.y_tensor[idx]

        x = x.permute(1, 2, 0).numpy()
        y = y.permute(1, 2, 0).numpy()

        augmented = self.transform(image=x, mask=y)
        x = augmented['image']
        y = augmented['mask']

        y = y.permute(2, 0, 1)

        return x.float(), y.float()


class SpeckleDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for speckle pattern reconstruction
    
    Args:
        train_data: Tuple of (x_train, y_train)
        val_data: Tuple of (x_val, y_val)
        test_data: Tuple of (x_test, y_test)
        batch_size: Batch size for training/validation/testing
        num_workers: Number of data loading workers
        use_augmentation: Whether to apply augmentation to training data
    """
    
    def __init__(
        self,
        train_data=None,
        val_data=None,
        test_data=None,
        batch_size: int = 16,
        num_workers: int = 4,
        use_augmentation: bool = True
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_augmentation = use_augmentation

        if use_augmentation:
            from modules.augmentation import get_train_augmentation
            self.train_transform = get_train_augmentation()
        else:
            self.train_transform = None

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            x_train, y_train = self.train_data
            x_val, y_val = self.val_data

            if self.use_augmentation and self.train_transform is not None:
                self.train_dataset = AugmentedSpeckleDataset(x_train, y_train, transform=self.train_transform)
            else:
                self.train_dataset = TensorDataset(x_train, y_train)
                
            self.val_dataset = TensorDataset(x_val, y_val)

        if stage == 'test' or stage is None:
            x_test, y_test = self.test_data
            self.test_dataset = TensorDataset(x_test, y_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
