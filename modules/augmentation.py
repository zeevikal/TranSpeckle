"""
Speckle-Specific Data Augmentation

Augmentation transforms optimized for speckle pattern reconstruction.
Border cropping preserves central speckle information while varying brightness
and rotation improves model robustness.

Google Colab Setup:
    !pip install albumentations opencv-python

"""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_augmentation():
    """
    Returns augmentation pipeline for speckle patterns
    
    Transforms:
        - Border cropping: -3% to 0% with zero padding (p=0.08)
        - Brightness/contrast: ±8% variation (p=0.05)
        - Random rotation: ±10° (p=0.07)
    """
    return A.Compose([
        A.CropAndPad(
            percent=(-0.03, 0.0),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            keep_size=True,
            p=0.08
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.08,
            contrast_limit=0.08,
            p=0.05
        ),
        A.Rotate(
            limit=10,
            p=0.07,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        ToTensorV2()
    ])
