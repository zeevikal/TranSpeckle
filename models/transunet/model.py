"""
TransUNet - Hybrid ViT + U-Net Architecture

Combines ViT-B/32 encoder with U-Net decoder for multi-scale feature aggregation.
Uses orientation-aware positional embeddings with angular encoding.

Google Colab Setup:
    !git clone https://github.com/Beckschen/TransUNet.git
    !pip install timm pytorch-lightning torchmetrics albumentations ml-collections
    
    import sys
    sys.path.append('/content/TransUNet')

Trained from scratch (no pretrained weights)

"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import JaccardIndex
from timm.models.layers import trunc_normal_

sys.path.append("/content/TransUNet")


class TransUNetWithOrientationPE(nn.Module):
    """
    TransUNet wrapper with orientation positional embeddings
    
    Args:
        vit_base_model: Base TransUNet model
        img_size: Input image size
        num_classes: Number of output classes
        add_theta: Add angular encoding to positional embeddings
    """
    
    def __init__(
        self,
        vit_base_model,
        img_size: int = 256,
        num_classes: int = 2,
        add_theta: bool = True
    ):
        super().__init__()
        
        self.vit_base = vit_base_model
        self.img_size = img_size
        self.add_theta = add_theta

        patch_size = 32
        num_patches = (img_size // patch_size) ** 2
        embed_dim = 768

        from modules.orientation_pe import OrientationPositionalEmbedding
        base_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        trunc_normal_(base_embedding, std=0.02)

        self.orientation_pos_embed = OrientationPositionalEmbedding(
            base_embedding=base_embedding,
            emb_dim=embed_dim,
            grid_size=int(num_patches ** 0.5),
            add_theta=add_theta
        )

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        x, features = self.vit_base.transformer.embeddings(x)
        x = self.orientation_pos_embed(x)
        x, attn_weights = self.vit_base.transformer.encoder(x)
        x = self.vit_base.decoder(x, features)
        logits = self.vit_base.segmentation_head(x)
        
        return logits


class LitTransUNet(pl.LightningModule):
    """
    TransUNet - PyTorch Lightning Module
    
    Args:
        lr: Learning rate
    """
    
    def __init__(self, lr: float = 2e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        from networks.vit_seg_modeling import VisionTransformer, CONFIGS

        config = CONFIGS["ViT-B_32"]
        config.n_classes = 2
        config.n_skip = 0

        vit_base = VisionTransformer(
            config=config,
            img_size=256,
            num_classes=2,
            zero_head=True,
            vis=False
        )

        self.model = TransUNetWithOrientationPE(
            vit_base_model=vit_base,
            img_size=256,
            num_classes=2,
            add_theta=True
        )

        self.criterion = self.balanced_cross_entropy
        self.iou = JaccardIndex(task="binary", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def balanced_cross_entropy(self, pred, target, eps=1e-6):
        pred_softmax = torch.softmax(pred, dim=1).clamp(min=eps, max=1 - eps)
        loss = - (target * torch.log(pred_softmax) + (1 - target) * torch.log(1 - pred_softmax))
        return loss.mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if logits.shape[2:] != y.shape[2:]:
            logits = F.interpolate(logits, size=y.shape[2:], mode="bilinear", align_corners=False)
        loss = self.criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        targets = torch.argmax(y, dim=1)

        iou = self.iou(preds, targets)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_iou", iou, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if logits.shape[2:] != y.shape[2:]:
            logits = F.interpolate(logits, size=y.shape[2:], mode="bilinear", align_corners=False)
        loss = self.criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        targets = torch.argmax(y, dim=1)

        iou = self.iou(preds, targets)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_iou", iou, prog_bar=True, on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=8,
            threshold=1e-4,
            cooldown=2,
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_iou",
                "interval": "epoch",
                "frequency": 1,
                "strict": True
            }
        }


def objective_transunet(trial, dm, base_log_dir):
    """
    Optuna objective function for TransUNet hyperparameter optimization
    
    Args:
        trial: Optuna trial object
        dm: SpeckleDataModule with training/validation data
        base_log_dir: Base directory for logging
    
    Returns:
        Best validation IoU score
    """
    import os
    import shutil
    from datetime import datetime
    from pathlib import Path
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import CSVLogger
    
    lr = trial.suggest_float("lr", 2e-4, 4e-4, log=True)

    model = LitTransUNet(lr=lr)

    version_dir = f"trial_{trial.number}_{datetime.now().strftime('%d%m%Y')}"
    trial_log_dir = os.path.join(base_log_dir, "optuna_trials", version_dir)
    os.makedirs(trial_log_dir, exist_ok=True)

    logger = CSVLogger(save_dir=trial_log_dir, name="", version=".")

    ckpt_temp_dir = os.path.join("/content/temp_checkpoints", version_dir)
    os.makedirs(ckpt_temp_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_temp_dir,
        filename="best",
        monitor="val_iou",
        mode="max",
        save_top_k=1,
        save_on_train_epoch_end=True
    )

    early_stop = EarlyStopping(monitor="val_iou", patience=30, mode="max")

    trainer = pl.Trainer(
        max_epochs=350,
        callbacks=[checkpoint_callback, early_stop],
        logger=logger,
        enable_progress_bar=True,
        accelerator='gpu',
        devices=1
    )

    trainer.fit(model, dm)

    best_ckpt_src = checkpoint_callback.best_model_path
    best_ckpt_dst = os.path.join(trial_log_dir, "checkpoints", "best.ckpt")
    os.makedirs(os.path.dirname(best_ckpt_dst), exist_ok=True)
    shutil.copy2(best_ckpt_src, best_ckpt_dst)

    trial.set_user_attr("best_model_path", best_ckpt_dst)

    print("-" * 60)
    print(f"\n Trial {trial.number} finished")
    print(f"   val_iou: {trainer.callback_metrics['val_iou'].item():.4f}")
    print(f"   Learning Rate: {lr}")
    print("-" * 60)

    return trainer.callback_metrics["val_iou"].item()
