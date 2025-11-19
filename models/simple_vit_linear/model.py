"""
SimpleViT with Linear Decoder

Simplified Vision Transformer with direct linear projection to pixel space.
Uses orientation-aware positional embeddings for improved spatial reasoning.

Google Colab Setup:
    !pip install vit-pytorch pytorch-lightning torchmetrics einops albumentations

Pretrained: None (trains from scratch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from vit_pytorch import SimpleViT
from torchmetrics import JaccardIndex
from einops import rearrange


class LitSimpleViTLinear(pl.LightningModule):
    """
    SimpleViT with Linear Decoder - PyTorch Lightning Module
    
    Args:
        lr: Learning rate
        dim: Model dimension
        depth: Number of transformer layers
        heads: Number of attention heads
        mlp_dim: MLP dimension
        patch_size: Patch size (default: 32)
        use_orientation_pos_emb: Use orientation positional embeddings
    """
    
    def __init__(
        self,
        lr: float = 1e-4,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 1024,
        patch_size: int = 32,
        use_orientation_pos_emb: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        self.use_orientation_pos_emb = use_orientation_pos_emb

        self.model = SimpleViT(
            image_size=256,
            patch_size=patch_size,
            num_classes=2,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=1
        )

        patch_area = patch_size * patch_size
        self.decoder = nn.Linear(dim, patch_area * 2)

        self.lr = lr
        self.criterion = self.balanced_cross_entropy
        self.iou = JaccardIndex(task="binary", num_classes=2)

        self.patch_size = patch_size
        self.image_size = 256

        h = w = self.image_size // self.patch_size
        num_patches = h * w

        if self.use_orientation_pos_emb:
            from modules.orientation_pe import OrientationPositionalEmbedding
            self.pos_encoder = OrientationPositionalEmbedding(
                base_embedding=nn.Parameter(torch.randn(1, num_patches, dim)),
                emb_dim=dim,
                grid_size=h,
                add_theta=True
            )
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, num_patches, dim))

    def forward(self, img):
        B, C, H, W = img.shape
        x = self.model.to_patch_embedding(img)

        if isinstance(self.pos_encoder, nn.Module):
            x = self.pos_encoder(x)
        else:
            x = x + self.pos_encoder.to(x.device, dtype=x.dtype)

        x = self.model.transformer(x)
        x = self.decoder(x)

        p = self.patch_size
        h = w = self.image_size // p
        x = rearrange(x, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)', h=h, w=w, p1=p, p2=p, c=2)

        return x

    def balanced_cross_entropy(self, pred, target, eps=1e-6):
        pred_softmax = torch.softmax(pred, dim=1).clamp(min=eps, max=1 - eps)
        loss = - (target * torch.log(pred_softmax) + (1 - target) * torch.log(1 - pred_softmax))
        return loss.mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = (logits[:, 1] > 0.5).long()
        targets = y[:, 1].long()

        iou = self.iou(preds, targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_iou", iou, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = (logits[:, 1] > 0.5).long()
        targets = y[:, 1].long()

        iou = self.iou(preds, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
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
            verbose=True
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


def objective_simple_vit_linear(trial, dm, base_log_dir):
    """
    Optuna objective function for SimpleViT Linear hyperparameter optimization
    
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
    
    param_options = [
        {"dim": 128, "heads": 2},
        {"dim": 128, "heads": 4},
        {"dim": 256, "heads": 4},
        {"dim": 256, "heads": 8},
        {"dim": 512, "heads": 8},
        {"dim": 512, "heads": 16}
    ]
    choice = trial.suggest_categorical("dim_heads", param_options)
    dim, heads = choice["dim"], choice["heads"]
    depth = trial.suggest_categorical("depth", [4, 6, 8])
    mlp_dim = 4 * dim
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    model = LitSimpleViTLinear(
        dim=dim,
        heads=heads,
        depth=depth,
        mlp_dim=mlp_dim,
        lr=lr,
        patch_size=32,
        use_orientation_pos_emb=True
    )

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

    print("-" * 50)
    print(f"\n Trial {trial.number} finished — val_iou: {trainer.callback_metrics['val_iou'].item():.4f} — params: {trial.params}\n")
    print("-" * 50)
    
    return trainer.callback_metrics["val_iou"].item()
