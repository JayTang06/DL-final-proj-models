"""
APTOS 2019 -- ConvNeXt-Tiny + HybridCEKappa training.

Design notes (read before changing):
  * Loss = CE warmup -> CE+SoftQWK mix (alpha schedule from HybridCEKappaLoss.get_alpha).
    With EPOCHS<30 the kappa term will barely activate, so keep EPOCHS aligned
    with the get_alpha horizon (default 30).
  * Augmentations use Albumentations (matches the 4th-place APTOS solution which
    relied heavily on shift/scale/rotate/contrast). No Ben Graham preprocessing
    -- the 2019 1st and 4th place winners both reported it did not help, and
    ConvNeXt is strong enough to learn through lighting variation.
  * Best checkpoint is selected by validation QWK, not loss or accuracy.
  * Extension hooks (Stage-A: 2015 DR pretrain; Stage-C: pseudo-labeling) are
    documented but not implemented -- they are the biggest remaining wins.

Requires: pip install albumentations
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from HybridCEKappaLoss import HybridCEKappaLoss, get_alpha

# =========================
# Config
# =========================

CSV_PATH = r"C:\Users\ohwow\Desktop\DL\train.csv"
IMAGE_DIR = r"C:\Users\ohwow\Desktop\DL\processed_train_images"
CHECKPOINT_PATH = r"C:\Users\ohwow\Desktop\DL\Checkpoints"

IMG_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 30                # aligned with get_alpha(total_epochs=30) default
WARMUP_EPOCHS = 5          # CE-only warmup; matches get_alpha default
LR = 1e-4                  # AdamW with cosine; 1e-5 was far too low for 30 epochs
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.05

NUM_WORKERS = 0            # bump on Linux; Windows + Jupyter often needs 0
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet stats -- ConvNeXt_Tiny_Weights.DEFAULT uses these
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =========================
# Albumentations pipelines
# =========================
# Training: shift/scale/rotate + flips + photometric jitter + coarse dropout.
# These are the augmentations the 4th-place APTOS team leaned on.

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
        border_mode=0, p=0.7,
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2, contrast_limit=0.2, p=0.5,
    ),
    A.HueSaturationValue(
        hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5,
    ),
    A.CoarseDropout(
        max_holes=8, max_height=32, max_width=32, p=0.3,
    ),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


# =========================
# Dataset
# =========================

class CustomDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_dir, f"{row['id_code']}.png")
        # Albumentations expects HWC numpy
        image = np.array(Image.open(image_path).convert("RGB"))
        label = int(row["diagnosis"])
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


# =========================
# Train / Validate helpers
# =========================

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    seen = 0
    preds_all, labels_all = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        seen += images.size(0)
        preds_all.extend(logits.argmax(dim=1).detach().cpu().numpy())
        labels_all.extend(labels.detach().cpu().numpy())

        pbar.set_postfix(loss=f"{total_loss/seen:.4f}")

    scheduler.step()
    avg_loss = total_loss / len(loader.dataset)
    acc = float(np.mean(np.array(preds_all) == np.array(labels_all)))
    qwk = cohen_kappa_score(labels_all, preds_all, weights="quadratic")
    return avg_loss, acc, qwk


@torch.no_grad()
def validate(model, loader, device, epoch, total_epochs):
    model.eval()
    preds_all, labels_all = [], []
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [ val ]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds_all.extend(logits.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
    acc = float(np.mean(np.array(preds_all) == np.array(labels_all)))
    qwk = cohen_kappa_score(labels_all, preds_all, weights="quadratic")
    return acc, qwk


# =========================
# Pseudo-label extension hook (Stage C)
# =========================
# After Stage-B converges, call this on the unlabeled test set:
#   1. predict probs on test images
#   2. keep rows where max(prob) > confidence_threshold (e.g. 0.9)
#   3. concat (pseudo_df, train_df), retrain
# The 4th-place 2019 team called this their "secret ingredient".
# Not implemented here -- wire up once Stage-B QWK plateaus.
def pseudo_label(model, test_image_dir, confidence_threshold=0.9):
    raise NotImplementedError("Stage-C pseudo-label hook -- implement after Stage-B converges")


# =========================
# Main
# =========================

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # -------------------------
    # Data
    # -------------------------
    df = pd.read_csv(CSV_PATH)
    num_classes = df["diagnosis"].nunique()

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["diagnosis"]
    )

    train_dataset = CustomDataset(train_df, IMAGE_DIR, train_transform)
    val_dataset = CustomDataset(val_df, IMAGE_DIR, val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # -------------------------
    # Class weights -- APTOS is heavily skewed toward class 0 (~50%)
    # -------------------------
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=train_df["diagnosis"].values,
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=DEVICE)
    print(f"class weights: {class_weights_np.round(3).tolist()}")

    # -------------------------
    # Model
    # -------------------------
    # Stage A (NOT IMPLEMENTED): pretrain on the 2015 Kaggle Diabetic Retinopathy
    # dataset (~35k images) before this fine-tune. Both 2019 1st and 4th place
    # teams credited 2015 pretraining as a major contributor. To enable later:
    #   1. download 2015 DR train set, map its 5-grade labels (same scale as APTOS)
    #   2. train this same model on it for ~10 epochs at LR=1e-4
    #   3. load that checkpoint here instead of ImageNet weights
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    model = model.to(DEVICE)

    # -------------------------
    # Loss / Optimizer / Scheduler
    # -------------------------
    criterion = HybridCEKappaLoss(
        num_classes=num_classes,
        alpha=1.0,                       # start in pure-CE warmup
        class_weights=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    ).to(DEVICE)                         # moves WeightedKappaLoss.weights buffer to GPU

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    # -------------------------
    # Training loop
    # -------------------------
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    best_ckpt_file = os.path.join(CHECKPOINT_PATH, "convnext_best.pth")

    best_val_qwk = -1.0
    for epoch in range(EPOCHS):
        # CRITICAL: update alpha each epoch -- without this the loss is pure CE.
        criterion.alpha = get_alpha(
            epoch=epoch,
            warmup_epochs=WARMUP_EPOCHS,
            total_epochs=EPOCHS,
            alpha_final=0.2,
        )

        train_loss, train_acc, train_qwk = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, DEVICE,
            epoch, EPOCHS,
        )
        val_acc, val_qwk = validate(model, val_loader, DEVICE, epoch, EPOCHS)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"alpha={criterion.alpha:.2f} lr={current_lr:.2e} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} qwk={train_qwk:.4f} | "
            f"val acc={val_acc:.4f} qwk={val_qwk:.4f}"
        )

        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_qwk": val_qwk,
                    "val_acc": val_acc,
                },
                best_ckpt_file,
            )
            print(f"  -> new best, saved to {best_ckpt_file}")

    print(f"done. best val QWK = {best_val_qwk:.4f}")
