"""
APTOS 2019 baseline: ResNet50 + MSE regression, label output via thresholds.

Dependencies: torch, torchvision, pandas, scikit-learn, Pillow

Before running, fill in the paths below, then:
    python train_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

# --- paths (fill in before running) ---
TRAIN_CSV = "C:\\Users\\ohwow\\Desktop\\DL\\train.csv"
IMAGE_DIR = "C:\\Users\\ohwow\\Desktop\\DL\\processed_train_images"
CHECKPOINT_DIR = "C:\\Users\\ohwow\\Desktop\\DL\\Checkpoints"

# --- hyperparameters ---
IMAGE_SIZE = 512
VAL_RATIO = 0.2
SEED = 42
EPOCHS = 15
BATCH_SIZE = 16
LR = 1e-4

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# continuous pred -> diagnosis label {0,1,2,3,4}
THRESHOLDS = (1.5, 2.5, 3.5, 4.5)


def to_labels(continuous: torch.Tensor) -> torch.Tensor:
    bounds = torch.tensor(THRESHOLDS, device=continuous.device, dtype=continuous.dtype)
    # pred < 1.5 -> 0, < 2.5 -> 1, < 3.5 -> 2, < 4.5 -> 3, else 4
    return torch.searchsorted(bounds, continuous, side="right").clamp(max=4).long()


class DiagnosisModel(nn.Module):
    """ResNet50 regressor; forward() returns integer labels 0-4."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        self.backbone = backbone

    def predict_continuous(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return to_labels(self.predict_continuous(x))


TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class FundusDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path) -> None:
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[i]
        path = self.image_dir / f"{row['id_code']}.png"
        with Image.open(path) as img:
            x = TRANSFORM(img.convert("RGB"))
        y = torch.tensor(float(row["diagnosis"]))
        return x, y


def run_epoch(
    model: DiagnosisModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Adam | None,
    device: torch.device,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    total, n = 0.0, 0
    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train_mode):
            pred = model.predict_continuous(x)
            loss = criterion(pred, y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def validate(
    model: DiagnosisModel, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    criterion = nn.MSELoss()
    mse_total, correct, n = 0.0, 0, 0
    for x, y in tqdm(loader, desc="Validating", leave=False):
        x = x.to(device)
        y = y.to(device)
        raw = model.predict_continuous(x)
        labels = model(x)
        mse_total += criterion(raw, y).item() * x.size(0)
        correct += (labels == y.long()).sum().item()
        n += x.size(0)
    return mse_total / n, correct / n


def resolve_paths() -> tuple[Path, Path, Path]:
    missing = [name for name, val in [
        ("TRAIN_CSV", TRAIN_CSV),
        ("IMAGE_DIR", IMAGE_DIR),
        ("CHECKPOINT_DIR", CHECKPOINT_DIR),
    ] if not str(val).strip()]
    if missing:
        sys.exit(f"Set paths at top of script: {', '.join(missing)}")

    train_csv = Path(TRAIN_CSV)
    image_dir = Path(IMAGE_DIR)
    checkpoint_dir = Path(CHECKPOINT_DIR)

    if not train_csv.is_file():
        sys.exit(f"TRAIN_CSV not found: {train_csv}")
    if not image_dir.is_dir():
        sys.exit(f"IMAGE_DIR not found: {image_dir}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return train_csv, image_dir, checkpoint_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()

    train_csv, image_dir, checkpoint_dir = resolve_paths()
    best_model_path = checkpoint_dir / "best_model.pt"

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    df = pd.read_csv(train_csv)
    train_df, val_df = train_test_split(
        df, test_size=VAL_RATIO, random_state=SEED, stratify=df["diagnosis"]
    )
    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader = DataLoader(
        FundusDataset(train_df, image_dir), shuffle=True, **loader_kw
    )
    val_loader = DataLoader(
        FundusDataset(val_df, image_dir), shuffle=False, **loader_kw
    )

    model = DiagnosisModel(pretrained=not args.no_pretrained).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_mse = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device)
        va_mse, va_acc = validate(model, val_loader, device)
        history.append(
            {"epoch": epoch, "train_mse": tr, "val_mse": va_mse, "val_acc": va_acc}
        )
        print(
            f"epoch {epoch}/{args.epochs}  train_mse={tr:.4f}  "
            f"val_mse={va_mse:.4f}  val_acc={va_acc:.4f}"
        )
        if va_mse < best_mse:
            best_mse = va_mse
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_mse": va_mse,
                    "val_acc": va_acc,
                    "thresholds": THRESHOLDS,
                },
                best_model_path,
            )
            print(f"  saved {best_model_path}")

    (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"done  best_val_mse={best_mse:.4f}")


if __name__ == "__main__":
    main()
