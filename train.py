"""
Training script for the Deepfake Detection model.

Fine‑tunes EfficientNet‑B4 on a dataset of real / fake face images
organised in ImageFolder format::

    dataset/
    ├── train/
    │   ├── real/   ← genuine face images
    │   └── fake/   ← deepfake face images
    └── val/
        ├── real/
        └── fake/

Usage
-----
    python train.py --data_dir ./dataset --epochs 10 --batch_size 16
"""

import argparse
import os
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from backend.config import DEVICE, IMAGE_SIZE, MODEL_DIR, MODEL_NAME, NUM_CLASSES
from backend.model_loader import load_model


# ── CLI arguments ───────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory with train/ and val/ sub‑folders",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze backbone layers, train only the classifier head",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for saved model (default: models/deepfake_efficientnet_b4.pth)",
    )
    return parser.parse_args()


# ── Data loaders ────────────────────────────────────────────────────────────
def get_data_loaders(
    data_dir: str, batch_size: int
) -> Tuple[DataLoader, DataLoader, datasets.ImageFolder]:
    """Build train and validation data loaders with augmentation.

    Parameters
    ----------
    data_dir : str
        Root directory containing ``train/`` and ``val/`` folders.
    batch_size : int
        Mini‑batch size.

    Returns
    -------
    tuple[DataLoader, DataLoader, ImageFolder]
        Training loader, validation loader, and training dataset (for class info).
    """
    train_transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"[train] Training samples  : {len(train_dataset)}")
    print(f"[train] Validation samples: {len(val_dataset)}")
    print(f"[train] Classes           : {train_dataset.classes}")
    return train_loader, val_loader, train_dataset


# ── Training loop ───────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
) -> float:
    """Train for one epoch and return the average loss."""
    model.train()
    running_loss: float = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(
                f"  Epoch [{epoch}] Batch [{batch_idx + 1}/{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    return running_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module
) -> Tuple[float, float, str]:
    """Evaluate the model and return loss, accuracy, and classification report."""
    model.eval()
    running_loss: float = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=["REAL", "FAKE"], zero_division=0
    )
    return avg_loss, acc, report


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    """Entry point for training."""
    args = parse_args()

    print("=" * 60)
    print("  Deepfake Detection — Model Training")
    print("=" * 60)
    print(f"  Device       : {DEVICE}")
    print(f"  Model        : {MODEL_NAME}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Freeze       : {args.freeze_backbone}")
    print("=" * 60)

    # Data ────────────────────────────────────────────────────────────────────
    train_loader, val_loader, train_dataset = get_data_loaders(
        args.data_dir, args.batch_size
    )

    # Model ───────────────────────────────────────────────────────────────────
    model = load_model()
    model.train()  # switch back from eval()

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("[train] Backbone frozen — training classifier head only.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # Training ────────────────────────────────────────────────────────────────
    best_acc: float = 0.0
    output_path: str = args.output or os.path.join(
        MODEL_DIR, f"deepfake_{MODEL_NAME}.pth"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc, report = evaluate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - start

        print(f"\n{'─' * 50}")
        print(f"Epoch {epoch}/{args.epochs}  ({elapsed:.1f}s)")
        print(f"  Train loss : {train_loss:.4f}")
        print(f"  Val loss   : {val_loss:.4f}")
        print(f"  Val acc    : {val_acc:.4f}")
        print(report)
        print(f"{'─' * 50}\n")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"  ✓ Best model saved → {output_path}  (acc={best_acc:.4f})")

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
