#!/usr/bin/env python3
"""
CNN (Convolutional Neural Network) traffic-sign classifier training with PyTorch.

Uses (defaults):
- splits/train_augmented.txt
- splits/val_augmented.txt

Outputs (defaults; override with --model-out / --metrics-out):
- reports/cnn_train_metrics.txt
- models/cnn_classifier.joblib

For original Train/ data only, run train-cnn-classifier-original.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast, Optional

import h5py
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset

from .path_utils import (
    resolve_dataset_root,
    resolve_labels_path,
    resolve_split_path,
)


class TrafficSignDataset(Dataset):
    """PyTorch Dataset for traffic sign images."""

    def __init__(self, root: Path, rows: list[tuple[str, int]], size: int = 64):
        self.root = root
        self.rows = rows
        self.size = size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rel, label = self.rows[idx]
        img_path = self.root / rel
        
        try:
            with Image.open(img_path) as img:
                arr = np.asarray(
                    img.convert("RGB").resize((self.size, self.size), Image.Resampling.LANCZOS),
                    dtype=np.float32,
                )
        except (UnidentifiedImageError, OSError, ValueError):
            # Return black image if unreadable
            arr = np.zeros((self.size, self.size, 3), dtype=np.float32)
        
        # Normalize using ImageNet mean/std per channel
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = arr / 255.0
        arr = (arr - mean) / std
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        # Convert class ID (1-48) to index (0-47)
        return tensor, int(label) - 1


class SimpleConvNet(nn.Module):
    """Lightweight CNN for 32x32 grayscale traffic sign images."""

    def __init__(self, num_classes: int = 48):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 -> 32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN traffic sign classifier with PyTorch.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Dataset root directory. If omitted, common dataset locations are auto-detected.",
    )
    parser.add_argument("--size", type=int, default=64, help="Square resize for images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--train-split",
        type=str,
        default="train_augmented.txt",
        help="Train split filename inside splits/ directory.",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="val_augmented.txt",
        help="Validation split filename inside splits/ directory.",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="cnn_classifier.h5",
        help="Model filename inside dataset models/ directory. Supports .h5 and .joblib.",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="cnn_train_metrics.txt",
        help="Metrics report filename inside dataset reports/ directory.",
    )
    parser.add_argument(
        "--strict-splits",
        action="store_true",
        help="If set, do not fall back to *_augmented.txt when the requested split is missing.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("labels.json"),
        help="Path to labels metadata JSON relative to root.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cpu', 'cuda', or 'auto'.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    return parser.parse_args()


def read_split(split_path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        rel, label = line.rsplit(" ", 1)
        rows.append((rel, int(label)))
    return rows


def load_label_metadata(root: Path, labels_path: Path) -> dict[int, dict[str, object]]:
    labels_file = resolve_labels_path(root, labels_path)
    metadata: dict[int, dict[str, object]] = {}

    payload = json.loads(labels_file.read_text(encoding="utf-8"))
    for row in payload.get("classes", []):
        class_id = int(row["class_id"])
        metadata[class_id] = {
            "class_id": class_id,
            "class_name": str(row.get("class_name", f"class_{class_id}")),
            "category": str(row.get("category", "unknown")),
        }
    return metadata





def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            # Convert back from indices (0-47) to class IDs (1-48)
            all_preds.extend((preds + 1).cpu().numpy())
            all_labels.extend((labels + 1).cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def save_model_h5(model_path: Path, model_data: dict) -> None:
    """Save model and metadata to HDF5 file."""
    with h5py.File(model_path, "w") as f:
        # Save model state dictionary
        g = f.create_group("model_state")
        for key, value in model_data["model_state"].items():
            g.create_dataset(key, data=value.cpu().numpy())

        # Save metadata
        f.attrs["model_class"] = model_data["model_class"]
        f.attrs["image_size"] = model_data["image_size"]
        f.attrs["num_classes"] = model_data["num_classes"]
        
        labels_present = np.array(model_data["labels_present"], dtype=int)
        f.create_dataset("labels_present", data=labels_present)

        # Save label metadata as JSON string
        label_meta_str = json.dumps(model_data["label_metadata"], indent=2)
        f.create_dataset("label_metadata", data=label_meta_str)


def main() -> None:
    print("Starting training script...")
    args = parse_args()
    print(f"Arguments: {args}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("Resolving dataset root...")
    root = resolve_dataset_root(args.root)
    print(f"Dataset root: {root}")
    splits = root / "splits"
    reports = root / "reports"
    models = root / "models"
    reports.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    train_fb = None if args.strict_splits else "train_augmented.txt"
    val_fb = None if args.strict_splits else "val_augmented.txt"
    print("Resolving split paths...")
    train_split_path = resolve_split_path(splits, args.train_split, train_fb)
    val_split_path = resolve_split_path(splits, args.val_split, val_fb)
    print(f"Train split: {train_split_path}")
    print(f"Validation split: {val_split_path}")
    train_rows = read_split(train_split_path)
    val_rows = read_split(val_split_path)

    print("Creating datasets...")
    train_dataset = TrafficSignDataset(root, train_rows, args.size)
    val_dataset = TrafficSignDataset(root, val_rows, args.size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("Loading label metadata...")
    label_metadata = load_label_metadata(root, args.labels_path)
    num_classes = len(label_metadata) if label_metadata else 48
    print(f"Number of classes: {num_classes}")

    device = get_device(args.device)
    print(f"Using device: {device}")

    model = SimpleConvNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0.0
    patience_counter = 0
    best_checkpoint = {}

    print(f"Training for up to {args.epochs} epochs (patience: {args.patience})...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device)
        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step(val_acc)

        print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model checkpoint
            best_checkpoint = {
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch,
            }
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch + 1} (patience exceeded)")
            break

    # Restore best model
    if best_checkpoint:
        model.load_state_dict(best_checkpoint["model_state"])

    # Final evaluation
    print("Final evaluation...")
    _, train_preds, train_labels = validate(model, train_loader, criterion, device)
    _, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    train_acc = accuracy_score(train_labels, train_preds)
    val_acc = accuracy_score(val_labels, val_preds)

    # Save model
    model_path = models / args.model_out
    print(f"Saving model to {model_path}...")
    model_data = {
        "model_state": model.state_dict(),
        "model_class": "SimpleConvNet",
        "image_size": args.size,
        "num_classes": num_classes,
        "labels_present": sorted(set(int(l) for _, l in train_rows)),
        "label_metadata": {
            str(class_id): label_metadata.get(
                class_id,
                {
                    "class_id": class_id,
                    "class_name": f"class_{class_id}",
                    "category": "unknown",
                },
            )
            for class_id in sorted(set(int(l) for _, l in train_rows))
        },
    }

    if model_path.suffix == ".h5":
        save_model_h5(model_path, model_data)
    else:
        joblib.dump(model_data, model_path)
    print("Model saved.")

    report_text = classification_report(val_labels, val_preds, digits=4, zero_division=0)
    metrics_path = reports / args.metrics_out
    print(f"Saving metrics to {metrics_path}...")
    metrics_lines: list[str] = [
        "CNN Classification Training Metrics",
        "=" * 80,
        f"Train split: {train_split_path.name}",
        f"Val split: {val_split_path.name}",
        f"Train samples: {len(train_rows)}",
        f"Val samples: {len(val_rows)}",
        f"Image size: {args.size}x{args.size}",
        f"Batch size: {args.batch_size}",
        f"Learning rate: {args.lr}",
        f"Epochs trained: {best_checkpoint.get('epoch', -1) + 1}",
        f"Train accuracy: {train_acc:.4f}",
        f"Val accuracy: {val_acc:.4f}",
        f"Device: {device}",
        "",
        "Validation Classification Report",
        "-" * 80,
        cast(str, report_text),
        "",
        f"Saved model: {model_path.relative_to(root).as_posix()}",
    ]
    metrics_path.write_text(
        "\n".join(metrics_lines) + "\n",
        encoding="utf-8",
    )
    print("Metrics saved.")

    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics:  {metrics_path}")
    print("Training script finished.")



if __name__ == "__main__":
    main()
