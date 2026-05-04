#!/usr/bin/env python3
"""
CNN traffic-sign classifier with hyperparameter tuning (Optuna).

Runs a short training loop per trial on the validation metric, then retrains
the best configuration with full epochs (same save flow as train-cnn-classifier).

Outputs (defaults):
- models/cnn_classifier_tuned.h5
- reports/cnn_tune_metrics.txt
- reports/cnn_tune_best_params.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset

try:
    from .path_utils import (
        resolve_dataset_root,
        resolve_labels_path,
        resolve_split_path,
    )
except ImportError:
    from path_utils import (
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
            arr = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = arr / 255.0
        arr = (arr - mean) / std
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        return tensor, int(label) - 1


class TunableConvNet(nn.Module):
    """
    CNN with hyperparameters: base channel width, classifier dropout, FC size.
    Spatial head matches SimpleConvNet (three pools on 64px -> 8x8 map).
    """

    def __init__(
        self,
        num_classes: int,
        base_channels: int,
        dropout: float,
        fc_hidden: int,
    ):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c3, c4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        flat = c4 * 8 * 8
        self.classifier = nn.Sequential(
            nn.Linear(flat, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CNN with Optuna hyperparameter tuning.")
    p.add_argument("--root", type=Path, default=None, help="Dataset root (auto-detect if omitted).")
    p.add_argument("--size", type=int, default=64, help="Square resize for images.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--train-split", type=str, default="train_augmented.txt")
    p.add_argument("--val-split", type=str, default="val_augmented.txt")
    p.add_argument(
        "--model-out",
        type=str,
        default="cnn_classifier_tuned.h5",
        help="Model path under models/; always written as HDF5 (.h5).",
    )
    p.add_argument("--metrics-out", type=str, default="cnn_tune_metrics.txt")
    p.add_argument("--best-params-out", type=str, default="cnn_tune_best_params.json")
    p.add_argument("--strict-splits", action="store_true")
    p.add_argument("--labels-path", type=Path, default=Path("labels.json"))
    p.add_argument("--n-trials", type=int, default=12, help="Optuna trials (each runs --trial-epochs).")
    p.add_argument("--trial-epochs", type=int, default=6, help="Epochs per tuning trial.")
    p.add_argument("--epochs", type=int, default=50, help="Epochs for final training with best params.")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (final train).")
    p.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto.")
    p.add_argument(
        "--study-name",
        type=str,
        default="cnn_traffic_signs",
        help="Optuna study name (used with --storage).",
    )
    p.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:///cnn_tune.db (default: in-memory).",
    )
    return p.parse_args()


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
    return total_loss / max(len(train_loader), 1)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend((preds + 1).cpu().numpy().tolist())
            all_labels.extend((labels + 1).cpu().numpy().tolist())
    avg_loss = total_loss / max(len(val_loader), 1)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def save_model_h5(model_path: Path, model_data: dict[str, Any]) -> None:
    with h5py.File(model_path, "w") as f:
        g = f.create_group("model_state")
        for key, value in model_data["model_state"].items():
            g.create_dataset(key, data=value.cpu().numpy())
        f.attrs["model_class"] = model_data["model_class"]
        f.attrs["image_size"] = model_data["image_size"]
        f.attrs["num_classes"] = model_data["num_classes"]
        if "tunable_arch" in model_data:
            f.attrs["tunable_arch"] = json.dumps(model_data["tunable_arch"])
        labels_present = np.array(model_data["labels_present"], dtype=int)
        f.create_dataset("labels_present", data=labels_present)
        label_meta_str = json.dumps(model_data["label_metadata"], indent=2)
        f.create_dataset("label_metadata", data=label_meta_str)


def run_final_training(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
) -> dict[str, Any]:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    best_val_acc = 0.0
    patience_counter = 0
    best_checkpoint: dict[str, Any] = {}
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device)
        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step(val_acc)
        print(
            f"  Final epoch {epoch + 1}/{epochs} - train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_checkpoint = {"model_state": model.state_dict(), "val_acc": val_acc, "epoch": epoch}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stop at epoch {epoch + 1}")
            break
    if best_checkpoint:
        model.load_state_dict(best_checkpoint["model_state"])
    return best_checkpoint


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = resolve_dataset_root(args.root)
    splits = root / "splits"
    reports = root / "reports"
    models = root / "models"
    reports.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    train_fb = None if args.strict_splits else "train_augmented.txt"
    val_fb = None if args.strict_splits else "val_augmented.txt"
    train_split_path = resolve_split_path(splits, args.train_split, train_fb)
    val_split_path = resolve_split_path(splits, args.val_split, val_fb)
    train_rows = read_split(train_split_path)
    val_rows = read_split(val_split_path)

    label_metadata = load_label_metadata(root, args.labels_path)
    num_classes = len(label_metadata) if label_metadata else 48
    device = get_device(args.device)
    print(f"Dataset root: {root}")
    print(f"Device: {device}, classes: {num_classes}")

    train_dataset_full = TrafficSignDataset(root, train_rows, args.size)
    val_dataset_full = TrafficSignDataset(root, val_rows, args.size)

    def objective(trial: optuna.Trial) -> float:
        base_channels = trial.suggest_int("base_channels", 16, 64, step=16)
        dropout = trial.suggest_float("dropout", 0.2, 0.55)
        fc_hidden = trial.suggest_categorical("fc_hidden", [256, 512, 1024])
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        train_loader = DataLoader(
            train_dataset_full, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset_full, batch_size=batch_size, shuffle=False, num_workers=0
        )
        model = TunableConvNet(
            num_classes=num_classes,
            base_channels=base_channels,
            dropout=dropout,
            fc_hidden=fc_hidden,
        ).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_acc = 0.0
        for step in range(args.trial_epochs):
            train_epoch(model, train_loader, criterion, optimizer, device)
            _, val_preds, val_labels = validate(model, val_loader, criterion, device)
            acc = float(accuracy_score(val_labels, val_preds))
            trial.report(acc, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
            best_acc = max(best_acc, acc)
        return best_acc

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
    print(f"Running Optuna with n_trials={args.n_trials}, trial_epochs={args.trial_epochs}...")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_params
    print("Best trial:", best)
    print(f"Best value (val accuracy during tune): {study.best_value:.4f}")

    best_params_path = reports / args.best_params_out
    payload = {
        "best_params": best,
        "best_trial_value": study.best_value,
        "n_trials": args.n_trials,
        "trial_epochs": args.trial_epochs,
    }
    best_params_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote best params to {best_params_path}")

    batch_size_final = int(best["batch_size"])
    arch = {
        "base_channels": int(best["base_channels"]),
        "dropout": float(best["dropout"]),
        "fc_hidden": int(best["fc_hidden"]),
    }
    lr_final = float(best["lr"])
    wd_final = float(best["weight_decay"])

    train_loader = DataLoader(
        train_dataset_full, batch_size=batch_size_final, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset_full, batch_size=batch_size_final, shuffle=False, num_workers=0
    )
    model = TunableConvNet(num_classes=num_classes, **arch).to(device)
    print(
        f"Final training: epochs={args.epochs}, lr={lr_final}, "
        f"weight_decay={wd_final}, batch_size={batch_size_final}, arch={arch}"
    )
    best_checkpoint = run_final_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        lr=lr_final,
        weight_decay=wd_final,
    )

    _, train_preds, train_labels = validate(
        model, train_loader, nn.CrossEntropyLoss(), device
    )
    _, val_preds, val_labels = validate(
        model, val_loader, nn.CrossEntropyLoss(), device
    )
    train_acc = accuracy_score(train_labels, train_preds)
    val_acc = accuracy_score(val_labels, val_preds)

    model_path = (models / Path(args.model_out)).with_suffix(".h5")
    model_data = {
        "model_state": model.state_dict(),
        "model_class": "TunableConvNet",
        "image_size": args.size,
        "num_classes": num_classes,
        "tunable_arch": arch,
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
    save_model_h5(model_path, model_data)
    print(f"Saved model to {model_path}")

    report_text = classification_report(
        val_labels, val_preds, digits=4, zero_division=cast(Any, 0.0)
    )
    metrics_path = reports / args.metrics_out
    lines = [
        "CNN hyperparameter tuning (Optuna) - final metrics",
        "=" * 80,
        f"Train split: {train_split_path.name}",
        f"Val split: {val_split_path.name}",
        f"Train samples: {len(train_rows)}",
        f"Val samples: {len(val_rows)}",
        f"Image size: {args.size}x{args.size}",
        f"Optuna trials: {args.n_trials}",
        f"Trial epochs: {args.trial_epochs}",
        f"Best params: {json.dumps(best)}",
        f"Final train epochs (best run): {best_checkpoint.get('epoch', -1) + 1}",
        f"Train accuracy: {train_acc:.4f}",
        f"Val accuracy: {val_acc:.4f}",
        f"Device: {device}",
        "",
        "Validation classification report",
        "-" * 80,
        cast(str, report_text),
        "",
        f"Saved model: {model_path.relative_to(root).as_posix()}",
        f"Best params JSON: {best_params_path.relative_to(root).as_posix()}",
    ]
    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")
    print(f"Train accuracy: {train_acc:.4f} | Val accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
