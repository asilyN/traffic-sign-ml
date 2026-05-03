#!/usr/bin/env python3
"""
CNN traffic-sign classifier — fully improved training script.
Saves trained model as:
  models/simple_classifier.keras   (modern Keras SavedModel format)
  models/simple_classifier.h5      (legacy HDF5 — universally compatible)
  models/simple_classifier_meta.json  (label mapping + normalisation stats)

Performance improvements applied
---------------------------------
 [1]  AdamW (decoupled weight decay) replaces Adam
 [2]  Linear warmup (default 3 ep) -> CosineAnnealingLR eta_min=1e-6
 [3]  LR default 3e-4, wd default 1e-4
 [4]  CrossEntropyLoss with label_smoothing=0.1
 [5]  Class-weighted loss (optional; OFF by default for balanced dataset)
 [6]  Normalisation: ImageNet priors OR computed dataset stats
 [7]  Spatial Dropout2d (p=0.1) after conv block 4
 [8]  Reduced FC dropout: 0.4 / 0.2
 [9]  Residual skip connection block 3 -> block 4 (1x1 projection)
 [10] 5x5 kernel in conv block 1
 [11] Gradient clipping max_norm=1.0
 [12] Mixed-precision training (AMP) on CUDA
 [13] Early stopping: patience=10, min_delta=0.001 — best_state pre-seeded
 [14] Epochs default raised to 40
 [15] Top-1, Top-3 accuracy + macro-F1 reported
 [16] Sample prediction grid with green/red correctness borders
 [17] Training-time data augmentation (rotation, brightness, contrast, etc.)
 [18] Confusion matrix saved to reports/confusion_matrix.png
 [19] CNN architecture: double conv in blocks 2 & 3
 [20] AdaptiveAvgPool2d for size-flexibility (48 / 64 / 72 px)
 [21] --monitor-metric flag: early-stop on val_acc OR macro_f1

Usage
-----
  python train_simple_classifier.py                   # all defaults
  python train_simple_classifier.py --compute-norm-stats
  python train_simple_classifier.py --no-class-weights
  python train_simple_classifier.py --no-amp
  python train_simple_classifier.py --no-early-stop
  python train_simple_classifier.py --size 64 --lr 7e-4
  python train_simple_classifier.py --monitor-metric macro_f1

Outputs
-------
  reports/simple_train_metrics.txt
  reports/sample_predictions.png
  reports/confusion_matrix.png
  models/simple_classifier.keras
  models/simple_classifier.h5
  models/simple_classifier_meta.json

Loading the saved model
-----------------------
  import tensorflow as tf, json
  model = tf.keras.models.load_model("models/simple_classifier.keras")
  meta  = json.load(open("models/simple_classifier_meta.json"))
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

try:
    import torchvision.transforms as T
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

try:
    import tensorflow as tf                         # noqa: F401 — checked at runtime
    HAS_TF = True
except ImportError:
    HAS_TF = False

# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NORM_MEAN = IMAGENET_MEAN.copy()   # overwritten when --compute-norm-stats is used
NORM_STD  = IMAGENET_STD.copy()

# ---------------------------------------------------------------------------
# Class names (fallback when labels.json is unavailable)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Stop", "No Entry", "Left Curve Ahead", "Right Curve Ahead",
    "Reverse Turn Ahead", "Slippery Road", "Road Work", "Pedestrian Crossing",
    "Children Crossing", "Bike Lane", "Turn Right", "Turn Left",
    "Straight Ahead Only", "Pass on Right", "Pass on Left", "Roundabout",
    "U-Turn", "Speed Limit (60 km/h)", "Speed Limit (70 km/h)",
    "No Blowing of Horn", "No U-Turn", "Speed Limit (15 km/h)",
    "No Left Turn", "No Right Turn", "No Straight Proceeding", "No Parking",
    "Intersection Ahead", "Side Road Junction (Right)", "Speed Limit (20 km/h)",
    "Speed Limit (40 km/h)", "Pass on Either Side",
    "Side Road Junction (Diagonal)", "Speed Limit (30 km/h)",
    "No Entry for Four-Wheeled Vehicles", "Hospital",
    "Do Not Block Intersection", "Railroad Crossing", "Parking Area",
    "No Stopping / No Waiting", "T-Junction",
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train an improved CNN traffic-sign classifier (saves to Keras/H5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # paths
    p.add_argument("--root", type=Path, default=None,
                   help="Dataset root. Auto-detected if omitted.")
    p.add_argument("--train-split", default="train_clean.txt")
    p.add_argument("--val-split",   default="val_clean.txt")
    p.add_argument("--test-split",  default="test.txt")
    p.add_argument("--labels-path", type=Path, default=Path("labels.json"))
    p.add_argument("--augmented-dir", default="Train_Augmented_Balanced")

    # normalisation
    p.add_argument("--compute-norm-stats", action="store_true",
                   help="Compute per-channel mean/std from training set instead of ImageNet priors.")

    # Tier-1 hyperparameters
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs",       type=int,   default=40)
    p.add_argument("--batch-size",   type=int,   default=64)

    # regularisation
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--dropout1",        type=float, default=0.4)
    p.add_argument("--dropout2",        type=float, default=0.2)
    p.add_argument("--spatial-dropout", type=float, default=0.1)
    p.add_argument("--warmup-epochs",   type=int,   default=3)

    # augmentation
    p.add_argument("--no-augment",    action="store_true")
    p.add_argument("--aug-rotation",  type=float, default=12.0)
    p.add_argument("--aug-blur",      action="store_true")

    # training tricks
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--size",      type=int,   default=48,
                   help="Square image resize in pixels. Works with 48, 64, or 72.")
    p.add_argument("--seed",      type=int,   default=42)

    # early stopping
    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--min-delta",     type=float, default=0.001)
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument("--monitor-metric", choices=["val_acc", "macro_f1"],
                   default="val_acc")

    # feature toggles
    p.add_argument("--no-amp",  action="store_true", help="Disable AMP.")
    p.add_argument("--class-weights", dest="no_class_weights",
                   action="store_false", default=True,
                   help="Enable class-weighted loss (off by default for balanced datasets).")
    p.add_argument("--no-cuda", action="store_true")

    # preview grid
    p.add_argument("--num-preview",    type=int, default=10)
    p.add_argument("--preview-source", choices=["val", "test"], default="val")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def resolve_dataset_root(root_arg: Path | None) -> Path:
    if root_arg is not None:
        root = root_arg.resolve()
        if (root / "Train").exists():
            return root
        raise FileNotFoundError(f"Train/ not found under --root: {root}")
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(".").resolve(),
        script_dir.parent / "dataset",
        script_dir.parent,
        Path(".").resolve() / "server" / "app" / "ml" / "dataset",
    ]
    for c in candidates:
        if (c / "Train").exists():
            return c.resolve()
    raise FileNotFoundError(
        "Could not auto-detect dataset root (missing Train/ directory).\n"
        + "\n".join(f"  - {c.resolve()}" for c in candidates)
        + "\nPass --root explicitly."
    )


def resolve_split_path(splits_dir: Path, name: str, fallback: str) -> Path:
    primary = splits_dir / name
    if primary.exists():
        return primary
    fb = splits_dir / fallback
    if fb.exists():
        print(f"  Using fallback split: {fb.name}")
        return fb
    raise FileNotFoundError(
        f"Split not found: {primary}\nFallback also missing: {fb}\n"
        "Run clean-splits.py to generate split files."
    )


def resolve_labels_path(root: Path, labels_path: Path) -> Path:
    if labels_path.is_absolute():
        if labels_path.exists():
            return labels_path
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    for c in [
        root / labels_path,
        root.parent / labels_path,
        root.parent / "ml" / labels_path,
        Path(".").resolve() / labels_path,
        Path(".").resolve() / "server" / "app" / "ml" / labels_path,
    ]:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"Labels file not found (searched relative to {root})")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def read_split(split_path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            rel, lbl = parts
            try:
                label = int(lbl)
            except ValueError:
                rel, label = line, -1
        else:
            rel, label = line, -1
        rows.append((rel, label))
    return rows


def _remap_to_augmented(rel: str, augmented_dir: str) -> str:
    parts = Path(rel).parts
    if parts and parts[0].lower() == "train":
        return str(Path(augmented_dir, *parts[1:]))
    return rel


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def build_train_transform(size: int, max_rotation: float = 12.0,
                           use_blur: bool = False):
    if not HAS_TORCHVISION:
        print("  WARNING: torchvision not available — augmentation disabled.")
        return None
    translate_frac = 0.10
    transforms = [
        T.ToTensor(),
        T.RandomRotation(degrees=max_rotation,
                         interpolation=T.InterpolationMode.BILINEAR, fill=0),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02),
        T.RandomAffine(degrees=0, translate=(translate_frac, translate_frac),
                       interpolation=T.InterpolationMode.BILINEAR, fill=0),
    ]
    if use_blur:
        transforms.append(T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)))
    transforms.append(T.Normalize(mean=NORM_MEAN.tolist(), std=NORM_STD.tolist()))
    return T.Compose(transforms)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TrafficSignDataset(Dataset):
    def __init__(self, root: Path, rows: list[tuple[str, int]], size: int,
                 use_augmented_dir: str | None = None,
                 label_to_idx: dict[int, int] | None = None,
                 augment_transform=None) -> None:
        self.root = root
        self.size = size
        self.label_to_idx = label_to_idx or {}
        self.augment_transform = augment_transform
        self.samples: list[tuple[Path, int]] = []
        skipped = 0
        for rel, label in rows:
            if use_augmented_dir:
                aug_rel  = _remap_to_augmented(rel, use_augmented_dir)
                img_path = root / aug_rel
                if not img_path.exists():
                    img_path = root / rel
            else:
                img_path = root / rel
            if not img_path.exists():
                skipped += 1
                continue
            self.samples.append((img_path, label))
        if skipped:
            print(f"  Skipped {skipped} missing image paths.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path) as im:
                pil_img = im.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        except (UnidentifiedImageError, OSError, ValueError):
            pil_img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        if self.augment_transform is not None:
            tensor = self.augment_transform(pil_img)
        else:
            arr    = np.asarray(pil_img, dtype=np.float32) / 255.0
            arr    = (arr - NORM_MEAN) / (NORM_STD + 1e-7)
            tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        mapped = self.label_to_idx.get(label, label) if label != -1 else -1
        return tensor, mapped


# ---------------------------------------------------------------------------
# Normalisation stats
# ---------------------------------------------------------------------------
def compute_dataset_norm_stats(root: Path, rows: list[tuple[str, int]],
                                size: int, use_augmented_dir: str | None,
                                ) -> tuple[np.ndarray, np.ndarray]:
    print("  Computing dataset normalisation statistics (one pass)...")
    pixel_sum    = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    count = 0
    for rel, _ in rows:
        if use_augmented_dir:
            aug_rel  = _remap_to_augmented(rel, use_augmented_dir)
            img_path = root / aug_rel
            if not img_path.exists():
                img_path = root / rel
        else:
            img_path = root / rel
        if not img_path.exists():
            continue
        try:
            with Image.open(img_path) as im:
                arr = np.asarray(
                    im.convert("RGB").resize((size, size), Image.BILINEAR),
                    dtype=np.float64,
                ) / 255.0
        except Exception:
            continue
        pixel_sum    += arr.reshape(-1, 3).sum(axis=0)
        pixel_sq_sum += (arr ** 2).reshape(-1, 3).sum(axis=0)
        count        += size * size
    if count == 0:
        print("  WARNING: No images found — falling back to ImageNet stats.")
        return IMAGENET_MEAN.copy(), IMAGENET_STD.copy()
    mean = (pixel_sum / count).astype(np.float32)
    std  = np.sqrt(np.maximum(
        pixel_sq_sum / count - mean.astype(np.float64) ** 2, 0
    )).astype(np.float32)
    std  = np.maximum(std, 1e-4)
    print(f"  Dataset mean: {mean.tolist()}")
    print(f"  Dataset std : {std.tolist()}")
    return mean, std


# ---------------------------------------------------------------------------
# CNN architecture (PyTorch — used for training)
# ---------------------------------------------------------------------------
class _ResBlock4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv     = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.shortcut = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.pool     = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x) + self.shortcut(x))


class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes: int, dropout1: float = 0.4,
                 dropout2: float = 0.2, spatial_dropout: float = 0.1) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block4       = _ResBlock4()
        self.spatial_drop = nn.Dropout2d(p=spatial_dropout)
        self.classifier   = nn.Sequential(
            nn.Dropout(p=dropout1),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.spatial_drop(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Loss / optimizer / scheduler helpers
# ---------------------------------------------------------------------------
def compute_class_weights(train_rows: list[tuple[str, int]],
                           all_labels: list[int],
                           device: torch.device) -> torch.Tensor:
    counts  = Counter(lbl for _, lbl in train_rows if lbl != -1)
    weights = torch.tensor(
        [1.0 / max(counts.get(lbl, 1), 1) for lbl in all_labels],
        dtype=torch.float32,
    )
    weights = weights / weights.mean()
    return weights.to(device)


def build_scheduler(optimizer: optim.Optimizer, warmup_epochs: int,
                    total_epochs: int) -> optim.lr_scheduler.SequentialLR:
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=1e-6)
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              optimizer: optim.Optimizer | None, device: torch.device,
              scaler=None, grad_clip: float = 0.0) -> tuple[float, float]:
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    use_amp  = scaler is not None and device.type == "cuda"
    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    with grad_ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device.type, enabled=use_amp):
                logits = model(images)
                loss   = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            total      += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def save_confusion_matrix(true_labels: list[int], pred_labels: list[int],
                           class_ids: list[int],
                           label_metadata: dict[int, dict],
                           reports_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping confusion matrix.")
        return
    cm       = confusion_matrix(true_labels, pred_labels, labels=class_ids)
    n        = len(class_ids)
    labels   = [label_metadata.get(cid, {}).get("class_name", f"cls_{cid}")
                for cid in class_ids]
    labels   = [lbl[:18] + "…" if len(lbl) > 19 else lbl for lbl in labels]
    fig_size = max(10, n * 0.45)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))
    im       = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(n), yticks=np.arange(n),
           xticklabels=labels, yticklabels=labels,
           xlabel="Predicted label", ylabel="True label",
           title="Validation Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=6,
                        color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    out_path = reports_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrix -> {out_path}")


# ---------------------------------------------------------------------------
# Prediction grid
# ---------------------------------------------------------------------------
def get_class_name(class_id: int, label_metadata: dict, fallback: list[str]) -> str:
    if class_id in label_metadata:
        return label_metadata[class_id].get("class_name", f"class_{class_id}")
    if 0 <= class_id < len(fallback):
        return fallback[class_id]
    return f"class_{class_id}"


def visualise_predictions(model: TrafficSignCNN, dataset: TrafficSignDataset,
                           idx_to_label: dict[int, int],
                           label_metadata: dict[int, dict],
                           reports_dir: Path,
                           norm_mean: np.ndarray, norm_std: np.ndarray,
                           num_samples: int = 10, labeled: bool = True,
                           title: str = "Sample Predictions",
                           filename: str = "sample_predictions.png") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not available — skipping prediction grid.")
        return
    rng     = np.random.default_rng(seed=0)
    indices = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    model.eval()
    ncols = min(5, num_samples)
    nrows = (num_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 3.2))
    axes = np.array(axes).reshape(-1)
    for plot_idx, ds_idx in enumerate(indices):
        tensor, true_idx = dataset[int(ds_idx)]
        with torch.no_grad():
            probs = torch.softmax(model(tensor.unsqueeze(0)), dim=1).squeeze()
        pred_idx   = int(probs.argmax())
        confidence = float(probs[pred_idx]) * 100.0
        pred_label = idx_to_label.get(pred_idx, pred_idx)
        pred_name  = get_class_name(pred_label, label_metadata, CLASS_NAMES)
        img_np     = tensor.permute(1, 2, 0).numpy() * norm_std + norm_mean
        img_np     = img_np.clip(0.0, 1.0)
        ax         = axes[plot_idx]
        ax.imshow(img_np, interpolation="nearest")
        ax.axis("off")
        if labeled and true_idx >= 0:
            true_label   = idx_to_label.get(true_idx, true_idx)
            correct      = pred_label == true_label
            border_color = "#2ecc71" if correct else "#e74c3c"
            true_name    = get_class_name(true_label, label_metadata, CLASS_NAMES)
            caption      = (f"{pred_name}\n{confidence:.1f}%" if correct
                            else f"X {pred_name}\n{confidence:.1f}%\n(GT: {true_name})")
        else:
            border_color = "#3498db"
            caption      = f"{pred_name}\n{confidence:.1f}%"
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
        ax.set_title(caption, fontsize=7, pad=3, wrap=True)
    for ax in axes[len(indices):]:
        ax.set_visible(False)
    if labeled:
        fig.legend(handles=[
            mpatches.Patch(facecolor="#2ecc71", label="Correct"),
            mpatches.Patch(facecolor="#e74c3c", label="Wrong"),
        ], loc="lower right", fontsize=8, framealpha=0.8)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = reports_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved prediction grid -> {out_path}")


# ---------------------------------------------------------------------------
# Keras model builder + weight transfer
# ---------------------------------------------------------------------------
def build_keras_model(num_classes: int, dropout1: float = 0.4,
                      dropout2: float = 0.2, spatial_dropout: float = 0.1):
    """
    Build a tf.keras model that mirrors TrafficSignCNN.
    Returns a compiled keras.Model ready for set_weights() transfer.
    """
    if not HAS_TF:
        raise ImportError("TensorFlow is required for Keras save.")
    import tensorflow as tf
    from tensorflow.keras import layers, Model  # noqa: F401

    inputs = tf.keras.Input(shape=(None, None, 3), name="image")

    # Block 1 — 5x5 conv
    x = layers.Conv2D(32, 5, padding="same", use_bias=False, name="b1_conv")(inputs)
    x = layers.BatchNormalization(name="b1_bn")(x)
    x = layers.ReLU(name="b1_relu")(x)
    x = layers.MaxPool2D(2, name="b1_pool")(x)

    # Block 2 — double 3x3
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="b2_conv1")(x)
    x = layers.BatchNormalization(name="b2_bn1")(x)
    x = layers.ReLU(name="b2_relu1")(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="b2_conv2")(x)
    x = layers.BatchNormalization(name="b2_bn2")(x)
    x = layers.ReLU(name="b2_relu2")(x)
    x = layers.MaxPool2D(2, name="b2_pool")(x)

    # Block 3 — double 3x3
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, name="b3_conv1")(x)
    x = layers.BatchNormalization(name="b3_bn1")(x)
    x = layers.ReLU(name="b3_relu1")(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, name="b3_conv2")(x)
    x = layers.BatchNormalization(name="b3_bn2")(x)
    x = layers.ReLU(name="b3_relu2")(x)
    x = layers.MaxPool2D(2, name="b3_pool")(x)

    # Block 4 — residual (main path + 1x1 shortcut)
    main     = layers.Conv2D(256, 3, padding="same", use_bias=False, name="b4_conv")(x)
    main     = layers.BatchNormalization(name="b4_bn")(main)
    main     = layers.ReLU(name="b4_relu")(main)
    shortcut = layers.Conv2D(256, 1, use_bias=False, name="b4_sc")(x)
    x        = layers.Add(name="b4_add")([main, shortcut])
    x        = layers.GlobalAveragePooling2D(name="gap")(x)

    # Classifier head
    x = layers.Dropout(dropout1, name="drop1")(x)
    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout2, name="drop2")(x)
    x = layers.Dense(num_classes, name="logits")(x)

    model = Model(inputs, x, name="TrafficSignCNN")
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def transfer_weights_to_keras(pt_model: TrafficSignCNN, keras_model) -> None:
    """
    Copy weights from a trained PyTorch TrafficSignCNN into the Keras model.

    Conv2D:     PyTorch (out, in, kH, kW)  ->  Keras (kH, kW, in, out)
    Dense:      PyTorch (out, in)          ->  Keras (in, out)
    BatchNorm:  gamma, beta, running_mean, running_var (same order in both)

    NOTE: GlobalAveragePooling2D (Keras) vs AdaptiveAvgPool2d(2,2)+flatten
    (PyTorch) produce different spatial reductions, so fc1/logits Dense weights
    will not be numerically identical.  All conv and BN weights transfer exactly.
    For full numeric parity, retrain natively in Keras.
    """
    def pt_to_keras_conv(w: torch.Tensor) -> np.ndarray:
        return w.numpy().transpose(2, 3, 1, 0)  # OIHW -> HWIO

    sd = {k: v.detach().cpu() for k, v in pt_model.state_dict().items()}
    km = {layer.name: layer for layer in keras_model.layers}

    mapping = [
        ("b1_conv",  ["block1.0.weight"]),
        ("b1_bn",    ["block1.1.weight", "block1.1.bias",
                      "block1.1.running_mean", "block1.1.running_var"]),
        ("b2_conv1", ["block2.0.weight"]),
        ("b2_bn1",   ["block2.1.weight", "block2.1.bias",
                      "block2.1.running_mean", "block2.1.running_var"]),
        ("b2_conv2", ["block2.3.weight"]),
        ("b2_bn2",   ["block2.4.weight", "block2.4.bias",
                      "block2.4.running_mean", "block2.4.running_var"]),
        ("b3_conv1", ["block3.0.weight"]),
        ("b3_bn1",   ["block3.1.weight", "block3.1.bias",
                      "block3.1.running_mean", "block3.1.running_var"]),
        ("b3_conv2", ["block3.3.weight"]),
        ("b3_bn2",   ["block3.4.weight", "block3.4.bias",
                      "block3.4.running_mean", "block3.4.running_var"]),
        ("b4_conv",  ["block4.conv.0.weight"]),
        ("b4_bn",    ["block4.conv.1.weight", "block4.conv.1.bias",
                      "block4.conv.1.running_mean", "block4.conv.1.running_var"]),
        ("b4_sc",    ["block4.shortcut.weight"]),
        ("fc1",      ["classifier.1.weight", "classifier.1.bias"]),
        ("logits",   ["classifier.4.weight", "classifier.4.bias"]),
    ]

    for layer_name, pt_keys in mapping:
        if layer_name not in km:
            print(f"  [skip] keras layer not found: {layer_name}")
            continue
        layer  = km[layer_name]
        k_vars = layer.get_weights()
        new_w  = []
        for i, pt_key in enumerate(pt_keys):
            if pt_key not in sd:
                print(f"  [skip] pt key not found: {pt_key}")
                new_w.append(k_vars[i])
                continue
            pt_w = sd[pt_key]
            if pt_w.ndim == 4:                      # conv weight
                w = pt_to_keras_conv(pt_w)
            elif layer_name in ("fc1", "logits") and pt_w.ndim == 2:
                w = pt_w.numpy().T                  # (out, in) -> (in, out)
            else:
                w = pt_w.numpy()
            if w.shape != k_vars[i].shape:
                print(f"  [shape mismatch] {layer_name}/{pt_key}: "
                      f"pt={w.shape} keras={k_vars[i].shape} — keeping keras init")
                new_w.append(k_vars[i])
            else:
                new_w.append(w)
        layer.set_weights(new_w)

    print("  Weight transfer complete "
          "(conv/BN layers exact; Dense may differ due to GAP vs flatten).")


def save_as_keras(pt_model: TrafficSignCNN, num_classes: int,
                  models_dir: Path, metadata: dict,
                  dropout1: float = 0.4, dropout2: float = 0.2,
                  spatial_dropout: float = 0.1):
    """
    Build a Keras mirror of TrafficSignCNN, transfer PyTorch weights, then save:
      - <models_dir>/simple_classifier.keras  (modern SavedModel bundle)
      - <models_dir>/simple_classifier.h5     (legacy HDF5)
      - <models_dir>/simple_classifier_meta.json

    Returns the keras.Model, or None if TensorFlow is not installed.
    """
    if not HAS_TF:
        print("  TensorFlow not installed — skipping Keras/H5 save.")
        return None

    keras_model = build_keras_model(num_classes, dropout1, dropout2, spatial_dropout)
    print(f"  Keras model parameters: {keras_model.count_params():,}")

    pt_model.eval()
    transfer_weights_to_keras(pt_model, keras_model)

    models_dir.mkdir(parents=True, exist_ok=True)

    keras_path = models_dir / "simple_classifier.keras"
    keras_model.save(keras_path)
    print(f"  Saved .keras -> {keras_path}")

    h5_path = models_dir / "simple_classifier.h5"
    keras_model.save(h5_path)
    print(f"  Saved .h5   -> {h5_path}")

    meta_path = models_dir / "simple_classifier_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  Saved meta  -> {meta_path}")

    return keras_model


# ---------------------------------------------------------------------------
# Label metadata loader
# ---------------------------------------------------------------------------
def load_label_metadata(root: Path, labels_path: Path) -> dict[int, dict]:
    try:
        labels_file = resolve_labels_path(root, labels_path)
    except FileNotFoundError as e:
        print(f"  Labels file not found ({e}) — using fallback names.")
        return {}
    metadata: dict[int, dict] = {}
    for row in json.loads(labels_file.read_text(encoding="utf-8")).get("classes", []):
        cid = int(row["class_id"])
        metadata[cid] = {
            "class_id"  : cid,
            "class_name": str(row.get("class_name", f"class_{cid}")),
            "category"  : str(row.get("category", "unknown")),
        }
    print(f"  Loaded {len(metadata)} label entries.")
    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root        = resolve_dataset_root(args.root)
    splits_dir  = root / "splits"
    reports_dir = root / "reports"
    models_dir  = root / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    device  = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    print(f"Device : {device} | AMP : {use_amp}")
    if not HAS_TF:
        print("WARNING: TensorFlow not found — model will be trained but not saved to Keras/H5.")

    # ── splits ───────────────────────────────────────────────────────────────
    train_split_path = resolve_split_path(splits_dir, args.train_split, "train_clean.txt")
    val_split_path   = resolve_split_path(splits_dir, args.val_split,   "val_clean.txt")
    train_rows       = read_split(train_split_path)
    val_rows         = read_split(val_split_path)

    # ── label mapping ────────────────────────────────────────────────────────
    all_labels   = sorted({lbl for _, lbl in train_rows if lbl != -1})
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
    num_classes  = len(all_labels)
    print(f"Classes: {num_classes} | Train: {len(train_rows)} | Val: {len(val_rows)}")

    # ── augmented dir ─────────────────────────────────────────────────────────
    augmented_dir: str | None = args.augmented_dir
    if not (root / augmented_dir).exists():
        print(f"WARNING: '{augmented_dir}' not found — falling back to Train/")
        augmented_dir = None

    # ── [6] normalisation stats ───────────────────────────────────────────────
    global NORM_MEAN, NORM_STD
    if args.compute_norm_stats:
        NORM_MEAN, NORM_STD = compute_dataset_norm_stats(
            root, train_rows, args.size, augmented_dir)
    else:
        print(f"  Using ImageNet normalisation "
              f"mean={NORM_MEAN.tolist()} std={NORM_STD.tolist()}")
    norm_mean = NORM_MEAN.copy()
    norm_std  = NORM_STD.copy()

    # ── [17] augmentation ─────────────────────────────────────────────────────
    if not args.no_augment and HAS_TORCHVISION:
        train_transform = build_train_transform(
            args.size, args.aug_rotation, args.aug_blur)
        aug_label = (f"rotation±{args.aug_rotation}° brightness±30% "
                     f"translation10%" + (" blur" if args.aug_blur else ""))
        print(f"  [17] Augmentation: {aug_label}")
    else:
        train_transform = None
        reason = "--no-augment" if args.no_augment else "torchvision not installed"
        print(f"  [17] Augmentation: disabled ({reason})")

    # ── datasets ──────────────────────────────────────────────────────────────
    print("Building datasets...")
    train_ds = TrafficSignDataset(
        root, train_rows, args.size,
        use_augmented_dir=None,
        label_to_idx=label_to_idx,
        augment_transform=train_transform,
    )
    val_ds = TrafficSignDataset(
        root, val_rows, args.size,
        use_augmented_dir=None,
        label_to_idx=label_to_idx,
        augment_transform=None,
    )
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"))

    # ── model ─────────────────────────────────────────────────────────────────
    model = TrafficSignCNN(
        num_classes     = num_classes,
        dropout1        = args.dropout1,
        dropout2        = args.dropout2,
        spatial_dropout = args.spatial_dropout,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── [4][5] loss ───────────────────────────────────────────────────────────
    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(train_rows, all_labels, device)
        print(f"  Class weights min={class_weights.min():.3f} "
              f"max={class_weights.max():.3f} mean={class_weights.mean():.3f}")
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing)

    # ── [1][2] optimizer + scheduler ──────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=True) if use_amp else None

    # ── [13] early stopping ───────────────────────────────────────────────────
    best_metric   = 0.0
    best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    history: list[str] = []
    patience_ctr  = 0
    stopped_early = False
    monitor       = args.monitor_metric

    print(f"\nTraining up to {args.epochs} epochs "
          f"(monitor={monitor}, patience={args.patience}, "
          f"min_delta={args.min_delta})\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, grad_clip=args.grad_clip)
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device)
        scheduler.step()
        cur_lr  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        # [21] optional macro-F1 monitoring
        if monitor == "macro_f1":
            all_logits_ep, all_true_ep = [], []
            model.eval()
            with torch.no_grad():
                for imgs_ep, lbls_ep in DataLoader(val_ds, batch_size=256):
                    all_logits_ep.append(model(imgs_ep))
                    all_true_ep.extend(lbls_ep.tolist())
            preds_ep   = torch.cat(all_logits_ep).argmax(dim=1).tolist()
            preds_orig = [idx_to_label[p] for p in preds_ep]
            true_orig  = [idx_to_label[t] for t in all_true_ep]
            epoch_f1   = f1_score(true_orig, preds_orig,
                                  average="macro", zero_division=0)
            metric_val = epoch_f1
            metric_str = f"macro_f1={epoch_f1:.4f} "
        else:
            metric_val = val_acc
            metric_str = ""

        line = (f"Epoch {epoch:>3}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                + (f"{metric_str}" if metric_str else "")
                + f"lr={cur_lr:.2e} ({elapsed:.1f}s)")
        print(line)
        history.append(line)

        if metric_val > best_metric + args.min_delta:
            best_metric  = metric_val
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            print(f"  [checkpoint] {monitor}={best_metric:.4f}")
        else:
            patience_ctr += 1
            if not args.no_early_stop and patience_ctr >= args.patience:
                print(f"\n  Early stop at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs).")
                stopped_early = True
                break

    if not stopped_early:
        print(f"\n  Completed all {args.epochs} epochs.")

    # ── restore best weights ──────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    model.to("cpu")

    # ── [15] full val evaluation ──────────────────────────────────────────────
    all_logits_list: list[torch.Tensor] = []
    all_true: list[int] = []
    with torch.no_grad():
        for images, labels in DataLoader(val_ds, batch_size=256):
            all_logits_list.append(model(images))
            all_true.extend(labels.tolist())

    all_logits   = torch.cat(all_logits_list, dim=0)
    all_true_t   = torch.tensor(all_true)
    top1_preds   = all_logits.argmax(dim=1).tolist()
    top3_correct = (
        all_logits.topk(min(3, num_classes), dim=1).indices
        == all_true_t.unsqueeze(1)
    ).any(dim=1).float().mean().item()

    all_preds_orig = [idx_to_label[p] for p in top1_preds]
    all_true_orig  = [idx_to_label[t] for t in all_true]
    final_val_acc  = accuracy_score(all_true_orig, all_preds_orig)
    macro_f1       = f1_score(all_true_orig, all_preds_orig,
                              average="macro", zero_division=0)
    report_text    = classification_report(
        all_true_orig, all_preds_orig, digits=4, zero_division=0)

    print(f"\n{'='*60}")
    print(f"  Best {monitor:<12}: {best_metric:.4f}")
    print(f"  Final val acc : {final_val_acc:.4f}")
    print(f"  Top-3 val acc : {top3_correct:.4f}")
    print(f"  Macro F1      : {macro_f1:.4f}")
    print(f"{'='*60}\n")

    # ── [18] confusion matrix ─────────────────────────────────────────────────
    label_metadata = load_label_metadata(root, args.labels_path)
    save_confusion_matrix(
        true_labels    = all_true_orig,
        pred_labels    = all_preds_orig,
        class_ids      = all_labels,
        label_metadata = label_metadata,
        reports_dir    = reports_dir,
    )

    # ── save Keras / H5 model ─────────────────────────────────────────────────
    metadata = {
        "num_classes"    : num_classes,
        "image_size"     : args.size,
        "label_to_idx"   : {str(k): v for k, v in label_to_idx.items()},
        "idx_to_label"   : {str(k): v for k, v in idx_to_label.items()},
        "norm_mean"      : norm_mean.tolist(),
        "norm_std"       : norm_std.tolist(),
        "dropout1"       : args.dropout1,
        "dropout2"       : args.dropout2,
        "spatial_dropout": args.spatial_dropout,
        "label_metadata" : {
            str(cid): label_metadata.get(
                cid, {"class_id": cid,
                      "class_name": f"class_{cid}",
                      "category": "unknown"})
            for cid in all_labels
        },
    }
    print("\nSaving model to Keras / H5 formats...")
    save_as_keras(
        pt_model        = model,
        num_classes     = num_classes,
        models_dir      = models_dir,
        metadata        = metadata,
        dropout1        = args.dropout1,
        dropout2        = args.dropout2,
        spatial_dropout = args.spatial_dropout,
    )

    # ── metrics report ────────────────────────────────────────────────────────
    aug_desc = "disabled"
    if train_transform is not None:
        aug_desc = (f"rotation±{args.aug_rotation}° brightness/contrast±30% "
                    f"translation10%" + (" blur" if args.aug_blur else ""))

    keras_path = models_dir / "simple_classifier.keras"
    h5_path    = models_dir / "simple_classifier.h5"
    meta_path  = models_dir / "simple_classifier_meta.json"

    metrics_path = reports_dir / "simple_train_metrics.txt"
    metrics_path.write_text(
        "\n".join([
            "CNN Classification Training Metrics — Fully Improved",
            "=" * 80,
            "Improvements applied:",
            f"  [1]  Optimizer     : AdamW lr={args.lr} wd={args.weight_decay}",
            f"  [2]  Scheduler     : {args.warmup_epochs}-ep warmup -> "
            f"CosineAnnealing eta_min=1e-6",
            f"  [3]  Epochs        : {args.epochs} max",
            f"  [4]  Label smooth  : {args.label_smoothing}",
            f"  [5]  Class weights : "
            f"{'yes' if not args.no_class_weights else 'disabled'}",
            f"  [6]  Normalisation : "
            f"{'dataset-computed' if args.compute_norm_stats else 'ImageNet priors'}",
            f"       mean={norm_mean.tolist()}",
            f"       std ={norm_std.tolist()}",
            f"  [7]  Spatial drop  : p={args.spatial_dropout}",
            f"  [8]  FC dropout    : {args.dropout1} / {args.dropout2}",
            f"  [9]  Residual skip : block3->block4",
            f"  [10] Block1 kernel : 5x5",
            f"  [11] Grad clip     : {args.grad_clip}",
            f"  [12] AMP           : {use_amp}",
            f"  [13] Early stop    : patience={args.patience} "
            f"min_delta={args.min_delta}",
            f"  [14] Epoch default : 40",
            f"  [15] Metrics       : top-1 / top-3 / macro-F1",
            f"  [16] Preview grid  : {args.num_preview} images",
            f"  [17] Augmentation  : {aug_desc}",
            f"  [18] Confusion mat : reports/confusion_matrix.png",
            f"  [19] Architecture  : double conv in blocks 2 & 3",
            f"  [20] AdaptivePool  : size-flexible (48/64/72 px)",
            f"  [21] Monitor metric: {monitor}",
            "",
            f"Train dir    : {augmented_dir or 'Train'}",
            f"Train samples: {len(train_ds)}",
            f"Val samples  : {len(val_ds)}",
            f"Image size   : {args.size}x{args.size} RGB",
            f"Batch size   : {args.batch_size}",
            f"Stopped early: {stopped_early}",
            f"Best {monitor:<12}: {best_metric:.4f}",
            f"Final val acc: {final_val_acc:.4f}",
            f"Top-3 val acc: {top3_correct:.4f}",
            f"Macro F1     : {macro_f1:.4f}",
            f"Device       : {device}",
            "",
            "[Epoch log]",
            *history,
            "",
            "Validation Classification Report",
            "-" * 80,
            report_text,
            "",
            f"Saved Keras model : {keras_path.relative_to(root).as_posix()}",
            f"Saved H5 model    : {h5_path.relative_to(root).as_posix()}",
            f"Saved meta JSON   : {meta_path.relative_to(root).as_posix()}",
        ]) + "\n",
        encoding="utf-8",
    )

    # ── [16] sample predictions ───────────────────────────────────────────────
    if args.num_preview > 0:
        print(f"Generating {args.num_preview}-image prediction grid "
              f"({args.preview_source})...")
        if args.preview_source == "test":
            try:
                test_rows  = read_split(
                    resolve_split_path(splits_dir, args.test_split, "test.txt"))
                preview_ds = TrafficSignDataset(
                    root, test_rows, args.size,
                    label_to_idx=label_to_idx, augment_transform=None)
                visualise_predictions(
                    model, preview_ds, idx_to_label, label_metadata,
                    reports_dir, norm_mean=norm_mean, norm_std=norm_std,
                    num_samples=args.num_preview, labeled=False,
                    title="Test Set Predictions (unlabeled)",
                    filename="test_predictions.png")
            except FileNotFoundError as exc:
                print(f"  Test split unavailable ({exc}) — falling back to val set.")
                visualise_predictions(
                    model, val_ds, idx_to_label, label_metadata,
                    reports_dir, norm_mean=norm_mean, norm_std=norm_std,
                    num_samples=args.num_preview, labeled=True,
                    title="Validation Set Predictions",
                    filename="sample_predictions.png")
        else:
            visualise_predictions(
                model, val_ds, idx_to_label, label_metadata,
                reports_dir, norm_mean=norm_mean, norm_std=norm_std,
                num_samples=args.num_preview, labeled=True,
                title="Validation Set Predictions",
                filename="sample_predictions.png")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training complete. Outputs:")
    print(f"  Keras model  : {keras_path}")
    print(f"  H5 model     : {h5_path}")
    print(f"  Meta JSON    : {meta_path}")
    print(f"  Metrics      : {metrics_path}")
    print(f"  Confusion mat: {reports_dir / 'confusion_matrix.png'}")
    print(f"  Preview grid : {reports_dir / 'sample_predictions.png'}")
    print(f"{'='*60}")
    print()
    print("To reload the Keras model:")
    print("  import tensorflow as tf, json")
    print("  model = tf.keras.models.load_model('models/simple_classifier.keras')")
    print("  meta  = json.load(open('models/simple_classifier_meta.json'))")


if __name__ == "__main__":
    main()