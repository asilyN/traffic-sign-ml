#!/usr/bin/env python3
"""
CNN traffic-sign classifier — fully improved training script.

Performance improvements applied (inherited + new)
----------------------------------------------------
  [1]  AdamW (decoupled weight decay) replaces Adam
  [2]  Linear warmup (default 3 ep) -> CosineAnnealingLR  eta_min=1e-6
  [3]  LR default 3e-4, wd default 1e-4  (tuned ranges in CLI help)
  [4]  CrossEntropyLoss with label_smoothing=0.1
  [5]  Class-weighted loss (optional; disable if dataset is already balanced)
  [6]  Normalisation: ImageNet priors OR computed dataset stats  *** NEW ***
  [7]  Spatial Dropout2d (p=0.1) after conv block 4
  [8]  Reduced FC dropout: 0.4 / 0.2  (was 0.5 / 0.3)
  [9]  Residual skip connection block 3 -> block 4  (1x1 projection)
  [10] 5x5 kernel in conv block 1  (better edge capture for signs)
  [11] Gradient clipping  max_norm=1.0
  [12] Mixed-precision training (AMP) on CUDA
  [13] Early stopping: patience=10, min_delta=0.001 — best_state pre-seeded  *** FIXED ***
  [14] Epochs default raised to 40  (early stopping is the real cutoff)
  [15] Top-1, Top-3 accuracy + macro-F1 reported
  [16] Sample prediction grid with green/red correctness borders
  [17] Training-time data augmentation (rotation, brightness, contrast,
       translation, optional blur) — val/test untouched  *** NEW ***
  [18] Confusion matrix saved to reports/confusion_matrix.png  *** NEW ***
  [19] CNN architecture improved: double conv in blocks 2 & 3  *** NEW ***
  [20] AdaptiveAvgPool2d ensures size-flexibility (48 / 64 / 72 px)  *** NEW ***
  [21] --monitor-metric flag: early-stop on val_acc OR macro_f1  *** NEW ***

Usage
-----
  python train_simple_classifier.py                          # all defaults
  python train_simple_classifier.py --compute-norm-stats     # dataset normalization
  python train_simple_classifier.py --no-class-weights       # pre-balanced dataset
  python train_simple_classifier.py --no-amp                 # disable AMP
  python train_simple_classifier.py --no-early-stop          # run full epoch count
  python train_simple_classifier.py --size 64 --lr 7e-4 --dropout1 0.3 --dropout2 0.1 --spatial-dropout 0.05
  python train_simple_classifier.py --monitor-metric macro_f1

Outputs
-------
  reports/simple_train_metrics.txt
  reports/sample_predictions.png
  reports/confusion_matrix.png          *** NEW ***
  models/simple_classifier.pt           (state-dict + metadata incl. norm stats)
  models/simple_classifier.joblib       (sklearn-compatible wrapper)
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# [17] torchvision transforms for augmentation — only applied to train set
try:
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ---------------------------------------------------------------------------
# [6] Normalisation constants
#     ImageNet priors used by default.  Pass --compute-norm-stats to derive
#     mean/std from the training images instead (recommended if your dataset
#     differs strongly in colour distribution from ImageNet).
# ---------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# These are overwritten at runtime when --compute-norm-stats is used
NORM_MEAN = IMAGENET_MEAN.copy()
NORM_STD  = IMAGENET_STD.copy()

# ---------------------------------------------------------------------------
# Class names (fallback when labels.json is unavailable)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Stop", "No Entry", "Left Curve Ahead", "Right Curve Ahead",
    "Reverse Turn Ahead", "Slippery Road", "Road Work",
    "Pedestrian Crossing", "Children Crossing", "Bike Lane",
    "Turn Right", "Turn Left", "Straight Ahead Only", "Pass on Right",
    "Pass on Left", "Roundabout", "U-Turn", "Speed Limit (60 km/h)",
    "Speed Limit (70 km/h)", "No Blowing of Horn", "No U-Turn",
    "Speed Limit (15 km/h)", "No Left Turn", "No Right Turn",
    "No Straight Proceeding", "No Parking", "Intersection Ahead",
    "Side Road Junction (Right)", "Speed Limit (20 km/h)",
    "Speed Limit (40 km/h)", "Pass on Either Side",
    "Side Road Junction (Diagonal)", "Speed Limit (30 km/h)",
    "No Entry for Four-Wheeled Vehicles", "Hospital",
    "Do Not Block Intersection", "Railroad Crossing",
    "Parking Area", "No Stopping / No Waiting", "T-Junction",
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train an improved CNN traffic-sign classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- paths ---
    p.add_argument("--root", type=Path, default=None,
                   help="Dataset root. Auto-detected if omitted.")
    p.add_argument("--train-split", default="train_clean.txt",
                   help="Train split file inside splits/.")
    p.add_argument("--val-split", default="val_clean.txt",
                   help="Val split file inside splits/.")
    p.add_argument("--test-split", default="test.txt",
                   help="Test split file inside splits/ (for --preview-source=test).")
    p.add_argument("--labels-path", type=Path, default=Path("labels.json"))
    p.add_argument("--augmented-dir", default="Train_Augmented_Balanced",
                   help="Sub-directory for augmented training images.")

    # --- [6] Normalisation ---
    p.add_argument("--compute-norm-stats", action="store_true",
                   help=(
                       "[NEW] Compute per-channel mean/std from the training set "
                       "instead of using ImageNet priors. Recommended when your "
                       "dataset colour distribution differs from ImageNet. "
                       "Adds one full pass over training images before training starts."
                   ))

    # --- Tier 1: highest-leverage hyperparameters (tune these first) ---
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Peak learning rate. Recommended sweep: [1e-4, 3e-3] log-uniform.")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="AdamW weight decay. Recommended sweep: [1e-5, 1e-2] log-uniform.")
    p.add_argument("--epochs", type=int, default=40,
                   help="Max training epochs. Early stopping is the real cutoff.")
    p.add_argument("--batch-size", type=int, default=64,
                   help=(
                       "Batch size. If you double the batch, scale LR up by ~sqrt(2). "
                       "Recommended range: [32, 128]."
                   ))

    # --- Tier 2: regularisation ---
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="Label smoothing epsilon in CrossEntropyLoss. Range [0.05, 0.15].")
    p.add_argument("--dropout1", type=float, default=0.4,
                   help="Dropout before FC layer 1. Range [0.3, 0.6].")
    p.add_argument("--dropout2", type=float, default=0.2,
                   help="Dropout before FC layer 2. Range [0.1, 0.3].")
    p.add_argument("--spatial-dropout", type=float, default=0.1,
                   help="Spatial (channel) dropout after conv block 4. Range [0.05, 0.2].")
    p.add_argument("--warmup-epochs", type=int, default=3,
                   help="Linear LR warmup epochs before cosine decay. Range [2, 5].")

    # --- [17] Augmentation ---
    p.add_argument("--no-augment", action="store_true",
                   help="[NEW] Disable training-time data augmentation.")
    p.add_argument("--aug-rotation", type=float, default=12.0,
                   help="[NEW] Max rotation degrees for augmentation. Range [5, 20].")
    p.add_argument("--aug-blur", action="store_true",
                   help="[NEW] Add random Gaussian blur to augmentation pipeline.")

    # --- Tier 3: training tricks ---
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max gradient norm for clipping. 0 = disabled.")
    p.add_argument("--size", type=int, default=48,
                   help=(
                       "Square image resize in pixels. Model uses AdaptiveAvgPool2d "
                       "so it works with 48, 64, or 72 without code changes."
                   ))
    p.add_argument("--seed", type=int, default=42)

    # --- [13] Early stopping ---
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping: epochs allowed without improvement.")
    p.add_argument("--min-delta", type=float, default=0.001,
                   help="Minimum improvement to reset patience counter.")
    p.add_argument("--no-early-stop", action="store_true",
                   help="Disable early stopping and run all --epochs.")
    # [21] Monitor metric for early stopping
    p.add_argument("--monitor-metric", choices=["val_acc", "macro_f1"],
                   default="val_acc",
                   help=(
                       "[NEW] Metric to monitor for early stopping and checkpointing. "
                       "'val_acc' is faster to compute per epoch; 'macro_f1' is more "
                       "informative for imbalanced datasets."
                   ))

    # --- feature toggles ---
    p.add_argument("--no-amp", action="store_true",
                   help="Disable mixed-precision training (AMP).")
    # [5] Class weights: OFF by default because Train_Augmented_Balanced is already balanced.
    #     Pass --class-weights to re-enable if using a different (imbalanced) dataset.
    p.add_argument("--class-weights", dest="no_class_weights", action="store_false",
                   default=True,
                   help=(
                       "Enable class-weighted loss. "
                       "OFF by default because Train_Augmented_Balanced is already "
                       "class-balanced — applying weights on top of a balanced dataset "
                       "adds no benefit and may hurt accuracy. "
                       "Enable only when using an imbalanced dataset."
                   ))
    p.add_argument("--no-cuda", action="store_true",
                   help="Force CPU even if CUDA is available.")

    # --- preview grid ---
    p.add_argument("--num-preview", type=int, default=10,
                   help="Number of images in the prediction grid. 0 = skip.")
    p.add_argument("--preview-source", choices=["val", "test"], default="val",
                   help="Dataset to draw preview images from.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Path helpers  (unchanged from original)
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
# Dataset helpers  (unchanged)
# ---------------------------------------------------------------------------
def read_split(split_path: Path) -> list[tuple[str, int]]:
    """Parse a split file. Lines without a label get label=-1 (unlabeled test set)."""
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
    """Rewrite 'Train/3/img.png' -> '<augmented_dir>/3/img.png'."""
    parts = Path(rel).parts
    if parts and parts[0].lower() == "train":
        return str(Path(augmented_dir, *parts[1:]))
    return rel


# ---------------------------------------------------------------------------
# [17] Training augmentation pipeline
#      Light but realistic transforms applied ONLY to the training set.
#      Augmentation philosophy for traffic signs:
#        - Rotation ≤ 15°: signs tilt slightly due to camera angle, not more.
#        - Brightness/contrast ±30%: varies with weather and lighting.
#        - Translation ≤ 10%: sign may not be perfectly centred.
#        - No horizontal flip: would invert arrow directions (wrong label).
#        - No large crops: would remove the sign from frame.
#        - Blur optional: simulates motion blur / low-resolution cameras.
# ---------------------------------------------------------------------------
def build_train_transform(
    size: int,
    max_rotation: float = 12.0,
    use_blur: bool = False,
) -> "T.Compose | None":
    """
    [17] Returns a torchvision transform for training augmentation.
    Returns None if torchvision is not installed (degrades gracefully).
    """
    if not HAS_TORCHVISION:
        print("  WARNING: torchvision not available — augmentation disabled.")
        return None

    # Translation: up to 10% of image dimension in each axis
    translate_frac = 0.10

    transforms = [
        # Convert PIL Image to tensor first so we can use tensor-based ops
        T.ToTensor(),
        # [17a] Random rotation — keep small to preserve sign semantics
        T.RandomRotation(degrees=max_rotation, interpolation=T.InterpolationMode.BILINEAR, fill=0),
        # [17b] Random brightness and contrast — simulates lighting variation
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02),
        # [17c] Slight translation — sign may be off-centre
        T.RandomAffine(
            degrees=0,
            translate=(translate_frac, translate_frac),
            interpolation=T.InterpolationMode.BILINEAR,
            fill=0,
        ),
    ]
    if use_blur:
        # [17d] Optional Gaussian blur — simulates out-of-focus or motion blur
        transforms.append(T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)))

    # Normalise AFTER all geometric/colour ops (on tensor [0,1])
    # NOTE: mean/std are set globally before this function is called
    transforms.append(T.Normalize(mean=NORM_MEAN.tolist(), std=NORM_STD.tolist()))
    return T.Compose(transforms)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TrafficSignDataset(Dataset):
    """
    Loads traffic-sign images with per-channel normalisation.

    [6]  Images are normalised using either ImageNet priors or dataset-specific
         mean/std computed by compute_dataset_norm_stats().
    [17] Training set optionally receives augmentation transforms.
         Validation/test sets always use the plain normalisation pipeline.
    """
    def __init__(
        self,
        root: Path,
        rows: list[tuple[str, int]],
        size: int,
        use_augmented_dir: str | None = None,
        label_to_idx: dict[int, int] | None = None,
        augment_transform=None,          # [17] None for val/test
    ) -> None:
        self.root             = root
        self.size             = size
        self.use_augmented_dir = use_augmented_dir
        self.label_to_idx     = label_to_idx or {}
        self.augment_transform = augment_transform  # [17]
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
            pil_img = Image.fromarray(
                np.zeros((self.size, self.size, 3), dtype=np.uint8)
            )

        if self.augment_transform is not None:
            # [17] Apply augmentation + normalisation together
            tensor = self.augment_transform(pil_img)
        else:
            # [6] Plain path: float32, normalise, HWC->CHW
            arr = np.asarray(pil_img, dtype=np.float32) / 255.0
            arr = (arr - NORM_MEAN) / (NORM_STD + 1e-7)
            tensor = torch.from_numpy(arr.transpose(2, 0, 1))

        mapped = self.label_to_idx.get(label, label) if label != -1 else -1
        return tensor, mapped


# ---------------------------------------------------------------------------
# [6] Dataset-specific normalisation statistics
# ---------------------------------------------------------------------------
def compute_dataset_norm_stats(
    root: Path,
    rows: list[tuple[str, int]],
    size: int,
    use_augmented_dir: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    [6] Compute per-channel mean and std from the raw training images.

    Why it helps: ImageNet priors are reasonable but traffic-sign datasets
    are usually dominated by specific colours (red, white, yellow) and may
    have a different dynamic range.  Dataset-specific stats improve BatchNorm
    convergence and can give +0.5-1% val acc with no other changes.

    Returns: (mean, std) each shape (3,) float32, values in [0, 1].
    """
    print("  Computing dataset normalisation statistics (one pass)...")
    pixel_sum    = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    count        = 0

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
    std  = np.sqrt(np.maximum(pixel_sq_sum / count - mean.astype(np.float64) ** 2, 0)).astype(np.float32)
    std  = np.maximum(std, 1e-4)   # avoid division by zero
    print(f"  Dataset mean : {mean.tolist()}")
    print(f"  Dataset std  : {std.tolist()}")
    return mean, std


# ---------------------------------------------------------------------------
# [19] Improved CNN architecture
#      Changes vs. original:
#        Block 2: double 3x3 conv (32->64->64) — richer mid-level features
#        Block 3: double 3x3 conv (64->128->128) — better edge/texture combos
#        Block 4: residual skip unchanged (block3->block4)
#        Classifier: input size computed dynamically via AdaptiveAvgPool2d [20]
# ---------------------------------------------------------------------------
class _ResBlock4(nn.Module):
    """
    [9] Conv block 4 with a residual skip connection from block 3 output.
    A 1x1 projection shortcut lets gradients bypass the 3x3 conv.
    AdaptiveAvgPool2d(2, 2) makes the output shape independent of input size. [20]
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # 1x1 projection to match channel dimension
        self.shortcut = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        # [20] AdaptiveAvgPool: output is always 2x2 regardless of input spatial size
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x) + self.shortcut(x))


class TrafficSignCNN(nn.Module):
    """
    Improved 4-block CNN for traffic-sign classification.

    Architecture changes vs. the previous version
    -----------------------------------------------
    Block 1      : 5x5 kernel (better initial receptive field for signs)   [10]
    Block 2      : double 3x3 conv  32->64->64                             [19] NEW
    Block 3      : double 3x3 conv  64->128->128                           [19] NEW
    Block 4      : residual skip block3->block4  (unchanged)               [9]
    After block4 : Dropout2d spatial dropout                               [7]
    FC head      : dropout 0.4 / 0.2                                       [8]
    Pool         : AdaptiveAvgPool2d — works with 48, 64, 72 px inputs     [20]

    Why double-conv blocks help:
      Two successive 3x3 convs see the same 5x5 receptive field as one 5x5
      conv but with an extra non-linearity + BatchNorm between them.  This
      gives the network more capacity to learn complex mid-level patterns
      (border shapes, inner symbols of traffic signs) with minimal extra
      parameters (~+30k total vs baseline ~550k).

    Spatial layout (48x48 input example):
        block1 -> MaxPool ->  24x24 @  32 ch
        block2 -> MaxPool ->  12x12 @  64 ch  (double conv)
        block3 -> MaxPool ->   6x6  @ 128 ch  (double conv)
        block4 (_ResBlock4) -> AdaptiveAvgPool -> 2x2 @ 256 ch
        flatten -> 1024 -> FC(512) -> FC(num_classes)
    """
    def __init__(
        self,
        num_classes: int,
        dropout1: float = 0.4,
        dropout2: float = 0.2,
        spatial_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # [10] 5x5 first kernel: wider initial receptive field for sign shapes
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 48 -> 24
        )

        # [19] NEW: double conv in block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   # second conv
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 24 -> 12
        )

        # [19] NEW: double conv in block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # second conv
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 12 -> 6
        )

        # [9][20] Block 4 with residual skip + AdaptiveAvgPool
        self.block4 = _ResBlock4()        # 6x6 -> 2x2

        # [7] Spatial dropout: zeros entire feature maps — better for CNNs
        #     than scalar dropout because CNN features are spatially correlated
        self.spatial_drop = nn.Dropout2d(p=spatial_dropout)

        # [8] FC head with reduced dropout (0.4/0.2 vs original 0.5/0.3)
        # 256 * 2 * 2 = 1024 (always, thanks to AdaptiveAvgPool2d)
        self.classifier = nn.Sequential(
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
# Sklearn-compatible wrapper  (keeps predict-simple-classifier.py working)
# ---------------------------------------------------------------------------
class CNNSklearnWrapper:
    """
    Thin wrapper exposing predict() / predict_proba() like an sklearn pipeline.

    [6]  Stores norm_mean / norm_std from training so inference uses the same
         statistics — critical when --compute-norm-stats was used.
    Accepts flattened numpy arrays of shape (N, size*size*3), values in [0, 1].
    Returns original class IDs (not contiguous training indices).
    """
    def __init__(
        self,
        model: TrafficSignCNN,
        size: int,
        classes: list[int],
        idx_to_label: dict[int, int],
        norm_mean: np.ndarray,   # [6] stored per-model
        norm_std: np.ndarray,    # [6] stored per-model
    ) -> None:
        self.model         = model
        self.size          = size
        self.classes_      = np.asarray(classes)
        self._idx_to_label = idx_to_label
        self._norm_mean    = norm_mean   # [6]
        self._norm_std     = norm_std    # [6]

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        # x: (N, size*size*3) float32, values in [0, 1]
        arr = x.reshape(-1, self.size, self.size, 3).astype(np.float32)
        # [6] Apply the SAME normalisation used during training
        arr = (arr - self._norm_mean) / (self._norm_std + 1e-7)
        return torch.from_numpy(arr.transpose(0, 3, 1, 2))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._to_tensor(x)).numpy()
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        idx = self.predict_proba(x).argmax(axis=1)
        return np.asarray([self._idx_to_label.get(int(i), int(i)) for i in idx])


# ---------------------------------------------------------------------------
# Label metadata  (unchanged)
# ---------------------------------------------------------------------------
def load_label_metadata(root: Path, labels_path: Path) -> dict[int, dict]:
    labels_file = resolve_labels_path(root, labels_path)
    metadata: dict[int, dict] = {}
    for row in json.loads(labels_file.read_text(encoding="utf-8")).get("classes", []):
        cid = int(row["class_id"])
        metadata[cid] = {
            "class_id":   cid,
            "class_name": str(row.get("class_name", f"class_{cid}")),
            "category":   str(row.get("category", "unknown")),
        }
    return metadata


# ---------------------------------------------------------------------------
# [5] Class-weighted loss
# ---------------------------------------------------------------------------
def compute_class_weights(
    train_rows: list[tuple[str, int]],
    all_labels: list[int],
    device: torch.device,
) -> torch.Tensor:
    """
    [5] Inverse-frequency weights normalised so their mean == 1.

    NOTE: Disable with --no-class-weights when using Train_Augmented_Balanced.
    An already-balanced dataset + class weights can over-correct and reduce
    accuracy on naturally frequent classes.  Only valid training rows
    (label != -1) are counted.
    """
    counts  = Counter(lbl for _, lbl in train_rows if lbl != -1)
    weights = torch.tensor(
        [1.0 / max(counts.get(lbl, 1), 1) for lbl in all_labels],
        dtype=torch.float32,
    )
    weights = weights / weights.mean()   # normalise: mean == 1
    return weights.to(device)


# ---------------------------------------------------------------------------
# [2] Scheduler: linear warmup -> CosineAnnealingLR
# ---------------------------------------------------------------------------
def build_scheduler(
    optimizer: optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> optim.lr_scheduler.SequentialLR:
    """
    [2] Linear warmup from 10% of peak LR, then cosine decay to eta_min=1e-6.
    Warmup prevents large gradient steps in epoch 1 when BatchNorm running
    stats are still unstable.
    """
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - warmup_epochs, 1),
        eta_min=1e-6,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


# ---------------------------------------------------------------------------
# [11][12] Training loop with AMP + gradient clipping
# ---------------------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
    grad_clip: float = 0.0,
) -> tuple[float, float]:
    """
    Single training or evaluation pass.
    [11] Gradient clipping prevents rare large spikes from destabilising BN.
    [12] AMP gives ~1.5-2x speedup on CUDA with identical accuracy.
    """
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = scaler is not None and device.type == "cuda"
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
# [18] Confusion matrix
# ---------------------------------------------------------------------------
def save_confusion_matrix(
    true_labels: list[int],
    pred_labels: list[int],
    class_ids: list[int],
    label_metadata: dict[int, dict],
    reports_dir: Path,
) -> None:
    """
    [18] Save a colour-coded confusion matrix to reports/confusion_matrix.png.

    Why it helps evaluation: top-1 accuracy hides systematic class confusions
    (e.g. Speed Limit 20 vs 30).  The confusion matrix reveals which pairs
    of classes are most often mixed up, guiding targeted augmentation.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping confusion matrix.")
        return

    cm     = confusion_matrix(true_labels, pred_labels, labels=class_ids)
    n      = len(class_ids)
    labels = [
        label_metadata.get(cid, {}).get("class_name", f"cls_{cid}")
        for cid in class_ids
    ]
    # Truncate long names for readability
    labels = [lbl[:18] + "…" if len(lbl) > 19 else lbl for lbl in labels]

    fig_size = max(10, n * 0.45)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))
    im       = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="Validation Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)

    # Annotate cells where count > 0 (skip zeros for readability)
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > thresh else "black",
                )

    plt.tight_layout()
    out_path = reports_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrix -> {out_path}")


# ---------------------------------------------------------------------------
# [16] Sample predictions visualisation  (unchanged from original)
# ---------------------------------------------------------------------------
def get_class_name(class_id: int, label_metadata: dict, fallback: list[str]) -> str:
    if class_id in label_metadata:
        return label_metadata[class_id].get("class_name", f"class_{class_id}")
    if 0 <= class_id < len(fallback):
        return fallback[class_id]
    return f"class_{class_id}"


def visualise_predictions(
    model: TrafficSignCNN,
    dataset: TrafficSignDataset,
    idx_to_label: dict[int, int],
    label_metadata: dict[int, dict],
    reports_dir: Path,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    num_samples: int = 10,
    labeled: bool = True,
    title: str = "Sample Predictions",
    filename: str = "sample_predictions.png",
) -> None:
    """
    [16] Grid of sample images with:
      - predicted class name and softmax confidence
      - green border = correct, red border = wrong, blue = unlabeled test
      - ground-truth label shown for wrong predictions
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not available — skipping prediction visualisation.")
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

        # Reverse normalisation for display: x = arr * std + mean
        img_np = tensor.permute(1, 2, 0).numpy() * norm_std + norm_mean
        img_np = img_np.clip(0.0, 1.0)

        ax = axes[plot_idx]
        ax.imshow(img_np, interpolation="nearest")
        ax.axis("off")

        if labeled and true_idx >= 0:
            true_label   = idx_to_label.get(true_idx, true_idx)
            correct      = pred_label == true_label
            border_color = "#2ecc71" if correct else "#e74c3c"
            true_name    = get_class_name(true_label, label_metadata, CLASS_NAMES)
            caption = (
                f"{pred_name}\n{confidence:.1f}%"
                if correct
                else f"X {pred_name}\n{confidence:.1f}%\n(GT: {true_name})"
            )
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
        fig.legend(
            handles=[
                mpatches.Patch(facecolor="#2ecc71", label="Correct"),
                mpatches.Patch(facecolor="#e74c3c", label="Wrong"),
            ],
            loc="lower right", fontsize=8, framealpha=0.8,
        )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = reports_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved prediction grid -> {out_path}")


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

    device  = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    print(f"Device : {device}  |  AMP : {use_amp}")

    # ── splits ────────────────────────────────────────────────────────────────
    train_split_path = resolve_split_path(splits_dir, args.train_split, "train_clean.txt")
    val_split_path   = resolve_split_path(splits_dir, args.val_split,   "val_clean.txt")
    train_rows       = read_split(train_split_path)
    val_rows         = read_split(val_split_path)

    # ── label mapping (contiguous 0-based for CrossEntropyLoss) ──────────────
    all_labels   = sorted({lbl for _, lbl in train_rows if lbl != -1})
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
    num_classes  = len(all_labels)
    print(f"Classes: {num_classes}  |  Train: {len(train_rows)}  |  Val: {len(val_rows)}")

    # ── augmented dir ─────────────────────────────────────────────────────────
    augmented_dir: str | None = args.augmented_dir
    if not (root / augmented_dir).exists():
        print(f"WARNING: '{augmented_dir}' not found — falling back to Train/")
        augmented_dir = None

    # ── [6] Normalisation statistics ──────────────────────────────────────────
    global NORM_MEAN, NORM_STD
    if args.compute_norm_stats:
        NORM_MEAN, NORM_STD = compute_dataset_norm_stats(
            root, train_rows, args.size, augmented_dir
        )
    else:
        print(f"  Using ImageNet normalisation  mean={NORM_MEAN.tolist()}  std={NORM_STD.tolist()}")

    # Keep local copies so they can be safely embedded in saved artefacts
    norm_mean = NORM_MEAN.copy()
    norm_std  = NORM_STD.copy()

    # ── [17] Augmentation transform ───────────────────────────────────────────
    if not args.no_augment and HAS_TORCHVISION:
        train_transform = build_train_transform(
            size=args.size,
            max_rotation=args.aug_rotation,
            use_blur=args.aug_blur,
        )
        aug_label = (
            f"rotation±{args.aug_rotation}°  brightness±30%  translation10%"
            + ("  blur" if args.aug_blur else "")
        )
        print(f"  [17] Augmentation : {aug_label}")
    else:
        train_transform = None
        if args.no_augment:
            print("  [17] Augmentation : disabled (--no-augment)")
        else:
            print("  [17] Augmentation : disabled (torchvision not installed)")

    # ── datasets ──────────────────────────────────────────────────────────────
    # NOTE: split files (train_clean.txt / val_clean.txt) already contain paths
    # relative to root that start with 'Train_Augmented_Balanced/' directly, so
    # use_augmented_dir=None for train_ds — no path rewriting needed.
    print("Building datasets...")
    train_ds = TrafficSignDataset(
        root, train_rows, args.size,
        use_augmented_dir=None,              # paths in split already point to Train_Augmented_Balanced/
        label_to_idx=label_to_idx,
        augment_transform=train_transform,   # [17] train only
    )
    val_ds = TrafficSignDataset(
        root, val_rows, args.size,
        use_augmented_dir=None,              # val also uses Train_Augmented_Balanced/ via split paths
        label_to_idx=label_to_idx,
        augment_transform=None,              # [17] no augmentation on val
    )
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    # ── [19][20] model ────────────────────────────────────────────────────────
    model = TrafficSignCNN(
        num_classes=num_classes,
        dropout1=args.dropout1,
        dropout2=args.dropout2,
        spatial_dropout=args.spatial_dropout,
    ).to(device)

    # ── [4][5] loss: label smoothing + optional class weights ─────────────────
    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(train_rows, all_labels, device)
        print(f"  Class weights — min={class_weights.min():.3f}  "
              f"max={class_weights.max():.3f}  mean={class_weights.mean():.3f}")
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    # ── [1] optimizer: AdamW ──────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # ── [2] scheduler: warmup -> cosine ───────────────────────────────────────
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)

    # ── [12] AMP scaler ───────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp else None

    # ── [13][21] early stopping — pre-seed best_state with initial weights ────
    #    FIX: initialise best_state BEFORE the loop so it is never empty.
    #    The original code could leave best_state = {} if val_acc never
    #    improved by min_delta in epoch 1, causing load_state_dict to fail.
    best_metric      = 0.0
    best_state: dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    history: list[str] = []
    patience_counter = 0
    stopped_early    = False
    monitor          = args.monitor_metric   # [21] "val_acc" or "macro_f1"

    print(
        f"\nTraining up to {args.epochs} epochs  "
        f"(monitor={monitor}, patience={args.patience}, min_delta={args.min_delta})\n"
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, grad_clip=args.grad_clip,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device,
        )
        scheduler.step()
        cur_lr  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        # [21] Compute macro_f1 per epoch only when monitoring it
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
            epoch_f1   = f1_score(true_orig, preds_orig, average="macro", zero_division=0)
            metric_val = epoch_f1
            metric_str = f"macro_f1={epoch_f1:.4f}"
        else:
            metric_val = val_acc
            metric_str = ""

        line = (
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            + (f"{metric_str}  " if metric_str else "")
            + f"lr={cur_lr:.2e}  ({elapsed:.1f}s)"
        )
        print(line)
        history.append(line)

        # [13][21] Checkpoint + early stopping
        if metric_val > best_metric + args.min_delta:
            best_metric      = metric_val
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  [checkpoint] {monitor}={best_metric:.4f}")
        else:
            patience_counter += 1
            if not args.no_early_stop and patience_counter >= args.patience:
                print(f"\n  Early stop at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs).")
                stopped_early = True
                break

    if not stopped_early:
        print(f"\n  Completed all {args.epochs} epochs.")

    # ── restore best weights ──────────────────────────────────────────────────
    # best_state is guaranteed non-empty because it was seeded before the loop
    model.load_state_dict(best_state)
    model.eval()
    model.to("cpu")

    # ── [15] full val evaluation: top-1, top-3, macro-F1 ─────────────────────
    all_logits_list: list[torch.Tensor] = []
    all_true: list[int] = []
    with torch.no_grad():
        for images, labels in DataLoader(val_ds, batch_size=256):
            all_logits_list.append(model(images))
            all_true.extend(labels.tolist())

    all_logits = torch.cat(all_logits_list, dim=0)
    all_true_t = torch.tensor(all_true)

    top1_preds   = all_logits.argmax(dim=1).tolist()
    top3_correct = (
        all_logits.topk(min(3, num_classes), dim=1).indices
        == all_true_t.unsqueeze(1)
    ).any(dim=1).float().mean().item()

    all_preds_orig = [idx_to_label[p] for p in top1_preds]
    all_true_orig  = [idx_to_label[t] for t in all_true]

    final_val_acc = accuracy_score(all_true_orig, all_preds_orig)
    macro_f1      = f1_score(all_true_orig, all_preds_orig, average="macro", zero_division=0)
    report_text   = classification_report(
        all_true_orig, all_preds_orig, digits=4, zero_division=0
    )
    best_val_acc  = best_metric if monitor == "val_acc" else final_val_acc

    print(f"\n{'='*60}")
    print(f"  Best {monitor:<12}: {best_metric:.4f}")
    print(f"  Final val acc  : {final_val_acc:.4f}")
    print(f"  Top-3 val acc  : {top3_correct:.4f}")
    print(f"  Macro F1       : {macro_f1:.4f}")
    print(f"{'='*60}\n")

    # ── [18] Confusion matrix ─────────────────────────────────────────────────
    label_metadata = load_label_metadata(root, args.labels_path)
    save_confusion_matrix(
        true_labels=all_true_orig,
        pred_labels=all_preds_orig,
        class_ids=all_labels,
        label_metadata=label_metadata,
        reports_dir=reports_dir,
    )

    # ── save PyTorch model ────────────────────────────────────────────────────
    pt_path = models_dir / "simple_classifier.pt"
    torch.save(
        {
            "state_dict":      best_state,
            "num_classes":     num_classes,
            "image_size":      args.size,
            "label_to_idx":    label_to_idx,
            "idx_to_label":    idx_to_label,
            # [6] Store the actual normalisation stats used during training
            "norm_mean":       norm_mean.tolist(),
            "norm_std":        norm_std.tolist(),
            "dropout1":        args.dropout1,
            "dropout2":        args.dropout2,
            "spatial_dropout": args.spatial_dropout,
        },
        pt_path,
    )

    # ── save joblib wrapper ───────────────────────────────────────────────────
    wrapper = CNNSklearnWrapper(
        model=model,
        size=args.size,
        classes=all_labels,
        idx_to_label=idx_to_label,
        norm_mean=norm_mean,   # [6] match training stats
        norm_std=norm_std,     # [6] match training stats
    )
    joblib_path = models_dir / "simple_classifier.joblib"
    joblib.dump(
        {
            "pipeline":       wrapper,
            "image_size":     args.size,
            "labels_present": all_labels,
            "label_metadata": {
                cid: label_metadata.get(
                    cid,
                    {"class_id": cid, "class_name": f"class_{cid}", "category": "unknown"},
                )
                for cid in all_labels
            },
        },
        joblib_path,
    )

    # ── write metrics report ──────────────────────────────────────────────────
    metrics_path = reports_dir / "simple_train_metrics.txt"
    aug_desc = "disabled"
    if train_transform is not None:
        aug_desc = (
            f"rotation±{args.aug_rotation}°  brightness/contrast±30%  "
            f"translation 10%"
            + ("  blur" if args.aug_blur else "")
        )
    metrics_path.write_text(
        "\n".join([
            "CNN Classification Training Metrics — Fully Improved",
            "=" * 80,
            "Improvements applied:",
            f"  [1]  Optimizer      : AdamW  lr={args.lr}  wd={args.weight_decay}",
            f"  [2]  Scheduler      : {args.warmup_epochs}-ep warmup -> CosineAnnealing  eta_min=1e-6",
            f"  [3]  Epochs         : {args.epochs} max",
            f"  [4]  Label smooth   : {args.label_smoothing}",
            f"  [5]  Class weights  : {'yes' if not args.no_class_weights else 'disabled'}",
            f"  [6]  Normalisation  : {'dataset-computed' if args.compute_norm_stats else 'ImageNet priors'}",
            f"         mean={norm_mean.tolist()}",
            f"         std ={norm_std.tolist()}",
            f"  [7]  Spatial drop   : p={args.spatial_dropout}",
            f"  [8]  FC dropout     : {args.dropout1} / {args.dropout2}",
            f"  [9]  Residual skip  : block3->block4",
            f"  [10] Block1 kernel  : 5x5",
            f"  [11] Grad clip      : {args.grad_clip}",
            f"  [12] AMP            : {use_amp}",
            f"  [13] Early stop     : patience={args.patience}  min_delta={args.min_delta}  (best_state pre-seeded)",
            f"  [14] Epoch default  : 40",
            f"  [15] Metrics        : top-1 / top-3 / macro-F1",
            f"  [16] Preview grid   : {args.num_preview} images",
            f"  [17] Augmentation   : {aug_desc}",
            f"  [18] Confusion mat  : reports/confusion_matrix.png",
            f"  [19] Architecture   : double conv in blocks 2 & 3",
            f"  [20] AdaptivePool   : size-flexible (48/64/72 px)",
            f"  [21] Monitor metric : {monitor}",
            "",
            f"Train dir       : {augmented_dir or 'Train'}",
            f"Train samples   : {len(train_ds)}",
            f"Val samples     : {len(val_ds)}",
            f"Image size      : {args.size}x{args.size} RGB",
            f"Batch size      : {args.batch_size}",
            f"Stopped early   : {stopped_early}",
            f"Best {monitor:<12}: {best_metric:.4f}",
            f"Final val acc   : {final_val_acc:.4f}",
            f"Top-3 val acc   : {top3_correct:.4f}",
            f"Macro F1        : {macro_f1:.4f}",
            f"Device          : {device}",
            "",
            "[Epoch log]",
            *history,
            "",
            "Validation Classification Report",
            "-" * 80,
            report_text,
            "",
            f"Saved PyTorch model : {pt_path.relative_to(root).as_posix()}",
            f"Saved joblib wrapper: {joblib_path.relative_to(root).as_posix()}",
        ]) + "\n",
        encoding="utf-8",
    )

    # ── [16] sample predictions ───────────────────────────────────────────────
    if args.num_preview > 0:
        print(f"Generating {args.num_preview}-image prediction grid ({args.preview_source})...")
        if args.preview_source == "test":
            try:
                test_rows = read_split(
                    resolve_split_path(splits_dir, args.test_split, "test.txt")
                )
                preview_ds = TrafficSignDataset(
                    root, test_rows, args.size,
                    label_to_idx=label_to_idx,
                    augment_transform=None,   # no augment on test
                )
                visualise_predictions(
                    model, preview_ds, idx_to_label, label_metadata, reports_dir,
                    norm_mean=norm_mean, norm_std=norm_std,
                    num_samples=args.num_preview, labeled=False,
                    title="Test Set Predictions (unlabeled)",
                    filename="test_predictions.png",
                )
            except FileNotFoundError as exc:
                print(f"  Test split unavailable ({exc}) — falling back to val set.")
                visualise_predictions(
                    model, val_ds, idx_to_label, label_metadata, reports_dir,
                    norm_mean=norm_mean, norm_std=norm_std,
                    num_samples=args.num_preview, labeled=True,
                    title="Validation Set Predictions",
                    filename="sample_predictions.png",
                )
        else:
            visualise_predictions(
                model, val_ds, idx_to_label, label_metadata, reports_dir,
                norm_mean=norm_mean, norm_std=norm_std,
                num_samples=args.num_preview, labeled=True,
                title="Validation Set Predictions",
                filename="sample_predictions.png",
            )

    print(f"\nSaved PyTorch model  : {pt_path}")
    print(f"Saved joblib wrapper : {joblib_path}")
    print(f"Saved metrics        : {metrics_path}")
    print(f"Saved confusion mat  : {reports_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()