#!/usr/bin/env python3
"""
Analyze training data distribution and class balance.
"""

from pathlib import Path
import json

# Use absolute path from script location
script_dir = Path(__file__).resolve().parent
dataset_root = script_dir
splits_dir = dataset_root / "splits"
labels_file = script_dir / ".." / "labels.json"

print("=" * 60)
print("Dataset Analysis")
print("=" * 60)
print(f"Dataset root: {dataset_root}\n")

# 1. Load labels
with open(labels_file) as f:
    labels_data = json.load(f)

class_names = {c['class_id']: c['class_name'] for c in labels_data.get('classes', [])}
print(f"Total classes: {len(class_names)}\n")

# 2. Analyze splits
train_split = splits_dir / "train_augmented.txt"
val_split = splits_dir / "val_augmented.txt"

if not train_split.exists():
    print(f"Error: {train_split} not found")
    exit(1)

train_counts = {}
for line in train_split.read_text().splitlines():
    _, label = line.rsplit(" ", 1)
    label = int(label)
    train_counts[label] = train_counts.get(label, 0) + 1

print(f"Training samples per class (sorted by count):\n")
total_train = 0
for class_id in sorted(train_counts.keys(), key=lambda x: train_counts[x], reverse=True):
    count = train_counts[class_id]
    name = class_names.get(class_id, "?")
    print(f"  Class {class_id:2d} ({name:30s}): {count:3d} samples")
    total_train += count

print(f"\nTotal training samples: {total_train}")
print(f"Min samples per class: {min(train_counts.values())}")
print(f"Max samples per class: {max(train_counts.values())}")
print(f"Imbalance ratio: {max(train_counts.values()) / min(train_counts.values()):.1f}x\n")

# 3. Check missing classes
missing = set(range(1, 49)) - set(train_counts.keys())
if missing:
    print(f"⚠ Missing classes: {missing}\n")

# 4. Analyze validation split
print("-" * 60)
print("\nValidation samples per class:\n")
val_counts = {}
if val_split.exists():
    for line in val_split.read_text().splitlines():
        _, label = line.rsplit(" ", 1)
        label = int(label)
        val_counts[label] = val_counts.get(label, 0) + 1
    
    total_val = 0
    for class_id in sorted(val_counts.keys(), key=lambda x: val_counts[x], reverse=True):
        count = val_counts[class_id]
        name = class_names.get(class_id, "?")
        print(f"  Class {class_id:2d} ({name:30s}): {count:3d} samples")
        total_val += count
    
    print(f"\nTotal validation samples: {total_val}")
    print(f"Train/Val ratio: {total_train / total_val:.2f}:1")

print("\n" + "=" * 60)
