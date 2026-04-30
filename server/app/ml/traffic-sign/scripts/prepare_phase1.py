#!/usr/bin/env python3
"""
Prepare phase 1 dataset artifacts:
- reports/class_counts.csv
- reports/data_audit.txt
- splits/train.txt
- splits/val.txt
- splits/test.txt

Expected project structure:
  .
  ├── Meta/
  ├── Train/
  │   ├── 1/
  │   ├── ...
  │   └── 40/
  ├── Test/
  └── class_map.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".webp"}


@dataclass
class AuditLog:
    missing_train_folders: List[str] = field(default_factory=list)
    empty_train_classes: List[str] = field(default_factory=list)
    unreadable_images: List[str] = field(default_factory=list)
    duplicate_filenames: Dict[str, List[str]] = field(default_factory=dict)
    duplicate_hashes: Dict[str, List[str]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines: List[str] = []
        lines.append("Data Audit Report")
        lines.append("=" * 80)
        lines.append("")

        lines.append("[Train Folder Checks]")
        if self.missing_train_folders:
            lines.append("Missing class folders:")
            lines.extend(f"  - {x}" for x in self.missing_train_folders)
        else:
            lines.append("Missing class folders: none")
        lines.append("")

        if self.empty_train_classes:
            lines.append("Empty class folders:")
            lines.extend(f"  - {x}" for x in self.empty_train_classes)
        else:
            lines.append("Empty class folders: none")
        lines.append("")

        lines.append("[Image Readability]")
        if self.unreadable_images:
            lines.append("Unreadable/corrupt images:")
            lines.extend(f"  - {x}" for x in self.unreadable_images)
        else:
            lines.append("Unreadable/corrupt images: none")
        lines.append("")

        lines.append("[Duplicate File Names]")
        if self.duplicate_filenames:
            for name, paths in sorted(self.duplicate_filenames.items()):
                lines.append(f"  - {name} ({len(paths)} files)")
                lines.extend(f"      * {p}" for p in paths)
        else:
            lines.append("Duplicate file names: none")
        lines.append("")

        lines.append("[Duplicate Image Hashes]")
        if self.duplicate_hashes:
            for digest, paths in sorted(self.duplicate_hashes.items()):
                lines.append(f"  - {digest} ({len(paths)} files)")
                lines.extend(f"      * {p}" for p in paths)
        else:
            lines.append("Duplicate image hashes: none")
        lines.append("")

        if self.notes:
            lines.append("[Notes]")
            lines.extend(f"  - {n}" for n in self.notes)
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare train/val/test split and audits.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Dataset root directory.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-classes", type=int, default=40, help="Expected number of classes.")
    parser.add_argument(
        "--keep-test-paths",
        action="store_true",
        help="Also write unlabeled test paths to splits/test_paths.txt when Test is unlabeled.",
    )
    return parser.parse_args()


def list_image_files(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def load_class_map(class_map_path: Path, expected_ids: Sequence[int]) -> List[str]:
    notes: List[str] = []
    if not class_map_path.exists():
        notes.append(
            f"class_map.csv not found at {class_map_path}; skipping class-map consistency checks."
        )
        return notes

    try:
        df = pd.read_csv(class_map_path)
    except Exception as exc:  # pragma: no cover - defensive parsing guard
        notes.append(f"Failed to read class_map.csv ({class_map_path}): {exc}")
        return notes

    if df.empty:
        notes.append(f"class_map.csv is empty: {class_map_path}")
        return notes

    present_ids = set()
    for col in ("class_id", "id", "ClassId", "ClassID", "label"):
        if col in df.columns:
            present_ids = set(int(x) for x in df[col].dropna().astype(int).tolist())
            break

    if present_ids:
        expected = set(expected_ids)
        missing_ids = sorted(expected - present_ids)
        extra_ids = sorted(present_ids - expected)
        if missing_ids:
            notes.append(f"class_map.csv missing class ids: {missing_ids}")
        if extra_ids:
            notes.append(f"class_map.csv has extra class ids: {extra_ids}")
    else:
        notes.append(
            "class_map.csv has no recognized ID column "
            "(expected one of: class_id,id,ClassId,ClassID,label)."
        )
    return notes


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def write_pairs(path: Path, pairs: Iterable[Tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        for rel, label in pairs:
            f.write(f"{rel} {label}\n")


def write_class_counts(path: Path, counts: List[Tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "image_count"])
        writer.writerows(counts)


def scan_train(
    root: Path, num_classes: int, audit: AuditLog
) -> Tuple[List[Tuple[str, int]], List[Tuple[int, int]]]:
    train_dir = root / "Train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    records: List[Tuple[str, int]] = []
    class_counts: List[Tuple[int, int]] = []
    filename_index: Dict[str, List[str]] = {}
    hash_index: Dict[str, List[str]] = {}

    for class_id in range(1, num_classes + 1):
        class_dir = train_dir / str(class_id)
        if not class_dir.exists():
            audit.missing_train_folders.append(relative_posix(class_dir, root))
            class_counts.append((class_id, 0))
            continue
        if not class_dir.is_dir():
            audit.notes.append(f"Expected directory but found non-directory: {relative_posix(class_dir, root)}")
            class_counts.append((class_id, 0))
            continue

        image_paths = list_image_files(class_dir)
        if not image_paths:
            audit.empty_train_classes.append(relative_posix(class_dir, root))
            class_counts.append((class_id, 0))
            continue

        valid_count = 0
        for img_path in image_paths:
            rel = relative_posix(img_path, root)

            filename_index.setdefault(img_path.name, []).append(rel)
            if not is_readable_image(img_path):
                audit.unreadable_images.append(rel)
                continue

            digest = compute_sha256(img_path)
            hash_index.setdefault(digest, []).append(rel)
            records.append((rel, class_id))
            valid_count += 1

        class_counts.append((class_id, valid_count))

    audit.duplicate_filenames = {k: v for k, v in filename_index.items() if len(v) > 1}
    audit.duplicate_hashes = {k: v for k, v in hash_index.items() if len(v) > 1}

    return records, class_counts


def make_train_val_split(
    records: List[Tuple[str, int]], val_ratio: float, seed: int
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    if not records:
        return [], []

    labels = [label for _, label in records]
    unique_labels = set(labels)

    if len(unique_labels) < 2:
        # train_test_split with stratify requires at least 2 classes.
        return records, []

    counts: Dict[int, int] = {}
    for y in labels:
        counts[y] = counts.get(y, 0) + 1
    if any(v < 2 for v in counts.values()):
        # Not enough samples to stratify every class.
        return records, []

    train_records, val_records = train_test_split(
        records, test_size=val_ratio, random_state=seed, stratify=labels
    )
    return sorted(train_records), sorted(val_records)


def scan_test(root: Path, num_classes: int, audit: AuditLog) -> List[Tuple[str, int]]:
    test_dir = root / "Test"
    if not test_dir.exists():
        audit.notes.append(f"Test directory not found: {relative_posix(test_dir, root)}")
        return []

    class_dirs = [test_dir / str(cid) for cid in range(1, num_classes + 1)]
    labeled_structure = any(d.exists() and d.is_dir() for d in class_dirs)

    pairs: List[Tuple[str, int]] = []
    if labeled_structure:
        for class_id in range(1, num_classes + 1):
            d = test_dir / str(class_id)
            if not d.exists() or not d.is_dir():
                continue
            for path in list_image_files(d):
                pairs.append((relative_posix(path, root), class_id))
        return sorted(pairs)

    for path in list_image_files(test_dir):
        pairs.append((relative_posix(path, root), -1))
    return sorted(pairs)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    reports_dir = root / "reports"
    splits_dir = root / "splits"
    class_map_path = root / "class_map.csv"

    audit = AuditLog()
    expected_ids = list(range(1, args.num_classes + 1))
    audit.notes.extend(load_class_map(class_map_path, expected_ids))

    train_records, class_counts = scan_train(root, args.num_classes, audit)
    write_class_counts(reports_dir / "class_counts.csv", class_counts)

    train_split, val_split = make_train_val_split(train_records, args.val_ratio, args.seed)
    write_pairs(splits_dir / "train.txt", train_split)
    write_pairs(splits_dir / "val.txt", val_split)

    if not val_split and train_split:
        audit.notes.append(
            "Validation split is empty. This can happen if data is too small/imbalanced "
            "for stratified split; all valid train images were kept in train.txt."
        )

    test_pairs = scan_test(root, args.num_classes, audit)
    write_pairs(splits_dir / "test.txt", test_pairs)

    if args.keep_test_paths and test_pairs and all(lbl == -1 for _, lbl in test_pairs):
        with (splits_dir / "test_paths.txt").open("w", encoding="utf-8") as f:
            for rel, _ in test_pairs:
                f.write(f"{rel}\n")

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "data_audit.txt").write_text(audit.to_text(), encoding="utf-8")

    print("Prepared dataset artifacts:")
    print(f"- {relative_posix(reports_dir / 'class_counts.csv', root)}")
    print(f"- {relative_posix(reports_dir / 'data_audit.txt', root)}")
    print(f"- {relative_posix(splits_dir / 'train.txt', root)}")
    print(f"- {relative_posix(splits_dir / 'val.txt', root)}")
    print(f"- {relative_posix(splits_dir / 'test.txt', root)}")
    if args.keep_test_paths:
        print(f"- {relative_posix(splits_dir / 'test_paths.txt', root)} (if unlabeled test)")


if __name__ == "__main__":
    main()
