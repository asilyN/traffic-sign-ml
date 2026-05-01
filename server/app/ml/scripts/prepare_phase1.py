#!/usr/bin/env python3
"""
Prepare phase 1 dataset artifacts:
- reports/class_counts.csv
- reports/data_audit.txt

Expected project structure:
  .
  ├── Meta/
  ├── Train/
  │   ├── 1/
  │   ├── ...
  │   └── 40/
  ├── Test/
  └── labels.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, UnidentifiedImageError


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
    parser = argparse.ArgumentParser(description="Prepare phase 1 dataset audits.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Dataset root directory. If omitted, common dataset locations are auto-detected.",
    )
    parser.add_argument("--num-classes", type=int, default=40, help="Expected number of classes.")
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


def load_label_class_ids(label_path: Path) -> Tuple[List[int], List[str]]:
    class_ids: List[int] = []
    notes: List[str] = []
    if not label_path.exists():
        notes.append(f"labels.json not found at {label_path}; using --num-classes fallback.")
        return class_ids, notes

    try:
        payload = json.loads(label_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive parsing guard
        notes.append(f"Failed to read label metadata ({label_path}): {exc}")
        return class_ids, notes

    for row in payload.get("classes", []):
        class_id = row.get("class_id")
        if class_id is None:
            continue
        class_ids.append(int(class_id))

    if not class_ids:
        notes.append(f"labels.json has no usable class_id entries: {label_path}")
        return class_ids, notes

    return sorted(set(class_ids)), notes


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def write_class_counts(path: Path, counts: List[Tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "image_count"])
        writer.writerows(counts)


def scan_train(
    root: Path, class_ids: Sequence[int], audit: AuditLog
) -> Tuple[List[Tuple[str, int]], List[Tuple[int, int]]]:
    train_dir = root / "Train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    records: List[Tuple[str, int]] = []
    class_counts: List[Tuple[int, int]] = []
    filename_index: Dict[str, List[str]] = {}
    hash_index: Dict[str, List[str]] = {}

    for class_id in class_ids:
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


def resolve_dataset_root(root_arg: Path | None) -> Path:
    if root_arg is not None:
        root = root_arg.resolve()
        if (root / "Train").exists():
            return root
        raise FileNotFoundError(
            f"Train directory not found under provided --root: {root / 'Train'}"
        )

    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(".").resolve(),
        script_dir.parent / "dataset",
        script_dir.parent,
        Path(".").resolve() / "server" / "app" / "ml" / "dataset",
    ]

    for candidate in candidates:
        if (candidate / "Train").exists():
            return candidate.resolve()

    pretty_candidates = "\n".join(f"- {c.resolve()}" for c in candidates)
    raise FileNotFoundError(
        "Could not auto-detect dataset root (missing 'Train' directory).\n"
        "Searched paths:\n"
        f"{pretty_candidates}\n"
        "Pass --root explicitly, e.g. --root server/app/ml/dataset"
    )


def main() -> None:
    args = parse_args()
    root = resolve_dataset_root(args.root)
    reports_dir = root / "reports"
    label_path = root / "labels.json"

    audit = AuditLog()
    class_ids, label_notes = load_label_class_ids(label_path)
    audit.notes.extend(label_notes)
    if not class_ids:
        class_ids = list(range(1, args.num_classes + 1))
        audit.notes.append(
            f"Using fallback class ids from --num-classes: {class_ids[0]}..{class_ids[-1]}"
        )

    _, class_counts = scan_train(root, class_ids, audit)
    write_class_counts(reports_dir / "class_counts.csv", class_counts)

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "data_audit.txt").write_text(audit.to_text(), encoding="utf-8")

    print("Prepared dataset artifacts:")
    print(f"- {relative_posix(reports_dir / 'class_counts.csv', root)}")
    print(f"- {relative_posix(reports_dir / 'data_audit.txt', root)}")


if __name__ == "__main__":
    main()
