#!/usr/bin/env python3
"""
Generate duplicate report and leakage-free train/val splits.

Outputs:
- reports/duplicate_report.csv
- reports/clean_split_summary.txt
- splits/train.txt
- splits/val.txt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build duplicate report and clean train/val splits."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Dataset root directory. If omitted, common dataset locations are auto-detected.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-classes", type=int, default=48, help="Fallback class count.")
    return parser.parse_args()


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_class_ids(root: Path, fallback_count: int) -> List[int]:
    labels_file = root / "labels.json"
    if labels_file.exists():
        payload = json.loads(labels_file.read_text(encoding="utf-8"))
        ids = []
        for row in payload.get("classes", []):
            class_id = row.get("class_id")
            if class_id is None:
                continue
            ids.append(int(class_id))
        if ids:
            return sorted(set(ids))
    return list(range(1, fallback_count + 1))


def list_images(folder: Path) -> List[Path]:
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_current_split_hashes(root: Path) -> Dict[str, set]:
    out: Dict[str, set] = defaultdict(set)
    for split in ("train.txt", "val.txt"):
        p = root / "splits" / split
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            rel = parts[0]
            full = root / rel
            if not full.exists():
                continue
            out[compute_sha256(full)].add(split)
    return out


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
    splits_dir = root / "splits"
    reports_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    class_ids = read_class_ids(root, args.num_classes)
    train_dir = root / "Train"

    current_split_hashes = load_current_split_hashes(root)
    digest_to_items: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    unreadable: List[str] = []

    for class_id in class_ids:
        class_dir = train_dir / str(class_id)
        if not class_dir.exists() or not class_dir.is_dir():
            continue
        for img in list_images(class_dir):
            rel = relative_posix(img, root)
            if not is_readable_image(img):
                unreadable.append(rel)
                continue
            digest = compute_sha256(img)
            digest_to_items[digest].append((rel, class_id))

    # Write duplicate report.
    duplicate_report = reports_dir / "duplicate_report.csv"
    with duplicate_report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sha256",
                "num_files",
                "num_unique_classes",
                "class_ids",
                "cross_class_duplicate",
                "already_leaks_in_old_split",
                "paths",
            ]
        )
        for digest, items in sorted(digest_to_items.items()):
            classes = sorted({label for _, label in items})
            old_split_set = current_split_hashes.get(digest, set())
            writer.writerow(
                [
                    digest,
                    len(items),
                    len(classes),
                    ";".join(str(x) for x in classes),
                    int(len(classes) > 1),
                    int({"train.txt", "val.txt"}.issubset(old_split_set)),
                    "|".join(path for path, _ in items),
                ]
            )

    # Build hash-grouped split (no hash leakage).
    digests = list(digest_to_items.keys())
    dominant_labels = []
    for d in digests:
        labels = [y for _, y in digest_to_items[d]]
        dominant_labels.append(Counter(labels).most_common(1)[0][0])

    can_stratify = len(set(dominant_labels)) > 1
    if can_stratify:
        label_counts = Counter(dominant_labels)
        if min(label_counts.values()) < 2:
            can_stratify = False

    if can_stratify:
        train_digests, val_digests = train_test_split(
            digests,
            test_size=args.val_ratio,
            random_state=args.seed,
            stratify=dominant_labels,
        )
    else:
        rng = random.Random(args.seed)
        shuffled = digests[:]
        rng.shuffle(shuffled)
        cut = int(round(len(shuffled) * (1.0 - args.val_ratio)))
        cut = max(1, min(cut, len(shuffled) - 1))
        train_digests, val_digests = shuffled[:cut], shuffled[cut:]

    train_set = set(train_digests)
    val_set = set(val_digests)
    assert train_set.isdisjoint(val_set), "Internal error: split overlap by digest."

    train_pairs: List[Tuple[str, int]] = []
    val_pairs: List[Tuple[str, int]] = []
    for digest, items in digest_to_items.items():
        if digest in train_set:
            train_pairs.extend(items)
        elif digest in val_set:
            val_pairs.extend(items)

    train_pairs.sort()
    val_pairs.sort()

    (splits_dir / "train.txt").write_text(
        "".join(f"{rel} {label}\n" for rel, label in train_pairs), encoding="utf-8"
    )
    (splits_dir / "val.txt").write_text(
        "".join(f"{rel} {label}\n" for rel, label in val_pairs), encoding="utf-8"
    )

    # Summary.
    cross_class_duplicate_groups = sum(
        1 for items in digest_to_items.values() if len({y for _, y in items}) > 1
    )
    duplicate_groups = sum(1 for items in digest_to_items.values() if len(items) > 1)

    train_counts = Counter(y for _, y in train_pairs)
    val_counts = Counter(y for _, y in val_pairs)

    summary = reports_dir / "clean_split_summary.txt"
    lines = [
        "Clean Split Summary",
        "=" * 80,
        f"Total readable train images: {len(train_pairs) + len(val_pairs)}",
        f"Unreadable images skipped: {len(unreadable)}",
        f"Unique image hashes: {len(digest_to_items)}",
        f"Duplicate hash groups: {duplicate_groups}",
        f"Cross-class duplicate groups: {cross_class_duplicate_groups}",
        f"Train clean samples: {len(train_pairs)}",
        f"Val clean samples: {len(val_pairs)}",
        f"Classes discovered: {len(class_ids)} ({class_ids[0]}..{class_ids[-1]})",
        "",
        "[Per-class counts]",
    ]
    for c in class_ids:
        lines.append(f"class {c:>2}: train={train_counts.get(c,0):>4} val={val_counts.get(c,0):>4}")
    lines.append("")
    lines.append("Artifacts:")
    lines.append("- reports/duplicate_report.csv")
    lines.append("- reports/clean_split_summary.txt")
    lines.append("- splits/train.txt")
    lines.append("- splits/val.txt")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Generated artifacts:")
    print("- reports/duplicate_report.csv")
    print("- reports/clean_split_summary.txt")
    print("- splits/train.txt")
    print("- splits/val.txt")


if __name__ == "__main__":
    main()
