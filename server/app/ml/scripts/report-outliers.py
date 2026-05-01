#!/usr/bin/env python3
"""
Generate dataset outlier reports for traffic-sign classification.

Outputs:
- reports/outlier_report.csv
- reports/outlier_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report potential outlier images.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root.")
    parser.add_argument("--min-size", type=int, default=20, help="Flag min width/height below this.")
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=3.0,
        help="Flag images where longer/shorter side exceeds this ratio.",
    )
    parser.add_argument(
        "--low-brightness",
        type=float,
        default=25.0,
        help="Flag mean grayscale brightness lower than this.",
    )
    parser.add_argument(
        "--high-brightness",
        type=float,
        default=230.0,
        help="Flag mean grayscale brightness higher than this.",
    )
    parser.add_argument(
        "--low-contrast",
        type=float,
        default=12.0,
        help="Flag grayscale std-dev lower than this.",
    )
    return parser.parse_args()


def read_class_ids(root: Path) -> list[int]:
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
    # Fallback: infer from Train directories
    train_dir = root / "Train"
    out = []
    if train_dir.exists():
        for d in train_dir.iterdir():
            if d.is_dir():
                try:
                    out.append(int(d.name))
                except ValueError:
                    pass
    return sorted(set(out))


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    train_dir = root / "Train"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    class_ids = read_class_ids(root)
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    records: list[dict[str, str]] = []
    digest_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    reason_counts: Counter[str] = Counter()
    total_images = 0

    for class_id in class_ids:
        class_dir = train_dir / str(class_id)
        if not class_dir.exists() or not class_dir.is_dir():
            continue

        for img_path in list_images(class_dir):
            total_images += 1
            rel = relative_posix(img_path, root)
            base = {
                "path": rel,
                "class_id": str(class_id),
                "width": "",
                "height": "",
                "aspect_ratio": "",
                "mean_brightness": "",
                "std_contrast": "",
                "sha256": "",
                "reasons": "",
            }
            reasons: list[str] = []

            try:
                with Image.open(img_path) as img:
                    gray = np.asarray(img.convert("L"), dtype=np.float32)
                    h, w = gray.shape
            except (UnidentifiedImageError, OSError, ValueError):
                reasons.append("unreadable_image")
                base["reasons"] = "|".join(reasons)
                records.append(base)
                reason_counts.update(reasons)
                continue

            digest = sha256(img_path)
            aspect = max(w, h) / max(1, min(w, h))
            mean_brightness = float(np.mean(gray))
            std_contrast = float(np.std(gray))

            if w < args.min_size or h < args.min_size:
                reasons.append("tiny_dimensions")
            if aspect > args.max_aspect_ratio:
                reasons.append("extreme_aspect_ratio")
            if mean_brightness < args.low_brightness:
                reasons.append("very_dark")
            if mean_brightness > args.high_brightness:
                reasons.append("very_bright")
            if std_contrast < args.low_contrast:
                reasons.append("low_contrast")

            row = {
                "path": rel,
                "class_id": str(class_id),
                "width": str(w),
                "height": str(h),
                "aspect_ratio": f"{aspect:.4f}",
                "mean_brightness": f"{mean_brightness:.2f}",
                "std_contrast": f"{std_contrast:.2f}",
                "sha256": digest,
                "reasons": "|".join(reasons),
            }
            digest_rows[digest].append(row)
            records.append(row)
            reason_counts.update(reasons)

    # Mark cross-class duplicates as outliers.
    for digest, rows in digest_rows.items():
        classes = {r["class_id"] for r in rows}
        if len(rows) > 1 and len(classes) > 1:
            for r in rows:
                reasons = [x for x in r["reasons"].split("|") if x]
                if "cross_class_duplicate_hash" not in reasons:
                    reasons.append("cross_class_duplicate_hash")
                    r["reasons"] = "|".join(reasons)
                    reason_counts.update(["cross_class_duplicate_hash"])

    # Keep only rows with at least one reason.
    outliers = [r for r in records if r["reasons"]]

    report_csv = reports_dir / "outlier_report.csv"
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "class_id",
                "width",
                "height",
                "aspect_ratio",
                "mean_brightness",
                "std_contrast",
                "sha256",
                "reasons",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(outliers, key=lambda x: x["path"]))

    summary_txt = reports_dir / "outlier_summary.txt"
    lines = [
        "Outlier Summary",
        "=" * 80,
        f"Total train images scanned: {total_images}",
        f"Total outlier rows: {len(outliers)}",
        "",
        "[Reason counts]",
    ]
    if reason_counts:
        for reason, count in reason_counts.most_common():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    lines += [
        "",
        "Artifacts:",
        "- reports/outlier_report.csv",
        "- reports/outlier_summary.txt",
    ]
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Generated artifacts:")
    print("- reports/outlier_report.csv")
    print("- reports/outlier_summary.txt")
    print(f"Outlier rows: {len(outliers)}")


if __name__ == "__main__":
    main()
