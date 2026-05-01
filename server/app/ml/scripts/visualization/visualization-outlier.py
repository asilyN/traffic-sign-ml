#!/usr/bin/env python3
"""
Create a simple visualization image from outlier reports.

Input:
- reports/outlier_report.csv

Output:
- reports/outlier_visualization.png
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize outlier report.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root.")
    return parser.parse_args()


def draw_bar_chart(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    width: int,
    title: str,
    counts: list[tuple[str, int]],
    bar_color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    font_bold: ImageFont.ImageFont,
) -> int:
    draw.text((x0, y0), title, fill=(20, 20, 20), font=font_bold)
    y = y0 + 28
    if not counts:
        draw.text((x0, y), "No data", fill=(90, 90, 90), font=font)
        return y + 24

    max_count = max(v for _, v in counts) or 1
    label_width = int(width * 0.42)
    bar_max = int(width * 0.42)
    row_h = 24

    for label, value in counts:
        draw.text((x0, y), label, fill=(40, 40, 40), font=font)
        bx = x0 + label_width
        bw = max(1, int((value / max_count) * bar_max))
        draw.rectangle((bx, y + 3, bx + bw, y + 18), fill=bar_color)
        draw.text((bx + bw + 8, y), str(value), fill=(40, 40, 40), font=font)
        y += row_h
    return y


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    report_csv = root / "reports" / "outlier_report.csv"
    out_png = root / "reports" / "outlier_visualization.png"

    if not report_csv.exists():
        raise FileNotFoundError(f"Missing input file: {report_csv}")

    reason_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    class_reason_counts: dict[str, Counter[str]] = defaultdict(Counter)
    total_rows = 0

    with report_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            cls = row.get("class_id", "")
            class_counts[cls] += 1
            reasons = [r for r in row.get("reasons", "").split("|") if r]
            for r in reasons:
                reason_counts[r] += 1
                class_reason_counts[cls][r] += 1

    top_reasons = reason_counts.most_common(10)
    top_classes = class_counts.most_common(10)

    width, height = 1200, 760
    img = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    font_bold = ImageFont.load_default()

    draw.rectangle((20, 20, width - 20, height - 20), outline=(200, 206, 214), width=2)
    draw.text((40, 36), "Dataset Outlier Visualization", fill=(15, 23, 42), font=font_bold)
    draw.text((40, 58), f"Total outlier rows: {total_rows}", fill=(51, 65, 85), font=font)

    left_x, right_x = 40, 620
    chart_w = 520

    y_left_end = draw_bar_chart(
        draw,
        left_x,
        100,
        chart_w,
        "Top Outlier Reasons",
        top_reasons,
        (59, 130, 246),
        font,
        font_bold,
    )

    y_right_end = draw_bar_chart(
        draw,
        right_x,
        100,
        chart_w,
        "Top Classes by Outlier Count",
        top_classes,
        (245, 158, 11),
        font,
        font_bold,
    )

    # Add most common reason per top class.
    y = max(y_left_end, y_right_end) + 20
    draw.text((40, y), "Top Class -> Most Common Outlier Reason", fill=(20, 20, 20), font=font_bold)
    y += 24
    for cls, cnt in top_classes[:8]:
        top_reason = class_reason_counts[cls].most_common(1)
        reason_text = top_reason[0][0] if top_reason else "n/a"
        reason_count = top_reason[0][1] if top_reason else 0
        draw.text(
            (40, y),
            f"Class {cls}: {cnt} outliers, top reason: {reason_text} ({reason_count})",
            fill=(50, 50, 50),
            font=font,
        )
        y += 20

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    print(f"Saved visualization: {out_png}")


if __name__ == "__main__":
    main()
