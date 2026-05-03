"""
Path resolution utilities for finding dataset-related files and directories.
"""
from __future__ import annotations

from pathlib import Path


def resolve_dataset_root(root_arg: Path | None) -> Path:
    """
    Find the root directory of the dataset.

    It auto-detects common dataset locations if --root is not provided.
    """
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


def resolve_split_path(
    splits_dir: Path, split_name: str, fallback_name: str | None
) -> Path:
    """
    Find the path to a train/validation split file.

    It can fall back to a default name if the requested one is not found.
    """
    primary = splits_dir / split_name
    if primary.exists():
        return primary

    if fallback_name:
        fallback = splits_dir / fallback_name
        if fallback.exists():
            print(f"Using fallback split: {fallback.name} (requested: {split_name})")
            return fallback

    extra = (
        f"\nFallback also missing: {splits_dir / fallback_name}"
        if fallback_name
        else "\n(No fallback requested.)"
    )
    raise FileNotFoundError(
        f"Split file not found: {primary}"
        f"{extra}\n"
        "Generate splits first with generate-augmented-splits.py or pass --train-split/--val-split explicitly."
    )


def resolve_labels_path(root: Path, labels_path: Path) -> Path:
    """
    Find the path to the label metadata JSON file.

    It searches in several common locations relative to the dataset root.
    """
    if labels_path.is_absolute():
        if labels_path.exists():
            return labels_path
        raise FileNotFoundError(f"Label metadata file not found: {labels_path}")

    candidates = [
        root / labels_path,
        root.parent / labels_path,
        root.parent / "ml" / labels_path,
        Path(".").resolve() / labels_path,
        Path(".").resolve() / "server" / "app" / "ml" / labels_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(f"- {p.resolve()}" for p in candidates)
    raise FileNotFoundError(
        "Label metadata file not found.\n"
        f"Requested path: {labels_path}\n"
        "Searched:\n"
        f"{searched}\n"
        "Pass --labels-path explicitly if your labels file is elsewhere."
    )
