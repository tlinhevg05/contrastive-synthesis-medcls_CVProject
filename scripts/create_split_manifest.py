from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable


CLASS_TO_LABEL = {
    "COVID": 0,
    "Lung_Opacity": 1,
    "Viral_Pneumonia": 2,
    "Normal": 3,
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def class_image_dir(root: Path, class_name: str) -> Path:
    class_root = root / class_name
    image_subdir = class_root / "images"
    return image_subdir if image_subdir.exists() else class_root


def find_class_images(labelled_dir: Path, class_name: str) -> list[Path]:
    image_dir = class_image_dir(labelled_dir, class_name)
    if not image_dir.exists():
        return []
    return sorted(path for path in image_dir.rglob("*") if is_image(path))


def row_for_image(image_path: Path, class_name: str, source: str) -> dict[str, str | int]:
    return {
        "image_path": str(image_path),
        "class_name": class_name,
        "label": CLASS_TO_LABEL[class_name],
        "source": source,
    }


def write_csv(path: Path, rows: Iterable[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "class_name", "label", "source"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_rows_by_class(rows: Iterable[dict[str, str | int]]) -> dict[str, int]:
    counts = {class_name: 0 for class_name in CLASS_TO_LABEL}
    for row in rows:
        counts[str(row["class_name"])] += 1
    return counts


def make_stratified_splits(
    labelled_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    source: str,
) -> tuple[
    list[dict[str, str | int]],
    list[dict[str, str | int]],
    list[dict[str, str | int]],
    list[dict[str, str | int]],
]:
    train_rows: list[dict[str, str | int]] = []
    val_rows: list[dict[str, str | int]] = []
    test_rows: list[dict[str, str | int]] = []
    all_rows: list[dict[str, str | int]] = []

    for class_name, label in CLASS_TO_LABEL.items():
        images = find_class_images(labelled_dir, class_name)
        if not images:
            raise FileNotFoundError(f"No images found for class {class_name} under {labelled_dir}")

        rng = random.Random(seed + label)
        rng.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        class_train = images[:n_train]
        class_val = images[n_train : n_train + n_val]
        class_test = images[n_train + n_val :]

        train_rows.extend(row_for_image(path, class_name, source) for path in class_train)
        val_rows.extend(row_for_image(path, class_name, source) for path in class_val)
        test_rows.extend(row_for_image(path, class_name, source) for path in class_test)
        all_rows.extend(row_for_image(path, class_name, source) for path in images)

    return train_rows, val_rows, test_rows, all_rows


def print_distribution(name: str, rows: list[dict[str, str | int]]) -> None:
    counts = count_rows_by_class(rows)
    print(name)
    for class_name, count in counts.items():
        print(f"  {class_name}: {count}")
    print(f"  TOTAL: {sum(counts.values())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed stratified train/val/test manifests.")
    parser.add_argument("--labelled-dir", type=Path, default=Path("data/processed/labelled_4232"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--source", default="real_labeled")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    labelled_dir = args.labelled_dir.expanduser()
    if not labelled_dir.exists():
        raise FileNotFoundError(f"Labelled dataset directory does not exist: {labelled_dir}")

    train_rows, val_rows, test_rows, all_rows = make_stratified_splits(
        labelled_dir=labelled_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
        source=args.source,
    )

    output_dir = args.output_dir
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    all_path = output_dir / "labelled_all.csv"
    summary_path = output_dir / "split_summary.json"

    write_csv(train_path, train_rows)
    write_csv(val_path, val_rows)
    write_csv(test_path, test_rows)
    write_csv(all_path, all_rows)

    summary = {
        "labelled_dir": str(labelled_dir),
        "split_seed": args.split_seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "class_to_label": CLASS_TO_LABEL,
        "files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "labelled_all": str(all_path),
        },
        "counts": {
            "train": count_rows_by_class(train_rows),
            "val": count_rows_by_class(val_rows),
            "test": count_rows_by_class(test_rows),
            "labelled_all": count_rows_by_class(all_rows),
        },
        "totals": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "labelled_all": len(all_rows),
        },
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_distribution("Train split:", train_rows)
    print_distribution("Val split:", val_rows)
    print_distribution("Test split:", test_rows)
    print_distribution("All labelled:", all_rows)
    print("Wrote split manifests:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print(f"  {all_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
