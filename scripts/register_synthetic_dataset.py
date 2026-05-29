from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from PIL import Image
import yaml


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


def find_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if is_image(path))


def validate_class_dirs(synthetic_dir: Path) -> tuple[list[str], list[str]]:
    present = sorted(path.name for path in synthetic_dir.iterdir() if path.is_dir()) if synthetic_dir.exists() else []
    expected = list(CLASS_TO_LABEL)
    missing = [name for name in expected if name not in present]
    unexpected = [name for name in present if name not in expected]
    return missing, unexpected


def build_manifest_rows(synthetic_dir: Path, generator: str, source: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []

    for class_name, label in CLASS_TO_LABEL.items():
        for image_path in find_images(class_image_dir(synthetic_dir, class_name)):
            rows.append(
                {
                    "image_path": str(image_path),
                    "class_name": class_name,
                    "label": label,
                    "source": source,
                    "generator": generator,
                }
            )

    return rows


def write_manifest(rows: Iterable[dict[str, str | int]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "class_name", "label", "source", "generator"]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_by_class(rows: Iterable[dict[str, str | int]]) -> dict[str, int]:
    counts = {class_name: 0 for class_name in CLASS_TO_LABEL}
    for row in rows:
        counts[str(row["class_name"])] += 1
    return counts


def write_sample_grid(rows: list[dict[str, str | int]], output_path: Path, samples_per_class: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_class = {class_name: [] for class_name in CLASS_TO_LABEL}
    for row in rows:
        class_name = str(row["class_name"])
        if len(by_class[class_name]) < samples_per_class:
            by_class[class_name].append(Path(str(row["image_path"])))

    fig, axes = plt.subplots(
        len(CLASS_TO_LABEL),
        samples_per_class,
        figsize=(3 * samples_per_class, 3 * len(CLASS_TO_LABEL)),
    )

    for row_idx, class_name in enumerate(CLASS_TO_LABEL):
        class_paths = by_class[class_name]
        for col_idx in range(samples_per_class):
            ax = axes[row_idx][col_idx] if samples_per_class > 1 else axes[row_idx]
            ax.axis("off")
            if col_idx < len(class_paths):
                image = Image.open(class_paths[col_idx]).convert("RGB")
                ax.imshow(image, cmap="gray")
                ax.set_title(class_name if col_idx == 0 else "")
            else:
                ax.set_title(f"{class_name}: missing" if col_idx == 0 else "")

    fig.suptitle("Stage 1 DCGAN Synthetic Samples")
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register Stage 1 DCGAN synthetic images.")
    parser.add_argument("--synthetic-dir", type=Path, required=True, help="Folder containing DCGAN class folders.")
    parser.add_argument("--manifest-path", type=Path, default=Path("data/manifests/synthetic_dcgan.csv"))
    parser.add_argument("--metadata-path", type=Path, default=Path("results/stage1_synthesis/dcgan_metadata.yaml"))
    parser.add_argument("--summary-path", type=Path, default=Path("results/stage1_synthesis/synthetic_summary.json"))
    parser.add_argument("--sample-grid-path", type=Path, default=Path("results/stage1_synthesis/sample_grid.png"))
    parser.add_argument("--generator", default="DCGAN")
    parser.add_argument("--source", default="stage1_synthesis")
    parser.add_argument("--samples-per-class", type=int, default=4)
    args = parser.parse_args()

    synthetic_dir = args.synthetic_dir.expanduser().resolve()
    if not synthetic_dir.exists():
        raise FileNotFoundError(f"Synthetic directory does not exist: {synthetic_dir}")

    missing, unexpected = validate_class_dirs(synthetic_dir)
    if missing:
        raise ValueError(f"Missing required synthetic class folders: {missing}")
    if unexpected:
        print(f"WARNING: Unexpected folders under synthetic directory: {unexpected}")

    rows = build_manifest_rows(synthetic_dir, generator=args.generator, source=args.source)
    counts = count_by_class(rows)
    total_images = sum(counts.values())

    write_manifest(rows, args.manifest_path)
    write_sample_grid(rows, args.sample_grid_path, args.samples_per_class)

    metadata = {
        "stage": "stage1_synthetic_lung_radiography_image_generation",
        "generator": args.generator,
        "source": args.source,
        "synthetic_dir": str(synthetic_dir),
        "manifest_path": str(args.manifest_path),
        "class_to_label": CLASS_TO_LABEL,
        "class_counts": counts,
        "total_images": total_images,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "GAN training was completed before this registration step.",
            "DCGAN synthetic images are the default source for COVID-QU-Syn experiments.",
        ],
    }

    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")

    summary = {
        "generator": args.generator,
        "source": args.source,
        "synthetic_dir": str(synthetic_dir),
        "manifest_path": str(args.manifest_path),
        "metadata_path": str(args.metadata_path),
        "sample_grid_path": str(args.sample_grid_path),
        "class_counts": counts,
        "total_images": total_images,
    }

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Stage 1 synthetic dataset registered.")
    print(f"Manifest: {args.manifest_path}")
    print(f"Metadata: {args.metadata_path}")
    print(f"Summary: {args.summary_path}")
    print(f"Sample grid: {args.sample_grid_path}")
    print(f"Class counts: {counts}")
    print(f"Total images: {total_images}")


if __name__ == "__main__":
    main()
