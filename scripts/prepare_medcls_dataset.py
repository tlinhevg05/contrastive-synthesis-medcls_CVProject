from pathlib import Path
import argparse
import csv
import hashlib
import random
import shutil


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Folder names required by the repo code
TARGET_LABELLED_COUNTS = {
    "Normal": 2038,
    "Lung_Opacity": 1202,
    "COVID": 723,
    "Viral_Pneumonia": 269,
}

# Possible raw dataset folder names
CLASS_ALIASES = {
    "Normal": ["Normal"],
    "Lung_Opacity": ["Lung_Opacity", "Lung Opacity"],
    "COVID": ["COVID", "COVID-19", "COVID19"],
    "Viral_Pneumonia": ["Viral_Pneumonia", "Viral Pneumonia"],
}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def bad_unlabelled_path(path: Path) -> bool:
    """
    Remove mask/metadata files, but keep images inside folders named
    'Segmentation Data'. COVID-QU stores real X-ray images inside those folders.
    """
    s = str(path).lower()
    name = path.name.lower()

    bad_tokens = [
        "mask",
        "masks",
        "metadata",
        "lung_mask",
        "lung masks",
        "infection_mask",
        "infection masks",
    ]

    return any(tok in s for tok in bad_tokens) or "mask" in name


def normalise_name(name: str) -> str:
    return (
        name.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def find_class_image_dir(raw_root: Path, aliases: list[str]) -> Path:
    """
    Find a class folder in the raw Kaggle dataset.
    Prefer class/images if it exists.
    """
    wanted = {normalise_name(a) for a in aliases}

    candidates = []
    for d in raw_root.rglob("*"):
        if d.is_dir() and normalise_name(d.name) in wanted:
            candidates.append(d)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find class folder for aliases {aliases} inside {raw_root}"
        )

    # Prefer folder with images/ subfolder, e.g. COVID/images
    for c in candidates:
        img_dir = c / "images"
        if img_dir.exists() and img_dir.is_dir():
            return img_dir

    # Otherwise use the class folder directly
    return candidates[0]


def stable_sample(paths: list[Path], n: int, seed: int) -> list[Path]:
    paths = sorted(paths, key=lambda p: str(p))
    rng = random.Random(seed)
    if len(paths) < n:
        raise ValueError(f"Need {n} images, but found only {len(paths)}")
    return rng.sample(paths, n)


def copy_or_symlink(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError("mode must be 'copy' or 'symlink'")


def prepare_labelled(raw_radiography: Path, out_root: Path, seed: int, mode: str) -> None:
    labelled_root = out_root / "labelled_4232"
    manifest_path = out_root / "labelled_4232_manifest.csv"

    rows = []

    for repo_class, count in TARGET_LABELLED_COUNTS.items():
        src_dir = find_class_image_dir(raw_radiography, CLASS_ALIASES[repo_class])
        imgs = [p for p in src_dir.iterdir() if is_image(p)]

        chosen = stable_sample(imgs, count, seed + hash(repo_class) % 10000)

        dst_dir = labelled_root / repo_class / "images"
        dst_dir.mkdir(parents=True, exist_ok=True)

        print(f"{repo_class}: found {len(imgs)}, sampling {len(chosen)} from {src_dir}")

        for p in chosen:
            dst = dst_dir / p.name
            copy_or_symlink(p, dst, mode)

            rows.append({
                "path": str(dst),
                "label": repo_class,
                "source_path": str(p),
            })

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved labelled dataset to: {labelled_root}")
    print(f"Saved labelled manifest to: {manifest_path}")


def prepare_unlabelled(raw_covidqu: Path, out_root: Path, n: int, seed: int, mode: str) -> None:
    unlabelled_root = out_root / f"unlabelled_{n}" / "images"
    manifest_path = out_root / f"unlabelled_{n}_manifest.csv"

    all_imgs = [
        p for p in raw_covidqu.rglob("*")
        if is_image(p) and not bad_unlabelled_path(p)
    ]

    chosen = stable_sample(all_imgs, n, seed)

    print(f"Unlabelled: found {len(all_imgs)}, sampling {len(chosen)} from {raw_covidqu}")

    rows = []
    unlabelled_root.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(chosen):
        # Avoid filename collision by adding a hash of the original path
        h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:10]
        dst_name = f"unlabelled_{i:05d}_{h}{p.suffix.lower()}"
        dst = unlabelled_root / dst_name

        copy_or_symlink(p, dst, mode)

        rows.append({
            "path": str(dst),
            "source_path": str(p),
        })

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "source_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved unlabelled dataset to: {unlabelled_root}")
    print(f"Saved unlabelled manifest to: {manifest_path}")


def print_tree_summary(out_root: Path, unlabelled_n: int) -> None:
    print("\nFinal dataset summary:")

    labelled_root = out_root / "labelled_4232"
    for cls in TARGET_LABELLED_COUNTS:
        img_dir = labelled_root / cls / "images"
        count = len([p for p in img_dir.glob("*") if is_image(p)])
        print(f"  {img_dir}: {count}")

    unlabelled_dir = out_root / f"unlabelled_{unlabelled_n}" / "images"
    count = len([p for p in unlabelled_dir.glob("*") if is_image(p)])
    print(f"  {unlabelled_dir}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-radiography",
        type=Path,
        default=Path("data/raw/radiography"),
        help="Raw COVID-19 Radiography Database folder",
    )
    parser.add_argument(
        "--raw-covidqu",
        type=Path,
        default=Path("data/raw/covidqu"),
        help="Raw COVID-QU folder",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed"),
        help="Output folder",
    )
    parser.add_argument(
        "--unlabelled-n",
        type=int,
        default=16934,
        help="Number of unlabelled images to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="copy",
        help="copy duplicates images; symlink saves disk space",
    )

    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    prepare_labelled(
        raw_radiography=args.raw_radiography,
        out_root=args.out,
        seed=args.seed,
        mode=args.mode,
    )

    prepare_unlabelled(
        raw_covidqu=args.raw_covidqu,
        out_root=args.out,
        n=args.unlabelled_n,
        seed=args.seed,
        mode=args.mode,
    )

    print_tree_summary(args.out, args.unlabelled_n)


if __name__ == "__main__":
    main()