from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ManifestImageDataset(Dataset):
    """Image dataset backed by a CSV manifest."""

    def __init__(self, manifest_path: str | Path, transform=None, root_dir: str | Path = "."):
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.root_dir = Path(root_dir)
        self.rows = self._read_rows(self.manifest_path)

    @staticmethod
    def _read_rows(manifest_path: Path) -> list[dict[str, str]]:
        required = {"image_path", "class_name", "label", "source"}
        with manifest_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames or [])
            missing = required - columns
            if missing:
                raise ValueError(f"{manifest_path} missing required columns: {sorted(missing)}")
            return list(reader)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = self.root_dir / image_path

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, int(row["label"])
