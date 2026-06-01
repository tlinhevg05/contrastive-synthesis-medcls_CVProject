from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from src.classification.supervised_baselines import dump_yaml, load_yaml


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def read_manifest_images(manifest_path: Path) -> list[Path]:
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image_path" not in set(reader.fieldnames or []):
            raise ValueError(f"{manifest_path} must contain image_path column")
        return [Path(row["image_path"]) for row in reader]


def find_unlabeled_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Unlabeled image directory does not exist: {root}")
    return sorted(path for path in root.rglob("*") if is_image(path))


class TwoCropDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform, root_dir: str | Path = "."):
        if not image_paths:
            raise ValueError("No images provided for SimCLR pretraining")
        self.image_paths = image_paths
        self.transform = transform
        self.root_dir = Path(root_dir)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        if not image_path.is_absolute():
            image_path = self.root_dir / image_path
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), self.transform(image)


class SimCLRResNet18(nn.Module):
    def __init__(self, init: str, projection_dim: int = 128):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if init == "imagenet" else None
        self.encoder = resnet18(weights=weights)
        feature_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, x):
        return self.projector(self.encoder(x))


def nt_xent_loss(z_i, z_j, temperature: float):
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)
    similarity = torch.matmul(z, z.T) / temperature

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    similarity = similarity.masked_fill(mask, -9e15)

    positives = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ]
    )
    return F.cross_entropy(similarity, positives)


def get_simclr_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain ResNet18 with SimCLR.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--real-unlabeled-dir", default="data/processed/unlabelled_16934")
    parser.add_argument("--synthetic-manifest", default="data/manifests/synthetic_dcgan.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    if cfg["backbone"] != "resnet18" or cfg["contrastive_method"] != "simclr":
        raise ValueError("run_simclr_resnet.py expects a ResNet18 SimCLR experiment config")

    pretrain = cfg["pretrain"]
    if not pretrain.get("enabled"):
        raise ValueError("pretrain.enabled must be true")

    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
        pretrain["checkpoint_path"] = str(Path(args.output_dir) / "pretrain/checkpoints/best_simclr_backbone.pth")
    if args.epochs is not None:
        pretrain["epochs"] = args.epochs
    if args.batch_size is not None:
        pretrain["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        pretrain["learning_rate"] = args.learning_rate

    set_seed(42)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(cfg["output_dir"])
    checkpoint_path = Path(pretrain["checkpoint_path"])
    last_checkpoint_path = checkpoint_path.parent / "last_simclr_checkpoint.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pretrain["dataset"] == "real_unlabeled":
        image_paths = find_unlabeled_images(Path(args.real_unlabeled_dir))
        source_path = args.real_unlabeled_dir
    elif pretrain["dataset"] == "synthetic":
        synthetic_manifest = Path(args.synthetic_manifest)
        if not synthetic_manifest.exists():
            raise FileNotFoundError(
                f"Synthetic manifest not found: {synthetic_manifest}. "
                "Run the Stage 1 registry notebook first, or pass --synthetic-manifest to a valid synthetic_dcgan.csv."
            )
        image_paths = read_manifest_images(synthetic_manifest)
        source_path = str(synthetic_manifest)
    else:
        raise ValueError(f"Unsupported SimCLR pretrain dataset: {pretrain['dataset']}")

    cfg["pretrain"]["resolved_source_path"] = source_path
    cfg["pretrain"]["num_pretrain_images"] = len(image_paths)
    dump_yaml(cfg, output_dir / "config_resolved_simclr.yaml")

    dataset = TwoCropDataset(image_paths, transform=get_simclr_transform(224), root_dir=REPO_ROOT)
    loader = DataLoader(
        dataset,
        batch_size=int(pretrain["batch_size"]),
        shuffle=True,
        num_workers=int(args.num_workers),
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    model = SimCLRResNet18(init=pretrain["init"], projection_dim=args.projection_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(pretrain["learning_rate"]), weight_decay=1e-4)

    best_loss = float("inf")
    history = []
    start_epoch = 0
    resume_path = Path(args.resume_checkpoint) if args.resume_checkpoint else last_checkpoint_path
    if resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_loss = float(checkpoint.get("best_loss", checkpoint.get("loss", best_loss)))
        history = list(checkpoint.get("history", []))
        print(f"Resuming SimCLR from {resume_path} at epoch {start_epoch}")

    epochs = int(pretrain["epochs"])
    if start_epoch >= epochs:
        print(f"Resume checkpoint already reached epoch {start_epoch}; target epochs={epochs}. Nothing to do.")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        total_images = 0
        for x_i, x_j in loader:
            x_i = x_i.to(device)
            x_j = x_j.to(device)
            z_i = model(x_i)
            z_j = model(x_j)
            loss = nt_xent_loss(z_i, z_j, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_i.size(0)
            total_images += x_i.size(0)

        avg_loss = total_loss / max(total_images, 1)
        history.append({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch {epoch + 1}/{epochs} simclr_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "encoder_state_dict": model.encoder.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "loss": best_loss,
                },
                checkpoint_path,
            )

        torch.save(
            {
                "epoch": epoch + 1,
                "encoder_state_dict": model.encoder.state_dict(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "loss": avg_loss,
                "best_loss": best_loss,
                "history": history,
            },
            last_checkpoint_path,
        )

    (output_dir / "pretrain").mkdir(parents=True, exist_ok=True)
    (output_dir / "pretrain/simclr_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("Saved SimCLR checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
