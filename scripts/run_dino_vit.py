from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.classification.supervised_baselines import dump_yaml, load_yaml
from src.contrastive.dino_losses import DINOLoss
from src.contrastive.utils import cancel_gradients_last_layer, clip_gradients, cosine_scheduler
from src.data.transforms import DataAugmentationDINO
from src.models.dino_head import DINOHead
from src.models.dino_wrapper import MultiCropWrapper


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def find_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    return sorted(path for path in root.rglob("*") if is_image(path))


def read_manifest_images(manifest_path: Path) -> list[Path]:
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image_path" not in set(reader.fieldnames or []):
            raise ValueError(f"{manifest_path} must contain image_path column")
        return [Path(row["image_path"]) for row in reader]


class DinoImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform, root_dir: str | Path = "."):
        if not image_paths:
            raise ValueError("No images provided for DINO pretraining")
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
        return self.transform(image), 0


def build_vit_backbone(init: str):
    try:
        import timm

        return timm.create_model("vit_small_patch16_224", pretrained=(init == "imagenet"), num_classes=0)
    except ModuleNotFoundError as exc:
        if init == "imagenet":
            raise ModuleNotFoundError("timm is required for ImageNet-initialized DINO ViT-S/16") from exc
        from src.models.vit import vit_small

        return vit_small(patch_size=16, img_size=[224])


def get_feature_dim(backbone: nn.Module) -> int:
    if hasattr(backbone, "num_features"):
        return int(backbone.num_features)
    if hasattr(backbone, "embed_dim"):
        return int(backbone.embed_dim)
    raise AttributeError("Could not infer ViT backbone feature dimension")


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain ViT-S/16 with DINO.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--real-unlabeled-dir", default="data/processed/unlabelled_16934")
    parser.add_argument("--synthetic-manifest", default="data/manifests/synthetic_dcgan.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--out-dim", type=int, default=4096)
    parser.add_argument("--local-crops-number", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    if cfg["backbone"] != "vit_s16" or cfg["contrastive_method"] != "dino":
        raise ValueError("run_dino_vit.py expects a ViT-S/16 DINO experiment config")

    pretrain: dict[str, Any] = cfg["pretrain"]
    if not pretrain.get("enabled"):
        raise ValueError("pretrain.enabled must be true")

    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
        pretrain["checkpoint_path"] = str(Path(args.output_dir) / "pretrain/checkpoints/best_dino_teacher.pth")
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
    last_checkpoint_path = checkpoint_path.parent / "last_dino_checkpoint.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pretrain["dataset"] == "real_unlabeled":
        image_paths = find_images(Path(args.real_unlabeled_dir))
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
        raise ValueError(f"Unsupported DINO pretrain dataset: {pretrain['dataset']}")

    cfg["pretrain"]["resolved_source_path"] = source_path
    cfg["pretrain"]["num_pretrain_images"] = len(image_paths)
    cfg["pretrain"]["out_dim"] = args.out_dim
    cfg["pretrain"]["local_crops_number"] = args.local_crops_number
    dump_yaml(cfg, output_dir / "config_resolved_dino.yaml")

    transform = DataAugmentationDINO(
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=args.local_crops_number,
    )
    dataset = DinoImageDataset(image_paths, transform=transform, root_dir=REPO_ROOT)
    loader = DataLoader(
        dataset,
        batch_size=int(pretrain["batch_size"]),
        shuffle=True,
        num_workers=int(args.num_workers),
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    if len(loader) == 0:
        raise ValueError("DINO DataLoader has zero batches. Reduce batch size or provide more images.")

    student_backbone = build_vit_backbone(pretrain["init"])
    teacher_backbone = build_vit_backbone(pretrain["init"])
    embed_dim = get_feature_dim(student_backbone)
    student = MultiCropWrapper(student_backbone, DINOHead(embed_dim, args.out_dim)).to(device)
    teacher = MultiCropWrapper(teacher_backbone, DINOHead(embed_dim, args.out_dim)).to(device)
    teacher.load_state_dict(student.state_dict())
    for param in teacher.parameters():
        param.requires_grad = False

    dino_loss = DINOLoss(
        out_dim=args.out_dim,
        ncrops=args.local_crops_number + 2,
        warmup_teacher_temp=0.04,
        teacher_temp=0.04,
        warmup_teacher_temp_epochs=min(10, int(pretrain["epochs"])),
        total_epochs=int(pretrain["epochs"]),
        student_temp=0.1,
        center_momentum=0.9,
    ).to(device)

    params_groups = []
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        weight_decay = 0.0 if name.endswith(".bias") or "norm" in name else 0.04
        params_groups.append({"params": [param], "weight_decay": weight_decay})
    optimizer = torch.optim.AdamW(params_groups, lr=float(pretrain["learning_rate"]))

    epochs = int(pretrain["epochs"])
    niter_per_ep = len(loader)
    lr_schedule = cosine_scheduler(
        float(pretrain["learning_rate"]) * int(pretrain["batch_size"]) / 256.0,
        1e-6,
        epochs,
        niter_per_ep,
        warmup_epochs=min(5, epochs),
    )
    wd_schedule = cosine_scheduler(0.04, 0.4, epochs, niter_per_ep)
    momentum_schedule = cosine_scheduler(0.996, 1.0, epochs, niter_per_ep)

    best_loss = float("inf")
    history = []
    start_epoch = 0
    resume_path = Path(args.resume_checkpoint) if args.resume_checkpoint else last_checkpoint_path
    if resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device)
        student.load_state_dict(checkpoint["student"])
        teacher.load_state_dict(checkpoint["teacher"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "dino_loss" in checkpoint:
            dino_loss.load_state_dict(checkpoint["dino_loss"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_loss = float(checkpoint.get("best_loss", checkpoint.get("loss", best_loss)))
        history = list(checkpoint.get("history", []))
        print(f"Resuming DINO from {resume_path} at epoch {start_epoch}")

    if start_epoch >= epochs:
        print(f"Resume checkpoint already reached epoch {start_epoch}; target epochs={epochs}. Nothing to do.")

    for epoch in range(start_epoch, epochs):
        student.train()
        teacher.eval()
        total_loss = 0.0
        total_batches = 0
        for iteration, (images, _) in enumerate(loader):
            global_iteration = epoch * niter_per_ep + iteration
            for group_idx, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[global_iteration]
                if group_idx == 0:
                    param_group["weight_decay"] = wd_schedule[global_iteration]

            images = [image.to(device, non_blocking=True) for image in images]
            with torch.no_grad():
                teacher_output = teacher(images[:2])
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

            optimizer.zero_grad()
            loss.backward()
            clip_gradients(student, 3.0)
            cancel_gradients_last_layer(epoch, student, 1)
            optimizer.step()

            with torch.no_grad():
                momentum = momentum_schedule[global_iteration]
                for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
                    teacher_param.data.mul_(momentum).add_((1.0 - momentum) * student_param.detach().data)

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        history.append({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch {epoch + 1}/{epochs} dino_loss={avg_loss:.4f}")

        save_dict = {
            "epoch": epoch + 1,
            "teacher": teacher.state_dict(),
            "student": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "dino_loss": dino_loss.state_dict(),
            "config": cfg,
            "loss": avg_loss,
            "best_loss": best_loss,
            "history": history,
        }
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_dict["best_loss"] = best_loss
            torch.save(save_dict, checkpoint_path)

        torch.save(save_dict, last_checkpoint_path)

    (output_dir / "pretrain").mkdir(parents=True, exist_ok=True)
    (output_dir / "pretrain/dino_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("Saved DINO checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
