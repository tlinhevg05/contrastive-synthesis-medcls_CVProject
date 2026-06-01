from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18 as torchvision_resnet18

from src.data.manifest_dataset import ManifestImageDataset


CLASS_NAMES = ["COVID", "Lung_Opacity", "Viral_Pneumonia", "Normal"]


try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def parse_scalar(value: str):
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def simple_yaml_load(text: str):
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        clean = strip_yaml_comment(raw_line).rstrip()
        if not clean.strip():
            continue
        indent = len(clean) - len(clean.lstrip(" "))
        lines.append((indent, clean.strip()))

    def parse_block(index: int, indent: int):
        if index >= len(lines):
            return {}, index

        is_list = lines[index][0] == indent and lines[index][1].startswith("- ")
        if is_list:
            values = []
            while index < len(lines):
                current_indent, content = lines[index]
                if current_indent != indent or not content.startswith("- "):
                    break
                values.append(parse_scalar(content[2:].strip()))
                index += 1
            return values, index

        mapping = {}
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected indentation near: {content}")
            if content.startswith("- "):
                break
            key, value = content.split(":", 1)
            key = key.strip()
            value = value.strip()
            index += 1
            if value:
                mapping[key] = parse_scalar(value)
            elif index < len(lines) and lines[index][0] > current_indent:
                mapping[key], index = parse_block(index, lines[index][0])
            else:
                mapping[key] = None
        return mapping, index

    data, index = parse_block(0, 0)
    if index != len(lines):
        raise ValueError("Could not parse complete YAML document")
    return data


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = simple_yaml_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    if yaml is not None:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms(image_size: int):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def make_loaders(
    manifest_dir: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    root_dir: str | Path = ".",
):
    train_transform, eval_transform = get_transforms(image_size)
    manifest_dir = Path(manifest_dir)

    train_dataset = ManifestImageDataset(manifest_dir / "train.csv", transform=train_transform, root_dir=root_dir)
    val_dataset = ManifestImageDataset(manifest_dir / "val.csv", transform=eval_transform, root_dir=root_dir)
    test_dataset = ManifestImageDataset(manifest_dir / "test.csv", transform=eval_transform, root_dir=root_dir)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def build_resnet18(init: str, num_classes: int) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if init == "imagenet" else None
    model = torchvision_resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_resnet18_encoder_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder_state = checkpoint.get("encoder_state_dict")
    if encoder_state is None:
        raise KeyError(f"{checkpoint_path} does not contain encoder_state_dict")

    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in encoder_state.items()
        if key in model_state and model_state[key].shape == value.shape and not key.startswith("fc.")
    }
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    if not compatible_state:
        raise ValueError(f"No compatible encoder weights found in {checkpoint_path}")
    if unexpected:
        print("Unexpected keys while loading SimCLR encoder:", unexpected)
    if missing:
        print("Missing keys after SimCLR encoder load:", missing)


def build_vit_s16(init: str, num_classes: int) -> nn.Module:
    try:
        import timm
    except ModuleNotFoundError as exc:
        if init == "imagenet":
            raise ModuleNotFoundError("timm is required for ImageNet-pretrained ViT-S/16") from exc
        from src.models.vit import vit_small
        from src.models.vit_cls_head import FinetuneViT

        return FinetuneViT(vit_small(patch_size=16, img_size=[224]), num_classes=num_classes)

    return timm.create_model("vit_small_patch16_224", pretrained=(init == "imagenet"), num_classes=num_classes)


def build_model(backbone: str, init: str, num_classes: int) -> nn.Module:
    if backbone == "resnet18":
        return build_resnet18(init, num_classes)
    if backbone == "vit_s16":
        return build_vit_s16(init, num_classes)
    raise ValueError(f"Unsupported backbone for supervised baseline: {backbone}")


def load_vit_dino_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher_state = checkpoint.get("teacher")
    if teacher_state is None:
        teacher_state = checkpoint.get("teacher_state_dict")
    if teacher_state is None:
        teacher_state = checkpoint.get("state_dict")
    if teacher_state is None:
        raise KeyError(f"{checkpoint_path} does not contain a DINO teacher state dict")

    model_state = model.state_dict()
    compatible_state = {}
    for key, value in teacher_state.items():
        if key.startswith("backbone."):
            new_key = key.replace("backbone.", "", 1)
        else:
            new_key = key
        if new_key.startswith("head.") or new_key.startswith("fc."):
            continue
        if new_key in model_state and model_state[new_key].shape == value.shape:
            compatible_state[new_key] = value

    if not compatible_state:
        raise ValueError(f"No compatible ViT DINO backbone weights found in {checkpoint_path}")
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    if unexpected:
        print("Unexpected keys while loading DINO teacher:", unexpected)
    if missing:
        print("Missing keys after DINO teacher load:", missing)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    preds: list[int] = []
    labels: list[int] = []

    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds.extend(outputs.argmax(dim=1).detach().cpu().tolist())
            labels.extend(targets.detach().cpu().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    metrics = compute_metrics(labels, preds)
    metrics["loss"] = avg_loss
    return metrics


def compute_metrics(labels: list[int], preds: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(labels, preds, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(labels, preds, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def collect_predictions(model, loader, device):
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    with torch.inference_mode():
        for images, targets in loader:
            outputs = model(images.to(device))
            preds.extend(outputs.argmax(dim=1).detach().cpu().tolist())
            labels.extend(targets.tolist())
    return labels, preds


def write_classification_report_csv(labels: list[int], preds: list[int], output_path: str | Path) -> None:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )

    output_path = Path(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_name", "precision", "recall", "f1_score", "support"])
        writer.writeheader()
        for idx, class_name in enumerate(CLASS_NAMES):
            writer.writerow(
                {
                    "class_name": class_name,
                    "precision": float(precision[idx]),
                    "recall": float(recall[idx]),
                    "f1_score": float(f1[idx]),
                    "support": int(support[idx]),
                }
            )


def save_confusion_matrix(labels: list[int], preds: list[int], output_path: str | Path) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_evaluation_outputs(model, loader, device, output_dir: str | Path, extra: dict[str, Any] | None = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels, preds = collect_predictions(model, loader, device)
    metrics = compute_metrics(labels, preds)
    if extra:
        metrics.update(extra)

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_classification_report_csv(labels, preds, output_dir / "classification_report.csv")
    save_confusion_matrix(labels, preds, output_dir / "confusion_matrix.png")
    return metrics


def resolve_training_config(config_path: str | Path, args) -> dict[str, Any]:
    config = load_yaml(config_path)
    finetune = config.setdefault("finetune", {})

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.epochs is not None:
        finetune["epochs"] = args.epochs
    if args.batch_size is not None:
        finetune["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        finetune["learning_rate"] = args.learning_rate

    config["manifest_dir"] = args.manifest_dir
    config["num_workers"] = args.num_workers
    config["device"] = args.device
    config["data"] = {
        "image_size": 224,
        "num_classes": 4,
        "class_names": CLASS_NAMES,
        "train_manifest": str(Path(args.manifest_dir) / "train.csv"),
        "val_manifest": str(Path(args.manifest_dir) / "val.csv"),
        "test_manifest": str(Path(args.manifest_dir) / "test.csv"),
    }
    return config
