from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from src.classification.supervised_baselines import (
    build_model,
    dump_yaml,
    make_loaders,
    resolve_training_config,
    run_epoch,
    save_evaluation_outputs,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train supervised ViT-S/16 baseline from fixed manifests.")
    parser.add_argument("--config", required=True, help="Experiment YAML path")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = resolve_training_config(args.config, args)
    if cfg["backbone"] != "vit_s16":
        raise ValueError(f"Expected vit_s16 config, got {cfg['backbone']}")
    if cfg["pretraining_strategy"] not in {"none", "imagenet"}:
        raise ValueError("This script only supports supervised none/imagenet baselines.")

    set_seed(42)
    device = torch.device(cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(cfg, output_dir / "config_resolved.yaml")

    finetune = cfg["finetune"]
    train_loader, val_loader, test_loader = make_loaders(
        manifest_dir=cfg["manifest_dir"],
        image_size=224,
        batch_size=int(finetune["batch_size"]),
        num_workers=int(cfg["num_workers"]),
    )

    model = build_model("vit_s16", finetune["init"], num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(finetune["learning_rate"]),
        weight_decay=float(finetune["weight_decay"]),
    )

    best_f1 = -1.0
    best_path = output_dir / "best_checkpoint.pth"
    epochs = int(finetune["epochs"])
    for epoch in range(epochs):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1_macro={val_metrics['f1_macro']:.4f}"
        )
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "val_metrics": val_metrics,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = save_evaluation_outputs(
        model,
        test_loader,
        device,
        output_dir,
        extra={"best_epoch": checkpoint["epoch"], "best_val_f1_macro": best_f1},
    )
    print("Saved baseline outputs to", output_dir)
    print(metrics)


if __name__ == "__main__":
    main()
