from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.classification.supervised_baselines import build_model, load_yaml, make_loaders, save_evaluation_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate supervised ResNet18 baseline on fixed test manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    if cfg["backbone"] != "resnet18":
        raise ValueError(f"Expected resnet18 config, got {cfg['backbone']}")

    finetune = cfg["finetune"]
    batch_size = args.batch_size or int(finetune["batch_size"])
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output_dir"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    _, _, test_loader = make_loaders(
        manifest_dir=args.manifest_dir,
        image_size=224,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )
    model = build_model("resnet18", finetune["init"], num_classes=4).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = save_evaluation_outputs(model, test_loader, device, output_dir)
    print(metrics)


if __name__ == "__main__":
    main()
