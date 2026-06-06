# Enhancing Lung Radiology Image Classification with DCGAN Synthesis and Contrastive Learning

This repository implements a reproducible computer vision pipeline for 4-class chest X-ray classification. The project studies whether synthetic lung radiography images and contrastive/self-supervised pretraining can improve downstream medical image classification.

The classification task uses four classes:

- `COVID`
- `Lung_Opacity`
- `Viral_Pneumonia`
- `Normal`

The project is organized into three stages:

1. **Stage 1: Synthetic Image Generation**
   - Generate or register DCGAN synthetic chest X-ray images.
   - Inspect synthetic image samples.
   - Report synthetic image quality using FID and Inception Score when metric artifacts are available.

2. **Stage 2: Contrastive Pretraining**
   - ResNet18 uses SimCLR.
   - ViT-S/16 uses DINO.
   - `COVID-QU` means pretraining on the real unlabeled dataset.
   - `COVID-QU-Syn` means pretraining on DCGAN synthetic images.

3. **Stage 3: Supervised Fine-Tuning and Evaluation**
   - All downstream classifiers are fine-tuned on the same fixed real labeled train/val/test split.
   - Evaluation reports accuracy, macro/weighted precision, recall, F1-score, classification report, and confusion matrix.

## Dataset Layout

The repository expects the following processed data layout:

```text
data/processed/
  labelled_4232/
    COVID/images/
    Lung_Opacity/images/
    Viral_Pneumonia/images/
    Normal/images/
  unlabelled_16934/
    images/
```

The fixed supervised split is stored in:

```text
data/manifests/
  labelled_all.csv
  train.csv
  val.csv
  test.csv
  split_summary.json
```

The registered DCGAN synthetic dataset is stored as:

```text
data/manifests/synthetic_dcgan.csv
```

The synthetic images themselves are not required to be committed to Git. On Colab/Drive they are expected at a configurable path such as:

```text
/content/drive/MyDrive/medcls_cvproject/data/processed/synthetic_dcgan/
```

## Experiments

The report-aligned experiment design contains 12 experiments:

| Backbone | Strategy | Experiment ID |
|---|---|---|
| ResNet18 | None | `resnet18_none` |
| ResNet18 | ImageNet | `resnet18_imagenet` |
| ResNet18 | COVID-QU | `resnet18_covidqu` |
| ResNet18 | ImageNet -> COVID-QU | `resnet18_imagenet_covidqu` |
| ResNet18 | COVID-QU-Syn | `resnet18_covidqu_syn` |
| ResNet18 | ImageNet -> COVID-QU-Syn | `resnet18_imagenet_covidqu_syn` |
| ViT-S/16 | None | `vit_s16_none` |
| ViT-S/16 | ImageNet | `vit_s16_imagenet` |
| ViT-S/16 | COVID-QU | `vit_s16_covidqu` |
| ViT-S/16 | ImageNet -> COVID-QU | `vit_s16_imagenet_covidqu` |
| ViT-S/16 | COVID-QU-Syn | `vit_s16_covidqu_syn` |
| ViT-S/16 | ImageNet -> COVID-QU-Syn | `vit_s16_imagenet_covidqu_syn` |

Experiment configs are stored under:

```text
configs/experiments/
  common.yaml
  paths.template.yaml
  resnet18/
  vit_s16/
```

## Repository Structure

```text
configs/experiments/       # 12 reproducible experiment configs
data/manifests/            # fixed train/val/test and synthetic manifests
notebooks/                 # Colab/Kaggle runner notebooks
scripts/                   # command-line entrypoints
src/                       # reusable model/data/training/evaluation code
ckpts/                     # local checkpoint placeholder only
```

The intended separation is:

- `notebooks/`: environment setup and experiment runners. Most training notebooks call Python scripts with `!python ...`.
- `scripts/`: executable entrypoints for one pipeline action, such as SimCLR pretraining, DINO pretraining, supervised fine-tuning, evaluation, or input validation.
- `src/`: reusable library code used by scripts, including model definitions, datasets, transforms, losses, and utility functions.

## Main Notebooks

```text
notebooks/01_train_dcgan.ipynb
```

Stage 1 DCGAN notebook. This notebook contains DCGAN training/inspection code directly because it is a self-contained synthetic generation workflow. By default it is configured to reuse existing DCGAN outputs instead of retraining.

```text
notebooks/02_train_supervised_baselines.ipynb
```

Runs the four supervised baseline experiments:

- `resnet18_none`
- `resnet18_imagenet`
- `vit_s16_none`
- `vit_s16_imagenet`

```text
notebooks/03_resnet18_covidqu.ipynb
notebooks/04_resnet18_imagenet_covidqu.ipynb
notebooks/05_resnet18_covidqu_syn.ipynb
notebooks/06_resnet18_imagenet_covidqu_syn.ipynb
```

Run the four ResNet18 SimCLR experiments. These notebooks call:

```text
scripts/run_simclr_resnet.py
scripts/run_classification_resnet.py
```

```text
notebooks/07_vit_s16_covidqu.ipynb
notebooks/08_vit_s16_imagenet_covidqu.ipynb
notebooks/09_vit_s16_covidqu_syn.ipynb
notebooks/10_vit_s16_imagenet_covidqu_syn.ipynb
```

Run the four ViT-S/16 DINO experiments. These notebooks call:

```text
scripts/run_dino_vit.py
scripts/run_classification_vit.py
```

## Setup

Create an environment with Python 3.10+ or use Colab/Kaggle. The notebooks include platform-specific setup cells. For local script checks:

```bash
pip install -r requirements.txt
```

If the runtime already provides PyTorch, especially on Colab or Kaggle, install non-PyTorch dependencies only or use the dependency filtering cell in the notebooks. This avoids CUDA/PyTorch wheel conflicts.

## Verify Inputs

Before training, run:

```bash
python scripts/check_experiment_inputs.py
```

This validates:

- all 12 experiment configs
- fixed supervised manifests
- required CSV columns
- class names and labels
- real unlabeled dataset path
- DCGAN synthetic manifest for COVID-QU-Syn experiments
- planned output directories

Expected result before training:

```text
FAIL: 0
```

## Create Fixed Splits

The train/val/test split is a CPU-only preprocessing step:

```bash
python scripts/create_split_manifest.py \
  --labelled-dir data/processed/labelled_4232 \
  --output-dir data/manifests \
  --split-seed 42
```

This creates:

```text
data/manifests/train.csv
data/manifests/val.csv
data/manifests/test.csv
data/manifests/labelled_all.csv
data/manifests/split_summary.json
```

## Register DCGAN Synthetic Dataset

If synthetic images are stored on Google Drive, register them into a manifest:

```bash
python scripts/register_synthetic_dataset.py \
  --synthetic-dir /path/to/synthetic_dcgan \
  --manifest-path data/manifests/synthetic_dcgan.csv \
  --stage1-dir results/stage1_synthesis
```

The manifest contains:

- `image_path`
- `class_name`
- `label`
- `source`
- `generator`

Label mapping:

```text
COVID: 0
Lung_Opacity: 1
Viral_Pneumonia: 2
Normal: 3
```

## Run Supervised Baselines

Examples:

```bash
python scripts/run_classification_resnet.py \
  --config configs/experiments/resnet18/none.yaml \
  --manifest-dir data/manifests \
  --output-dir results/experiments/resnet18_none
```

```bash
python scripts/run_classification_vit.py \
  --config configs/experiments/vit_s16/imagenet.yaml \
  --manifest-dir data/manifests \
  --output-dir results/experiments/vit_s16_imagenet
```

## Run ResNet18 SimCLR Experiments

SimCLR pretraining:

```bash
python scripts/run_simclr_resnet.py \
  --config configs/experiments/resnet18/covidqu.yaml \
  --output-dir results/experiments/resnet18_covidqu
```

Fine-tuning from SimCLR checkpoint:

```bash
python scripts/run_classification_resnet.py \
  --config configs/experiments/resnet18/covidqu.yaml \
  --manifest-dir data/manifests \
  --output-dir results/experiments/resnet18_covidqu \
  --pretrained-checkpoint results/experiments/resnet18_covidqu/pretrain/checkpoints/best_simclr_backbone.pth
```

## Run ViT-S/16 DINO Experiments

DINO pretraining:

```bash
python scripts/run_dino_vit.py \
  --config configs/experiments/vit_s16/covidqu.yaml \
  --output-dir results/experiments/vit_s16_covidqu
```

Fine-tuning from DINO checkpoint:

```bash
python scripts/run_classification_vit.py \
  --config configs/experiments/vit_s16/covidqu.yaml \
  --manifest-dir data/manifests \
  --output-dir results/experiments/vit_s16_covidqu \
  --pretrained-checkpoint results/experiments/vit_s16_covidqu/pretrain/checkpoints/best_dino_teacher.pth
```

## Outputs

Each classification experiment writes to:

```text
results/experiments/<experiment_id>/
```

Expected files:

```text
config_resolved.yaml
best_checkpoint.pth
metrics.json
classification_report.csv
confusion_matrix.png
```

Contrastive experiments also include:

```text
pretrain/checkpoints/
pretrain/*_history.json
```

`results/` and model checkpoint files are ignored by Git.

## Metrics

The classification metrics are:

- `accuracy`
- `precision_macro`
- `recall_macro`
- `f1_macro`
- `precision_weighted`
- `recall_weighted`
- `f1_weighted`

Synthetic image quality can be summarized using:

- FID between real labeled images and DCGAN synthetic images
- Inception Score on DCGAN synthetic images
- visual sample grids by class

## Notes

- Training should be run on GPU through Colab or Kaggle.
- Local execution is intended for repository inspection, manifest creation, config validation, and lightweight script checks.
- The fixed split seed is `42`.
- The image size used by the experiment configs is `224`.
- The synthetic source for all `COVID-QU-Syn` experiments is DCGAN.
