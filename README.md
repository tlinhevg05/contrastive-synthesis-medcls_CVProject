# Evaluating Pretraining Strategies for Lung Radiology Classification

This repository implements a reproducible computer vision pipeline for 4-class chest X-ray classification under limited labeled-data conditions. The project compares how different initialization and pretraining strategies affect downstream classification performance when all models are fine-tuned and evaluated on the same fixed real labeled split.

The target classes are:

- `COVID`
- `Lung_Opacity`
- `Viral_Pneumonia`
- `Normal`

The main question is not whether synthetic data always improves classification. Instead, the project evaluates different pretraining conditions, including ImageNet initialization, real unlabeled chest X-ray pretraining, and DCGAN-generated synthetic chest X-ray pretraining. Synthetic images are treated as one candidate pretraining source whose usefulness must be tested empirically against real unlabeled images and standard transfer learning baselines.

## Pipeline Overview

The project is organized around a controlled experiment matrix. Each strategy differs in initialization or pretraining source, but all strategies share the same downstream supervised fine-tuning and evaluation protocol.

### Stage 1: Synthetic Pretraining Source Construction

Class-specific DCGAN generators are trained using the real labeled chest X-ray subset. Each generator produces images for one target class. The generated images are registered as a synthetic pretraining source with 1,000 images per class.

This synthetic source is denoted as:

- `CXR-Syn`: DCGAN-generated synthetic chest X-ray images

The synthetic data is characterized using:

- visual inspection
- Fréchet Inception Distance (FID)
- Inception Score (IS)

These metrics are used only as supporting diagnostics. They are not used as the main criterion for deciding whether synthetic data is useful. The main evaluation of `CXR-Syn` is its downstream effect when used as a self-supervised pretraining source.

### Stage 2: Self-Supervised Pretraining

Two self-supervised learning methods are used:

- ResNet18 uses SimCLR.
- ViT-S/16 uses DINO.

Two pretraining sources are compared:

- `CXR-Unlabeled`: real unlabeled chest X-ray images
- `CXR-Syn`: DCGAN-generated synthetic chest X-ray images

The ImageNet-initialized variants test whether self-supervised pretraining provides additional adaptation after the backbone has already learned generic visual representations from ImageNet.

### Stage 3: Supervised Fine-Tuning and Evaluation

All strategies are fine-tuned on the same fixed real labeled train/validation/test manifests. The supervised fine-tuning set contains only real labeled chest X-ray images. Synthetic images are not added to the supervised training, validation, or test split.

Evaluation reports:

- accuracy
- macro precision, recall, and F1-score
- weighted precision, recall, and F1-score
- classification report
- confusion matrix

## Dataset Layout

The expected processed data layout is:

```text
data/processed/
  labelled_4232/
    COVID/images/
    Lung_Opacity/images/
    Viral_Pneumonia/images/
    Normal/images/
  unlabelled_16933/
    images/
````

In the report and experiment descriptions, the real unlabeled subset is referred to as:

```text
CXR-Unlabeled
```

The physical folder name may still be `unlabelled_16933` because it reflects the processed data directory used by the code.

The fixed supervised split is stored in:

```text
data/manifests/
  labelled_all.csv
  train.csv
  val.csv
  test.csv
  split_summary.json
```

The registered synthetic dataset manifest is:

```text
data/manifests/synthetic_dcgan.csv
```

Synthetic images are not required to be committed to Git. In Colab or Kaggle, point the notebooks to the folder containing:

```text
synthetic_dcgan/
  COVID/images/
  Lung_Opacity/images/
  Viral_Pneumonia/images/
  Normal/images/
```

In the report and experiment descriptions, this synthetic source is referred to as:

```text
CXR-Syn
```

## Preprocessing

The `data/processed` folders contain curated and organized image files. Pixel-level preprocessing is applied online by each training stage rather than permanently written back to disk.

* DCGAN training resizes images and normalizes them to the `[-1, 1]` range.
* SimCLR and DINO apply self-supervised multi-view augmentations during pretraining.
* Supervised fine-tuning resizes images to `224 x 224`, applies random horizontal flipping during training, converts images to tensors, and uses ImageNet normalization.

Random resized cropping and stronger multi-view augmentations are mainly used during the self-supervised pretraining stages, not in the supervised fine-tuning baseline pipeline.

## Experiment Matrix

The complete experiment design contains 12 experiments:

| Backbone | Strategy name in report   | Experiment ID in code           |
| -------- | ------------------------- | ------------------------------- |
| ResNet18 | None                      | `resnet18_none`                 |
| ResNet18 | ImageNet                  | `resnet18_imagenet`             |
| ResNet18 | CXR-Unlabeled             | `resnet18_covidqu`              |
| ResNet18 | ImageNet -> CXR-Unlabeled | `resnet18_imagenet_covidqu`     |
| ResNet18 | CXR-Syn                   | `resnet18_covidqu_syn`          |
| ResNet18 | ImageNet -> CXR-Syn       | `resnet18_imagenet_covidqu_syn` |
| ViT-S/16 | None                      | `vit_s16_none`                  |
| ViT-S/16 | ImageNet                  | `vit_s16_imagenet`              |
| ViT-S/16 | CXR-Unlabeled             | `vit_s16_covidqu`               |
| ViT-S/16 | ImageNet -> CXR-Unlabeled | `vit_s16_imagenet_covidqu`      |
| ViT-S/16 | CXR-Syn                   | `vit_s16_covidqu_syn`           |
| ViT-S/16 | ImageNet -> CXR-Syn       | `vit_s16_imagenet_covidqu_syn`  |

The experiment IDs keep the original `covidqu` naming to remain compatible with existing configuration files and result folders. The report-facing names `CXR-Unlabeled` and `CXR-Syn` are used to make the role of each pretraining source clearer.

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
configs/experiments/       # experiment configs
data/manifests/            # fixed split and synthetic manifests
notebooks/                 # Colab/Kaggle experiment runners
scripts/                   # command-line entrypoints
src/                       # reusable model, data, training, and evaluation code
ckpts/                     # local checkpoint placeholder
```

The intended separation is:

* `notebooks/`: platform setup and experiment runners
* `scripts/`: executable entrypoints for one pipeline action
* `src/`: reusable library code used by scripts

Most training notebooks call Python scripts. The DCGAN notebook is more self-contained because it includes generation, inspection, and synthetic-data registration logic in one workflow.

## Main Notebooks

```text
notebooks/01_train_dcgan.ipynb
```

Stage 1 DCGAN training, inspection, synthetic-image generation, and synthetic dataset registration.

```text
notebooks/02_train_supervised_baselines.ipynb
```

Kaggle runner for the four supervised baselines:

* `resnet18_none`
* `resnet18_imagenet`
* `vit_s16_none`
* `vit_s16_imagenet`

```text
notebooks/03_resnet18_covidqu.ipynb
notebooks/04_resnet18_imagenet_covidqu.ipynb
notebooks/05_resnet18_covidqu_syn.ipynb
notebooks/06_resnet18_imagenet_covidqu_syn.ipynb
```

ResNet18 SimCLR experiment runners for the `CXR-Unlabeled`, `ImageNet -> CXR-Unlabeled`, `CXR-Syn`, and `ImageNet -> CXR-Syn` strategies.

```text
notebooks/07_vit_s16_covidqu.ipynb
notebooks/08_vit_s16_imagenet_covidqu.ipynb
notebooks/09_vit_s16_covidqu_syn.ipynb
notebooks/10_vit_s16_imagenet_covidqu_syn.ipynb
```

ViT-S/16 DINO experiment runners for the `CXR-Unlabeled`, `ImageNet -> CXR-Unlabeled`, `CXR-Syn`, and `ImageNet -> CXR-Syn` strategies.

```text
notebooks/11_colab_read_experiment_outputs.ipynb
```

Utility notebook for reading and summarizing result folders.

## Setup

For local checks:

```bash
pip install -r requirements.txt
```

On Colab or Kaggle, PyTorch is usually already installed. In that case, use the notebook dependency cells or install only the non-PyTorch packages to avoid CUDA wheel conflicts.

## Verify Inputs

Before training, run:

```bash
python scripts/check_experiment_inputs.py
```

This validates:

* all experiment configs
* fixed supervised manifests
* required CSV columns
* class names and labels
* real unlabeled dataset path
* synthetic DCGAN manifest for `CXR-Syn` experiments

The check should end with:

```text
FAIL: 0
```

Warnings about `synthetic_dcgan.csv` are acceptable before the Stage 1 synthetic registry has been run.

## Create Fixed Splits

The train/validation/test split is a CPU-only local preprocessing step:

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

## Register Synthetic Data

If DCGAN images are stored outside the repo, register them into a manifest:

```bash
python scripts/register_synthetic_dataset.py \
  --synthetic-dir /path/to/synthetic_dcgan \
  --manifest-path data/manifests/synthetic_dcgan.csv \
  --stage1-dir results/stage1_synthesis
```

The manifest contains:

* `image_path`
* `class_name`
* `label`
* `source`
* `generator`

Label mapping:

```text
COVID: 0
Lung_Opacity: 1
Viral_Pneumonia: 2
Normal: 3
```

## Run Baselines

Example commands:

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

## Run SSL Pretraining + Fine-Tuning

Example ResNet18 SimCLR run using the real unlabeled source `CXR-Unlabeled`:

```bash
python scripts/run_simclr_resnet.py \
  --config configs/experiments/resnet18/covidqu.yaml \
  --real-unlabeled-dir data/processed/unlabelled_16933 \
  --output-dir results/experiments/resnet18_covidqu
```

```bash
python scripts/run_classification_resnet.py \
  --config configs/experiments/resnet18/covidqu.yaml \
  --manifest-dir data/manifests \
  --output-dir results/experiments/resnet18_covidqu \
  --pretrained-checkpoint results/experiments/resnet18_covidqu/pretrain/checkpoints/best_simclr_backbone.pth
```

Example ViT-S/16 DINO run using the real unlabeled source `CXR-Unlabeled`:

```bash
python scripts/run_dino_vit.py \
  --config configs/experiments/vit_s16/covidqu.yaml \
  --real-unlabeled-dir data/processed/unlabelled_16933 \
  --output-dir results/experiments/vit_s16_covidqu
```

```bash
python scripts/run_classification_vit.py \
  --config configs/experiments/vit_s16/covidqu.yaml \
  --manifest-dir data/manifests \
  --output-dir results/experiments/vit_s16_covidqu \
  --pretrained-checkpoint results/experiments/vit_s16_covidqu/pretrain/checkpoints/best_dino_teacher.pth
```

For synthetic pretraining, use the corresponding `covidqu_syn` configs and provide the registered synthetic manifest:

```text
data/manifests/synthetic_dcgan.csv
```

## Output Files

Each completed classification experiment saves:

```text
results/experiments/<experiment_id>/
  config_resolved.yaml
  best_checkpoint.pth
  metrics.json
  classification_report.csv
  confusion_matrix.png
```

SSL-based experiments additionally save pretraining checkpoints and history under:

```text
results/experiments/<experiment_id>/pretrain/
```

```
```
