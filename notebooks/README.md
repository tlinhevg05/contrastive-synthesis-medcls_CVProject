# MedCLS Colab notebooks

Generated notebooks for the project repo:

https://github.com/tlinhevg05/contrastive-synthesis-medcls_CVProject.git

These notebooks reproduce the experiment structure from the project report:

- GAN-based synthetic image generation using ACGAN and DCGAN
- ResNet18 classification under 6 pretraining strategies
- ViT-S/16 classification under 6 pretraining strategies
- Result aggregation across the 12 classification runs

## Files

- `00_train_compare_gans_acgan_dcgan.ipynb`

  **Strategy:** Synthetic image generation and GAN comparison.

  **Pipeline:**

  ```text
  labelled_4232 real X-ray images
      ↓
  train 1 ACGAN model
      +
  train 4 class-specific DCGAN models
      ↓
  generate synthetic_acgan and synthetic_dcgan images
      ↓
  compare ACGAN vs DCGAN using IS/FID
      ↓
  save synthetic images for later synthetic classification runs
  ```

  **Output used by later notebooks:**

  ```text
  /content/drive/MyDrive/medcls_cvproject/data/processed/synthetic_dcgan/
  ```

---

- `01_resnet18_none.ipynb`

  **Strategy:** ResNet18 trained from scratch, no pretraining.

  **Pipeline:**

  ```text
  labelled_4232 real X-ray images
      ↓
  initialize ResNet18 randomly
      ↓
  supervised full fine-tuning on labelled real data
      ↓
  evaluate on test split
  ```

---

- `02_resnet18_imagenet.ipynb`

  **Strategy:** ResNet18 with ImageNet pretraining.

  **Pipeline:**

  ```text
  labelled_4232 real X-ray images
      ↓
  load ImageNet-pretrained ResNet18
      ↓
  replace classification head for 4 classes
      ↓
  supervised full fine-tuning on labelled real data
      ↓
  evaluate on test split
  ```

---

- `03_resnet18_covid_qu.ipynb`

  **Strategy:** ResNet18 with SimCLR pretraining on real unlabelled COVID-QU images.

  **Pipeline:**

  ```text
  unlabelled_16934 real X-ray images
      ↓
  SimCLR self-supervised pretraining
      ↓
  load pretrained ResNet18 encoder
      ↓
  supervised full fine-tuning on labelled_4232
      ↓
  evaluate on test split
  ```

---

- `04_resnet18_imagenet_to_covid_qu.ipynb`

  **Strategy:** ResNet18 with ImageNet pretraining, then SimCLR pretraining on real unlabelled COVID-QU images.

  **Pipeline:**

  ```text
  ImageNet-pretrained ResNet18
      ↓
  SimCLR pretraining on unlabelled_16934 real X-ray images
      ↓
  supervised full fine-tuning on labelled_4232
      ↓
  evaluate on test split
  ```

---

- `05_resnet18_covid_qu_syn.ipynb`

  **Strategy:** ResNet18 with SimCLR pretraining on DCGAN synthetic images.

  **Pipeline:**

  ```text
  synthetic_dcgan images from notebook 00
      ↓
  SimCLR self-supervised pretraining
      ↓
  load pretrained ResNet18 encoder
      ↓
  supervised full fine-tuning on labelled_4232 real data
      ↓
  evaluate on test split
  ```

  **Requirement:** Run notebook `00_train_compare_gans_acgan_dcgan.ipynb` first.

---

- `06_resnet18_imagenet_to_covid_qu_syn.ipynb`

  **Strategy:** ResNet18 with ImageNet pretraining, then SimCLR pretraining on DCGAN synthetic images.

  **Pipeline:**

  ```text
  ImageNet-pretrained ResNet18
      ↓
  SimCLR pretraining on synthetic_dcgan images
      ↓
  supervised full fine-tuning on labelled_4232 real data
      ↓
  evaluate on test split
  ```

  **Requirement:** Run notebook `00_train_compare_gans_acgan_dcgan.ipynb` first.

---

- `07_vit_s16_none.ipynb`

  **Strategy:** ViT-S/16 trained from scratch, no pretraining.

  **Pipeline:**

  ```text
  labelled_4232 real X-ray images
      ↓
  initialize ViT-S/16 randomly
      ↓
  supervised full fine-tuning on labelled real data
      ↓
  evaluate on test split
  ```

---

- `08_vit_s16_imagenet.ipynb`

  **Strategy:** ViT-S/16 with ImageNet pretraining.

  **Pipeline:**

  ```text
  labelled_4232 real X-ray images
      ↓
  load ImageNet-pretrained ViT-S/16
      ↓
  replace classification head for 4 classes
      ↓
  supervised full fine-tuning on labelled real data
      ↓
  evaluate on test split
  ```

---

- `09_vit_s16_covid_qu.ipynb`

  **Strategy:** ViT-S/16 with DINO pretraining on real unlabelled COVID-QU images.

  **Pipeline:**

  ```text
  unlabelled_16934 real X-ray images
      ↓
  DINO self-supervised pretraining
      ↓
  load pretrained ViT-S/16 backbone
      ↓
  supervised full fine-tuning on labelled_4232
      ↓
  evaluate on test split
  ```

---

- `10_vit_s16_imagenet_to_covid_qu.ipynb`

  **Strategy:** ViT-S/16 with ImageNet pretraining, then DINO pretraining on real unlabelled COVID-QU images.

  **Pipeline:**

  ```text
  ImageNet-pretrained ViT-S/16
      ↓
  DINO pretraining on unlabelled_16934 real X-ray images
      ↓
  supervised full fine-tuning on labelled_4232
      ↓
  evaluate on test split
  ```

---

- `11_vit_s16_covid_qu_syn.ipynb`

  **Strategy:** ViT-S/16 with DINO pretraining on DCGAN synthetic images.

  **Pipeline:**

  ```text
  synthetic_dcgan images from notebook 00
      ↓
  DINO self-supervised pretraining
      ↓
  load pretrained ViT-S/16 backbone
      ↓
  supervised full fine-tuning on labelled_4232 real data
      ↓
  evaluate on test split
  ```

  **Requirement:** Run notebook `00_train_compare_gans_acgan_dcgan.ipynb` first.

---

- `12_vit_s16_imagenet_to_covid_qu_syn.ipynb`

  **Strategy:** ViT-S/16 with ImageNet pretraining, then DINO pretraining on DCGAN synthetic images.

  **Pipeline:**

  ```text
  ImageNet-pretrained ViT-S/16
      ↓
  DINO pretraining on synthetic_dcgan images
      ↓
  supervised full fine-tuning on labelled_4232 real data
      ↓
  evaluate on test split
  ```

  **Requirement:** Run notebook `00_train_compare_gans_acgan_dcgan.ipynb` first.


## Expected data layout

The real labelled and unlabelled datasets are expected to be inside the cloned repo:

```text
/content/contrastive-synthesis-medcls_CVProject/data/processed/
├── labelled_4232/
└── unlabelled_16934/
    └── images/
```

Notebook `00` creates synthetic images and saves them to Google Drive:

```text
/content/drive/MyDrive/medcls_cvproject/data/processed/
├── synthetic_dcgan/
└── synthetic_acgan/
```

The synthetic classification notebooks use:

```text
/content/drive/MyDrive/medcls_cvproject/data/processed/synthetic_dcgan/
```

## Notes

- Each notebook has `FULL_RUN = True` by default. Set it to `False` for a short smoke test.
- Synthetic classification runs require notebook `00_train_compare_gans_acgan_dcgan.ipynb` to finish first because they need `synthetic_dcgan`.
- The notebooks are designed for Google Colab.
- Before running, enable GPU in Colab:

  ```text
  Runtime → Change runtime type → Hardware accelerator → GPU
  ```

- The notebooks use the repo scripts as much as possible. Some notebook code is still used for Colab setup, Google Drive mounting, path configuration, output management, and GAN orchestration where the repo does not provide direct training scripts.
- Outputs and metrics are saved to Google Drive under:

  ```text
  /content/drive/MyDrive/medcls_cvproject/outputs/
  ```

- If Colab returns a GitHub 404, check the repository visibility or spelling.
