# MedCLS Colab notebooks

Generated notebooks for the project repo:

https://github.com/tlinhevg05/contrastive-synthesis-medcls_CVProject.git

Files:

- `00_train_compare_gans_acgan_dcgan.ipynb`: trains ACGAN and class-specific DCGANs, generates synthetic images, computes IS/FID.
- `01`-`06`: ResNet18 classification runs.
- `07`-`12`: ViT-S/16 classification runs.

Expected Google Drive data layout:

```text
/content/contrastive-synthesis-medcls_CVProject/data/processed/
├── labelled_4232/
├── unlabelled_16934/images/
├── synthetic_dcgan/      # created by notebook 00
└── synthetic_acgan/      # created by notebook 00
```

Notes:

- Each notebook has `FULL_RUN = True` by default. Set it to `False` for a 1-epoch smoke test.
- Synthetic classification runs require notebook 00 to finish first because they need `synthetic_dcgan`.
- Notebook number 00 should be run on local then push the output to git. Others can be run using Google Colab.
- The notebooks are self-contained and clone the repo for project context. If Colab returns a GitHub 404, check the repository visibility or spelling.

