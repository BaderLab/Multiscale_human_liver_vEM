# Multiscale Human Liver Volume Electron Microscopy

## Introduction

Using a deep learning-based segmentation framework, we generated comprehensive labels across vascular, cellular, and subcellular levels, enabling quantitative analysis of bile duct–cholangiocyte organization and sinusoidal branch geometry. At the organelle scale, analysis of 37,920 mitochondria revealed distinct morphological profiles and spatial distributions. Examination of mitochondrial–endoplasmic reticulum (ER) spatial relationships uncovered characteristic ER-associated mitochondrial narrowing, indicative of fission and fusion activity.

<p align="center">
  <img src="figures/automated_segmentation.png" alt="Automated segmentation overview" width="800">
</p>

## Installation

1. **Create a virtual environment** to install the required packages. An example setup script is provided in `create_run_example.slurm`.

2. **Clone the repository:**
   ```bash
   git clone https://github.com/BaderLab/Multiscale_human_liver_vEM.git
   ```

3. **Install dependencies:**
   ```bash
   cd Multiscale_human_liver_vEM
   pip install -r requirements.txt
   ```
   The `requirements.txt` includes packages needed for both [nnUNet](https://github.com/MIC-DKFZ/nnUNet) and [SAM2](https://github.com/facebookresearch/sam2).

## Getting Started

### 1. Vascular and Cellular Level Segmentation

We provide a script that uses SAM2 to generate 3D instance masks from input prompts:

```bash
python sam2maskpropagator.py
```

### 2. Organelle Segmentation

Organelle segmentation was performed using [nnUNet](https://github.com/MIC-DKFZ/nnUNet) with pretraining and fine-tuning. Trained model checkpoints for all segmented organelles are available on [Zenodo](https://zenodo.org/uploads/17360859).

### 3. Mitochondrial Morphology Feature Extraction

After obtaining organelle masks, morphological features of mitochondria were extracted using [PyRadiomics](https://pyradiomics.readthedocs.io/):

```bash
python morphology_features.py
```

### 4. Mitochondria–ER Interaction Analysis

To analyze mitochondria–ER spatial interactions:

```bash
python mito_er_analysis.py
```

## Acknowledgements

We thank the [SAM2](https://arxiv.org/abs/2408.00714) and [nnUNet](https://www.nature.com/articles/s41592-020-01008-z) teams for making their source code publicly available. We also thank the [PyRadiomics](https://doi.org/10.1158/0008-5472.CAN-17-0339) team for their open-source morphological feature extraction package. We gratefully acknowledge [OpenOrganelle](https://openorganelle.janelia.org/) and [Parlakgül et al. (2022)](https://www.nature.com/articles/s41586-022-04488-5) for making the mouse liver volume electron microscopy data publicly available.

## Citation

<!-- Update this once the manuscript is published -->
```bibtex
@article{multiscale_human_liver,
  title   = {},
  author  = {},
  journal = {},
  volume  = {},
  pages   = {},
  year    = {}
}
```
