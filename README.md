# HiCGen
a hierarchical and cell-type-specific genome organization generator 

HiCGen is a deep learning framework for predicting multiscale 3D genome organization (1 kb to 128 kb resolution) using DNA sequences and genomic features. Built on Swin-Transformer, HiCGen enables cross-cell-type predictions and *in silico* perturbation analysis to study structural consequences of genetic/epigenetic changes. 

**Paper**: [Preprint/BioRxiv Link]() | **Demo Data**: [Example Inputs/Outputs](data/)

![HiCGen Overview](docs/overview.png)

## Key Features
- **Multiscale Prediction**: Generate hierarchical contact maps (1 kb to 128 kb resolutions) from sequence and epigenetic signals.
- **Cross-Cell Generalization**: Predict chromatin architecture for unseen cell types using cell-specific ATAC/CTCF profiles.
- **Perturbation Analysis**: Simulate structural changes caused by enhancer/promoter activation/silencing or CTCF boundary editing.
- **Cancer Studies**: Model carcinogenesis-associated genome reorganization and identify pivotal regulatory elements.

## Installation

### Dependencies
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.7+ (GPU recommended)
- [cooler](https://github.com/open2c/cooler), [pyBigWig](https://github.com/deeptools/pyBigWig)

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/JasonWei2014/HiCGen.git
   cd HiCGen
