# Heterogeneous Graph Diffusion (HGD)

This repository contains the implementation of **Heterogeneous Graph Diffusion**, a graph-structure-aware diffusion model for self-supervised graph representation learning.

## Overview

Standard diffusion models treat node features independently of graph structure during the forward noising process. HGD instead couples the forward process to the graph topology via **Laplacian dynamics**: the clean signal at each timestep is the Laplacian-diffused version of the original features, so information propagates along edges before Gaussian noise is added.

**Forward process:**
```
x_t = sqrt(α̅_t) · (-L)^t x_0  +  sqrt(1 − α̅_t) · ε,    ε ~ N(0, 2I)
```

where `L` is the normalized graph Laplacian. This heterogeneous *n*-particle system can be derived from:
```
dX^i_t = −(X^i_t + Σ_{j≠i} ξ_{ij} (X^i_t − X^j_t)) dt + √2 dW^i_t
```

The reverse process is parameterized by a **Denoising U-Net** built from Graph Attention Network (GAT) layers with skip connections and time-step injection.

## Project Structure

```
├── src/
│   ├── models/
│   │   ├── hgd.py            # HGD (our method)
│   │   ├── ddm.py            # DDM baseline
│   │   ├── denoising_net.py  # Shared U-Net backbone
│   │   └── utils.py
│   ├── datasets/
│   │   └── data_util.py      # TUDataset loading & preprocessing
│   ├── utils/
│   │   └── training.py       # Optimizer, pooler, checkpoint helpers
│   └── evaluate.py           # SVM-based graph classification evaluation
├── configs/
│   └── MUTAG.yaml            # Hyperparameters
├── train_hgd.py              # Train HGD
├── train_ddm.py              # Train DDM baseline
├── scripts/
│   └── run_mutag.sh          # Run both models on MUTAG
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

> For GPU support, install the DGL and PyTorch variants matching your CUDA version.

## Running Experiments

**Graph classification on MUTAG:**
```bash
# Train HGD (our method)
python train_hgd.py --config configs/MUTAG.yaml

# Train DDM baseline
python train_ddm.py --config configs/MUTAG.yaml

# Or run both at once
bash scripts/run_mutag.sh
```

Results (F1 score) are logged to `logs/hgd/train.log` and `logs/ddm/train.log`.

## Supported Datasets

Any TU benchmark dataset supported by DGL's `TUDataset` can be used. Set `DATA.data_name` in the config:

| Dataset  | Graphs | Classes |
|----------|--------|---------|
| MUTAG    | 188    | 2       |
| IMDB-B   | 1000   | 2       |
| REDDIT-B | 2000   | 2       |
| COLLAB   | 5000   | 3       |

## Key Hyperparameters

| Parameter       | Description                              | Default |
|-----------------|------------------------------------------|---------|
| `T`             | Number of diffusion timesteps            | 50      |
| `beta_schedule` | Noise schedule (`linear`/`sigmoid`/...)  | sigmoid |
| `num_hidden`    | Hidden dimension of the U-Net            | 512     |
| `num_layers`    | Number of GAT layers per path            | 2       |
| `nhead`         | Attention heads per GAT layer            | 4       |

## Citation

If you use this code, please cite:

```bibtex
@article{hgd2024,
  title   = {Heterogeneous Graph Diffusion via Laplacian Dynamics},
  author  = {Chen Xu and others},
  year    = {2024}
}
```
