# Klein & PV Neural Network Layers

Hyperbolic fully connected (FC) layers on the **Klein ball** and **Proper Velocity (PV)** manifolds, extending the [GyroBN](https://github.com/GitZH-Chen/GyroBN) and [HBNN](https://github.com/GitZH-Chen/HBNN) frameworks.

## Overview

We introduce geodesic hyperplane FC and Busemann FC layers on two under-explored hyperbolic models:

| Space | FC Layer | Busemann FC Layer | MLR | Busemann MLR |
|-------|----------|-------------------|-----|--------------|
| **Klein ball** | KleinFC (Thm 5.5) | KleinBFC (Thm 5.6) | KleinMLR | KleinBMLR |
| **PV space** | PVFC | PVBFC | PVMLR | PVBMLR |

These layers are drop-in replacements for existing Poincare/Lorentz FC layers in the GyroBN training framework.

## Repository Structure

```
code/
├── klein_nn/                   # Our contribution: Klein & PV layers
│   ├── manifold.py             # Klein manifold operations
│   └── layers/
│       ├── fc.py               # KleinFC  — geodesic hyperplane FC
│       ├── bfc.py              # KleinBFC — Busemann horosphere FC
│       ├── mlr.py              # KleinMLR — geodesic hyperplane MLR
│       ├── bmlr.py             # KleinBMLR — Busemann MLR
│       ├── pv_fc.py            # PVFC
│       └── pv_bfc.py           # PVBFC
│
├── GyroBN-main/                # Training framework (modified fork)
│   ├── GyroBNH.py              # Entry point
│   ├── conf/nnet/              # Model configs (HNN, KleinFC, PVFC, etc.)
│   ├── Geometry/constantcurvature/
│   │   ├── klein.py            # KleinBall manifold (new)
│   │   └── pv.py               # PVSpace manifold (new)
│   └── RieNets/hnns/models/
│       └── encoders.py         # Encoder classes (HNN_KleinFC, HNN_PVFC, etc.)
│
├── HBNN-main/lib/              # BFC/BMLR layers (extended with Klein/PV)
├── test_layers.py              # Unit tests
└── PoinnCARE-main/             # Enzyme classification (Track B)
```

## Installation

```bash
pip install torch hydra-core hydra-joblib-launcher omegaconf scikit-learn tensorboard geoopt
```

## Data

Download graph datasets from [HGCN](https://github.com/HazyResearch/hgcn):

```bash
git clone https://github.com/HazyResearch/hgcn.git
cp -r hgcn/data/ code/data/
```

Datasets: Disease (delta=0), Airport (delta=1), PubMed (delta=3.5), Cora (delta=11).

## Usage

### Link Prediction (HBNN Table 7)

```bash
cd GyroBN-main

# Klein FC on Airport
python GyroBNH.py dataset.dataset=airport dataset.path=../data \
  nnet=KleinFC nnet.BN_param.is_bn=False nnet.optimizer.weight_decay=1e-3 \
  fit.epochs=5000 fit.double_precision=1

# PV FC on Airport
python GyroBNH.py dataset.dataset=airport dataset.path=../data \
  nnet=PVFC nnet.BN_param.is_bn=False nnet.optimizer.weight_decay=1e-3 \
  fit.epochs=5000 fit.double_precision=1

# Mobius FC baseline (reproduction)
python GyroBNH.py dataset.dataset=airport dataset.path=../data \
  nnet=HNN nnet.BN_param.is_bn=False nnet.optimizer.weight_decay=1e-3 \
  fit.epochs=5000 fit.double_precision=1
```

### Available Models

| Config | Encoder | Manifold | FC Layer |
|--------|---------|----------|----------|
| `HNN` | HNN | PoincareBall | Mobius matvec (HypLinear) |
| `LorentzFC` | HNN_LorentzFC | Hyperboloid | Ambient Minkowski FC |
| `BFC_P` | HNN_BFC | PoincareBall | Busemann FC |
| `BFC_L` | HNN_BFC | Hyperboloid | Busemann FC |
| `KleinFC` | HNN_KleinFC | KleinBall | Klein geodesic FC |
| `KleinBFC` | HNN_BFC | KleinBall | Klein Busemann FC |
| `PVFC` | HNN_PVFC | PVSpace | PV geodesic FC |
| `PVBFC` | HNN_BFC | PVSpace | PV Busemann FC |

### Hyperparameters

Following HBNN (Chen et al., CVPR 2026):

| Parameter | Value |
|-----------|-------|
| Embedding dim | 128 |
| Num layers | 2 |
| Dropout | 0.2 |
| Optimizer | Adam, lr=0.01 |
| Weight decay | 1e-3 (Cora: 0) |
| Epochs | 5000 |
| Early stopping | patience=100, min_epochs=100 |
| Precision | float64 |
| Curvature | K=-1 (fixed) |
| Seed | 1234 |
| Activation | relu (FC); identity/tanh (BFC) |

## Unit Tests

```bash
python test_layers.py
```

Verifies forward/backward pass, numerical stability, and Euclidean limit (K -> 0) for all Klein/PV layers.

## References

- **GyroBN**: Chen et al., "Gyrogroup Batch Normalization", ICLR 2025
- **HBNN**: Chen et al., "Hyperbolic Busemann Neural Networks", CVPR 2026
- **PV**: Su et al., "Proper Velocity Neural Networks", ICLR 2026
- **HGCN**: Chami et al., "Hyperbolic Graph Convolutional Neural Networks", NeurIPS 2019

## License

Research use only. GyroBN and HBNN code follows their respective licenses.
