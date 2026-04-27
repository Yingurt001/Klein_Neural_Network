# Klein Neural Network Layers

Closed-form neural network layers on the **Klein ball** model of hyperbolic geometry, built on the Einstein gyrovector structure.

| Layer | Description |
|-------|-------------|
| `KleinFC` | Geodesic-hyperplane fully connected layer |
| `KleinBFC` | Busemann-horosphere fully connected layer |
| `KleinMLR` | Geodesic-hyperplane multinomial logistic regression |
| `KleinBMLR` | Busemann-horosphere multinomial logistic regression |
| `KleinEinsteinBN` | Closed-form Einstein-midpoint batch normalization |

All layers are drop-in replacements for the Poincaré / Lorentz counterparts in the [GyroBN](https://github.com/GitZH-Chen/GyroBN) and [HBNN](https://github.com/GitZH-Chen/HBNN) training frameworks.

## Repository Structure

```
.
├── klein_nn/                 # Core library (this work)
│   ├── manifold.py           # Klein manifold operations
│   └── layers/
│       ├── fc.py             # KleinFC  — geodesic-hyperplane FC
│       ├── bfc.py            # KleinBFC — Busemann-horosphere FC
│       ├── mlr.py            # KleinMLR
│       ├── bmlr.py           # KleinBMLR
│       ├── klein_bn.py       # KleinEinsteinBN — closed-form BN
│       └── pv_fc.py          # PVFC (auxiliary baseline)
│
├── bn_microbench.py          # Closed-form vs iterative BN benchmark
├── test_klein_bn.py          # BN unit tests
├── test_layers.py            # Layer unit tests
├── run.sh                    # Reproduce link-prediction experiments
├── download_data.sh          # Fetch HGCN graph datasets
├── TrackC_Genome/            # Genome (TEB) classification pipeline
├── scripts/                  # Experiment launchers
├── MODIFICATIONS.md          # Patches applied to GyroBN / HBNN forks
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```python
import torch
from klein_nn import KleinManifold
from klein_nn.layers import KleinFC, KleinBFC, KleinMLR, KleinBMLR

# 1. Lift a Euclidean tensor onto the Klein ball at curvature K = -1
x = torch.randn(32, 64) * 0.1            # (batch, dim)
x_klein = KleinManifold.projx(x, K=-1.0) # project into the ball

# 2. Apply a Klein FC layer
fc = KleinFC(in_dim=64, out_dim=128, K=-1.0)
h = fc(x_klein)

# 3. Classify with a Klein Busemann MLR head
head = KleinBMLR(in_dim=128, num_classes=10, K=-1.0)
logits = head(h)
```

## Datasets

```bash
bash download_data.sh    # clones HGCN datasets into ./data/
```

This pulls Disease, Airport, PubMed, and Cora (Gromov $\delta$ spanning $0$ to $11$) from the [HGCN repo](https://github.com/HazyResearch/hgcn).

## Reproduce link-prediction experiments

The full training pipeline lives in two third-party frameworks (not bundled here; see `MODIFICATIONS.md` for the file-level patches we apply):

```bash
git clone https://github.com/GitZH-Chen/GyroBN.git
git clone https://github.com/GitZH-Chen/HBNN.git
# Apply the patches listed in MODIFICATIONS.md, then:
bash run.sh                  # all models, all four graphs
bash run.sh KleinFC          # one model
bash run.sh --quick          # 100 epochs, single precision (sanity check)
```

Available `run.sh` targets: `HNN`, `KleinFC`, `KleinBFC`, `LorentzFC` (plus `PVFC` as an auxiliary baseline).

## Hyperparameters

Following [HBNN](https://github.com/GitZH-Chen/HBNN) (Chen et al., CVPR 2026):

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
| Activation | relu (FC); identity/tanh (BFC) |

## Tests

```bash
python test_layers.py        # forward / backward / Euclidean limit (K -> 0)
python test_klein_bn.py      # BN-specific tests
```

## Citation

If you use this code, please cite (BibTeX coming with the camera-ready):

```
@inproceedings{kleinnn2026,
  title  = {Klein Neural Networks},
  author = {anonymous (under review)},
  year   = {2026}
}
```

## References

- Ungar, *Analytic Hyperbolic Geometry and Albert Einstein's Special Theory of Relativity*, 2nd ed., 2022.
- Chen et al., *Gyrogroup Batch Normalization* (GyroBN), ICLR 2025.
- Chen et al., *Hyperbolic Busemann Neural Networks* (HBNN), CVPR 2026.
- Su et al., *Proper Velocity Neural Networks* (PVNN), ICLR 2026.
- Chami et al., *Hyperbolic Graph Convolutional Neural Networks* (HGCN), NeurIPS 2019.

## License

See `LICENSE`. Third-party forks (GyroBN, HBNN) follow their respective licenses.
