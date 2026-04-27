# Modifications to Third-Party Code

This file documents all changes made to forked codebases.

## GyroBN-main (Chen et al., ICLR 2025)

Original: https://github.com/GitZH-Chen/GyroBN

### New files (ours)

| File | Description |
|------|-------------|
| `Geometry/constantcurvature/klein.py` | KleinBall manifold with full Manifold API |
| `Geometry/constantcurvature/pv.py` | PVSpace manifold with full Manifold API |
| `conf/nnet/KleinFC.yaml` | Config for Klein geodesic FC encoder |
| `conf/nnet/KleinBFC.yaml` | Config for Klein Busemann FC encoder |
| `conf/nnet/PVFC.yaml` | Config for PV geodesic FC encoder |
| `conf/nnet/PVBFC.yaml` | Config for PV Busemann FC encoder |
| `conf/nnet/BFC_P.yaml` | Config for Poincaré BFC encoder |
| `conf/nnet/BFC_L.yaml` | Config for Lorentz BFC encoder |
| `conf/nnet/LorentzFC.yaml` | Config for Lorentz ambient FC encoder |

### Modified files

| File | Change |
|------|--------|
| `Geometry/constantcurvature/__init__.py` | Added `from .klein import KleinBall` and `from .pv import PVSpace` |
| `RieNets/hnns/models/encoders.py` | Added encoder classes: `HNN_BFC`, `HNN_KleinFC`, `HNN_PVFC`, `HNN_LorentzFC` |
| `RieNets/hnns/utils/train_utils.py` | `import fcntl` → try/except for Windows compatibility; CUDA fallback to CPU |
| `RieNets/hnns/layers/layers.py` | No functional change (FermiDiracDecoder unchanged) |

### Unchanged files

All training logic (`GyroBNH.py`, `train_kfold.py`, `base_models.py`, `data_utils.py`, `hyp_layers.py`) is unmodified.

---

## HBNN-main (Chen et al., CVPR 2026)

Original: https://github.com/GitZH-Chen/HBNN

### Modified files

| File | Change |
|------|--------|
| `lib/bnn/BFC.py` | Added `metric="klein"` and `metric="pv"` support via `_klein_busemann_linear()` and `_pv_busemann_linear()` |
| `lib/bnn/BMLR.py` | Added `_kbusemann_logits()` and `_pvbusemann_logits()` |
| `lib/bnn/Auxlayers.py` | Added Klein/PV wrapper classes |
| `lib/geoopt/optim/rlinesearch.py` | scipy import compatibility fix |

### Unchanged files

All geometry code (`lib/bnn/Geometry/`), Lorentz layers (`lib/lorentz/`), and geoopt are unmodified.
