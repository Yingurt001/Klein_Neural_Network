"""Decision boundary 2D visualization for paper Figure 1.

Five-panel figure: Euclidean / Klein-MLR / Klein-BMLR / Poincare-MLR / PV-MLR.
Trained on a shared 3-class Gaussian-blob synthetic benchmark in the unit disk.
Output: decision_boundary_2d.{pdf,png} next to this script.

Run from anywhere — the script auto-resolves klein_nn at <code/>.
Self-contained: only requires torch, numpy, matplotlib, and klein_nn package.
"""
import sys
from pathlib import Path

# Locate Phase 2/code/ (parent of scripts/viz/) and add to sys.path
_HERE = Path(__file__).resolve().parent
_CODE_ROOT = _HERE.parent.parent  # Phase 2/code/
sys.path.insert(0, str(_CODE_ROOT))

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from klein_nn.manifold import KleinManifold
from klein_nn.layers.mlr import KleinMLR
from klein_nn.layers.bmlr import KleinBMLR
from klein_nn.layers.pv_fc import _pv_mlr_logits


K_CURV = -1.0


# =====================================================================
# Data: 3-class Gaussian blobs in the unit disk
# =====================================================================
def make_blobs(n_per_class=100, seed=0):
    rng = np.random.RandomState(seed)
    R = 0.65
    centers = np.array([
        [ R,         0.0       ],
        [-R/2,       R * 0.866 ],
        [-R/2,      -R * 0.866 ],
    ])
    pts, labels = [], []
    for i, c in enumerate(centers):
        p = rng.randn(n_per_class, 2) * 0.05 + c
        pts.append(p)
        labels.append(np.full(n_per_class, i))
    X = np.concatenate(pts).astype(np.float32)
    y = np.concatenate(labels).astype(np.int64)
    # Keep all points safely inside the unit disk (|x| < 0.83).
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    X = np.where(norm > 0.83, X / norm * 0.83, X)
    return torch.from_numpy(X), torch.from_numpy(y)


# =====================================================================
# Five MLR variants. All accept (B, 2) Euclidean input and return (B, 3) logits.
# =====================================================================
class EuclideanMLR(nn.Module):
    def __init__(self, in_dim=2, num_classes=3):
        super().__init__()
        self.lin = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.lin(x)


class KleinMLRWrap(nn.Module):
    """KleinMLR — input clipped safely into the Klein ball before forward."""
    def __init__(self, K=K_CURV):
        super().__init__()
        self.K = K
        self.mlr = KleinMLR(in_dim=2, num_classes=3, K=K)

    def forward(self, x):
        return self.mlr(KleinManifold.projx(x, K=self.K))


class KleinBMLRWrap(nn.Module):
    def __init__(self, K=K_CURV):
        super().__init__()
        self.K = K
        self.mlr = KleinBMLR(in_dim=2, num_classes=3, K=K)

    def forward(self, x):
        return self.mlr(KleinManifold.projx(x, K=self.K))


class PoincareMLR(nn.Module):
    """Inline Shimizu/Ganea Poincare MLR (UnidirectionalPoincareMLR formula).

    Input is treated directly as a point on the Poincare ball of curvature
    c = -K > 0. Decision boundaries in this disk are circular arcs orthogonal
    to the boundary.
    """
    def __init__(self, K=K_CURV, in_dim=2, num_classes=3):
        super().__init__()
        self.K = K
        self.c = float(-K)
        std = (in_dim ** -0.5) / math.sqrt(self.c)
        weight = torch.empty(in_dim, num_classes).normal_(mean=0.0, std=std)
        self.weight_g = nn.Parameter(weight.norm(dim=0))
        self.weight_v = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        # Project safely into Poincare ball |x| < 1/sqrt(c).
        rc = math.sqrt(self.c)
        max_r = (1.0 / rc) * (1.0 - 1e-5)
        norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        x_safe = torch.where(norm > max_r, x / norm * max_r, x)

        rcx = rc * x_safe                                      # (B, in_dim)
        cx2 = rcx.pow(2).sum(dim=-1, keepdim=True)             # (B, 1)
        z_unit = self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15)
        drcr = 2.0 * rc * self.bias                            # (C,)
        cosh_drcr = drcr.cosh()
        sinh_drcr = drcr.sinh()
        numer = 2.0 * (rcx @ z_unit) * cosh_drcr - (1.0 + cx2) * sinh_drcr
        denom = (1.0 - cx2).clamp_min(1e-15)
        return 2.0 * self.weight_g / rc * torch.asinh(numer / denom)


class PVMLR(nn.Module):
    """PV MLR using the geodesic-hyperplane signed distance (pv_fc._pv_mlr_logits).

    PV space is unbounded R^n; we extend the visualization grid to show that.
    """
    def __init__(self, K=K_CURV, in_dim=2, num_classes=3):
        super().__init__()
        self.K = K
        self.z = nn.Parameter(torch.empty(num_classes, in_dim))
        self.r = nn.Parameter(torch.zeros(num_classes))
        nn.init.xavier_uniform_(self.z)

    def forward(self, x):
        return _pv_mlr_logits(x, self.z, self.r, self.K, 1e-4)


# =====================================================================
# Train + plot
# =====================================================================
def train(model, X, y, epochs=800, lr=0.08):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(X), y)
        loss.backward()
        opt.step()
    return model


def plot_panel(ax, model, X_np, y_np, title, kind, K=K_CURV):
    lim = 1.5 if kind == "pv" else 1.05
    res = 320
    xx, yy = np.meshgrid(
        np.linspace(-lim, lim, res),
        np.linspace(-lim, lim, res),
    )
    grid = torch.from_numpy(np.stack([xx.ravel(), yy.ravel()], axis=1)).float()
    with torch.no_grad():
        preds = model(grid).argmax(dim=1).numpy().reshape(xx.shape).astype(np.float32)

    if kind in ("klein", "poincare"):
        sqrt_mK = math.sqrt(-K)
        max_r = 1.0 / sqrt_mK
        outside = (xx ** 2 + yy ** 2) > (max_r * 0.985) ** 2
        preds = np.where(outside, np.nan, preds)

    cmap = plt.get_cmap("Set2")
    colors = [cmap(0), cmap(1), cmap(2)]
    ax.contourf(
        xx, yy, preds,
        levels=[-0.5, 0.5, 1.5, 2.5],
        colors=colors,
        alpha=0.45,
    )
    # Black decision-boundary lines (between adjacent class-id integers).
    boundaries = np.where(np.isnan(preds), -10.0, preds)
    ax.contour(
        xx, yy, boundaries,
        levels=[0.5, 1.5],
        colors="black",
        linewidths=1.1,
    )
    ax.scatter(
        X_np[:, 0], X_np[:, 1],
        c=y_np, cmap="Set2",
        edgecolor="black", linewidth=0.5, s=22,
    )
    if kind in ("klein", "poincare"):
        circle = plt.Circle((0.0, 0.0), 1.0 / math.sqrt(-K),
                            fill=False, color="black", linewidth=1.6)
        ax.add_patch(circle)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=15)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = make_blobs(n_per_class=100, seed=0)

    panels = [
        ("Euclidean MLR",          EuclideanMLR(),                  "eucl"),
        ("Klein MLR (geodesic)",   KleinMLRWrap(K=K_CURV),          "klein"),
        ("Klein BMLR (Busemann)",  KleinBMLRWrap(K=K_CURV),         "klein"),
        ("Poincaré MLR",      PoincareMLR(K=K_CURV),           "poincare"),
        ("PV MLR",                 PVMLR(K=K_CURV),                 "pv"),
    ]

    trained = []
    for name, model, kind in panels:
        model = train(model, X, y, epochs=800, lr=0.08)
        with torch.no_grad():
            acc = (model(X).argmax(dim=1) == y).float().mean().item()
        print(f"{name:28s} train acc = {acc:.3f}")
        trained.append((name, model, kind))

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
    for ax, (name, model, kind) in zip(axes, trained):
        plot_panel(ax, model, X.numpy(), y.numpy(), name, kind, K=K_CURV)

    plt.tight_layout()
    out_pdf = _HERE / "decision_boundary_2d.pdf"
    out_png = _HERE / "decision_boundary_2d.png"
    plt.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
