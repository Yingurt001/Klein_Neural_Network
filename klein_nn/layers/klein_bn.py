"""Klein Batch Normalization variants — for the BN ablation experiment.

Provides four BN modules that operate on points in the Klein ball:
  - KleinEinsteinBN  (ours, closed-form Einstein midpoint)
  - KleinTangentBN   (log_0 → Euclidean BN → exp_0, Lou 2020 style)
  - KleinFrechetBNViaPoincare (iterative Fréchet mean on Poincaré, via
                               isometric embedding; baseline for speed vs ours)
  - KleinActWrapper  (companion: applies Euclidean activation through log/exp)

All modules assume input shape (batch, dim) or (N, dim) and return the same.
Dim=last axis is the manifold dimension d; samples are taken along dim 0.

Curvature convention: K < 0 (default -1.0). The Klein ball radius is 1/√(-K).

Reference (ours):
  §5 Theorem 5.x of the paper — closed-form Einstein-midpoint BN with
  Fréchet variance scaling and Einstein-translation biasing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from klein_nn.manifold import KleinManifold, MIN_NORM


# ---------------------------------------------------------------------------
# 1. Klein Einstein-midpoint BN (ours)
# ---------------------------------------------------------------------------

class KleinEinsteinBN(nn.Module):
    """Einstein-midpoint Batch Normalization on the Klein ball.

    Forward (training):
      1. μ  = EinsteinMidpoint(x)                       closed-form, O(Bd)
      2. σ² = mean_i d_K(x_i, μ)²                       Fréchet variance
      3. x̂  = gyr(-μ, x)                                Einstein translation
      4. y  = (γ / √(σ²+ε)) ⊗_E x̂                       Einstein scalar
      5. z  = exp_0(β) ⊕_E y                            Einstein bias

    Forward (eval): replace μ, σ² with running statistics.

    Parameters
    ----------
    dim : int
        Manifold dimension (last axis of input).
    K : float, default -1.0
        Curvature (must be negative).
    momentum : float, default 0.1
        EMA coefficient for running stats.
    eps : float, default 1e-6
        Variance stabilizer.
    """

    def __init__(self, dim: int, K: float = -1.0,
                 momentum: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.K = float(K)
        self.momentum = float(momentum)
        self.eps = float(eps)

        # learnable: β ∈ T_0 K^d (tangent at origin), γ = exp(log_gamma) > 0
        self.beta = nn.Parameter(torch.zeros(self.dim))
        self.log_gamma = nn.Parameter(torch.zeros(1))

        # running stats
        self.register_buffer('running_mean', torch.zeros(self.dim))
        self.register_buffer('running_var', torch.ones(1))
        self.register_buffer('initialized', torch.tensor(False))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, K={self.K}, momentum={self.momentum}, eps={self.eps}"

    def _batch_variance(self, x: Tensor, mu: Tensor) -> Tensor:
        """Fréchet variance on Klein: E[d_K(x, μ)²]."""
        mu_b = mu.unsqueeze(0).expand_as(x)
        d = KleinManifold.dist(x, mu_b, K=self.K, keepdim=False)
        return (d * d).mean()

    def _update_running_mean(self, batch_mu: Tensor) -> None:
        """Geodesic EMA via two-point Einstein midpoint with weights."""
        if not bool(self.initialized):
            self.running_mean.copy_(batch_mu.detach())
            self.initialized.fill_(True)
        else:
            stacked = torch.stack([self.running_mean, batch_mu.detach()], dim=0)
            weights = torch.tensor(
                [1.0 - self.momentum, self.momentum],
                dtype=batch_mu.dtype, device=batch_mu.device,
            )
            new_mean = KleinManifold.einstein_midpoint(
                stacked, K=self.K, weights=weights)
            self.running_mean.copy_(new_mean)

    def forward(self, x: Tensor) -> Tensor:
        # flatten any leading dims (e.g. (B, N, d) → (B*N, d))
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, orig_shape[-1])

        if self.training:
            mu = KleinManifold.einstein_midpoint(x, K=self.K)
            var = self._batch_variance(x, mu)
            with torch.no_grad():
                self._update_running_mean(mu)
                if not bool(self.initialized) or self.running_var is None:
                    self.running_var.copy_(var.detach().view(1))
                else:
                    self.running_var.mul_(1.0 - self.momentum).add_(
                        self.momentum * var.detach().view(1))
        else:
            mu = self.running_mean
            var = self.running_var

        # Step 3 — Einstein centering by -μ (Klein gyro-negation is Euclidean -x)
        neg_mu = (-mu).unsqueeze(0).expand_as(x)
        x_c = KleinManifold.einstein_add(neg_mu, x, K=self.K)

        # Step 4 — Einstein scalar with learnable γ
        factor = torch.exp(self.log_gamma) / (var + self.eps).sqrt()
        x_s = KleinManifold.einstein_scalar(factor, x_c, K=self.K)

        # Step 5 — Einstein bias: add exp_0(β)
        beta_M = KleinManifold.exp0(self.beta, K=self.K)
        x_out = KleinManifold.einstein_add(
            beta_M.unsqueeze(0).expand_as(x_s), x_s, K=self.K)

        if x.dim() > 2 or len(orig_shape) > 2:
            x_out = x_out.reshape(orig_shape)
        return x_out


# ---------------------------------------------------------------------------
# 2. Tangent-space BN baseline  (Lou 2020 RBN-H style on Klein ball)
# ---------------------------------------------------------------------------

class KleinTangentBN(nn.Module):
    """Stable tangent-space normalization for Klein-ball points.

    Rationale (see §6.5 discussion):
      BatchNorm on ``log_0(x)`` is unstable on graph LP: a single near-boundary
      sample in the batch blows up the batch std, BN scales other samples with
      a huge γ, and ``exp_0`` ends right at the ball boundary — the next
      layer's Lorentz factor overflows and the decoder receives NaNs.
      We fix this with three orthogonal changes:

      1. **LayerNorm instead of BatchNorm** — per-sample normalization is
         robust to batch composition and matches modern Transformer practice.
      2. **Small learnable scale** — initialize γ at 0.3 so BN cannot
         aggressively re-scale activations into the boundary.
      3. **Soft-cap** — ``u ← c·tanh(u/c)`` enforces ``||exp_0(u)||/r ≤ tanh(c)``
         even if (1) and (2) fail at a pathological seed. c=0.5 gives ≤ 0.46 r.

    This corresponds closest to what Lou (2020) proposed as RBN-H in spirit:
    a per-sample tangent normalization followed by bounded re-projection.
    Pure BatchNorm in the tangent space (our earlier V1 attempt) is
    structurally unstable — we document that failure mode in the appendix.
    """

    def __init__(self, dim: int, K: float = -1.0,
                 momentum: float = 0.1, eps: float = 1e-6,
                 cap: float = 0.5, gamma_init: float = 0.3,
                 use_layer_norm: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.K = float(K)
        self.cap = float(cap)
        self.use_ln = bool(use_layer_norm)
        if self.use_ln:
            self.norm = nn.LayerNorm(self.dim, eps=eps, elementwise_affine=True)
            with torch.no_grad():
                self.norm.weight.fill_(gamma_init)
        else:  # legacy pure-BN path (kept for ablation)
            self.norm = nn.BatchNorm1d(self.dim, momentum=momentum, eps=eps)
            with torch.no_grad():
                self.norm.weight.fill_(gamma_init)

    def extra_repr(self) -> str:
        kind = "LayerNorm" if self.use_ln else "BatchNorm1d"
        return f"dim={self.dim}, K={self.K}, cap={self.cap}, norm={kind}"

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, orig_shape[-1])
        u = KleinManifold.log0(x, K=self.K)
        u = self.norm(u)
        u = self.cap * torch.tanh(u / self.cap)
        out = KleinManifold.exp0(u, K=self.K)
        if len(orig_shape) > 2:
            out = out.reshape(orig_shape)
        return out


# ---------------------------------------------------------------------------
# 3. Fréchet BN via Poincaré isometry (iterative baseline)
# ---------------------------------------------------------------------------

class KleinFrechetBNViaPoincare(nn.Module):
    """Iterative Fréchet-mean BN on Klein, realized via Poincaré isometry.

    Steps:
      1. Klein x        → Poincaré x_P                  (isometric)
      2. GyroBNH(x_P)   → y_P                           (Möbius + Fréchet)
      3. Poincaré y_P   → Klein                         (isometric)

    This is the iterative baseline V3 in the BN ablation. Matches the
    "Chen 2025 GyroBN-CCS-Klein" idea — closed-form centering via Möbius,
    but mean computation relies on iterative Fréchet Newton updates.
    """

    def __init__(self, dim: int, K: float = -1.0,
                 momentum: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.K = float(K)

        # delegate to the framework's GyroBNH on Poincaré
        import sys as _sys
        import os as _os
        _here = _os.path.dirname(_os.path.abspath(__file__))
        _gyro_root = _os.path.join(_here, '..', '..', 'GyroBN-main')
        if _os.path.isdir(_gyro_root) and _gyro_root not in _sys.path:
            _sys.path.insert(0, _gyro_root)
        from RieNets.hnns.layers.GyroBNH import GyroBNH
        from frechetmean.manifolds.ball import Poincare

        self.inner = GyroBNH(self.dim, manifold=Poincare(K=K),
                             momentum=momentum, eps=eps)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, K={self.K}"

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, orig_shape[-1])
        x_P = KleinManifold.klein_to_poincare(x, K=self.K)
        y_P = self.inner(x_P)
        out = KleinManifold.poincare_to_klein(y_P, K=self.K)
        out = KleinManifold.projx(out, K=self.K)
        if len(orig_shape) > 2:
            out = out.reshape(orig_shape)
        return out


# ---------------------------------------------------------------------------
# 4. Klein-compatible Lorentz-style Centroid BN (for cross-space speed compare)
# ---------------------------------------------------------------------------

class KleinLorentzStyleBN(nn.Module):
    """Lorentz-centroid BN (Bdeir 2024 style) wrapped for Klein-ball inputs.

    Flow:  Klein point → Lorentz (isometry, closed-form) → centroid BN
                      → Klein (isometry, closed-form)

    This is NOT our main contribution — it's a microbench-only baseline so
    that the closed-form speed comparison (V4 Lorentz vs V5 Einstein) is
    fair: both receive Klein-ball inputs, differ only in centroid algorithm.

    Lorentz centroid (Bdeir 2024):
        μ_L = √k · avg(x_L) / √(|<avg, avg>_L|)
    where <·,·>_L is the Lorentz inner product and avg is Euclidean mean.
    """

    def __init__(self, dim: int, K: float = -1.0,
                 momentum: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.K = float(K)
        self.k = -1.0 / self.K  # Lorentz radius² = 1/c = -1/K
        self.eps = eps
        self.momentum = momentum
        self.beta = nn.Parameter(torch.zeros(self.dim))
        self.log_gamma = nn.Parameter(torch.zeros(1))
        self.register_buffer('running_mean', torch.zeros(self.dim))
        self.register_buffer('running_var', torch.ones(1))
        self.register_buffer('initialized', torch.tensor(False))

    def _lorentz_inner(self, x: Tensor, y: Tensor, keepdim: bool = True) -> Tensor:
        """<x, y>_L = -x_0·y_0 + Σ x_i·y_i  (time-first convention)"""
        time = -x[..., :1] * y[..., :1]
        space = (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
        result = time + space
        return result if keepdim else result.squeeze(-1)

    def _centroid(self, x_L: Tensor) -> Tensor:
        """Closed-form Lorentz centroid on (N, d+1) points."""
        avg = x_L.mean(dim=0)  # (d+1,)
        inner = (-self._lorentz_inner(avg, avg, keepdim=False)).abs().clamp_min(self.eps)
        return (self.k ** 0.5) * avg / inner.sqrt()

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, orig_shape[-1])
        # Klein → Lorentz (d → d+1, prepend γ time component)
        x_L = KleinManifold.klein_to_lorentz(x, K=self.K)  # (N, d+1)

        if self.training:
            mu_L = self._centroid(x_L)
            # Fréchet-style variance in Lorentz (sum of space components sqnorm)
            # use simpler "tangent-after-centering" variance for speed
            var = (x_L[..., 1:] - mu_L[1:]).pow(2).sum(dim=-1).mean()
            with torch.no_grad():
                if not bool(self.initialized):
                    self.running_mean.copy_(mu_L[1:].detach())  # store spatial part
                    self.running_var.copy_(var.detach().view(1))
                    self.initialized.fill_(True)
                else:
                    self.running_mean.mul_(1 - self.momentum).add_(
                        self.momentum * mu_L[1:].detach())
                    self.running_var.mul_(1 - self.momentum).add_(
                        self.momentum * var.detach().view(1))
            mu_space = mu_L[1:]
        else:
            mu_space = self.running_mean
            var = self.running_var

        # rescale space part and re-compute time to stay on hyperboloid
        space = x_L[..., 1:] - mu_space.unsqueeze(0)
        factor = torch.exp(self.log_gamma) / (var + self.eps).sqrt()
        space = factor * space + self.beta.unsqueeze(0)
        # add bias β, recompute time from space (k = 1/c)
        time_sq = space.pow(2).sum(dim=-1, keepdim=True) + self.k
        time = time_sq.clamp_min(self.eps).sqrt()
        x_L_out = torch.cat([time, space], dim=-1)

        # back to Klein
        out = KleinManifold.lorentz_to_klein(x_L_out)
        out = KleinManifold.projx(out, K=self.K)
        if len(orig_shape) > 2:
            out = out.reshape(orig_shape)
        return out


# ---------------------------------------------------------------------------
# 5. Klein activation wrapper (applied externally when BN replaces FC's act)
# ---------------------------------------------------------------------------

class KleinActWrapper(nn.Module):
    """log_0 → pointwise Euclidean activation → exp_0, projected back."""

    def __init__(self, act_name: str, K: float = -1.0):
        super().__init__()
        self.K = float(K)
        self.act_name = act_name
        if not hasattr(F, act_name):
            raise ValueError(f"Unsupported activation '{act_name}'.")
        self._act = getattr(F, act_name)

    def extra_repr(self) -> str:
        return f"act={self.act_name}, K={self.K}"

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, orig_shape[-1])
        u = KleinManifold.log0(x, K=self.K)
        u = self._act(u)
        out = KleinManifold.exp0(u, K=self.K)
        if len(orig_shape) > 2:
            out = out.reshape(orig_shape)
        return out


# ---------------------------------------------------------------------------
# Factory (used by encoders.py)
# ---------------------------------------------------------------------------

def build_klein_bn(bn_type: str, dim: int, K: float = -1.0,
                   momentum: float = 0.1, eps: float = 1e-6) -> nn.Module:
    """Return a BN module for the given bn_type tag."""
    if bn_type == 'KleinEinsteinBN':
        return KleinEinsteinBN(dim, K=K, momentum=momentum, eps=eps)
    if bn_type == 'KleinTgBN':
        return KleinTangentBN(dim, K=K, momentum=momentum, eps=eps)
    if bn_type == 'KleinFrechetBN':
        return KleinFrechetBNViaPoincare(dim, K=K, momentum=momentum, eps=eps)
    raise ValueError(f"Unknown bn_type: {bn_type}")
