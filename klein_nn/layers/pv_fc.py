"""
PV FC — 基于测地超平面的全连接层

PV MLR logit (Theorem 5.2, Su et al. ICLR 2026):
  v_k(x) = (‖z_k‖/√(-K)) · arcsinh(
      √(-K)/‖z_k‖ · [cosh(√(-K)r_k)·<x,z_k> - sinh(√(-K)r_k)·√(1-K‖x‖²)·‖z_k‖/√(-K)]
  )

PV FC output (无界空间, 无需归一化):
  y_k = (1/√(-K)) · sinh(√(-K) · act(v_k(x)))

参数化:
  z_k ∈ R^n  — 超平面方向 + 强度
  r_k ∈ R    — 超平面离原点的距离
  tangent ∈ R^{out} — gyrobias (可选)

欧氏极限 (K → 0⁻):
  y_k → <x, z_k> - r_k · ‖z_k‖ (线性变换)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from klein_nn.layers.pv_bfc import _pv_exp0, _pv_gyro_add, TINY

EPS = {torch.float32: 1e-4, torch.float64: 1e-8}


def _eps(x):
    return EPS.get(x.dtype, 1e-8)


@torch.jit.script
def _pv_mlr_logits(x: Tensor, z: Tensor, r: Tensor,
                    K: float, eps: float) -> Tensor:
    """PV MLR logit 的 JIT 加速核心计算。

    x: (batch, n)   — PV 空间的输入点
    z: (C, n)       — 超平面参数
    r: (C,)         — 距离参数
    K: float        — 曲率 (< 0)
    返回: (batch, C) — 每个类的 logit
    """
    sqrt_mK = (-K) ** 0.5

    # ‖z_k‖: (C,)
    z_norm = z.pow(2).sum(dim=-1).clamp(min=eps * eps).sqrt()

    # ρ_k = √(-K) · r_k: (C,) — clamp for numerical stability
    rho = (sqrt_mK * r).clamp(-15.0, 15.0)
    cosh_rho = torch.cosh(rho)
    sinh_rho = torch.sinh(rho)

    # Term A: cosh(ρ) · <x, z_k>: (batch, C)
    xz = x @ z.t()
    term_A = cosh_rho.unsqueeze(0) * xz

    # Term B: sinh(ρ) · β_x^{-1} · ‖z_k‖/√(-K)
    # β_x^{-1} = √(1 - K‖x‖²): (batch, 1)
    x_sqnorm = x.pow(2).sum(dim=-1, keepdim=True)
    beta_inv = (1.0 - K * x_sqnorm).clamp(min=eps).sqrt()

    term_B = (sinh_rho / sqrt_mK).unsqueeze(0) * z_norm.unsqueeze(0) * beta_inv

    # Factor C = √(-K)/‖z_k‖: (1, C)
    factor_C = sqrt_mK / z_norm.unsqueeze(0)

    # argument = factor_C · (term_A - term_B)
    argument = factor_C * (term_A - term_B)
    argument = argument.clamp(-1e6, 1e6)

    # v_k(x) = (‖z_k‖/√(-K)) · arcsinh(argument)
    scale = (z_norm / sqrt_mK).unsqueeze(0)
    logits = scale * torch.asinh(argument)

    return logits


class PVFC(nn.Module):
    """PV 全连接层（测地超平面版）。

    输入: PV 空间的点 x ∈ ℝ^n
    输出: PV 空间的点 y ∈ ℝ^m

    使用 MLR-style geodesic hyperplane signed distance → sinh 映射。
    PV 空间无界, 不需要归一化。

    与 PVBFC 对比:
    - PVFC: geodesic hyperplane, FLOPs 略高 (arcsinh + cosh + sinh)
    - PVBFC: Busemann horosphere, FLOPs 低 (log + 内积)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        K: float = -1.0,
        bias: bool = True,
        dropout: float = 0.0,
        gyrobias: bool = True,
        act=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = K
        self.dropout = dropout
        self.use_gyrobias = gyrobias

        sqrt_mK = (-K) ** 0.5
        self.sqrt_mK = sqrt_mK

        if act is None:
            self.act = lambda x: x
        else:
            act_name = act.lower() if isinstance(act, str) else act
            if isinstance(act_name, str) and hasattr(F, act_name):
                self.act = getattr(F, act_name)
            elif callable(act):
                self.act = act
            else:
                raise ValueError(f"Unsupported activation '{act}'.")

        self.z = nn.Parameter(torch.empty(out_dim, in_dim))
        self.r = nn.Parameter(torch.empty(out_dim))

        if self.use_gyrobias:
            self.tangent = nn.Parameter(torch.zeros(out_dim))

        self._init_weights()

    def _init_weights(self):
        std = (2 * self.in_dim * self.out_dim) ** -0.5
        nn.init.normal_(self.z, mean=0.0, std=std)
        nn.init.uniform_(self.r, a=-1e-2, b=1e-2)

    def forward(self, x: Tensor) -> Tensor:
        eps = _eps(x)

        z = F.dropout(self.z, self.dropout, training=self.training)
        v = _pv_mlr_logits(x, z, self.r, self.K, eps)

        # Activation
        v = self.act(v)

        # sinh mapping (decoupled, no normalization)
        arg = (self.sqrt_mK * v).clamp(-15.0, 15.0)
        y = torch.sinh(arg) / self.sqrt_mK

        # Gyrobias
        if self.use_gyrobias:
            p = _pv_exp0(self.tangent.unsqueeze(0), self.K, TINY)
            y = _pv_gyro_add(y, p.squeeze(0), self.K, TINY)

        return y
