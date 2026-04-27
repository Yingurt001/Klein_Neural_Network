"""
PV Busemann FC — 基于 PV Busemann 函数的全连接层

PV Busemann 函数:
  B^v_PV(x) = (1/√(-K)) · log(√(1-K‖x‖²) - √(-K)·<x,v>)

PV BMLR logit:
  u_k(x) = (-α_k/√(-K)) · log(√(1-K‖x‖²) - √(-K)·<x,v_k>) + b_k

PV BFC output (无界空间, 无需归一化):
  y_k = (1/√(-K)) · sinh(√(-K) · act(u_k(x)))

与 Klein BFC 的关键区别:
  - PV 空间 PV^n_K = ℝ^n, 无界 → 不需要 √(1+Σsinh²) 归一化
  - Klein 空间 K^n_K 是开球 → 需要归一化保证 ‖y‖ < 1/√(-K)

参数化 (对齐 HBNN BFC):
  weight_v ∈ ℝ^{out×in}  — 方向向量 (forward 时 L2 归一化)
  weight_g ∈ ℝ^{out}     — log-scale 缩放
  bias ∈ ℝ^{out}
  tangent ∈ ℝ^{out}       — gyrobias 切向量 (可选)

欧氏极限 (K → 0⁻):
  y_k → u_k(x) → α_k <v_k, x> + b_k
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# HBNN convention
EPS = {torch.float32: 1e-4, torch.float64: 1e-8}
TINY = 1e-15


def _eps(x):
    return EPS.get(x.dtype, 1e-8)


@torch.jit.script
def _pv_bmlr_logits(v: Tensor, x: Tensor, K: float, scaling: Tensor,
                     bias: Tensor, eps: float) -> Tensor:
    """PV Busemann MLR logits (batch friendly).

    PV Busemann function:
      B^v(x) = (1/√(-K)) · log(√(1-K‖x‖²) - √(-K)<x,v>)

    Logit: u_k(x) = -α_k · B^{v_k}(x) + b_k

    v: (d, C)   — unit direction vectors (transposed)
    x: (bs, d)  — PV space points
    K: float    — curvature (< 0)
    scaling: (C,) — α = exp(weight_g)
    bias: (C,)    — bias
    """
    sqrt_c = (-K) ** 0.5

    # β_x^{-1} = √(1-K‖x‖²): (bs, 1)
    # Note: K<0, so 1-K‖x‖² = 1+|K|‖x‖² ≥ 1
    x_sqnorm = x.pow(2).sum(dim=-1, keepdim=True)
    beta_inv = (1.0 - K * x_sqnorm).clamp(min=eps).sqrt()

    # √(-K)·<x, v_k>: (bs, C)
    inner = sqrt_c * (x @ v)

    # log argument: β_x^{-1} - √(-K)<x,v>  (always > 0 by Cauchy-Schwarz)
    argument = (beta_inv - inner).clamp(min=eps)

    # B^v(x) = (1/√(-K)) · log(argument)
    busemann = torch.log(argument) / sqrt_c

    # logit = -α·B + b
    logits = -busemann * scaling.unsqueeze(0) + bias.unsqueeze(0)
    return logits


def _pv_busemann_linear(x: Tensor, v_unit: Tensor, scaling: Tensor,
                         bias: Tensor, K: float, eps: float,
                         act) -> Tensor:
    """PV BFC: Busemann logit → activation → sinh → PV space.

    PV 是无界空间, 输出不需要归一化。
    y_k = (1/√(-K)) · sinh(√(-K) · act(u_k))
    """
    sqrt_c = (-K) ** 0.5
    tmp = _pv_bmlr_logits(v_unit, x, K, scaling, bias, eps)
    v = act(tmp)
    # sinh mapping (decoupled, no normalization)
    arg = sqrt_c * v
    arg = arg.clamp(-15.0, 15.0)  # numerical stability
    y = torch.sinh(arg) / sqrt_c
    return y


# ---- PV manifold helpers for gyrobias ----

@torch.jit.script
def _pv_beta_factor(x: Tensor, K: float, eps: float) -> Tensor:
    """β_x = 1/√(1-K‖x‖²)"""
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    return (1.0 - K * x2).clamp(min=eps).rsqrt()


@torch.jit.script
def _pv_exp0(v: Tensor, K: float, eps: float) -> Tensor:
    """PV exp map at origin: exp_0(v) = sinh(‖v‖/s) / (‖v‖/s) · v
    where s = 1/√(-K).
    """
    sqrt_c = (-K) ** 0.5
    s = 1.0 / sqrt_c
    r = torch.norm(v, p=2, dim=-1, keepdim=True).clamp(min=eps)
    coef = torch.sinh(r / s) / (r / s).clamp(min=eps)
    return coef * v


@torch.jit.script
def _pv_gyro_add(x: Tensor, y: Tensor, K: float, eps: float) -> Tensor:
    """PV gyro-addition: x ⊕ y.

    x ⊕ y = x + y + [β_x/(1+β_x) · (-K)<x,y> + 1/β_y - 1] · x
    """
    neg_k = -K
    b_x = _pv_beta_factor(x, K, eps)
    b_y = _pv_beta_factor(y, K, eps)
    xy = (x * y).sum(dim=-1, keepdim=True)
    coef = (b_x / (1.0 + b_x)) * neg_k * xy + (1.0 / b_y) - 1.0
    return x + y + coef * x


class PVBFC(nn.Module):
    """PV Busemann 全连接层。

    输入: PV 空间的点 x ∈ ℝ^n
    输出: PV 空间的点 y ∈ ℝ^m

    使用 PV Busemann 函数计算 logit, 通过 sinh 映射到 PV 空间。
    PV 空间无界, 不需要归一化。
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
        self.is_bias = bias
        self.dropout = dropout
        self.use_gyrobias = gyrobias

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

        self._init_parameters()

    def _init_parameters(self):
        gain = 1.0
        weight = torch.empty(self.out_dim, self.in_dim).normal_(
            mean=0, std=(2 * self.in_dim * self.out_dim) ** -0.5 * gain)
        self.weight_v = nn.Parameter(weight)
        self.weight_g = nn.Parameter(
            weight.norm(dim=-1).clamp_min(EPS[weight.dtype]).log())
        self.bias = nn.Parameter(
            torch.zeros(self.out_dim), requires_grad=self.is_bias)

        if self.use_gyrobias:
            self.tangent = nn.Parameter(torch.zeros(self.out_dim))

    def forward(self, x: Tensor) -> Tensor:
        eps = _eps(x)
        drop_weight = F.dropout(self.weight_v, self.dropout, training=self.training)
        v_unit = drop_weight / drop_weight.norm(dim=-1, keepdim=True).clamp_min(eps)
        scaling = self.weight_g.exp()

        y = _pv_busemann_linear(
            x, v_unit.t(), scaling, self.bias,
            self.K, eps, self.act,
        )

        if self.use_gyrobias:
            p = _pv_exp0(self.tangent.unsqueeze(0), self.K, TINY)
            y = _pv_gyro_add(y, p.squeeze(0), self.K, TINY)

        return y
