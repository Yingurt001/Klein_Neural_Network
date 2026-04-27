"""
Klein Busemann MLR — 基于 Busemann 函数的分类头

核心公式 (Klein_FC_Conv_sections.tex, Eq. klein-bmlr-expanded):

  u_k(x) = α_k / √(-K) · [½ log(1 + K‖x‖²) - log(1 - <x, v_k>)] + b_k

参数化 (对齐 HBNN BMLR, Chen et al. 2026):
  weight_v ∈ ℝ^{C×n}  — 方向向量 (forward 时 L2 归一化)
  weight_g ∈ ℝ^C      — log-scale 缩放 (forward 时 exp 恢复 α > 0)
  bias ∈ ℝ^C           — 偏置

初始化:
  weight_v ~ Normal(0, 1/√n)
  weight_g = log(‖weight_v‖_row)
  bias = 0

欧氏极限 (K → 0⁻):
  u_k(x) → α_k <v_k, x> + b_k
"""

import torch
import torch.nn as nn
from torch import Tensor

# HBNN convention: EPS = 1e-4 for float32
EPS = {torch.float32: 1e-4, torch.float64: 1e-8}


def _eps(x):
    return EPS.get(x.dtype, 1e-8)


@torch.jit.script
def _klein_bmlr_logits(x: Tensor, v: Tensor, alpha: Tensor, b: Tensor,
                        K: float, eps: float) -> Tensor:
    """Klein Busemann MLR logit 的 JIT 加速核心计算。

    x: (batch, n)   — Klein 球内的输入点
    v: (C, n)       — 单位方向向量 (已归一化)
    alpha: (C,)     — 正尺度参数
    b: (C,)         — 偏置参数
    K: float        — 曲率 (< 0)
    返回: (batch, C) — 每个类的 logit
    """
    sqrt_mK = (-K) ** 0.5

    # ½ log(1 + K‖x‖²): (batch, 1)  — 所有类共享
    x_sqnorm = x.pow(2).sum(dim=-1, keepdim=True)
    shared_term = 0.5 * torch.log((1.0 + K * x_sqnorm).clamp(min=eps))

    # √(-K) · <x, v_k>: (batch, C)
    inner = sqrt_mK * (x @ v.t())

    # -log(1 - √(-K)<x, v_k>): (batch, C)
    class_term = -torch.log((1.0 - inner).clamp(min=eps))

    # u_k(x) = α_k/√(-K) · (shared_term + class_term) + b_k
    logits = (alpha / sqrt_mK).unsqueeze(0) * (shared_term + class_term) + b.unsqueeze(0)

    return logits


class KleinBMLR(nn.Module):
    """Klein Busemann 多项逻辑回归（分类头）。

    使用 horosphere (等距球面) 而非 geodesic hyperplane 作为决策边界。
    参数化对齐 HBNN (Chen et al. 2026) 的 g/v 分解 + log-scale。
    """

    def __init__(self, in_dim: int, num_classes: int, K: float = -1.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.K = K

        # 对齐 HBNN BMLR 初始化
        weight = torch.empty(num_classes, in_dim)
        nn.init.normal_(weight, mean=0.0, std=in_dim ** -0.5)  # Normal(0, 1/√n)
        self.weight_v = nn.Parameter(weight)
        self.weight_g = nn.Parameter(
            weight.norm(dim=-1).clamp_min(1e-4).log()  # log(‖v‖) ≈ 0
        )
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (batch, in_dim) — Klein 球内的点
        返回: (batch, num_classes) — logits
        """
        eps = _eps(x)
        v_unit = self.weight_v / self.weight_v.norm(dim=-1, keepdim=True).clamp_min(eps)
        scaling = self.weight_g.exp()  # α = exp(weight_g) > 0
        return _klein_bmlr_logits(x, v_unit, scaling, self.bias, self.K, eps)
