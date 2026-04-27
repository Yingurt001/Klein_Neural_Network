"""
Klein MLR — 多项逻辑回归分类头 (PDF §5)

核心公式 (PDF Eq. 82, 一般 K < 0)：

  v_k(y) = ||z_k|| / √(-K) · sinh⁻¹(
      γ^K_y · ( cosh(√(-K) r_k) · √(-K)<y, z_k>/||z_k|| - sinh(√(-K) r_k) )
  )

参数化 (PDF §5.4 trivialization)：
  z_k ∈ R^n  — 编码超平面方向 (ẑ_k = z_k/||z_k||) 和法向量强度 (||z_k||)
  r_k ∈ R    — 控制超平面离原点的距离

两者均为无约束欧氏参数。
"""

import torch
import torch.nn as nn
from torch import Tensor
from klein_nn.manifold import KleinManifold, MIN_NORM


@torch.jit.script
def _klein_mlr_logits(y: Tensor, z: Tensor, r: Tensor,
                       K: float, eps: float) -> Tensor:
    """Klein MLR logit 的 JIT 加速核心计算。

    y: (batch, n)   — Klein 球内的输入点
    z: (C, n)       — 超平面参数
    r: (C,)         — 距离参数
    K: float        — 曲率
    返回: (batch, C) — 每个类的 logit
    """
    sqrt_mK = (-K) ** 0.5

    # γ_y: (batch, 1)
    y_sqnorm = y.pow(2).sum(dim=-1, keepdim=True)
    gamma_y = (1.0 + K * y_sqnorm).clamp(min=eps).rsqrt()

    # ||z_k||: (C,)
    z_norm = z.pow(2).sum(dim=-1).clamp(min=eps * eps).sqrt()

    # ρ_k = √(-K) · r_k: (C,)
    rho = sqrt_mK * r

    # √(-K) <y, z_k> / ||z_k||: (batch, C)
    inner = sqrt_mK * (y @ z.t()) / z_norm.unsqueeze(0)

    # sinh⁻¹ 的参数: γ_y · (cosh(ρ_k) · inner - sinh(ρ_k))
    # cosh(ρ_k), sinh(ρ_k): (C,)
    cosh_rho = torch.cosh(rho)
    sinh_rho = torch.sinh(rho)

    # (batch, C)
    argument = gamma_y * (cosh_rho.unsqueeze(0) * inner - sinh_rho.unsqueeze(0))

    # v_k(y) = ||z_k||/√(-K) · sinh⁻¹(argument)
    logits = (z_norm / sqrt_mK).unsqueeze(0) * torch.asinh(argument)

    return logits


class KleinMLR(nn.Module):
    """Klein 多项逻辑回归（分类头）。

    将 Klein 球内的特征映射为 C 个类的 logits。
    """

    def __init__(self, in_dim: int, num_classes: int, K: float = -1.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.K = K

        # z_k ∈ R^n: 超平面方向 + 强度
        self.z = nn.Parameter(torch.empty(num_classes, in_dim))
        # r_k ∈ R: 超平面离原点的距离
        self.r = nn.Parameter(torch.zeros(num_classes))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.z)
        nn.init.zeros_(self.r)

    def forward(self, y: Tensor) -> Tensor:
        """
        y: (batch, in_dim) — Klein 球内的点
        返回: (batch, num_classes) — logits
        """
        return _klein_mlr_logits(y, self.z, self.r, self.K, MIN_NORM)
