"""
Klein FC — 基于测地超平面的全连接层 (Theorem 5.5)

核心公式:
  1. Klein MLR logit (与 KleinMLR 共享):
     v_k(y) = ‖z_k‖/√(-K) · sinh⁻¹(γ_y · (cosh(√(-K)r_k)·√(-K)<y,z_k>/‖z_k‖ - sinh(√(-K)r_k)))

  2. Klein FC output (归一化到 Klein 球):
     y_k = sinh(√(-K) · v_k(x)) / [√(-K) · √(1 + Σ_j sinh²(√(-K) · v_j(x)))]

     与 BFC 共享归一化结构，区别在于 logit 的计算方式:
     - FC 使用测地超平面 (geodesic hyperplane) → 更几何精确
     - BFC 使用 Busemann 等距球面 (horosphere) → 更计算高效

  3. 可选 gyrobias:
     result = p ⊕_E y, 其中 p = exp_0(tangent)

参数化 (trivialization):
  z_k ∈ R^n  — 超平面法向量方向 + 强度
  r_k ∈ R    — 超平面离原点的距离
  tangent ∈ R^{out} — gyrobias (可选)

欧氏极限 (K → 0⁻):
  y_k → (‖z_k‖ · <x, z_k/‖z_k‖> - ‖z_k‖ · r_k) / √(1 + Σ)
  退化为带有归一化的线性变换
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from klein_nn.manifold import KleinManifold, MIN_NORM, EDGE_EPS

EPS = {torch.float32: 1e-4, torch.float64: 1e-8}


def _eps(x):
    return EPS.get(x.dtype, 1e-8)


def _klein_fc_linear(x: Tensor, z: Tensor, r: Tensor,
                      K: float, eps: float, act) -> Tensor:
    """Klein FC (geodesic hyperplane) 核心计算。

    x: (batch, in_dim)    — Klein 球内的输入点
    z: (out_dim, in_dim)  — 超平面参数 (方向 + 强度)
    r: (out_dim,)         — 距离参数
    K: float              — 曲率 (< 0)
    eps: float            — 数值安全
    act: callable         — 激活函数
    返回: (batch, out_dim) — Klein 球内的输出点
    """
    sqrt_mK = (-K) ** 0.5

    # Step 1: 计算 MLR logit (与 KleinMLR 相同)
    # γ_x: (batch, 1)
    x_sqnorm = x.pow(2).sum(dim=-1, keepdim=True)
    gamma_x = (1.0 + K * x_sqnorm).clamp(min=eps).rsqrt()

    # ‖z_k‖: (out,)
    z_norm = z.pow(2).sum(dim=-1).clamp(min=eps * eps).sqrt()

    # ρ_k = √(-K) · r_k: (out,)
    rho = sqrt_mK * r

    # √(-K) <x, z_k> / ‖z_k‖: (batch, out)
    inner = sqrt_mK * (x @ z.t()) / z_norm.unsqueeze(0)

    # sinh⁻¹ 参数: γ_x · (cosh(ρ_k) · inner - sinh(ρ_k))
    cosh_rho = torch.cosh(rho)
    sinh_rho = torch.sinh(rho)
    argument = gamma_x * (cosh_rho.unsqueeze(0) * inner - sinh_rho.unsqueeze(0))

    # v_k(x) = ‖z_k‖/√(-K) · sinh⁻¹(argument)
    u = (z_norm / sqrt_mK).unsqueeze(0) * torch.asinh(argument)

    # Step 2: 激活函数
    v = act(u)

    # Step 3: 归一化到 Klein 球 (与 BFC 共享 Theorem 5.5/5.6 结构)
    # y_k = sinh(√(-K)·v_k) / [√(-K) · √(1 + Σ sinh²(√(-K)·v_j))]
    sinh_val = torch.sinh(sqrt_mK * v)
    norm_factor = (1.0 + sinh_val.pow(2).sum(dim=-1, keepdim=True)).sqrt()
    y = sinh_val / (sqrt_mK * norm_factor)

    y = KleinManifold.projx(y, K)
    return y


class KleinFC(nn.Module):
    """Klein 全连接层（测地超平面版）。

    输入: Klein 球内的点 x ∈ K^n
    输出: Klein 球内的点 y ∈ K^m (m = out_dim)

    使用 MLR-style geodesic hyperplane signed distance 作为 logit，
    通过 sinh 归一化映射回 Klein 球。

    与 KleinBFC 对比:
    - KleinFC: 更几何精确 (geodesic hyperplane), FLOPs 略高
    - KleinBFC: 更计算高效 (horosphere Busemann), FLOPs 低
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

        # 参数: z_k 编码超平面方向+强度, r_k 编码距离
        self.z = nn.Parameter(torch.empty(out_dim, in_dim))
        self.r = nn.Parameter(torch.zeros(out_dim))

        if self.use_gyrobias:
            self.tangent = nn.Parameter(torch.zeros(out_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.z)
        nn.init.zeros_(self.r)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (batch, in_dim) — Klein 球内的输入点
        返回: (batch, out_dim) — Klein 球内的输出点
        """
        eps = _eps(x)

        z = F.dropout(self.z, self.dropout, training=self.training)

        y = _klein_fc_linear(x, z, self.r, self.K, eps, self.act)

        if self.use_gyrobias:
            p = KleinManifold.exp0(self.tangent, self.K)
            y = KleinManifold.einstein_add(p, y, self.K)

        return y
