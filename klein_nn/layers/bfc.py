"""
Klein Busemann FC — 基于 Busemann 函数的全连接层 (Theorem 5.6)

核心公式:
  1. Busemann logit:
     u_k(x) = α_k / √(-K) · [½ log(1 + K‖x‖²) - log(1 - √(-K)<x, v_k>)] + b_k
     (与 KleinBMLR 共享)

  2. Klein BFC output (归一化到 Klein 球):
     y_k = sinh(√(-K) · u_k) / [√(-K) · √(1 + Σ_j sinh²(√(-K) · u_j))]

     这保证了 ‖y‖² < -1/K，即输出严格在 Klein 球内。

  3. 可选 gyrobias:
     result = p ⊕_E y, 其中 p = exp_0(tangent)

参数化 (对齐 HBNN BFC):
  weight_v ∈ ℝ^{out×in}  — 方向向量 (forward 时 L2 归一化)
  weight_g ∈ ℝ^{out}     — log-scale 缩放 (forward 时 exp → α > 0)
  bias ∈ ℝ^{out}          — 偏置
  tangent ∈ ℝ^{out}       — gyrobias 切向量 (可选)

欧氏极限 (K → 0⁻):
  y_k → u_k / √(1 + Σ u_j²)，退化为 softmax-normalized FC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from klein_nn.manifold import KleinManifold, MIN_NORM, EDGE_EPS

# HBNN convention
EPS = {torch.float32: 1e-4, torch.float64: 1e-8}


def _eps(x):
    return EPS.get(x.dtype, 1e-8)


def _klein_busemann_linear(x: Tensor, v_unit: Tensor, scaling: Tensor,
                            bias: Tensor, K: float, eps: float,
                            act) -> Tensor:
    """Klein BFC 核心计算。

    x: (batch, in_dim)     — Klein 球内的输入点
    v_unit: (in_dim, out)  — 单位方向向量 (转置后)
    scaling: (out,)        — 正尺度参数 α
    bias: (out,)           — 偏置
    K: float               — 曲率 (< 0)
    eps: float             — 数值安全
    act: callable          — 激活函数 (作用在 logit 上)
    返回: (batch, out_dim) — Klein 球内的输出点
    """
    sqrt_mK = (-K) ** 0.5

    # Step 1: 计算 Busemann logit u_k(x)
    # ½ log(1 + K‖x‖²): (batch, 1)
    x_sqnorm = x.pow(2).sum(dim=-1, keepdim=True)
    shared_term = 0.5 * torch.log((1.0 + K * x_sqnorm).clamp(min=eps))

    # √(-K) · <x, v_k>: (batch, out)
    inner = sqrt_mK * (x @ v_unit)

    # -log(1 - √(-K)<x, v_k>): (batch, out)
    class_term = -torch.log((1.0 - inner).clamp(min=eps))

    # u_k(x) = α_k/√(-K) · (shared + class) + b_k
    u = (scaling / sqrt_mK).unsqueeze(0) * (shared_term + class_term) + bias.unsqueeze(0)

    # Step 2: 激活函数 (可选，如 ReLU/tanh)
    v = act(u)

    # Step 3: 归一化到 Klein 球 (Theorem 5.6)
    # y_k = sinh(√(-K) · v_k) / [√(-K) · √(1 + Σ sinh²(√(-K) · v_j))]
    sinh_val = torch.sinh(sqrt_mK * v)          # (batch, out)
    norm_factor = (1.0 + sinh_val.pow(2).sum(dim=-1, keepdim=True)).sqrt()  # (batch, 1)
    y = sinh_val / (sqrt_mK * norm_factor)

    # 数值安全投影
    y = KleinManifold.projx(y, K)
    return y


class KleinBFC(nn.Module):
    """Klein Busemann 全连接层。

    输入: Klein 球内的点 x ∈ K^n
    输出: Klein 球内的点 y ∈ K^m (m = out_dim)

    使用 Busemann 函数计算 logit，然后通过 sinh 归一化映射回 Klein 球。
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
        # 对齐 HBNN BFC 初始化
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
        """
        x: (batch, in_dim) — Klein 球内的输入点
        返回: (batch, out_dim) — Klein 球内的输出点
        """
        eps = _eps(x)

        drop_weight = F.dropout(self.weight_v, self.dropout, training=self.training)
        v_unit = drop_weight / drop_weight.norm(dim=-1, keepdim=True).clamp_min(eps)
        scaling = self.weight_g.exp()

        # Busemann linear: (batch, out_dim) → Klein 球内
        y = _klein_busemann_linear(
            x, v_unit.t(), scaling, self.bias,
            self.K, eps, self.act,
        )

        # Gyrobias: Einstein 加法偏移
        if self.use_gyrobias:
            p = KleinManifold.exp0(self.tangent, self.K)
            y = KleinManifold.einstein_add(p, y, self.K)

        return y
