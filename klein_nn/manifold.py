"""
Klein 模型流形运算 — 完整几何工具箱

所有公式编号对应 Klein_PV范式推导.pdf。
约定：曲率 K < 0，默认 K = -1。

Klein 模型空间：K^n_K = {x in R^n | ||x||^2 < -1/K}
当 K = -1 时即为单位开球 {x : ||x|| < 1}。
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

# 数值常量
MIN_NORM = 1e-15
EDGE_EPS = 1e-5
CLAMP_MAX = 1e6


class KleinManifold:
    """Klein 模型上的所有黎曼/陀螺运算。

    所有方法均为 staticmethod 或接受 K 参数，
    以便在 JIT 编译和层代码中灵活调用。
    """

    def __init__(self, K: float = -1.0):
        self.K = K

    # ================================================================
    # §1. 基本量
    # ================================================================

    @staticmethod
    def gamma(x: Tensor, K: float = -1.0, keepdim: bool = True) -> Tensor:
        """Lorentz 因子 γ_x。 (PDF Eq. 3)

        γ_x = 1 / sqrt(1 + K ||x||^2)

        注意 K < 0，所以 1 + K||x||^2 ∈ (0, 1]，γ_x >= 1。
        """
        x_sqnorm = x.pow(2).sum(dim=-1, keepdim=keepdim)
        return (1.0 + K * x_sqnorm).clamp(min=MIN_NORM).rsqrt()

    @staticmethod
    def riemannian_inner(x: Tensor, u: Tensor, v: Tensor,
                         K: float = -1.0, keepdim: bool = False) -> Tensor:
        """Klein 黎曼内积。 (PDF Eq. 4)

        g^K_x(u, v) = γ²_x <u, v> + (-K) γ⁴_x <x, u><x, v>
        """
        gamma_x = KleinManifold.gamma(x, K, keepdim=True)
        gamma_sq = gamma_x.pow(2)
        gamma_4 = gamma_sq.pow(2)
        uv = (u * v).sum(dim=-1, keepdim=True)
        xu = (x * u).sum(dim=-1, keepdim=True)
        xv = (x * v).sum(dim=-1, keepdim=True)
        result = gamma_sq * uv + (-K) * gamma_4 * xu * xv
        if not keepdim:
            result = result.squeeze(-1)
        return result

    @staticmethod
    def riemannian_norm(x: Tensor, v: Tensor,
                        K: float = -1.0, keepdim: bool = True) -> Tensor:
        """切向量的黎曼范数 ||v||_{K,x} = sqrt(g^K_x(v, v))。"""
        inner = KleinManifold.riemannian_inner(x, v, v, K, keepdim=True)
        result = inner.clamp(min=0).sqrt()
        if not keepdim:
            result = result.squeeze(-1)
        return result

    @staticmethod
    def dist(x: Tensor, y: Tensor, K: float = -1.0,
             keepdim: bool = False) -> Tensor:
        """Klein 测地距离。 (PDF Eq. 5)

        d_K(x, y) = (1/√(-K)) cosh⁻¹(γ_x γ_y (1 + K <x, y>))
        """
        sqrt_mK = (-K) ** 0.5
        gamma_x = KleinManifold.gamma(x, K, keepdim=True)
        gamma_y = KleinManifold.gamma(y, K, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        arg = gamma_x * gamma_y * (1.0 + K * xy)
        # cosh⁻¹(t) 要求 t >= 1
        arg = arg.clamp(min=1.0)
        d = torch.acosh(arg) / sqrt_mK
        if not keepdim:
            d = d.squeeze(-1)
        return d

    # ================================================================
    # §2. 指数映射与对数映射
    # ================================================================

    @staticmethod
    def exp0(v: Tensor, K: float = -1.0) -> Tensor:
        """原点处指数映射。 (PDF Eq. 7)

        exp_0(v) = (1/√(-K)) tanh(√(-K) ||v||) · v/||v||
        """
        sqrt_mK = (-K) ** 0.5
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=MIN_NORM)
        result = torch.tanh(sqrt_mK * v_norm) * v / (sqrt_mK * v_norm)
        return KleinManifold.projx(result, K)

    @staticmethod
    def log0(y: Tensor, K: float = -1.0) -> Tensor:
        """原点处对数映射。 (PDF Eq. 8)

        log_0(y) = (1/√(-K)) tanh⁻¹(√(-K) ||y||) · y/||y||
        """
        sqrt_mK = (-K) ** 0.5
        y_norm = y.norm(dim=-1, keepdim=True).clamp(min=MIN_NORM)
        # tanh⁻¹ 的参数必须在 (-1, 1) 内
        arg = (sqrt_mK * y_norm).clamp(max=1.0 - EDGE_EPS)
        return torch.atanh(arg) * y / (sqrt_mK * y_norm)

    @staticmethod
    def expx(x: Tensor, v: Tensor, K: float = -1.0) -> Tensor:
        """一般基点指数映射。 (PDF Eq. 9)"""
        v_norm_K = KleinManifold.riemannian_norm(x, v, K, keepdim=True)
        v_norm_K = v_norm_K.clamp(min=MIN_NORM)
        gamma_x = KleinManifold.gamma(x, K, keepdim=True)
        xv = (x * v).sum(dim=-1, keepdim=True)

        sinh_norm = torch.sinh(v_norm_K)
        cosh_norm = torch.cosh(v_norm_K)

        coeff = sinh_norm / (cosh_norm + gamma_x.pow(2) * xv * sinh_norm / v_norm_K)
        result = x + coeff * v / v_norm_K
        return KleinManifold.projx(result, K)

    @staticmethod
    def logx(x: Tensor, y: Tensor, K: float = -1.0) -> Tensor:
        """一般基点对数映射。 (PDF Eq. 10)"""
        d = KleinManifold.dist(x, y, K, keepdim=True)
        diff = y - x
        diff_norm_K = KleinManifold.riemannian_norm(x, diff, K, keepdim=True)
        diff_norm_K = diff_norm_K.clamp(min=MIN_NORM)
        return d * diff / diff_norm_K

    # ================================================================
    # §3. 平行传输
    # ================================================================

    @staticmethod
    def pt_0_to_x(v: Tensor, x: Tensor, K: float = -1.0) -> Tensor:
        """从原点到 x 的平行传输。 (PDF Eq. 11)"""
        gamma_x = KleinManifold.gamma(x, K, keepdim=True)
        vx = (v * x).sum(dim=-1, keepdim=True)
        return v / gamma_x - vx / (1.0 + gamma_x) * x

    @staticmethod
    def pt_x_to_0(v: Tensor, x: Tensor, K: float = -1.0) -> Tensor:
        """从 x 到原点的平行传输（逆传输）。 (PDF Eq. 21)"""
        gamma_x = KleinManifold.gamma(x, K, keepdim=True)
        xv = (x * v).sum(dim=-1, keepdim=True)
        return gamma_x * v + gamma_x.pow(3) / (1.0 + gamma_x) * xv * x

    # ================================================================
    # §4. Einstein Gyrovector Space
    # ================================================================

    @staticmethod
    def einstein_add(x: Tensor, y: Tensor, K: float = -1.0) -> Tensor:
        """Einstein 加法 x ⊕_E y。 (PDF Eq. 22)"""
        gamma_x = KleinManifold.gamma(x, K, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        denom = (1.0 - K * xy).clamp(min=MIN_NORM)
        num = x + y / gamma_x - K * gamma_x / (1.0 + gamma_x) * xy * x
        result = num / denom
        return KleinManifold.projx(result, K)

    @staticmethod
    def einstein_scalar(t: Tensor, x: Tensor, K: float = -1.0) -> Tensor:
        """Einstein 标量乘法 t ⊗_E x。 (PDF Eq. 24)"""
        sqrt_mK = (-K) ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True).clamp(min=MIN_NORM)
        arg = (sqrt_mK * x_norm).clamp(max=1.0 - EDGE_EPS)
        atanh_val = torch.atanh(arg)

        if not isinstance(t, Tensor):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)
        if t.dim() < x.dim():
            t = t.unsqueeze(-1)

        result = torch.tanh(t * atanh_val) * x / (sqrt_mK * x_norm)
        return KleinManifold.projx(result, K)

    @staticmethod
    def einstein_midpoint(xs: Tensor, K: float = -1.0,
                          weights: Optional[Tensor] = None) -> Tensor:
        """Einstein 中点（γ-加权平均）。 (PDF Eq. 108)"""
        gammas = KleinManifold.gamma(xs, K, keepdim=True)

        if weights is not None:
            w = weights.unsqueeze(-1)
            gammas = gammas * w

        numerator = (gammas * xs).sum(dim=-2)
        denominator = gammas.sum(dim=-2).clamp(min=MIN_NORM)
        mu = numerator / denominator
        return KleinManifold.projx(mu, K)

    # ================================================================
    # 数值安全
    # ================================================================

    @staticmethod
    def projx(x: Tensor, K: float = -1.0) -> Tensor:
        """将点投影到 Klein 球内（数值安全）。"""
        sqrt_mK = (-K) ** 0.5
        max_norm = (1.0 - EDGE_EPS) / sqrt_mK
        x_norm = x.norm(dim=-1, keepdim=True).clamp(min=MIN_NORM)
        cond = x_norm > max_norm
        projected = x / x_norm * max_norm
        return torch.where(cond, projected, x)

    @staticmethod
    def check_in_ball(x: Tensor, K: float = -1.0) -> Tensor:
        """检查点是否在 Klein 球内，返回布尔张量。"""
        max_norm_sq = -1.0 / K
        x_sqnorm = x.pow(2).sum(dim=-1)
        return x_sqnorm < max_norm_sq

    # ================================================================
    # Klein ↔ Poincaré 等距映射
    # ================================================================

    @staticmethod
    def klein_to_poincare(x: Tensor, K: float = -1.0) -> Tensor:
        """Klein → Poincaré。"""
        x_sqnorm = x.pow(2).sum(dim=-1, keepdim=True)
        sqrt_term = (1.0 + K * x_sqnorm).clamp(min=MIN_NORM).sqrt()
        return x / (1.0 + sqrt_term)

    @staticmethod
    def poincare_to_klein(y: Tensor, K: float = -1.0) -> Tensor:
        """Poincaré → Klein。"""
        y_sqnorm = y.pow(2).sum(dim=-1, keepdim=True)
        return 2.0 * y / (1.0 - K * y_sqnorm).clamp(min=MIN_NORM)

    # ================================================================
    # Klein ↔ Lorentz 映射
    # ================================================================

    @staticmethod
    def klein_to_lorentz(x: Tensor, K: float = -1.0) -> Tensor:
        """Klein → Lorentz (gnomonic projection)。 (PDF Eq. 43)"""
        gamma_x = KleinManifold.gamma(x, K, keepdim=True)
        return torch.cat([gamma_x, gamma_x * x], dim=-1)

    @staticmethod
    def lorentz_to_klein(x_L: Tensor) -> Tensor:
        """Lorentz → Klein。"""
        x_t = x_L[..., :1].clamp(min=MIN_NORM)
        x_s = x_L[..., 1:]
        return x_s / x_t
