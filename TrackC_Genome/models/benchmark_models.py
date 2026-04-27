import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from lib.lorentz.layers import LorentzFullyConnected, LorentzConv1d, LorentzBatchNorm1d, LorentzReLU, LorentzGlobalAvgPool2d, LorentzMLR
from lib.lorentz.manifold import CustomLorentz

# Klein MLR / BMLR imports
from klein_nn.layers.mlr import KleinMLR
from klein_nn.layers.bmlr import KleinBMLR
from klein_nn.manifold import KleinManifold

def get_euclidean_convolution_block(channels_sizes,
                                    kernel_size: int = 9,
                                    padding: int = 4):
    return nn.Sequential(
            nn.Conv1d(channels_sizes[0], channels_sizes[1], kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels_sizes[2], channels_sizes[3], kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels_sizes[3]))


def get_hyperbolic_convolution_block(manifold, 
                                     channels_sizes,
                                     kernel_size: int = 9,
                                     padding: int = 4):
    return nn.Sequential(
            LorentzConv1d(manifold=manifold, in_channels=channels_sizes[0], out_channels=channels_sizes[1], kernel_size=kernel_size, padding=padding),
            LorentzBatchNorm1d(manifold=manifold, num_features=channels_sizes[1]),
            LorentzReLU(manifold=manifold),
            LorentzConv1d(manifold=manifold, in_channels=channels_sizes[2], out_channels=channels_sizes[3], kernel_size=kernel_size, padding=padding),
            LorentzBatchNorm1d(manifold=manifold, num_features=channels_sizes[3])
            )


class HyperbolicCNN(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 length: int, 
                 model_dim: int, 
                 fc_dim: int,
                 num_layers: int = 3, 
                 multi_k_model: bool = False,
                 learnable_k: bool = True,
                 k: float = 1.0):
        super(HyperbolicCNN, self).__init__()

        self.multi_k_model = multi_k_model
        self.output_length = length
        self.num_layers = num_layers
        self.manifolds = nn.ModuleList([CustomLorentz(k=k, learnable=learnable_k)])
        if self.multi_k_model:
            self.manifolds.extend([CustomLorentz(k=k, learnable=learnable_k) for _ in range(self.num_layers)])
        else:
            self.manifolds.extend([self.manifolds[0] for _ in range(self.num_layers)])
        
        # Lorentz: input one-hot channels=5, in_channels=6 after adding time component
        initial_channel_sizes = [6] + [model_dim] * 3
        channel_sizes = [model_dim] * 4
        
        self.conv_layers = nn.ModuleList([
            get_hyperbolic_convolution_block(self.manifolds[0], initial_channel_sizes),
        ])
        self.conv_layers.extend([get_hyperbolic_convolution_block(self.manifolds[i + 1], channel_sizes) for i in range(self.num_layers - 1)])
        
        self.shortcut = nn.Sequential()
        
        self.activations = nn.ModuleList([
            LorentzReLU(manifold=manifold) for manifold in self.manifolds
        ])
        
        self.fc_layer = LorentzFullyConnected(
            manifold=self.manifolds[-1],
            in_features=1 + ((model_dim - 1) * self.output_length),
            out_features=fc_dim,
            bias=True
        )
        
        self.mlr = LorentzMLR(
            manifold=self.manifolds[-1],
            num_features=fc_dim,
            num_classes=num_classes
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.pad(x, pad=(1, 0))
        x = self.manifolds[0].projx(x)
        
        for i in range(self.num_layers):
            out = self.conv_layers[i](x)
            if i > 0:
                out = out.narrow(-1, 1, res.shape[-1]-1) + res.narrow(-1, 1, res.shape[-1]-1)
                out = self.manifolds[i].add_time(out)
            x = self.activations[i](out)
            if self.multi_k_model:
                x = self.manifolds[i+1].expmap0(self.manifolds[i].logmap0(x))
            res = self.shortcut(x)
        
        x = x.unsqueeze(2) #[b, l, c] -> [b, l, 1, c]
        x = self.manifolds[-1].lorentz_flatten(x)
        x = self.fc_layer(x)
        x = self.activations[-1](x)
        x = self.mlr(x)
        
        return x



class EuclideanCNN(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 length: int, 
                 model_dim: int, 
                 fc_dim: int,
                 num_layers: int = 3,
                 ):
        super(EuclideanCNN, self).__init__()
        
        self.output_length = length
        self.num_layers = num_layers
        initial_channel_sizes = [5] + [model_dim] * 3
        channel_sizes = [model_dim] * 4
        
        self.conv_layers = nn.ModuleList([
            get_euclidean_convolution_block(initial_channel_sizes)
        ])
        self.conv_layers.extend([get_euclidean_convolution_block(channel_sizes) for _ in range(self.num_layers - 1)])
    
        self.activations = nn.ModuleList([
            nn.ReLU(inplace=True) for _ in range(self.num_layers + 1)
        ])        
        
        self.fc_layer = nn.Linear(self.output_length * model_dim, fc_dim)
        self.mlr = nn.Linear(fc_dim, num_classes)
        
        
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)         
            if i > 0:
                x += out
            x = self.activations[i](x)
            out = x

        x = x.view(x.shape[0], -1)       
        x = self.fc_layer(x)
        x = self.activations[-1](x)
        x = self.mlr(x)
        return x


# =================== PV Hyperbolic CNN ===================
from lib.pv.manifold import PVManifold
from lib.pv.blocks_1d import get_pv_convolution_block
from lib.pv.layers_1d import PVReLU, PVConv1d
from lib.pv.mlr import PVManifoldMLR

# =================== Poincaré Hyperbolic CNN ===================
from lib.geoopt.manifolds.stereographic.manifold import PoincareBall
from lib.poincare.blocks_1d import get_poincare_convolution_block
from lib.poincare.layers.PMLR import UnidirectionalPoincareMLR


class PVHyperbolicCNN(nn.Module):
    def __init__(self,
                 num_classes: int,
                 length: int,
                 model_dim: int,
                 fc_dim: int,
                 num_layers: int = 3,
                 learnable_k: bool = True,
                 k: float = 0.5):
        super().__init__()

        self.output_length = length
        self.num_layers = num_layers
        self.manifolds = nn.ModuleList([PVManifold(c=k, learnable=learnable_k) for _ in range(self.num_layers + 1)])

        # Input projection: 5(one-hot) -> model_dim, compatible with block channels_sizes[1]
        self.input_proj = PVConv1d(self.manifolds[0].c, in_channels=5, out_channels=model_dim, kernel_size=1, padding=0)

        # Before block, already model_dim channels; block uses constant channel config
        initial_channel_sizes = [model_dim] * 4
        channel_sizes = [model_dim] * 4

        self.conv_layers = nn.ModuleList([
            get_pv_convolution_block(self.manifolds[0], initial_channel_sizes),
        ])
        self.conv_layers.extend([get_pv_convolution_block(self.manifolds[i + 1], channel_sizes) for i in range(self.num_layers - 1)])

        self.activations = nn.ModuleList([
            PVReLU(manifold=manifold) for manifold in self.manifolds
        ])

        # Tangent-space FC + PV MLR classification head
        self.fc_layer = nn.Linear(self.output_length * model_dim, fc_dim)
        self.mlr = PVManifoldMLR(self.manifolds[-1], in_features=fc_dim, num_classes=num_classes)

    def forward(self, x):
        # Input x: (B, C_in=5, L)
        # First 5 -> model_dim input projection
        x = self.input_proj(x)
        for i in range(self.num_layers):
            out = self.conv_layers[i](x)
            x = self.activations[i](out)
            out = x

        x = x.reshape(x.shape[0], -1)
        x = self.fc_layer(x)
        # Map tangent-space features back to PV then PV MLR
        x_pv = self.manifolds[-1].expmap0(x)
        x = self.mlr(x_pv)
        return x


class PoincareHyperbolicCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        length: int,
        model_dim: int,
        fc_dim: int,
        num_layers: int = 3,
        learnable_k: bool = True,
        k: float = 1.0,
    ):
        super().__init__()

        self.output_length = length
        self.num_layers = num_layers
        # Use geoopt PoincareBall, c=k
        self.manifolds = nn.ModuleList([PoincareBall(c=k, learnable=learnable_k) for _ in range(self.num_layers + 1)])

        initial_channel_sizes = [5] + [model_dim] * 3
        channel_sizes = [model_dim] * 4

        self.conv_layers = nn.ModuleList(
            [get_poincare_convolution_block(self.manifolds[0], initial_channel_sizes)]
        )
        self.conv_layers.extend(
            [get_poincare_convolution_block(self.manifolds[i + 1], channel_sizes) for i in range(self.num_layers - 1)]
        )

        self.activations = nn.ModuleList([])  # ReLU and BN already in block

        # Tangent-space FC + Poincaré MLR classification head
        self.fc_layer = nn.Linear(self.output_length * model_dim, fc_dim)
        self.mlr = UnidirectionalPoincareMLR(fc_dim, num_classes, bias=True, ball=self.manifolds[-1])

    def forward(self, x):
        # Input x: (B, C_in=5, L)
        for i in range(self.num_layers):
            out = self.conv_layers[i](x)
            x = out

        x = x.reshape(x.shape[0], -1)
        x = self.fc_layer(x)
        # Map tangent-space features back to Poincaré then Poincaré MLR
        x_p = self.manifolds[-1].expmap0(x)
        x = self.mlr(x_p)
        return x


# =================== Klein Hybrid CNN (Euclidean backbone + Klein head) ===================

# Shared utility functions for hybrid models
EPS_HYBRID = 1e-4


def _clip_features(x, r=1.0):
    """CLIP(x; r) = min{1, r/||x||} * x"""
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    scale = torch.clamp(r / norm, max=1.0)
    return x * scale


def _klein_exp0(v, K=-1.0):
    """Klein exp_0: tangent at origin -> Klein ball point.
    exp_0(v) = tanh(sqrt(-K)*||v||) / (sqrt(-K)*||v||) * v
    """
    sqrt_mK = (-K) ** 0.5
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    return torch.tanh(sqrt_mK * v_norm) / (sqrt_mK * v_norm) * v


def _klein_projx(x, K=-1.0, edge_eps=1e-5):
    """Project x into Klein ball interior: ||x|| < (1-eps)/sqrt(-K)."""
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    maxnorm = (1 - edge_eps) / (-K) ** 0.5
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def _pv_exp0(v, K=-1.0):
    """PV exp_0: tangent at origin -> PV space point.
    exp_0(v) = sinh(||v||/s) / (||v||/s) * v, where s = 1/sqrt(-K)
    PV space is unbounded (||x|| can be > 1/sqrt(-K)).
    """
    s = 1.0 / (-K) ** 0.5
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    ratio = v_norm / s
    return torch.sinh(ratio) / ratio.clamp_min(1e-15) * v


class KleinMLRCNN(nn.Module):
    """Euclidean CNN backbone -> clip(r) -> exp_0^K -> KleinMLR head.

    Architecture follows HBNN Table 5: Euclidean feature extractor
    with hyperbolic classification head on Klein model.
    """
    def __init__(self,
                 num_classes: int,
                 length: int,
                 model_dim: int,
                 fc_dim: int,
                 num_layers: int = 3,
                 learnable_k: bool = True,
                 k: float = 1.0,
                 clip_r: float = 1.0):
        super().__init__()

        self.K = -1.0  # Fixed curvature K = -1
        self.clip_r = clip_r

        # Euclidean CNN backbone (same as EuclideanCNN)
        self.output_length = length
        self.num_layers = num_layers
        initial_channel_sizes = [5] + [model_dim] * 3
        channel_sizes = [model_dim] * 4

        self.conv_layers = nn.ModuleList([
            get_euclidean_convolution_block(initial_channel_sizes)
        ])
        self.conv_layers.extend([get_euclidean_convolution_block(channel_sizes) for _ in range(self.num_layers - 1)])

        self.activations = nn.ModuleList([
            nn.ReLU(inplace=True) for _ in range(self.num_layers + 1)
        ])

        self.fc_layer = nn.Linear(self.output_length * model_dim, fc_dim)

        # Klein MLR classification head
        self.mlr = KleinMLR(in_dim=fc_dim, num_classes=num_classes, K=self.K)

    def forward(self, x):
        # Euclidean backbone
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if i > 0:
                x += out
            x = self.activations[i](x)
            out = x

        x = x.view(x.shape[0], -1)
        x = self.fc_layer(x)
        x = self.activations[-1](x)

        # clip -> exp_0 -> projx -> Klein MLR
        x = _clip_features(x, self.clip_r)
        x = _klein_exp0(x, self.K)
        x = _klein_projx(x, self.K)
        x = self.mlr(x)
        return x


class KleinBMLRCNN(nn.Module):
    """Euclidean CNN backbone -> clip(r) -> exp_0^K -> KleinBMLR head.

    Architecture follows HBNN Table 5: Euclidean feature extractor
    with Klein Busemann MLR classification head.
    """
    def __init__(self,
                 num_classes: int,
                 length: int,
                 model_dim: int,
                 fc_dim: int,
                 num_layers: int = 3,
                 learnable_k: bool = True,
                 k: float = 1.0,
                 clip_r: float = 1.0):
        super().__init__()

        self.K = -1.0  # Fixed curvature K = -1
        self.clip_r = clip_r

        # Euclidean CNN backbone (same as EuclideanCNN)
        self.output_length = length
        self.num_layers = num_layers
        initial_channel_sizes = [5] + [model_dim] * 3
        channel_sizes = [model_dim] * 4

        self.conv_layers = nn.ModuleList([
            get_euclidean_convolution_block(initial_channel_sizes)
        ])
        self.conv_layers.extend([get_euclidean_convolution_block(channel_sizes) for _ in range(self.num_layers - 1)])

        self.activations = nn.ModuleList([
            nn.ReLU(inplace=True) for _ in range(self.num_layers + 1)
        ])

        self.fc_layer = nn.Linear(self.output_length * model_dim, fc_dim)

        # Klein BMLR classification head
        self.bmlr = KleinBMLR(in_dim=fc_dim, num_classes=num_classes, K=self.K)

    def forward(self, x):
        # Euclidean backbone
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if i > 0:
                x += out
            x = self.activations[i](x)
            out = x

        x = x.view(x.shape[0], -1)
        x = self.fc_layer(x)
        x = self.activations[-1](x)

        # clip -> exp_0 -> projx -> Klein BMLR
        x = _clip_features(x, self.clip_r)
        x = _klein_exp0(x, self.K)
        x = _klein_projx(x, self.K)
        x = self.bmlr(x)
        return x


# =================== PV BMLR ===================

# PV BMLR head (self-contained, from Phase 1 mlr_heads.py)
_PV_EPS = {torch.float32: 1e-4, torch.float64: 1e-8}


def _pv_eps(x):
    return _PV_EPS.get(x.dtype, 1e-8)


class _PV_BMLR_Head(nn.Module):
    """PV Busemann MLR via PV->Lorentz isometry.

    PV Busemann (Eq. pv-busemann):
      B^v(x) = 1/sqrt(-K) * log(sqrt(1-K||x||^2) - sqrt(-K)<x,v>)

    BMLR logit:
      u_k(x) = -alpha_k * B^v_k(x) + b_k

    Parameters aligned with HBNN BMLR (Normal + log-scale).
    Input: (bs, n) PV space point (unbounded)
    """

    def __init__(self, in_dim: int, num_classes: int, K: float = -1.0):
        super().__init__()
        self.K = K

        weight = torch.empty(num_classes, in_dim)
        nn.init.normal_(weight, mean=0.0, std=in_dim ** -0.5)
        self.weight_v = nn.Parameter(weight)
        self.weight_g = nn.Parameter(weight.norm(dim=-1).clamp_min(1e-4).log())
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        eps = _pv_eps(x)
        K = self.K
        sqrt_c = (-K) ** 0.5

        v_unit = self.weight_v / self.weight_v.norm(dim=-1, keepdim=True).clamp_min(eps)
        scaling = self.weight_g.exp()

        # PV Busemann: B^v(x) = 1/sqrt(-K) * log(sqrt(1-K||x||^2) - sqrt(-K)<x,v>)
        x_sq = x.pow(2).sum(dim=-1, keepdim=True)        # (bs, 1)
        sqrt_term = torch.sqrt((1.0 - K * x_sq).clamp_min(eps))  # sqrt(1+|K|||x||^2)
        x_dot_v = x @ v_unit.t()                          # (bs, C)

        # argument = sqrt(1-K||x||^2) - sqrt(-K)<x,v>  (always > 0)
        argument = sqrt_term - sqrt_c * x_dot_v            # (bs, C)
        busemann = torch.log(argument.clamp_min(eps)) / sqrt_c

        return -busemann * scaling + self.bias


class PV_BMLRCNN(nn.Module):
    """Euclidean CNN backbone -> clip(r) -> PV exp_0 -> PV BMLR head.

    Architecture follows HBNN Table 5: Euclidean feature extractor
    with PV Busemann MLR classification head.
    """
    def __init__(self,
                 num_classes: int,
                 length: int,
                 model_dim: int,
                 fc_dim: int,
                 num_layers: int = 3,
                 learnable_k: bool = True,
                 k: float = 1.0,
                 clip_r: float = 1.0):
        super().__init__()

        self.K = -1.0  # Fixed curvature K = -1
        self.clip_r = clip_r

        # Euclidean CNN backbone (same as EuclideanCNN)
        self.output_length = length
        self.num_layers = num_layers
        initial_channel_sizes = [5] + [model_dim] * 3
        channel_sizes = [model_dim] * 4

        self.conv_layers = nn.ModuleList([
            get_euclidean_convolution_block(initial_channel_sizes)
        ])
        self.conv_layers.extend([get_euclidean_convolution_block(channel_sizes) for _ in range(self.num_layers - 1)])

        self.activations = nn.ModuleList([
            nn.ReLU(inplace=True) for _ in range(self.num_layers + 1)
        ])

        self.fc_layer = nn.Linear(self.output_length * model_dim, fc_dim)

        # PV BMLR classification head
        self.bmlr = _PV_BMLR_Head(in_dim=fc_dim, num_classes=num_classes, K=self.K)

    def forward(self, x):
        # Euclidean backbone
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if i > 0:
                x += out
            x = self.activations[i](x)
            out = x

        x = x.view(x.shape[0], -1)
        x = self.fc_layer(x)
        x = self.activations[-1](x)

        # clip -> PV exp_0 -> PV BMLR
        x = _clip_features(x, self.clip_r)
        x = _pv_exp0(x, self.K)
        x = self.bmlr(x)
        return x

