"""Unit tests for Klein BN variants (must pass before SLURM submission).

Run:  python test_klein_bn.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from klein_nn.manifold import KleinManifold
from klein_nn.layers.klein_bn import (
    KleinEinsteinBN, KleinTangentBN, KleinFrechetBNViaPoincare, KleinActWrapper,
    build_klein_bn,
)


def _random_klein_points(B, D, K=-1.0, dtype=torch.float64):
    """Random points strictly inside Klein ball ||x||² < 1/|K|."""
    torch.manual_seed(0)
    radius = 0.9 / (-K) ** 0.5
    x = torch.randn(B, D, dtype=dtype) * 0.3
    x = KleinManifold.projx(x, K=K)
    # enforce max norm
    n = x.norm(dim=-1, keepdim=True)
    x = torch.where(n > radius, x / n * radius, x)
    return x


def test_einstein_midpoint_shape_and_validity():
    x = _random_klein_points(32, 8)
    mu = KleinManifold.einstein_midpoint(x, K=-1.0)
    assert mu.shape == (8,), f"expected (8,), got {mu.shape}"
    assert KleinManifold.check_in_ball(mu, K=-1.0).all(), "midpoint outside ball"
    print("[PASS] einstein_midpoint shape/validity")


def test_einstein_bn_forward_training():
    torch.manual_seed(0)
    bn = KleinEinsteinBN(dim=8, K=-1.0).double()
    x = _random_klein_points(32, 8)
    bn.train()
    y = bn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all(), "NaN/Inf in BN output"
    assert KleinManifold.check_in_ball(y, K=-1.0).all(), "BN output leaves ball"
    print("[PASS] einstein BN forward (train)")


def test_einstein_bn_backward():
    torch.manual_seed(0)
    bn = KleinEinsteinBN(dim=8, K=-1.0).double()
    x = _random_klein_points(32, 8).requires_grad_(True)
    bn.train()
    y = bn(x)
    loss = y.pow(2).sum()
    loss.backward()
    assert bn.log_gamma.grad is not None and torch.isfinite(bn.log_gamma.grad).all()
    assert bn.beta.grad is not None and torch.isfinite(bn.beta.grad).all()
    assert torch.isfinite(x.grad).all()
    print("[PASS] einstein BN backward")


def test_einstein_bn_train_eval_consistency():
    torch.manual_seed(0)
    bn = KleinEinsteinBN(dim=8, K=-1.0).double()
    x = _random_klein_points(32, 8)
    # Run a couple of training steps to populate running stats
    bn.train()
    for _ in range(3):
        _ = bn(x)
    bn.eval()
    y1 = bn(x)
    y2 = bn(x)
    assert torch.allclose(y1, y2), "eval forward not deterministic"
    assert KleinManifold.check_in_ball(y1, K=-1.0).all()
    print("[PASS] einstein BN train/eval consistency")


def test_tangent_bn_roundtrip():
    torch.manual_seed(0)
    bn = KleinTangentBN(dim=8, K=-1.0).double()
    x = _random_klein_points(32, 8)
    bn.train()
    y = bn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert KleinManifold.check_in_ball(y, K=-1.0).all()
    # backward
    loss = y.pow(2).sum()
    loss.backward()
    for p in bn.parameters():
        assert torch.isfinite(p.grad).all()
    print("[PASS] tangent BN forward+backward")


def test_frechet_bn_via_poincare():
    try:
        torch.manual_seed(0)
        bn = KleinFrechetBNViaPoincare(dim=8, K=-1.0).double()
    except ImportError as e:
        print(f"[SKIP] frechet BN (unavailable): {e}")
        return
    x = _random_klein_points(32, 8)
    bn.train()
    y = bn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert KleinManifold.check_in_ball(y, K=-1.0).all()
    print("[PASS] frechet BN via Poincare forward")


def test_klein_act_wrapper():
    act = KleinActWrapper('relu', K=-1.0).double()
    x = _random_klein_points(32, 8)
    y = act(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert KleinManifold.check_in_ball(y, K=-1.0).all()
    print("[PASS] KleinActWrapper")


def test_factory():
    for tag in ['KleinEinsteinBN', 'KleinTgBN']:
        mod = build_klein_bn(tag, dim=8, K=-1.0).double()
        x = _random_klein_points(16, 8)
        y = mod(x)
        assert y.shape == x.shape
    print("[PASS] build_klein_bn factory")


def test_encoder_integration():
    # Pretend args object, exercise HNN_KleinFC with BN
    import sys as _sys
    gyro_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'GyroBN-main')
    if gyro_root not in _sys.path:
        _sys.path.insert(0, gyro_root)
    try:
        from RieNets.hnns.models.encoders import HNN_KleinFC
    except ImportError as e:
        print(f"[SKIP] encoder integration (framework unavailable): {e}")
        return

    class _Args:
        pass
    args = _Args()
    args.manifold = 'KleinBall'
    args.num_layers = 2
    args.dim = 16
    args.feat_dim = 10
    args.c = 1.0
    args.task = 'lp'
    args.act = 'relu'
    args.dropout = 0.0
    args.bias = 1
    args.cuda = -1
    args.is_bn = True
    args.bn_type = 'KleinEinsteinBN'

    torch.set_default_dtype(torch.float64)
    enc = HNN_KleinFC(torch.tensor([args.c]), args)
    feat = torch.randn(20, args.feat_dim) * 0.1
    # expmap0 via encoder's manifold
    x_hyp = enc.manifold.proj(
        enc.manifold.expmap0(enc.manifold.proj_tan0(feat, args.c), c=args.c),
        c=args.c)
    out = enc.layers(x_hyp)
    assert out.shape == (20, args.dim)
    assert torch.isfinite(out).all()
    print("[PASS] HNN_KleinFC + BN end-to-end")


if __name__ == '__main__':
    test_einstein_midpoint_shape_and_validity()
    test_einstein_bn_forward_training()
    test_einstein_bn_backward()
    test_einstein_bn_train_eval_consistency()
    test_tangent_bn_roundtrip()
    test_frechet_bn_via_poincare()
    test_klein_act_wrapper()
    test_factory()
    test_encoder_integration()
    print("\nAll Klein BN tests passed.")
