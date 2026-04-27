"""
Phase 2 层代码快速验证脚本
测试所有 4 个新实现的 FC/BFC 层:
  1. KleinFC   (Theorem 5.5)
  2. KleinBFC  (Theorem 5.6)
  3. PVFC      (geodesic hyperplane)
  4. PVBFC     (Busemann horosphere)

验证内容:
  - import 无报错
  - forward pass 无 NaN/Inf
  - Klein 输出在球内 (||y|| < 1/sqrt(-K))
  - backward pass 无报错
  - 欧氏极限 (K → 0⁻) 行为合理
"""
import sys
import torch
import torch.nn as nn

torch.manual_seed(42)
K = -1.0
BATCH = 8
IN_DIM = 16
OUT_DIM = 10
SQRT_MK = (-K) ** 0.5

def check(name, y, bounded=False):
    """检查输出是否合法"""
    ok = True
    issues = []

    if torch.isnan(y).any():
        issues.append("NaN detected!")
        ok = False
    if torch.isinf(y).any():
        issues.append("Inf detected!")
        ok = False

    if bounded:
        # projx clamps to (1-EDGE_EPS)/sqrt(-K), allow tiny float tolerance
        max_norm = 1.0 / SQRT_MK  # strict ball radius
        norms = y.norm(dim=-1)
        if (norms >= max_norm).any():
            issues.append(f"Out of ball! max_norm={norms.max():.6f} >= {max_norm:.6f}")
            ok = False

    status = "✅ PASS" if ok else "❌ FAIL"
    norm_info = f"||y||: [{y.norm(dim=-1).min():.4f}, {y.norm(dim=-1).max():.4f}]"
    print(f"  {status} {name}: shape={tuple(y.shape)}, {norm_info}", end="")
    if issues:
        print(f"  ⚠️  {', '.join(issues)}")
    else:
        print()
    return ok


def test_backward(name, layer, x):
    """测试反向传播"""
    x_test = x.clone().requires_grad_(True)
    y = layer(x_test)
    loss = y.sum()
    loss.backward()
    grad_ok = x_test.grad is not None and not torch.isnan(x_test.grad).any()
    status = "✅" if grad_ok else "❌"
    print(f"  {status} {name} backward: grad exists={x_test.grad is not None}, no NaN={grad_ok}")
    return grad_ok


# ========================================
print("=" * 60)
print("Phase 2 Layer Verification")
print("=" * 60)

all_pass = True

# --- 1. Klein layers ---
print("\n--- Klein FC/BFC (bounded Klein ball) ---")

from klein_nn.layers.fc import KleinFC
from klein_nn.layers.bfc import KleinBFC

# Klein 输入: 需要在 Klein 球内
x_klein = torch.randn(BATCH, IN_DIM) * 0.3  # 小范数确保在球内

klein_fc = KleinFC(IN_DIM, OUT_DIM, K=K, gyrobias=True, act='relu')
klein_bfc = KleinBFC(IN_DIM, OUT_DIM, K=K, gyrobias=True, act='relu')

y_fc = klein_fc(x_klein)
y_bfc = klein_bfc(x_klein)

all_pass &= check("KleinFC", y_fc, bounded=True)
all_pass &= check("KleinBFC", y_bfc, bounded=True)
all_pass &= test_backward("KleinFC", klein_fc, x_klein)
all_pass &= test_backward("KleinBFC", klein_bfc, x_klein)


# --- 2. PV layers ---
print("\n--- PV FC/BFC (unbounded PV space) ---")

from klein_nn.layers.pv_fc import PVFC
from klein_nn.layers.pv_bfc import PVBFC

# PV 输入: 可以是任意 R^n 的点
x_pv = torch.randn(BATCH, IN_DIM) * 0.5

pv_fc = PVFC(IN_DIM, OUT_DIM, K=K, gyrobias=True, act='relu')
pv_bfc = PVBFC(IN_DIM, OUT_DIM, K=K, gyrobias=True, act='relu')

y_pvfc = pv_fc(x_pv)
y_pvbfc = pv_bfc(x_pv)

all_pass &= check("PVFC", y_pvfc, bounded=False)
all_pass &= check("PVBFC", y_pvbfc, bounded=False)
all_pass &= test_backward("PVFC", pv_fc, x_pv)
all_pass &= test_backward("PVBFC", pv_bfc, x_pv)


# --- 3. Klein MLR/BMLR (from Phase 1) ---
print("\n--- Klein MLR/BMLR (Phase 1 复现验证) ---")

from klein_nn.layers.mlr import KleinMLR
from klein_nn.layers.bmlr import KleinBMLR

klein_mlr = KleinMLR(IN_DIM, OUT_DIM, K=K)
klein_bmlr = KleinBMLR(IN_DIM, OUT_DIM, K=K)

y_mlr = klein_mlr(x_klein)
y_bmlr = klein_bmlr(x_klein)

all_pass &= check("KleinMLR", y_mlr, bounded=False)  # logits, 不需要在球内
all_pass &= check("KleinBMLR", y_bmlr, bounded=False)


# --- 4. HBNN 集成测试 ---
print("\n--- HBNN BFC/BMLR Klein+PV 集成 ---")

sys.path.insert(0, "HBNN-main")
from lib.bnn.BFC import BFC, Gyrobias
from lib.bnn.BMLR import BMLR

# Klein BFC via HBNN
hbnn_klein_bfc = BFC(IN_DIM, OUT_DIM, metric='klein', K=K, gyrobias=True, act='relu')
y_hk = hbnn_klein_bfc(x_klein)
all_pass &= check("HBNN BFC(klein)", y_hk, bounded=True)

# PV BFC via HBNN
hbnn_pv_bfc = BFC(IN_DIM, OUT_DIM, metric='pv', K=K, gyrobias=True, act='relu')
y_hp = hbnn_pv_bfc(x_pv)
all_pass &= check("HBNN BFC(pv)", y_hp, bounded=False)

# Klein BMLR via HBNN
hbnn_klein_bmlr = BMLR(n_classes=OUT_DIM, dim=IN_DIM, K=K, metric='klein')
y_hkm = hbnn_klein_bmlr(x_klein)
all_pass &= check("HBNN BMLR(klein)", y_hkm, bounded=False)

# PV BMLR via HBNN
hbnn_pv_bmlr = BMLR(n_classes=OUT_DIM, dim=IN_DIM, K=K, metric='pv')
y_hpm = hbnn_pv_bmlr(x_pv)
all_pass &= check("HBNN BMLR(pv)", y_hpm, bounded=False)

# Gyrobias Klein/PV
gb_klein = Gyrobias(dim=OUT_DIM, metric='klein', K=K)
gb_pv = Gyrobias(dim=OUT_DIM, metric='pv', K=K)
y_gbk = gb_klein(y_hk)
y_gbp = gb_pv(y_hp)
all_pass &= check("Gyrobias(klein)", y_gbk, bounded=True)
all_pass &= check("Gyrobias(pv)", y_gbp, bounded=False)


# --- 5. 欧氏极限测试 ---
print("\n--- 欧氏极限 (K → 0⁻) ---")

K_small = -1e-6
x_small = torch.randn(BATCH, IN_DIM) * 0.1

klein_bfc_eu = KleinBFC(IN_DIM, OUT_DIM, K=K_small, gyrobias=False, act=None)
pv_bfc_eu = PVBFC(IN_DIM, OUT_DIM, K=K_small, gyrobias=False, act=None)

y_eu_klein = klein_bfc_eu(x_small)
y_eu_pv = pv_bfc_eu(x_small)

all_pass &= check("KleinBFC(K→0)", y_eu_klein, bounded=False)  # K very small, ball very large
all_pass &= check("PVBFC(K→0)", y_eu_pv, bounded=False)

# 在 K→0 时 Klein BFC 和 PV BFC 应该行为相近（但参数随机初始化不同，只检查形状和无NaN）
print(f"  Klein/PV output ratio (K≈0): {y_eu_klein.norm() / y_eu_pv.norm().clamp_min(1e-8):.4f}")


# --- Summary ---
print("\n" + "=" * 60)
if all_pass:
    print("🎉 ALL TESTS PASSED — 代码就绪，可以传到 Win 跑实验")
else:
    print("⚠️  SOME TESTS FAILED — 需要排查")
print("=" * 60)
