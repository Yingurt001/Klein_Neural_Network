# 代码阅读指南

这份指南帮你理解整个代码是怎么组织的、数据怎么流的、以及我们改了什么。

---

## 1. 从头到尾：一次实验的完整流程

运行一个实验：
```bash
python GyroBNH.py dataset.dataset=airport nnet=KleinFC ...
```

### 第 1 步：配置加载 (Hydra)

```
GyroBNH.py
  → 读 conf/GyroBNH.yaml (总配置)
  → 读 conf/dataset/LP.yaml (数据集: val_prop=0.05, test_prop=0.1, seed=1234)
  → 读 conf/nnet/KleinFC.yaml (模型: dim=128, dropout=0.2, manifold=KleinBall)
  → 合并为一个 args 对象
```

**你要看的文件**: `conf/nnet/KleinFC.yaml` — 这就是一个模型的全部配置。

### 第 2 步：数据加载

```
train_kfold.py → train_process()
  → load_data(args, path)                     # RieNets/hnns/utils/data_utils.py
    → load_data_lp("airport", ...)             # 加载邻接矩阵 + 特征
    → mask_edges(adj, 0.05, 0.10, seed=1234)   # 划分 train/val/test 边
    → normalize adj + features                  # 归一化
  → 输出: data dict {features, adj_train_norm, train_edges, val_edges, ...}
```

### 第 3 步：模型构建

```
base_models.py → LPModel(args)
  → self.manifold = KleinBall()                # Geometry/constantcurvature/klein.py
  → self.c = tensor([1.0])                     # 曲率 c = -K = 1
  → self.encoder = HNN_KleinFC(c, args)        # RieNets/hnns/models/encoders.py
    → 创建 2 个 KleinFC 层: KleinFC(11→128), KleinFC(128→128)
  → self.dc = FermiDiracDecoder(r=2.0, t=1.0)  # 距离 → 边概率
```

### 第 4 步：前向传播 (encode)

```
model.encode(features, adj)
  → x_hyp = manifold.proj(manifold.expmap0(manifold.proj_tan0(x, c), c), c)
    # 欧几里得特征 → Klein 球上的点
    # proj_tan0: 恒等 (切空间在原点就是 R^n)
    # expmap0: tanh(||v||) / ||v|| · v (映射到球内)
    # proj: 裁剪到 ||x|| < 1/√c

  → x = KleinFC_1(x_hyp)                       # klein_nn/layers/fc.py
    # 1. 计算 MLR logits (geodesic hyperplane signed distance)
    # 2. 激活 (relu)
    # 3. sinh 归一化 → Klein 球上的新点
    # 4. gyrobias (可选)

  → x = KleinFC_2(x)                           # 第二层，同上
  → 输出: embeddings (n_nodes × 128) 在 Klein 球上
```

### 第 5 步：解码 + 损失

```
model.compute_metrics(embeddings, data, 'train')
  → pos_scores = decode(embeddings, pos_edges)
    → emb_in = embeddings[edge[:, 0]]          # 源节点嵌入
    → emb_out = embeddings[edge[:, 1]]          # 目标节点嵌入
    → sqdist = manifold.sqdist(emb_in, emb_out, c)  # Klein 球上的平方距离
    → probs = FermiDirac(sqdist)                # 1/(exp((d-r)/t) + 1)

  → loss = BCE(pos_scores, 1) + BCE(neg_scores, 0)
  → roc = roc_auc_score(labels, preds)
```

### 第 6 步：训练循环

```
for epoch in range(5000):
    loss.backward()
    optimizer.step()          # Adam, lr=0.01
    if val_score improved:
        save best_test_metrics
    elif patience exhausted:
        early stop
```

---

## 2. 我们改了什么（★ 标记）

### 新增文件

| 文件 | 说明 |
|------|------|
| `klein_nn/manifold.py` | ★ Klein 流形: exp, log, gamma, proj, Einstein add |
| `klein_nn/layers/fc.py` | ★ KleinFC: Theorem 5.5, 测地超平面 FC |
| `klein_nn/layers/bfc.py` | ★ KleinBFC: Theorem 5.6, Busemann FC |
| `klein_nn/layers/pv_fc.py` | ★ PVFC: PV 空间测地 FC |
| `klein_nn/layers/pv_bfc.py` | ★ PVBFC: PV Busemann FC |
| `klein_nn/layers/mlr.py` | ★ KleinMLR (Phase 1) |
| `klein_nn/layers/bmlr.py` | ★ KleinBMLR (Phase 1) |
| `GyroBN-main/Geometry/constantcurvature/klein.py` | ★ KleinBall manifold (GyroBN 接口) |
| `GyroBN-main/Geometry/constantcurvature/pv.py` | ★ PVSpace manifold |
| `GyroBN-main/conf/nnet/KleinFC.yaml` | ★ Klein FC 配置 |
| `GyroBN-main/conf/nnet/KleinBFC.yaml` | ★ Klein BFC 配置 |
| `GyroBN-main/conf/nnet/PVFC.yaml` | ★ PV FC 配置 |
| `GyroBN-main/conf/nnet/PVBFC.yaml` | ★ PV BFC 配置 |

### 修改的文件

| 文件 | 改了什么 |
|------|---------|
| `GyroBN-main/Geometry/constantcurvature/__init__.py` | 注册 KleinBall, PVSpace |
| `GyroBN-main/RieNets/hnns/models/encoders.py` | 添加 HNN_BFC, HNN_KleinFC, HNN_PVFC, HNN_LorentzFC encoder |
| `GyroBN-main/RieNets/hnns/utils/train_utils.py` | fcntl 兼容 Windows; CUDA fallback CPU |
| `HBNN-main/lib/bnn/BFC.py` | 添加 Klein/PV metric 支持 |
| `HBNN-main/lib/bnn/BMLR.py` | 添加 Klein/PV Busemann logits |
| `HBNN-main/lib/bnn/Auxlayers.py` | 添加 Klein/PV wrapper |
| `HBNN-main/lib/geoopt/optim/rlinesearch.py` | scipy 兼容性修复 |

### 没改的文件 (GyroBN 原始)

| 文件 | 说明 |
|------|------|
| `GyroBNH.py` | 入口, 不动 |
| `train_kfold.py` | 训练循环, 不动 |
| `base_models.py` | LPModel/NCModel, 不动 |
| `hyp_layers.py` | HypLinear/HypAct, 不动 |
| `data_utils.py` | 数据加载, 不动 |

---

## 3. 核心数学：3 种 FC 层的对比

### Möbius FC (HypLinear, 已有)
```
y = mobius_matvec(W, x)    # Möbius 矩阵乘法
y = proj(y)                # 投影回球
y = mobius_add(y, bias)    # Möbius 加法加偏置
# 激活在 HypAct 中: log→relu→exp
```

### Klein FC (我们的, Theorem 5.5)
```
v_k = (||z_k||/√(-K)) · arcsinh(...)   # 测地超平面有符号距离
y_k = sinh(√(-K)·v_k) / [√(-K)·√(1 + Σ sinh²)]  # sinh 归一化 → Klein 球
# 激活在层内: relu/tanh 作用在 v_k 上
```

### Klein BFC (我们的, Theorem 5.6)
```
u_k = -α_k · B^{v_k}(x) + b_k         # Busemann 函数 (horosphere 距离)
y_k = sinh(√(-K)·u_k) / [√(-K)·√(1 + Σ sinh²)]  # 同样的 sinh 归一化
# Busemann 函数: B(x) = (1/√(-K)) · log(√(1+K||x||²) - √(-K)<x,v>)
# 计算更快 (不需要 arcsinh), 但决策边界是 horosphere 而非测地超平面
```

---

## 4. Manifold 接口

GyroBN 的所有 manifold 都实现同一套接口:

```python
class Manifold:
    def sqdist(self, p1, p2, c)    # 两点间平方距离
    def proj(self, x, c)           # 投影到流形
    def expmap0(self, u, c)        # 原点处指数映射 (切空间 → 流形)
    def logmap0(self, p, c)        # 原点处对数映射 (流形 → 切空间)
    def mobius_add(self, x, y, c)  # 加法 (Möbius/Einstein)
    def mobius_matvec(self, m, x, c)  # 矩阵乘法
    def proj_tan0(self, u, c)      # 投影到原点切空间
```

我们实现了 `KleinBall` 和 `PVSpace` 两个 manifold, 接口完全一致。
所以只要换 yaml 里的 `manifold: KleinBall`, HypLinear 就自动用 Klein 的 mobius_matvec。

---

## 5. 怎么加一个新模型

假设你要加一个 "FooFC" 层:

1. **写层**: `klein_nn/layers/foo_fc.py` — 实现 `FooFC(nn.Module)` 的 `forward(x) → y`
2. **写 encoder**: 在 `encoders.py` 添加 `HNN_FooFC(Encoder)` 类
3. **写配置**: `conf/nnet/FooFC.yaml` — 设 `model: HNN_FooFC, manifold: KleinBall`
4. **跑实验**: `python GyroBNH.py nnet=FooFC dataset.dataset=airport ...`

---

## 6. 调试技巧

### 检查嵌入分布
在 `encoders.py` 的 `encode()` 末尾加:
```python
print(f"output norm: {output.norm(dim=-1).min():.4f} ~ {output.norm(dim=-1).max():.4f}")
```
正常范围: HNN 输出 0.3-0.7, BFC 输出 0.01-0.03 (这就是 BFC 距离尺度问题的来源)

### 检查距离范围
在 `base_models.py` 的 `decode()` 里加:
```python
print(f"sqdist: {sqdist.min():.4f} ~ {sqdist.max():.4f}")
```
FermiDirac 的 r=2, 所以 sqdist 应该在 0.5-5 范围内才能区分正负边

### 快速测试
```bash
# 100 epochs 快速验证
python GyroBNH.py dataset.dataset=disease_lp dataset.path=../data \
  nnet=KleinFC fit.epochs=100 fit.double_precision=0
```
