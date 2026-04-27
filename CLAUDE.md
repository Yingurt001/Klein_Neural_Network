# Phase 2: Klein/PV FC 实验代码

## 你是谁、要做什么

你是研究助理，帮用户在 GyroBN 框架上跑 Klein/PV 空间的 FC 层对比实验。
核心任务：对标 HBNN Table 7 (link prediction) 和 PV Table 5 (node classification)。

## 项目结构

```
code/
├── klein_nn/                 ← 我们的核心贡献 (Klein/PV 层)
├── GyroBN-main/              ← 训练框架 (Chen et al., ICLR 2025, 已修改)
├── HBNN-main/lib/            ← BFC/BMLR 层 (Chen et al., CVPR 2026)
├── PoinnCARE-main/           ← 酶分类 (Track B, 备用)
└── test_layers.py            ← 单元测试
```

## 关键文件（按优先级）

1. `klein_nn/layers/fc.py` — KleinFC (Theorem 5.5)
2. `klein_nn/layers/bfc.py` — KleinBFC (Theorem 5.6)
3. `klein_nn/manifold.py` — Klein 流形完整几何工具箱
4. `GyroBN-main/RieNets/hnns/models/encoders.py` — 所有 encoder (HNN, HNN_BFC, HNN_KleinFC, HNN_PVFC, HNN_LorentzFC)
5. `GyroBN-main/Geometry/constantcurvature/klein.py` — KleinBall manifold (GyroBN 接口)
6. `GyroBN-main/Geometry/constantcurvature/pv.py` — PVSpace manifold
7. `GyroBN-main/conf/nnet/` — 各模型的 Hydra 配置

## 运行实验

```bash
cd GyroBN-main

# Link Prediction (Track A)
python GyroBNH.py dataset.dataset=disease_lp dataset.path=/path/to/data \
  nnet=KleinFC nnet.BN_param.is_bn=False nnet.optimizer.weight_decay=1e-3 \
  nnet.model.act=relu fit.epochs=5000 fit.double_precision=1

# 模型选项: HNN, BFC_P, BFC_L, LorentzFC, KleinFC, KleinBFC, PVFC, PVBFC
# 数据集: disease_lp, airport, pubmed, cora
# 激活: relu (非BFC), null/tanh (BFC: Disease/PubMed=null, Airport/Cora=tanh)
# Cora weight_decay=0, 其他=1e-3
```

## 数据

从 HGCN 下载: `git clone https://github.com/HazyResearch/hgcn.git`
将 `hgcn/data/` 放到 `code/data/`

## 曲率约定

| 代码 | 参数 | 值 | 关系 |
|------|------|------|------|
| GyroBN | c | 1.0 | c = -K |
| klein_nn | K | -1.0 | K = -c |
| HBNN BFC | K | -1.0 | 同 klein_nn |
| HBNN CustomLorentz | k | 1.0 | k = 1/c |

## 已知问题

1. BFC 在 link prediction 上距离尺度和 FermiDirac r=2 不匹配 (HBNN 训练代码未公开)
2. PV FC 在高维特征 (PubMed/Cora) 上需要 identity 激活, 不能用 relu
3. Ada HPC 上 pip install 过的 nvidia 包需要清理才能用 GPU

## 用中文沟通
