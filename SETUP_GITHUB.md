# GitHub Repo 设置指引 (一次性, manual)

> **本地状态**: `Phase 2/code/` 已 `git init`, 已 stage 159 个文件 (无大数据集 / 无 1.4GB PoinnCARE 数据)。
> **目标**: 把核心代码 push 到 GitHub, 让 paper abstract 的 `\url{...}` 链接生效。

---

## Step 1: 在 GitHub 创建 empty repo (你做)

1. 打开 https://github.com/new
2. **Repository name**: 推荐 `KleinNN-pp` (Klein NN++) 或者你喜欢的别的名字 (e.g. `KleinNN`, `klein-hyperbolic-nn`)
3. **Description**: `Klein neural networks: closed-form Einstein-midpoint primitives for hyperbolic deep learning`
4. **Public** (paper code 通常公开)
5. **不要勾** "Add a README", "Add .gitignore", "Choose license" — 我们本地都已经有了
6. 点 **Create repository**
7. **复制** 出现的 SSH 或 HTTPS URL, 例如 `git@github.com:Yingurt/KleinNN-pp.git`

---

## Step 2: 本地 commit + push (3 行命令)

打开终端, cd 到 code 目录, 跑下面三行 (替换 `<URL>` 为 Step 1 复制的 URL):

```bash
cd "/Users/zhangying/Personal/Andrea/Academic 学术/University of Nottingham 诺丁汉大学/Spring 2026/Hyperbolic双曲神经网络/Week 11/Phase 2/code"

# 1. First commit (内容已 staged)
git commit -m "Initial commit: Klein neural network framework

- klein_nn/: Klein manifold + 6 layers (FC/BFC/MLR/BMLR + PV variants)
- bn_microbench.py: BN ablation benchmark
- TrackC_Genome/: genome classification (TEB benchmark)
- scripts/: experiment launchers
- README.md: usage + reproducibility instructions
"

# 2. 加 remote (替换 URL)
git remote add origin git@github.com:<你的用户名>/<repo名>.git

# 3. Push
git push -u origin main
```

如果 `git push` 报 ``error: src refspec main does not match any``, 是因为分支名是 `master` 不是 `main`, 改用:
```bash
git branch -M main
git push -u origin main
```

---

## Step 3: 更新 paper 里的 GitHub URL

`main.tex` 里 abstract 末尾我用了占位符:
```latex
Code: \url{https://github.com/Yingurt/KleinNN-pp}.
```

如果你的实际 GitHub username 或 repo name 不一样, 替换成正确 URL:

```bash
TEX="Week 11/notes/neurips_template/2026_NeurIPS26_KNNpp/main.tex"
sed -i '' 's|github.com/Yingurt/KleinNN-pp|github.com/<你的用户名>/<repo名>|' "$TEX"
```

---

## 当前 staged 内容 (会被 push 上去的)

| 类型 | 内容 | 大小 |
|------|------|:----:|
| 核心代码 | `klein_nn/` (6 layers + manifold) | 184 KB |
| Benchmark | `bn_microbench.py`, `test_*.py` | ~30 KB |
| Genome | `TrackC_Genome/` (TEB code, 不含数据) | ~3 MB |
| Scripts | `scripts/` | 12 KB |
| Docs | `README.md`, `MODIFICATIONS.md`, `CODE_GUIDE.md`, `requirements.txt` | ~30 KB |
| **总计** | | **~3.5 MB** (健康范围, 不会卡) |

## 不会 push 的东西 (.gitignore 排除)

- ❌ `data/`, `gene_data/` (~70 MB 数据集)
- ❌ `PoinnCARE-main/` (1.4 GB, 别人的 repo + 大数据)
- ❌ `HBNN-main/`, `GyroBN-main/` (别人的 repo, 我们 fork 改了)
- ❌ `__pycache__/`, `*.pyc` (编译产物)
- ❌ `*.pt`, `*.ckpt`, `runs/`, `logs/` (训练 artifacts)

---

## (可选) Step 4: GitHub repo 完善

push 之后, 在 GitHub repo 页面:
1. **About** (右上角齿轮): 加 description + topics (`hyperbolic-neural-networks`, `klein-model`, `pytorch`, `geometric-deep-learning`)
2. **License**: 推荐 MIT 或 Apache 2.0 (paper code 通常 MIT). 在 `Code > Add file > Create new file > LICENSE`, GitHub 会自动 template
3. **Citation**: 在 GitHub repo 主页可以加一个 `CITATION.cff` 文件让别人引用

## 后续 update 流程

之后改了代码, 标准 git workflow:
```bash
cd "Week 11/Phase 2/code"
git add <改动的文件>
git commit -m "短描述"
git push
```

---

## 处理 fork repos (HBNN/GyroBN/PoinnCARE) 的我们的 modifications

这 3 个 fork repo 现在被 `.gitignore` 排除了。如果以后想让审稿人能完全复现, 有 3 个方案:

### 方案 A: 不管 (推荐, paper code release 标准做法)
- 在主 README 写明 ``依赖 GyroBN/HBNN/PoinnCARE 三个 framework, 我们提供 patches``
- 用户 follow README 自己 clone 那 3 个 repo, 然后 apply 我们的 patches

### 方案 B: 把它们也 push (要分别 fork)
1. 在 GitHub fork 各自 upstream repo (e.g. fork `GitZH-Chen/HBNN`)
2. 把本地 `HBNN-main/` 链到 fork (`git remote add origin <fork-url>`)
3. push 我们的修改
4. 主 repo 用 `git submodule add <fork-url> HBNN/` 引用

### 方案 C: 整个 push 但要交 license / 大文件审计 (不推荐)

**默认走方案 A**, 简单干净。如果审稿人特别要 reproducibility, 再做 B。
