# IDKC

本项目是 **IDKC（Isolation Distributional Kernel Clustering）** 的单数据集实验实现，参考：

- 论文：Ye Zhu, Kai Ming Ting. *Kernel-based clustering via Isolation Distributional Kernel*. Information Systems 117 (2023) 102212.
- 作者实现：<https://github.com/zhuye88/IDKC>

项目采用 **MATLAB + Python** 混合结构：MATLAB 负责核特征、搜参、聚类、评价和结果发布；Python 只负责数据格式转换、绘图和绘图前的一致性校验。

---

## 一、算法原理

IDKC 的核心思想是：用 **Isolation Kernel** 把每个样本映射成稀疏二值特征，再用簇内样本的平均核特征表示该簇的分布，并从若干种子点开始逐步吸收与簇分布相似的样本。

### 1.1 Isolation Kernel 特征

对数据集做 `t` 次随机划分。每次从数据中不放回抽取 `ψ` 个中心点：

1. 对每个中心点，计算它到其余中心点的最近邻距离，作为该中心的隔离球半径。
2. 对每个样本，找到距离最近的中心点。
3. 如果样本落在该中心的隔离球内，则对应位置记为 1；否则该次特征全为 0。

重复 `t` 次后，样本 `x` 得到稀疏二值向量 `φ(x)`，特征维度最多为 `tψ`。样本间的 Isolation Kernel 定义为：

```text
K(x, y) = φ(x)ᵀ φ(y) / t
```

核特征依赖输入数据的分布，因此每次换数据或换抽样都需要重新构建。实现见 `IDKC/iNNEspace.m`。

### 1.2 用分布表示簇

簇 `C` 的分布表示为其成员核特征的均值：

```text
Φ(C) = Σ_{x∈C} φ(x) / |C|
```

样本 `x` 与簇 `C` 的相似度为：

```text
s(x, C) = φ(x)ᵀ Φ(C) / t
```

实现中省略了统一的 `1/t` 缩放，这不影响簇间比较和原有阈值行为。簇表示会随着新样本加入而增量更新。

### 1.3 选择 k 个种子

在 `s` 个候选样本上执行：

1. 计算每个候选点与候选整体均值特征的相似度，作为密度 `Density`。
2. 在核特征欧氏距离下取 `q` 个近邻，统计自身密度严格高于近邻的次数，得到局部对比度 `LC`。
3. 将候选点按密度降序排列，对每个点计算它到排在其前面（密度更高或排序更靠前）的点的最小距离，作为分离度 `MinDist`。
4. 对 `Density` 和 `MinDist` 分别做 `tiedrank`，取秩乘积最大的 `k` 个点作为初始种子。

实现见 `IDKC/FindMode.m` 和 `IDKC/getLC.m`。候选点距离矩阵规模为 `s×s`，因此 `s` 直接决定该阶段的内存和耗时。

### 1.4 簇增长

每一轮：

1. 计算所有未分配点到各簇平均特征的相似度，取最近簇。
2. 只把相似度超过当前阈值的点加入对应簇。
3. 阈值从当前最大相似度开始，每轮乘以衰减系数 `v`。
4. 当剩余点的相似度之和为 0，或阈值低于 `1e-5` 时停止增长。

### 1.5 Refinement

簇增长后执行最多 100 轮 refinement：

- 用当前簇的均值特征重新分配所有样本；
- 如果本轮改变归属的样本数小于 `ceil(0.01n)`，或更新会使簇数少于 `k`，则停止且不应用该轮更新。

因此最终标签中仍可能出现 `0`，表示该样本没有被分配到任何簇。实现见 `IDKC/DKC.m`。

> 说明：以上描述以本仓库保留的作者实现为准，包括核特征距离、秩乘积选种子、阈值衰减和 refinement 停止条件；不等同于对论文伪代码的逐行重写。

---

## 二、项目逻辑

### 2.1 总体结构

```text
MATLAB main.m
  ├─ Python prepare：读取 MAT/PKL、校验数据、生成临时 input.mat
  ├─ MATLAB search：构建核特征、遍历 (ψ, seed, q)、计算指标、选择最佳结果
  ├─ Python plot：校验保存的标签与指标、匈牙利对齐、降维、画图
  └─ MATLAB publish：原子发布 results/<dataset>/，更新 results/idkc.csv
```

运行过程中的中间文件都写入 `.local/runs/`；只有搜索、标签校验和绘图全部成功后，结果才会替换到 `results/`。

### 2.2 环境要求

- **MATLAB**：核心实现使用 `datasample`、`pdist2`、`tiedrank` 和 `gather`，需要 Statistics and Machine Learning Toolbox。
- **Python 3.10+**：建议使用与 MATLAB `config.python` 相同的解释器，安装辅助依赖：

```bash
python -m pip install -r requirements.txt
```

Python 路径默认读取环境变量 `IDKC_PYTHON`，未设置时使用 `python`。如果本地依赖目录 `.local/plot_dependencies/` 存在，辅助脚本会优先加载它；也可以通过 `IDKC_PLOT_DEPENDENCIES` 指定其他目录。

数据文件不随仓库提供。默认数据目录为项目上两级目录下的 `data/DC`，建议运行时显式指定：

```matlab
config = struct( ...
    'data_root', '/path/to/data/DC', ...
    'python',    '/path/to/python');
```

### 2.3 运行命令

#### 运行一个数据集

在 MATLAB 中进入项目根目录：

```matlab
config = struct( ...
    'data_root', '/path/to/data/DC', ...
    'python',    '/path/to/python');
main('AC', config);
```

不传配置时使用默认值：

```matlab
main('AC');
```

不传数据集名称时默认运行 `dlpfc_151507`：

```matlab
main;
```

非交互运行：

```bash
matlab -batch "main('AC', struct('data_root','/path/to/data/DC','python','/path/to/python'))"
```

#### 运行多个数据集

建议顺序执行，避免多个进程同时写共享汇总表：

```matlab
datasets = {'AC', '4C', 'RingG'};
config = struct('data_root', '/path/to/data/DC', 'python', '/path/to/python');
for i = 1:numel(datasets)
    main(datasets{i}, config);
end
```

#### 只重画已保存结果

不重新搜参，但仍需要原始输入来核对样本顺序：

```matlab
main('AC', struct('plot_only', true, 'data_root', '/path/to/data/DC'));
```

#### 使用自定义数据

默认 MAT 字段为 `data` 和 `class`：

```matlab
config = struct( ...
    'input_file', '/path/to/my_data.mat', ...
    'python',     '/path/to/python');
main('my_data', config);
```

自定义字段：

```matlab
config = struct( ...
    'input_file', '/path/to/my_data.mat', ...
    'x_key', 'X', ...
    'y_key', 'Y', ...
    'pos_key', 'locations', ...
    'python', '/path/to/python');
main('my_data', config);
```

`pos_key` 只用于绘图，不参与 IDKC 聚类；数据集名称同时作为输出目录名和汇总表唯一键。

#### 运行测试

```bash
matlab -batch "addpath('tests'); test_idkc_search(tempname)"
```

### 2.4 完整执行流程

```text
1. prepare
   读取 MAT/PKL
   → 检查特征矩阵、标签长度、有限数值
   → 对非数值标签按排序后的唯一值编码
   → 生成 .local/runs/<dataset>/<run>/input.mat 和 dataset.json

2. search
   读取 input.mat
   → 逐属性 min-max 归一化到 [0,1]，常量属性设为 0.5
   → 对每个 (ψ, seed, q) 构建 Isolation Kernel、选种子、簇增长、refinement
   → 计算 NMI / ARI / macro-F1 / 未分配点数 / 单次耗时
   → 逐层选择最佳 q、最佳 ψ、最佳 seed

3. plot
   读取 best_labels.csv
   → 核对样本顺序、真实标签和 NMI/ARI/macro-F1
   → 执行 KBC 风格匈牙利对齐
   → 全量绘图：二维用原坐标，高维用 t-SNE，空间数据用 locations
   → 生成聚类图、真实标签图和绘图元数据

4. publish
   把 staging 结果移动到 results/<dataset>/
   → 旧目录保留为本次运行中的 previous_result/
   → 以临时文件 + 替换的方式更新 results/idkc.csv
```

如果任一步失败，正式结果不会发布；已经存在的旧结果会尝试恢复。

### 2.5 搜参逻辑

默认参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| ψ | `2,4,6,8,16,24,32,48,64,80,100,200,250,500,750` | 每次 Isolation Kernel 的中心抽样数 |
| seed | `42,3407,4079,2024,0` | 固定五次重复实验 |
| q | `round(n × [0.05, 0.10, ..., 0.50])` | 选种子的近邻数 |
| t | `100` | 随机划分次数 |
| v | `0.9` | 簇增长阈值衰减系数 |
| s | `min(n, 10000)` | 选种子的候选样本数 |
| k | 真实类别数 | 聚类数 |

有效候选：

- `ψ` 只保留 `2 ≤ ψ ≤ n`；
- `q` 先取整、去重、升序，只保留 `1 ≤ q < s`；
- 每个数据集最多 `15 × 5 × 10 = 750` 次运行。

选择顺序：

1. **固定 `(ψ, seed)`，遍历 `q`**：按 NMI 选最佳 `q`，同时保留该次运行的 ARI、macro-F1 和耗时。
2. **固定 `ψ`，汇总五个 seed**：对每个 seed 的最佳 `q` 结果求算术平均。
3. **比较 `ψ`**：按五次平均 NMI 选最佳 `ψ`。
4. **选画图运行**：在最佳 `ψ` 内按单次 NMI 选最佳 seed，用该 seed 的最佳 `q` 标签画图。

补充规则：

- 所有比较使用完整浮点精度，输出 CSV 时才保留四位小数。
- NMI 完全相同时，`q` 保留较小值；`ψ` 和 seed 保留配置顺序中的第一个。
- 每次尝试 `q` 前都执行 `rng(seed, 'twister')`，使同一 `(ψ, seed)` 下的核划分和候选点保持一致。
- 本项目用真实标签确定 `k`，并用真实标签计算 NMI 选参，属于有标签的实验评价口径。
- 汇总中的“最佳结果”是 **最佳 ψ 内的最佳 seed**，不是全网格全局 NMI 最高的 `(ψ, seed, q)`。

可在 `config` 中覆盖主要搜索参数：

```matlab
config = struct( ...
    'psi_values',  [16, 32, 64], ...
    'q_fractions', [0.1, 0.2, 0.3], ...
    't', 100, 'v', 0.9, 'sample_size', 10000);
main('AC', config);
```

### 2.6 评价与汇总

`metrics/clustering_scores.m` 用匈牙利算法做预测簇与真实类别的一对一匹配：

- DKC 标签 `0` 不参与匹配；未分配点和未匹配簇仍保留在分区中，作为独立簇计入 NMI/ARI。
- `AlignedLabel` 中未分配点和未匹配簇记为 `NaN`。
- **NMI**：互信息除以两个分区熵的几何平均。
- **ARI**：调整兰德指数。
- **macro-F1**：每个真实类别等权，未匹配样本计入漏判。
- **std**：最佳 `ψ` 下五个 seed 结果的样本标准差（分母 `n−1`）。
- **time**：单次从加载临时 MAT 到 DKC 返回标签的时间，包含 min-max、核构建、选种子、增长和 refinement，不包含 Python 前处理、评价、绘图和发布。

`results/idkc.csv` 每个数据集一行，列固定为：

```text
dataset,points,dimensions,clusters,nmi,nmi_std,ari,ari_std,f1,f1_std,time,time_std,psi,best_seed,best_q,best_nmi,best_ari,best_f1,best_time
```

前半部分是最佳 `ψ` 下五次运行的均值和标准差；`best_*` 是同一最佳 `ψ` 下最佳 seed 的参数和单次指标。重跑已有数据集会覆盖对应行，新数据集追加，匹配名称时忽略大小写。

### 2.7 绘图规则

绘图与当前 KBC 实现保持一致：

- **全部样本**参与绘图，不抽样；
- 一维/二维特征直接使用原坐标；
- 高维特征使用 t-SNE，`random_state=42`，PCA 初始化，自动学习率，`perplexity=min(30, n−1)`；
- 空间转录组数据使用 `locations`，不跑 t-SNE；
- 绘图前使用与 `../KBC/utils.py` 相同的匈牙利对齐，未匹配簇分配互不冲突的负 dummy 标签；
- 聚类图和真实标签图共用同一套按真实类别排序的颜色，未匹配簇固定灰色；
- 样式为 8×6 英寸、300 dpi、点大小 15、透明度 0.7、等比例坐标、无标题和刻度。

绘图对齐与评价对齐是两套独立结果：绘图使用 KBC 风格对齐，不会改写 `best_labels.csv` 中的评价标签。

`plot_sample_size` 仅为兼容旧配置保留，不再限制绘图点数。

### 2.8 文件作用

```text
main.m                         单数据集完整入口：prepare → search → plot → publish
pipeline_helper.py             MATLAB 内部调用的 Python 助手：数据转换与绘图
requirements.txt               Python 辅助依赖版本

IDKC/
  iNNEspace.m                  Isolation Kernel 稀疏特征
  FindMode.m                   种子选择：密度秩 × 分离度秩
  getLC.m                      局部对比度
  DKC.m                        簇增长与 refinement

experiments/
  run_idkc_search.m            搜参主循环、指标计算、最佳结果选择和 CSV/JSON 导出
  update_idkc_summary.m        更新 results/idkc.csv，保留其他数据集行

metrics/
  clustering_scores.m          匈牙利匹配、NMI/ARI/macro-F1、未分配统计
  hungarian.m                  匈牙利算法实现
  nmi.m / ari.m                聚类指标实现

pipeline/
  __init__.py                  包说明
  data.py                      26 个数据集配置、MAT/PKL 加载、MAT 转换
  files.py                     参数与指标 CSV 读取
  plotting.py                  KBC 风格全量绘图、对齐与标签校验

tests/
  test_idkc_search.m           评价、选参、可复现性、精度导出和汇总回归测试
```

### 2.9 结果目录

`results/` 默认不进入版本控制，只保留每个数据集的子目录和一份共享汇总表：

```text
results/
├── <dataset>/
└── idkc.csv
```

每个数据集目录的主要文件：

| 文件 | 内容 |
|---|---|
| `q_search.csv` | 所有 `(ψ, seed, q)` 候选的指标、耗时和未分配点数 |
| `seed_best.csv` | 每个 `(ψ, seed)` 下 NMI 最高的 `q` |
| `psi_summary.csv` | 每个 `ψ` 的五次平均结果 |
| `best_psi.csv` | 最佳 `ψ` 及其五次平均结果 |
| `best_psi_seeds.csv` | 最佳 `ψ` 下五个 seed 各自的最佳 `q` 和结果 |
| `best_run.csv` | 最佳 `ψ` 下 NMI 最高的单次运行 |
| `best_labels.csv` | 最佳运行的样本序号、真实标签、原始标签和评价对齐标签 |
| `best_psi_labels_seed_*.csv` | 最佳 `ψ` 下各 seed 的标签 |
| `search_statistics.json` | 完整精度的五次结果和最佳单次结果 |
| `search_time.csv` | 完整参数循环耗时 |
| `run_metadata.json` | 原始输入、规模、字段、搜索与绘图配置 |
| `original_space_clustering.jpg` | 最佳运行的聚类图 |
| `original_space_true_labels.jpg` | 对应的真实标签图 |
| `best_run_plot_coordinates.npz` | 全部样本行号和绘图坐标 |
| `best_run_plot_metadata.json` | 绘图参数、代码哈希、库版本和最佳运行信息 |
| `kbc_style_label.csv` | 全部样本的 KBC 风格对齐标签 |

### 2.10 复现注意事项

1. 数据文件需要单独准备，不包含在仓库中。
2. 一次运行只处理一个数据集；多个数据集建议顺序执行。
3. 不要并行写同一个 `results/`，否则共享 `idkc.csv` 可能竞争。
4. 搜索中断不会自动续跑；已完成的中间结果保留在 `.local/runs/`，正式结果不会被部分覆盖。
5. 参数选择只用 NMI；ARI 和 macro-F1 只用于评价。
6. 绘图使用最佳单次运行；汇总中的 `nmi/ari/f1` 是五次平均，`best_*` 才对应绘图运行。
7. 重画历史结果时如果缺少 `search_statistics.json`，均值沿用 `best_psi.csv`，std 从保存的四位小数 seed 结果计算。
