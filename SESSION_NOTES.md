## 项目会话记录

> 本文档记录项目启动过程中的关键决策、环境配置和待办事项，方便跨会话查阅。

---

### 基本信息

- **项目名称：** taac2025-recsys（基于多模态序列建模的广告推荐系统）
- **GitHub 仓库：** https://github.com/JustWe-f/taac2025-recsys
- **数据集：** TAAC2025 TencentGR-1M（HuggingFace 公开，百万级样本）
- **基线代码：** [官方 baseline](https://github.com/TencentAdvertisingAlgorithmCompetition/baseline_2025)（SASRec 架构）
- **参考方案：** [O_o 团队开源代码](https://github.com/salmon1802/O_o)（初赛第 14 名 / Top 1%）
- **预计周期：** 15-20 天
- **GPU：** 本地 RTX 3060 Laptop GPU

---

### 环境配置记录

| 项目 | 状态 | 备注 |
|------|------|------|
| PyCharm 解释器 | 已配置 | Anaconda: `C:\Users\Lenovo\anaconda3` |
| Git | 已配置 | v2.52.0，用户名 JustWe-f，邮箱 1767252278@qq.com |
| Git Credential Manager | 已配置 | 存储了 GitHub fine-grained PAT |
| gh CLI | 已安装并认证 | v2.96.0，通过浏览器 OAuth 登录 JustWe-f |
| Conda 环境 | 已创建 | `D:\conda-envs\taac2025-recsys`（Python 3.12） |
| PyTorch | 已安装 | 2.5.1+cu121，CUDA 可用 |
| pip 缓存 | 已配置 | `D:\pip-cache\` |
| conda 包缓存 | 已配置 | `D:\conda-pkgs\` |

**重要路径修正：** 用户桌面实际在 `D:\Desktop`（非 `C:\Users\Lenovo\Desktop`）。
- 项目路径：`D:\Desktop\GR\taac2025-recsys`
- 数据路径：`D:\taac2025-data\`

**Git proxy 配置（需要时）：**
```bash
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897
```

---

### 已安装的依赖

| 包 | 版本 | 用途 |
|----|------|------|
| torch | 2.5.1+cu121 | 深度学习框架 |
| numpy | 2.4.4 | 数值计算 |
| pyarrow | 25.0.0 | Parquet 数据加载 |
| tqdm | 4.68.4 | 进度条 |
| tensorboard | 2.21.0 | 训练日志可视化 |
| scikit-learn | 1.9.0 | 评估指标 |
| faiss-cpu | 1.14.3 | ANN 检索（推理） |
| huggingface_hub | 1.23.0 | 数据集下载 |

---

### 代码修改记录

1. **main.py**：数据路径改为 `D:/taac2025-data`，num_workers 降低（train=4, valid=2）
2. **dataset.py**：修复 PyArrow null 值问题，`fill_null(0).to_numpy(zero_copy_only=False)`

---

### 技术路线决策

**核心策略：以官方 baseline 为骨架，逐步添加 2-3 个有针对性的改进。**

**选定的 3 个改进点（按优先级排序）：**

1. **流行度加权负采样**（改进 A）
   - 统计 item 频次 -> 构建 CDF -> np.searchsorted O(log N) 采样
   - 代码量小（约50行），面试能讲 word2vec 负采样思想的类比

2. **特征工程**（改进 B）
   - 稀疏特征交叉（如行业 x 子行业）
   - 点击统计特征（log2 分桶，16 个桶）

3. **轻量 HSTU 模型升级**（改进 C）
   - 从 SASRec 1层1头升级到 2层4头 HSTU
   - 关键组件：相对时间偏置 + RoPE + 门控注意力

---

### 待办事项

- [x] 创建项目目录和 Git 仓库
- [x] 配置 Git/GitHub 认证
- [x] 创建 D 盘 conda 环境
- [x] 安装 PyTorch CUDA 12.1
- [x] 安装全部 Python 依赖
- [x] 下载数据集到 D:/taac2025-data
- [x] 修复 dataset.py PyArrow 兼容问题
- [ ] 跑通 baseline 训练（1 epoch，验证流程）
- [ ] 跑通推理和评估
- [ ] 精读 model.py 和 dataset.py，写理解笔记
- [ ] 改进 A：流行度加权负采样
- [ ] 改进 B：特征工程
- [ ] 改进 C：轻量 HSTU
