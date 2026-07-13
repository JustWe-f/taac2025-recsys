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
- **GPU 方案：** 云 GPU 租用（推荐 AutoDL，RTX 4090 或 A100）

---

### 环境配置记录

| 项目 | 状态 | 备注 |
|------|------|------|
| PyCharm 解释器 | 已配置 | Anaconda: `C:\Users\Lenovo\anaconda3` |
| Git | 已配置 | v2.52.0，用户名 JustWe-f，邮箱 1767252278@qq.com |
| Git Credential Manager | 已配置 | 存储了 GitHub fine-grained PAT（github_pat_ 开头） |
| gh CLI | 已安装并认证 | v2.96.0，通过浏览器 OAuth 登录 JustWe-f |

**重要提醒：** Git proxy 配置曾被临时设置（端口 7897），但后来移除以直连 GitHub。如遇 GitHub 访问问题，可重新设置代理：
```bash
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897
```

---

### 技术路线决策

**核心策略：以官方 baseline 为骨架，逐步添加 2-3 个有针对性的改进。**

不直接复刻 O_o 代码的理由：
1. O_o 代码改动幅度大（HSTU 8层8头、InfoNCE 对比学习、SSL 增强、特征交叉等），对 PyTorch 基础偏弱的情况来说，理解成本高
2. 面试时如果讲不清每个模块的原理和 trade-off，反而是减分项
3. 从 baseline 出发逐步改进，每个决策都能讲出"为什么"，面试更有说服力

**选定的 3 个改进点（按优先级排序）：**

1. **流行度加权负采样**（Day 6-8）
   - 统计 item 频次 → 构建 CDF → np.searchsorted O(log N) 采样
   - 代码量小（约50行），面试能讲 word2vec 负采样思想的类比
   - 参考 O_o 代码中的 `build_neg_sampling_cdf`

2. **特征工程**（Day 9-12）
   - 稀疏特征交叉（如行业 × 子行业）
   - 点击统计特征（log2 分桶，16 个桶）
   - 工业推荐系统标准操作，面试必问

3. **轻量 HSTU 模型升级**（Day 13-17）
   - 从 SASRec 的 1 层 1 头升级到 2 层 4 头 HSTU
   - 关键组件：相对时间偏置（128 可学习桶）+ RoPE 旋转位置编码 + 门控注意力
   - 最有面试价值的改进，能画架构对比图

**明确不做的事：**
- RQ-KMeans（语义 ID 生成）：独立模块，不影响主模型训练，学习成本偏高
- 完整复刻 O_o 的 8 层 8 头 HSTU：计算资源和学习时间都不够
- InfoNCE 对比学习损失：作为可选加分项，非必须

---

### 项目目录结构

```
taac2025-recsys/
├── .gitignore              # 忽略数据/模型/日志等大文件
├── README.md               # 项目说明
├── SESSION_NOTES.md         # 本文档（会话记录）
├── model.py                # 模型代码（从 baseline 复制）
├── dataset.py              # 数据加载（从 baseline 复制）
├── main.py                 # 训练入口（从 baseline 复制）
├── infer.py                # 推理脚本（从 baseline 复制）
├── eval.py                 # 评估脚本（从 baseline 复制）
├── requirements.txt        # Python 依赖
├── data/                   # 数据集（不入 Git）
├── notebooks/              # 探索性分析
├── experiments/
│   ├── logs/               # TensorBoard 日志
│   └── checkpoints/        # 模型权重
├── scripts/                # 辅助脚本
└── reference/
    └── baseline_2025/      # 官方 baseline 原版代码（shallow clone，不入 Git）
```

Git 初始 commit：`bfd468c`（2026-07-10）

---

### 数据集说明

**TAAC2025 TencentGR-1M** 是腾讯广告与港中文 Prof. King 团队联合发布的开源数据集。

**数据格式：** Parquet（非 HuggingFace Dataset 格式），需用 baseline 自带的 `dataset.py` 加载。

**数据目录结构：**

| 目录 | 内容 | 说明 |
|------|------|------|
| seq/ | 用户行为序列 | user_id + seq(item_id, action_type, timestamp) |
| item_feat/ | 广告特征表 | 13 个稀疏特征（ID: 100,101,102,112,114-122） |
| user_feat/ | 用户特征表 | 4 个稀疏 + 4 个多值特征（ID: 103-110） |
| mm_emb/ | 多模态 Embedding | 特征 81(32维) 到 86(4096维) |
| indexer.pkl | ID 映射表 | item/user/feature 的重编号 |
| candidate/ | 候选广告 | 推理时使用 |

**重要提醒：** 不要用 `load_dataset("TAAC2025/TencentGR-1M", "candidate")`，直接用 baseline 的 Parquet 加载器。

---

### 15-20 天时间线

| 阶段 | 天数 | 内容 |
|------|------|------|
| 环境搭建 + 理解 baseline | Day 1-5 | 租 GPU、搭环境、下载数据、跑通训练/推理、精读代码 |
| 改进 A：负采样升级 | Day 6-8 | 流行度加权负采样 + 消融实验 |
| 改进 B：特征工程 | Day 9-12 | 特征交叉 + 点击统计 + 消融实验 |
| 改进 C：HSTU 升级 | Day 13-17 | 轻量 HSTU 实现 + 训练 + 消融实验 |
| 收尾 + 文档 | Day 18-20 | 全链路测试、写 README、准备面试问答 |

---

### 云 GPU 使用建议

- **AutoDL**：RTX 4090 约 2 元/时，A100 约 5 元/时
- 日常看代码用 CPU 实例（0.5 元/时），训练时再开 GPU
- 用 `nohup` 或 `tmux` 后台训练，断开 SSH 不中断
- 1M 数据单轮训练约 2-4 小时
- 预估总 GPU 费用：60-150 元

---

### 待办事项（下一步）

- [ ] 租云 GPU 实例（AutoDL）
- [ ] 在 PyCharm 中配置 conda 环境（C:\Users\Lenovo\anaconda3）
- [ ] 安装 requirements.txt 中的依赖
- [ ] 下载 TAAC2025 TencentGR-1M 数据集到 data/
- [ ] 修改 main.py 中的硬编码路径
- [ ] 跑通 baseline 训练（1 epoch，验证流程）
- [ ] 跑通推理和评估
- [ ] 精读 model.py 和 dataset.py，写理解笔记
