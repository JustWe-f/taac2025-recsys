## 基于多模态序列建模的广告推荐系统

基于 TAAC2025（腾讯广告算法大赛）百万级真实广告数据，构建 SASRec 序列推荐系统。

### 项目背景

TAAC2025 是腾讯广告举办的算法大赛，任务是根据用户的全模态行为序列，预测用户下一个会点击的广告创意。本项目使用大赛开源的 TencentGR-1M 数据集（百万级样本），在官方 SASRec baseline 基础上进行多项优化。

### 技术栈

- 框架：PyTorch 2.0+（Anaconda 环境）
- 模型：SASRec → 轻量 HSTU（渐进式优化）
- 数据：TAAC2025 TencentGR-1M（Parquet 格式）
- 检索：Faiss ANN / PyTorch 批量 Top-K

### 改进点

1. **流行度加权负采样**：替代均匀随机负采样，基于 item 频次构建 CDF 分布
2. **特征工程**：稀疏特征交叉 + 点击统计特征（log2 分桶）
3. **轻量 HSTU 模型升级**：相对时间偏置 + RoPE 旋转位置编码 + 门控注意力

### 项目结构

```
taac2025-recsys/
├── model.py              # 模型定义（SASRec / HSTU）
├── dataset.py            # 数据加载与特征处理
├── main.py               # 训练入口
├── infer.py              # 推理与 ANN 检索
├── eval.py               # 评估脚本
├── requirements.txt      # 依赖
├── data/                 # 数据集（不入库）
├── notebooks/            # 探索性分析
├── experiments/          # 实验日志和模型
│   ├── logs/
│   └── checkpoints/
├── scripts/              # 辅助脚本
└── reference/            # 官方 baseline 参考代码
```

### 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 下载数据集（需要 HuggingFace datasets + pyarrow）
# 将数据放入 data/ 目录

# 训练
python main.py --data_path ./data --device cuda

# 推理
python infer.py --data_path ./data --device cuda
```

### GitHub 仓库

https://github.com/JustWe-f/taac2025-recsys
