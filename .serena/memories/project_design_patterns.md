# 项目设计模式和架构

## 核心设计模式

### 1. 流水线模式
项目采用顺序执行的流水线架构，每个脚本对应一个处理阶段：
- 数据预处理 → 特征提取 → 模型训练 → 评估 → 可视化

### 2. 模块化设计
- 每个脚本专注于单一职责
- 使用类封装功能（如 `GraphMAE`, `SentenceBERTEncoder`）
- 使用 dataclass 定义数据结构

### 3. 配置管理
- 配置文件：`config/env` 存储 API 密钥等配置
- 路径配置：在脚本中定义路径常量
- 模型配置：使用字典存储模型参数

## 关键架构组件

### 场景图构建
- **Entity** - 实体类（包含 id, class_name, attributes）
- **Relation** - 关系类（包含 subject, predicate, object）
- **SentenceBERTEncoder** - 文本编码器

### 图神经网络
- **GraphMAE** - 掩码图自编码器
- 使用 `torch_geometric` 进行图数据处理
- 支持 GPU 加速

### 比较学习
- **Bradley-Terry 模型** - 用于 pairwise 比较
- 数据集类：`ComparisonDataset`
- 支持多种基线模型（ResNet, ViT, CLIP 等）

### 评估系统
- 多种评估指标：Accuracy, AUC, Precision, Recall, F1 等
- 可视化：ROC 曲线、PR 曲线、混淆矩阵、雷达图

## 数据处理流程
1. **图像 → 描述**：使用 Gemini API 生成图像描述
2. **描述 → 场景图**：解析描述为实体-关系三元组
3. **场景图 → 图嵌入**：使用 Sentence-BERT 编码节点和边
4. **图嵌入 → 图级表示**：GraphMAE 预训练
5. **图级表示 → 偏好预测**：Bradley-Terry 模型学习

## 输出管理
- 中间结果保存在 `output/` 目录的不同子目录
- 最终结果保存在 `result/` 目录
- 训练日志保存在 `logs/` 目录
- 使用 JSON、CSV、PNG、PTH 等格式