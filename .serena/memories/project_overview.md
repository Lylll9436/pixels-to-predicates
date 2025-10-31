# 项目概述

## 项目目的
这是一个关于**场景图结构建模用于图像感知偏好预测**的研究项目。项目使用图神经网络（GraphMAE）对场景图进行编码，然后通过 Bradley-Terry 模型进行 pairwise 比较学习，以预测人们对城市街景图像的感知偏好（如安全感、活力感、美感等）。

## 技术栈
- **Python 3** - 主要编程语言
- **PyTorch** - 深度学习框架
- **torch-geometric** - 图神经网络库（用于 GraphMAE）
- **sentence-transformers** - Sentence-BERT 用于文本编码
- **scikit-learn** - 机器学习评估指标
- **matplotlib/seaborn** - 数据可视化
- **PIL (Pillow)** - 图像处理
- **numpy, pandas** - 数据科学库
- **tqdm** - 进度条
- **requests** - HTTP 请求（用于 API 调用）

## 项目结构
- **主要脚本**：按数字顺序排列（01-10）
  - `01_describe_pic.py` - 图像描述生成（使用 Gemini API）
  - `02_merge_data.py` - 数据合并脚本
  - `03_build_scene_graphs.py` - 场景图构建
  - `04_convert_to_pytorch.py` - 转换为 PyTorch 格式
  - `05_graph_vae.py` - GraphMAE 预训练
  - `06_comparison_trainer.py` - Bradley-Terry 比较模型训练
  - `07_evaluate_and_visualize.py` - 评估和可视化
  - `08_radar.py` - 雷达图可视化
  - `09_reasoning.py` - 感知推理脚本
  - `10_rel_visual.py` - 关系可视化
- **数据目录**：`data/`
- **输出目录**：`output/`
- **结果目录**：`result/`
- **日志目录**：`logs/`
- **配置文件**：`config/env`

## 数据处理流程
1. 图像描述生成 → 2. 数据合并 → 3. 场景图构建 → 4. PyTorch 格式转换 → 5. GraphMAE 预训练 → 6. Bradley-Terry 比较训练 → 7. 评估与可视化