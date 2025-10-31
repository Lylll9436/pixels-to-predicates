# 建议命令

## 环境设置
- Python 版本：Python 3（推荐使用虚拟环境）
- 系统：Windows

## 主要执行命令（按流程顺序）

### 1. 图像描述生成
```bash
python 01_describe_pic.py
```

### 2. 数据合并
```bash
python 02_merge_data.py
```

### 3. 场景图构建
```bash
python 03_build_scene_graphs.py
```

### 4. 转换为 PyTorch 格式
```bash
python 04_convert_to_pytorch.py
```

### 5. GraphMAE 预训练
```bash
python 05_graph_vae.py
```

### 6. Bradley-Terry 比较模型训练
```bash
python 06_comparison_trainer.py
```

### 7. 评估与可视化
```bash
python 07_evaluate_and_visualize.py
```

### 8. 雷达图可视化
```bash
python 08_radar.py
```

### 9. 感知推理
```bash
python 09_reasoning.py
```

### 10. 关系可视化
```bash
python 10_rel_visual.py
```

## Windows 系统实用命令
- `dir` - 列出目录内容
- `cd` - 切换目录
- `type` - 查看文件内容（相当于 Unix 的 `cat`）
- `findstr` - 文本搜索（相当于 Unix 的 `grep`）
- `python` - 运行 Python 脚本

## 依赖管理
- 项目未发现 requirements.txt 文件，需要根据导入的库手动安装依赖
- 主要依赖包括：torch, torch-geometric, sentence-transformers, scikit-learn, matplotlib, seaborn, PIL, numpy, pandas, tqdm, requests

## Git 命令（如果使用版本控制）
- `git status` - 查看状态
- `git add .` - 添加所有更改
- `git commit -m "message"` - 提交更改
- `git push` - 推送更改