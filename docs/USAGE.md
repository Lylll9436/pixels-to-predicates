# Usage Guide

This guide provides detailed instructions for using the Graph-based Urban Perception Prediction framework.

## 📖 Table of Contents

- [Overview](#overview)
- [Pipeline Steps](#pipeline-steps)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Advanced Usage](#advanced-usage)
- [Examples](#examples)

## 🎯 Overview

The framework processes urban street scene images through the following stages:

```
Images → Descriptions → Scene Graphs → Graph Embeddings → Preference Predictions
```

Each stage is implemented as a separate script that can be run independently or as part of a pipeline.

## 📊 Pipeline Steps

### Step 1: Image Description Generation

Generate natural language descriptions of images using vision-language models.

```bash
python src/01_describe_pic.py
```

**Input**: 
- Images from `data/PP2/final_photo_dataset/` or other configured directories

**Output**: 
- JSON files in `output/stage_01_descriptions/` containing image descriptions

**Configuration**:
```bash
# Optional arguments
python src/01_describe_pic.py \
    --image-dir data/PP2/final_photo_dataset \
    --output-dir output/stage_01_descriptions \
    --max-samples 1000  # Limit number of images
```

**Example Output** (`output/stage_01_descriptions/image_001.json`):
```json
{
  "image_id": "image_001",
  "description": "A busy urban street with tall buildings, pedestrians crossing at a crosswalk, and cars waiting at a traffic light. The scene appears lively with green trees lining the sidewalk.",
  "timestamp": "2024-10-30T12:00:00"
}
```

### Step 2: Data Merging

Merge generated descriptions with metadata (labels, categories, comparison votes).

```bash
python src/02_merge_data.py csv output/stage_01_descriptions/PP2/*.json data/PP2/metadata/final_data.csv -o output/stage_02_merged/pp2_full.json
```

**Parameters**:
- `csv`: Specify CSV metadata source
- First positional arg: Pattern matching description files
- Second positional arg: Metadata CSV path
- `-o`: Output merged JSON path

**Output**: Single JSON file containing merged data

### Step 3: Scene Graph Construction

Convert natural language descriptions into structured scene graphs.

```bash
python src/03_build_scene_graphs.py \
    --input output/stage_02_merged/pp2_full.json \
    --output output/stage_03_scene_graphs
```

**Process**:
1. Parse descriptions to extract entities (objects)
2. Identify relationships between entities
3. Extract attributes for each entity
4. Build graph structure with nodes and edges

**Output**: JSON files representing scene graphs

**Example Scene Graph**:
```json
{
  "image_id": "image_001",
  "nodes": [
    {"id": 0, "label": "building", "attributes": ["tall", "glass"]},
    {"id": 1, "label": "street", "attributes": ["busy", "paved"]},
    {"id": 2, "label": "pedestrian", "attributes": ["walking"]}
  ],
  "edges": [
    {"source": 0, "target": 1, "relation": "beside"},
    {"source": 2, "target": 1, "relation": "on"}
  ]
}
```

### Step 4: PyTorch Conversion

Convert scene graphs to PyTorch Geometric format for neural network processing.

```bash
python src/04_convert_to_pytorch.py \
    --inputs output/stage_03_scene_graphs \
    --output-dir output/stage_04_pytorch
```

**Process**:
1. Encode node labels and attributes using Sentence-BERT
2. Create adjacency matrices
3. Save as PyTorch `.pt` files

**Output**: PyTorch Data objects ready for training

### Step 5: GraphMAE Pre-training

Pre-train graph encoder using masked autoencoder objective.

```bash
python src/05_graph_vae.py \
    --data-dir output/stage_04_pytorch \
    --output-dir ./packed \
    --num-epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4
```

**Key Arguments**:
- `--data-dir`: Directory with PyTorch graph files
- `--output-dir`: Where to save trained model and representations
- `--num-epochs`: Training epochs (default: 100)
- `--mask-ratio`: Fraction of nodes to mask (default: 0.3)
- `--hidden-dim`: Hidden dimension size (default: 128)

**Output**:
- `packed/graph_mae_best.pth`: Trained model weights
- `packed/graph_representations.pt`: Graph embeddings
- `packed/graph_ids.json`: Mapping between graphs and IDs

### Step 6: Comparison Model Training

Train Bradley-Terry model for pairwise preference prediction.

```bash
# Train GraphMAE-based model
python src/06_comparison_trainer.py \
    --backbone graphmae \
    --graphs-dir output/stage_03_scene_graphs \
    --repr-file packed/graph_representations.pt \
    --graph-epochs 220

# Or train baseline models
python src/06_comparison_trainer.py \
    --backbone cnn \  # or vit, clip
    --image-root data/PP2/final_photo_dataset \
    --graphs-dir output/stage_03_scene_graphs \
    --epochs 100
```

**Backbone Options**:
- `graphmae`: Graph-based model (our approach)
- `cnn`: ResNet50 baseline
- `vit`: Vision Transformer baseline
- `clip`: CLIP ViT-B/32 baseline

**Output**:
- `result/{backbone}_baseline_best.pth`: Trained model
- `result/metrics_{backbone}.json`: Evaluation metrics
- `logs/`: Training logs

### Step 7: Evaluation and Visualization

Evaluate trained models and generate visualizations.

```bash
python src/07_evaluate_and_visualize.py \
    --graphs-dir output/stage_03_scene_graphs \
    --repr-file packed/graph_representations.pt
```

**Generates**:
- Accuracy metrics (overall and per-category)
- ROC curves
- Precision-recall curves
- Confusion matrices
- Score grids showing best/worst predictions

**Output Directory**: `output/evaluation/`

## 🎓 Model Training

### Training Configuration

All models use consistent training settings for fair comparison:

```python
{
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 5e-5,
    "patience": 20,  # Early stopping
    "dropout": 0.1,
    "hidden_dims": [512, 256, 128]
}
```

### Monitoring Training

Training progress is logged to:
- Console output (with tqdm progress bars)
- `logs/{backbone}/training_log.json`
- TensorBoard logs (optional)

Example log entry:
```
Epoch 10: train_loss=0.4234, train_acc=0.8123, val_acc=0.7845, val_auc=0.8456
```

### Early Stopping

Training automatically stops if validation accuracy doesn't improve for 20 consecutive epochs.

### Checkpointing

Best model (highest validation accuracy) is automatically saved to `result/` directory.

## 📈 Evaluation

### Metrics

The framework computes:

- **Accuracy**: Fraction of correct pairwise predictions
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under precision-recall curve
- **F1 Score**: Harmonic mean of precision and recall
- **Category-wise metrics**: Performance breakdown by perceptual dimension

### Cross-validation

For robust evaluation:

```bash
python src/06_comparison_trainer.py \
    --backbone graphmae \
    --graphs-dir output/stage_03_scene_graphs \
    --repr-file packed/graph_representations.pt \
    --split-seed 42  # Different seeds for different splits
```

Run with multiple seeds (e.g., 42, 123, 456) and average results.

## 🔬 Advanced Usage

### Custom Scene Graph Generation

Modify `src/03_build_scene_graphs.py` to use your own parsing logic:

```python
def custom_parse(description):
    # Your custom implementation
    nodes = extract_entities(description)
    edges = extract_relationships(description)
    return build_graph(nodes, edges)
```

### Hyperparameter Tuning

Create a tuning script:

```python
import itertools

learning_rates = [1e-4, 5e-5, 1e-5]
dropout_rates = [0.1, 0.2, 0.3]

for lr, dropout in itertools.product(learning_rates, dropout_rates):
    train_model(learning_rate=lr, dropout=dropout)
```

### Data Augmentation

For vision-based models, augmentation is applied in `ImageComparisonDataset`:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])
```

### Inference on New Images

```python
import torch
from src.models import BradleyTerryModel

# Load trained model
model = BradleyTerryModel(input_dim=128)
model.load_state_dict(torch.load('result/graphmae_baseline_best.pth'))
model.eval()

# Load representations
representations = torch.load('packed/graph_representations.pt')

# Compare two images
left_repr = representations['image_001']
right_repr = representations['image_002']

with torch.no_grad():
    prob = model.predict_prob(left_repr, right_repr)
    
print(f"Probability that left > right: {prob:.4f}")
```

## 💡 Examples

### Example 1: Quick Test Run

Test the pipeline on a small subset:

```bash
# Generate descriptions for 50 images
python src/01_describe_pic.py --max-samples 50

# Run through pipeline
python src/02_merge_data.py csv output/stage_01_descriptions/PP2/*.json data/PP2/metadata/final_data.csv -o output/stage_02_merged/test.json
python src/03_build_scene_graphs.py --input output/stage_02_merged/test.json --output output/stage_03_scene_graphs_test
python src/04_convert_to_pytorch.py --inputs output/scene_graphs_test --output-dir output/pytorch_test

# Quick training
python src/05_graph_vae.py --data-dir output/pytorch_test --output-dir ./test_packed --num-epochs 10
```

### Example 2: Baseline Comparison

Train all baselines:

```bash
#!/bin/bash
for backbone in cnn vit clip graphmae; do
    echo "Training $backbone baseline..."
    python src/06_comparison_trainer.py \
        --backbone $backbone \
        --image-root data/PP2/final_photo_dataset \
        --graphs-dir output/stage_03_scene_graphs \
        --repr-file packed/graph_representations.pt \
        --epochs 100
done

echo "Generating comparison report..."
python src/08_radar.py
```

### Example 3: Category-specific Analysis

Analyze performance on specific perceptual dimensions:

```python
import json

# Load evaluation results
with open('output/evaluation/evaluation_results.json') as f:
    results = json.load(f)

# Extract category metrics
categories = ['safety', 'lively', 'beautiful', 'wealthy', 'boring', 'depressing']

for cat in categories:
    acc = results['per_category'][cat]['accuracy']
    auc = results['per_category'][cat]['auc']
    print(f"{cat:12s}: Acc={acc:.4f}, AUC={auc:.4f}")
```

## 🔧 Tips and Best Practices

1. **Start Small**: Test on a subset before running full pipeline
2. **Monitor GPU**: Use `nvidia-smi` to check GPU utilization
3. **Save Checkpoints**: Training can take hours; ensure checkpoints are saved
4. **Version Control**: Track experiment configurations
5. **Reproducibility**: Set random seeds for consistent results

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
```

## ❓ FAQ

**Q: How long does training take?**

A: On a single GPU (e.g., NVIDIA RTX 3090):
- GraphMAE pre-training: 2-4 hours
- Comparison model training: 1-2 hours per baseline

**Q: Can I use CPU only?**

A: Yes, but training will be significantly slower. Use `--device cpu`.

**Q: How much data do I need?**

A: Minimum 500 pairwise comparisons per category. More data (5000+) yields better results.

**Q: Can I add custom perceptual dimensions?**

A: Yes! Ensure your metadata CSV includes the new category labels, and the pipeline will automatically handle them.

## 📞 Support

For issues or questions:
- Check [`docs/SETUP.md`](SETUP.md) for installation issues
- Review error logs in `logs/` directory
- Open a GitHub issue with details

---

**Last Updated**: October 2024

