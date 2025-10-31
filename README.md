# From Pixels to Predicates: Structuring urban perception with scene graphs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of "From Pixels to Predicates: Structuring urban perception with scene graphs". A deep learning framework that transforms street view imagery (SVI) into structured scene graph representations for predicting six perceptual indicators (safety, liveliness, boredom, wealth, depression, and beauty) of urban environments.

**NOTICE**: We upgrade this repo's method using a LLM api(for example, ChatGPT or Gemini) to generate scene graphs, if you want use a OpenPSG model to do it, please refer to this awesome work: https://github.com/Jingkang50/OpenPSG

## 🎯 Overview

This research addresses the challenge of predicting human perception of urban environments by modeling street scenes as structured graphs rather than relying solely on pixel features or object co-occurrence statistics. Our three-stage pipeline: (1) extracts object–predicate–object triplets using Open-set Panoptic Scene Graph (OpenPSG), (2) learns compact scene-level embeddings through GraphMAE (heterogeneous graph autoencoder), and (3) predicts perception scores using pairwise comparison learning with Bradley-Terry models.

**Key Contribution**: Our approach improves perception prediction accuracy by an average of **26%** over baseline models and maintains strong cross-city generalization performance.

### Key Features

- **Open-set Panoptic Scene Graph (OpenPSG)**: Extracts object–predicate–object triplets from street view imagery, enabling recognition of diverse urban elements beyond predefined label sets
- **Graph Representation Learning**: Self-supervised pre-training using GraphMAE on scene graph structures (128-dim embeddings)
- **Pairwise Preference Prediction**: Bradley-Terry pairwise comparison model for six perceptual dimensions
- **Comprehensive Baselines**: Fair comparisons with CNN (ResNet50), Vision Transformer (ViT-B/16), and CLIP (ViT-B/32) approaches
- **Cross-City Evaluation**: Validated on Tokyo and Amsterdam subsets from Mapillary dataset

## 🏗️ Architecture

```
Street View Images → OpenPSG → Scene Graphs → GraphMAE → Scene Embeddings → Bradley-Terry → Perception Scores
                         ↓           ↓           ↓              ↓              ↓
                    Object-      Structured  Masked     128-dim      Pairwise
                    Predicate-   Graphs      Auto-      Vectors      Comparison
                    Object                    Encoder
                    Triplets
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.0+
- Additional dependencies listed in `requirements.txt`

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Lylll9436/pixels-to-predicates.git
cd pixels-to-predicates
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API credentials (if using Gemini for descriptions)

Copy the example configuration and add your API keys:

```bash
cp config/.env.example config/.env
```

Edit `config/.env` with your actual API credentials:

```env
API_BASE=https://your-api-endpoint.com
MODEL=gemini-2.5-flash
API_KEYS=your_api_key_1,your_api_key_2
PER_KEY_WORKERS=1
```

**Note**: For OpenPSG-based scene graph extraction, please refer to the OpenPSG repository and configure accordingly.

## 📊 Data Preparation

This project uses two datasets:

- **Place Pulse 2.0**: Provides pairwise comparisons across six perceptual dimensions (safe, lively, boring, wealthy, depressing, beautiful) for model training and validation
- **Mapillary Street-Level Sequences**: Used for cross-city generalization testing (Tokyo and Amsterdam subsets)

Please refer to [`data/README.md`](data/README.md) for detailed information on data structure and preparation.

## 🔬 Usage

### Complete Pipeline

Run the full pipeline from scene graph extraction to evaluation:

```bash
# Step 1: Generate scene descriptions (or use OpenPSG directly)
python src/01_describe_pic.py

# Step 2: Merge data with metadata
python src/02_merge_data.py csv output/stage_01_descriptions/PP2/*.json data/PP2/metadata/final_data.csv -o output/stage_02_merged/pp2_full.json

# Step 3: Build scene graphs (using OpenPSG or NLP parsing)
python src/03_build_scene_graphs.py --input output/stage_02_merged/pp2_full.json --output output/stage_03_scene_graphs

# Step 4: Convert to PyTorch format (Sentence-BERT encoding)
python src/04_convert_to_pytorch.py --inputs output/stage_03_scene_graphs --output-dir output/stage_04_pytorch

# Step 5: Pre-train GraphMAE
python src/05_graph_vae.py --data-dir output/stage_04_pytorch --output-dir ./packed --num-epochs 220

# Step 6: Train comparison model
python src/06_comparison_trainer.py --backbone graphmae --graphs-dir output/stage_03_scene_graphs --repr-file packed/graph_representations.pt

# Step 7: Evaluate and visualize
python src/07_evaluate_and_visualize.py --graphs-dir output/stage_03_scene_graphs --repr-file packed/graph_representations.pt
```

### Training Baseline Models

#### CNN Baseline (ResNet50)

```bash
python src/06_comparison_trainer.py \
    --backbone cnn \
    --image-root data/PP2/final_photo_dataset \
    --graphs-dir output/stage_03_scene_graphs \
    --epochs 100 \
    --batch-size 32
```

#### Vision Transformer Baseline (ViT-B/16)

```bash
python src/06_comparison_trainer.py \
    --backbone vit \
    --image-root data/PP2/final_photo_dataset \
    --graphs-dir output/stage_03_scene_graphs \
    --epochs 100
```

#### CLIP Baseline (ViT-B/32)

```bash
python src/06_comparison_trainer.py \
    --backbone clip \
    --image-root data/PP2/final_photo_dataset \
    --graphs-dir output/stage_03_scene_graphs \
    --epochs 100
```

### GraphMAE Baseline

```bash
python src/06_comparison_trainer.py \
    --backbone graphmae \
    --graphs-dir output/stage_03_scene_graphs \
    --repr-file packed/graph_representations.pt \
    --graph-epochs 220
```

## 📈 Results

### Main Results on Place Pulse 2.0

Our graph-based approach consistently outperforms image-only baselines across all six perceptual dimensions:

| Model | Beautiful | Boring | Depressing | Lively | Safety | Wealthy | **Average** |
|-------|-----------|--------|------------|--------|--------|---------|-------------|
| CLIP | 0.59 | 0.64 | 0.59 | 0.64 | 0.64 | 0.66 | 0.63 |
| CNN | 0.74 | 0.69 | 0.67 | 0.77 | 0.76 | 0.77 | 0.73 |
| ViT | 0.72 | 0.67 | 0.63 | 0.73 | 0.74 | 0.74 | 0.71 |
| **GraphMAE** | **0.87** | **0.84** | **0.83** | **0.88** | **0.87** | **0.90** | **0.87** |

### Overall Performance Metrics

| Model | AUC | Accuracy | Recall | F1 | Precision |
|-------|-----|----------|--------|----|-----------|
| CLIP | 0.61 | 0.63 | 0.60 | 0.55 | 0.60 |
| CNN | 0.82 | 0.73 | 0.78 | 0.68 | 0.64 |
| ViT | 0.79 | 0.71 | 0.74 | 0.72 | 0.67 |
| **GraphMAE** | **0.84** | **0.87** | **0.89** | **0.85** | **0.83** |

### Cross-City Generalization

Performance on Mapillary subsets (Tokyo and Amsterdam):

| Metric | Place Pulse 2.0 | Tokyo–Amsterdam | Change |
|--------|-----------------|-----------------|--------|
| Accuracy | 0.84 | 0.79 | -5.6% |
| AUC | 0.87 | 0.84 | -3.5% |

The structured graph-based model exhibits strong cross-city generalization, indicating that relational semantics provide a transferable foundation for urban perception prediction.

Results are saved in the `result/` directory with comprehensive JSON metrics and visualizations.

## 📁 Project Structure

```
pixels-to-predicates/
├── src/                           # Source code
│   ├── 01_describe_pic.py         # Image description generation (optional)
│   ├── 02_merge_data.py           # Data merging utilities
│   ├── 03_build_scene_graphs.py   # Scene graph construction (OpenPSG/NLP)
│   ├── 04_convert_to_pytorch.py   # PyTorch data conversion (Sentence-BERT)
│   ├── 05_graph_vae.py            # GraphMAE pre-training
│   ├── 06_comparison_trainer.py   # Bradley-Terry comparison model training
│   ├── 07_evaluate_and_visualize.py # Evaluation & visualization
│   ├── 08_radar.py                # Radar chart visualization
│   ├── 09_reasoning.py            # Perceptual reasoning analysis
│   └── 10_rel_visual.py           # Relationship visualization
├── config/                        # Configuration files
│   └── .env.example              # Environment configuration template
├── data/                          # Data directory (gitignored)
│   ├── PP2/                       # Place Pulse 2.0 dataset
│   ├── SVI/                       # Mapillary Street View Imagery
│   └── README.md                 # Data preparation guide
├── docs/                          # Additional documentation
├── output/                        # Generated outputs (gitignored)
├── result/                        # Model results (gitignored)
├── logs/                          # Training logs (gitignored)
├── .gitignore                    # Git ignore rules
├── LICENSE                        # MIT License
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

## 🔍 Key Implementation Details

### Fair Model Comparison

All baseline models are trained under **identical conditions** to ensure fair comparison:

- **Training Configuration**:
  - Epochs: 100 (with early stopping patience: 20)
  - Learning rate: 1e-4
  - Weight decay: 5e-5
  - Dropout: 0.1
  - Hidden dimensions: [512, 256, 128]
  - Batch size: 32

- **Training Strategy**:
  - **Baseline models**: Pre-trained backbones are **frozen** (not fine-tuned)
  - Optimizer: AdamW with gradient clipping
  - Loss function: Binary cross-entropy with logits
  - Data split: 60% train, 30% validation, 10% test (6:3:1 ratio)

### GraphMAE Pre-training

- Self-supervised masked graph reconstruction
- Embedding dimension: 128
- Extended training: 220 epochs for convergence
- Node-level and graph-level representation learning
- Sentence-BERT encoding: Object and predicate texts encoded using Sentence-BERT

### Scene Graph Construction

- **OpenPSG**: Open-set Panoptic Scene Graph model for extracting object–predicate–object triplets
- **Sentence-BERT**: Text embeddings for nodes (objects) and edges (predicates)
- Heterogeneous graph structure capturing spatial and semantic relationships

## 📝 Citation

### **⚠️ 31.10.2025-NOTICE**: This paper has NOT yet been published. The citation format will be updated upon publication.

If you use this code in your research, please cite:

```bibtex
@inproceedings{pixels_to_predicates2026,
  title={From Pixels to Predicates: Structuring urban perception with scene graphs},
  author={Liu, Yunlong and Li, Shuyang and Liu, Pengyuan and Zhang, Yu and Stouffs, Rudi},
  booktitle={Proceedings of the 31st International Conference on Computer-Aided Architectural Design Research in Asia (CAADRIA 2026)},
  year={2026},
  note={Submitted, under review},
  url={https://github.com/Lylll9436/pixels-to-predicates}
}
```

**Authors:**
- Yunlong Liu (School of Architecture, Southeast University, China)
- Shuyang Li (College of Design and Engineering, National University of Singapore, Singapore / Future Cities Laboratory, Singapore-ETH Centre, Singapore)
- Pengyuan Liu (Division of Urban Studies and Social Policy, University of Glasgow, United Kingdom)
- Yu Zhang (School of Architecture, Southeast University, China)
- Rudi Stouffs (College of Design and Engineering, National University of Singapore, Singapore)

## 🤝 Contributing

This is a research project developed for academic purposes. Suggestions and discussions are welcome through issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenPSG**: Open-set Panoptic Scene Graph generation [Zhou et al., 2024]
- Graph neural network implementation based on **PyTorch Geometric**
- Text embeddings using **Sentence-BERT** (Reimers & Gurevych, 2019)
- **GraphMAE**: Self-supervised masked graph autoencoders (Hou et al., 2022)
- Baseline models utilize pre-trained weights from ImageNet, CLIP (OpenAI), and other public sources
- Datasets: Place Pulse 2.0 (Dubey et al., 2016) and Mapillary Street-Level Sequences (Warburg et al., 2020)

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub or contact the corresponding author:

- **Yunlong Liu**: lyl_arch@seu.edu.cn
- School of Architecture, Southeast University, China

---

**Note**: Large data files and trained model weights are excluded from this repository due to size constraints. Please follow the data preparation guide to set up your own datasets.
