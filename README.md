# Graph-based Urban Perception Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A deep learning framework for predicting human perceptual preferences of urban street scenes using graph-based scene representations. This work implements a novel pipeline that combines scene graph generation, graph masked autoencoders (GraphMAE), and Bradley-Terry comparison models.

## 🎯 Overview

This research project addresses the challenge of understanding and predicting human perception of urban environments. By modeling street scenes as structured graphs rather than raw pixel arrays, we capture the semantic relationships between objects, their attributes, and spatial arrangements.

### Key Features

- **Scene Graph Generation**: Converts natural language image descriptions into structured entity-relationship graphs
- **Graph Representation Learning**: Self-supervised pre-training using GraphMAE on scene graph structures
- **Preference Prediction**: Bradley-Terry pairwise comparison model for perceptual quality ranking
- **Comprehensive Baselines**: Fair comparisons with CNN (ResNet50), Vision Transformer (ViT), and CLIP-based approaches
- **Multi-dimensional Analysis**: Evaluation across multiple perceptual dimensions (safety, liveliness, beauty, etc.)

## 🏗️ Architecture

```
Input Images → Scene Descriptions → Scene Graphs → Graph Encoding → Preference Prediction
     ↓              (Gemini)           (NLP)        (GraphMAE)     (Bradley-Terry)
  Raw Pixels    Text Descriptions   Structured    Vector Repr.    Pairwise Scores
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.0+
- Additional dependencies listed in `requirements.txt`

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Lylll9436/structure_image.git
cd structure_image
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

### 4. Configure API credentials

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

## 📊 Data Preparation

This project expects image data organized in specific directories. Please refer to [`data/README.md`](data/README.md) for detailed information on data structure and preparation.

## 🔬 Usage

### Complete Pipeline

Run the full pipeline from image description to evaluation:

```bash
# Step 1: Generate scene descriptions
python src/01_describe_pic.py

# Step 2: Merge data with metadata
python src/02_merge_data.py csv output/stage_01_descriptions/PP2/*.json data/PP2/metadata/final_data.csv -o output/stage_02_merged/pp2_full.json

# Step 3: Build scene graphs
python src/03_build_scene_graphs.py --input output/stage_02_merged/pp2_full.json --output output/stage_03_scene_graphs

# Step 4: Convert to PyTorch format
python src/04_convert_to_pytorch.py --inputs output/stage_03_scene_graphs --output-dir output/stage_04_pytorch

# Step 5: Pre-train GraphMAE
python src/05_graph_vae.py --data-dir output/stage_04_pytorch --output-dir ./packed --num-epochs 100

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

#### Vision Transformer Baseline

```bash
python src/06_comparison_trainer.py \
    --backbone vit \
    --image-root data/PP2/final_photo_dataset \
    --graphs-dir output/stage_03_scene_graphs \
    --epochs 100
```

#### CLIP Baseline

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

The model's performance is evaluated using multiple metrics:

- **Accuracy**: Overall pairwise comparison accuracy
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Category-wise Analysis**: Per-category performance breakdown
- **Cross-validation**: Robust evaluation across multiple splits

Results are saved in the `result/` directory with comprehensive JSON metrics and visualizations.

## 📁 Project Structure

```
structure_image/
├── src/                           # Source code
│   ├── 01_describe_pic.py         # Image description generation
│   ├── 02_merge_data.py           # Data merging utilities
│   ├── 03_build_scene_graphs.py   # Scene graph construction
│   ├── 04_convert_to_pytorch.py   # PyTorch data conversion
│   ├── 05_graph_vae.py            # GraphMAE pre-training
│   ├── 06_comparison_trainer.py   # Comparison model training
│   ├── 07_evaluate_and_visualize.py # Evaluation & visualization
│   ├── 08_radar.py                # Radar chart visualization
│   ├── 09_reasoning.py            # Perceptual reasoning analysis
│   └── 10_rel_visual.py           # Relationship visualization
├── config/                        # Configuration files
│   └── .env.example              # Environment configuration template
├── data/                          # Data directory (gitignored)
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
  - Fine-tuning: All pre-trained backbones are fine-tuned end-to-end (not frozen)
  - Optimizer: AdamW with gradient clipping
  - Loss function: Binary cross-entropy with logits
  - Data split: 70% train, 15% validation, 15% test

### GraphMAE Pre-training

- Self-supervised masked graph reconstruction
- Embedding dimension: 128
- Extended training: 220 epochs for convergence
- Node-level and graph-level representation learning

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{structure_image2024,
  title={Graph-based Urban Perception Prediction},
  author={Liu, Yunlong},
  year={2024},
  publisher={GitHub},
  url={https://github.com/Lylll9436/structure_image}
}
```

## 🤝 Contributing

This is a research project developed for academic purposes. Suggestions and discussions are welcome through issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Scene graph generation powered by Google Gemini API
- Graph neural network implementation based on PyTorch Geometric
- Text embeddings using Sentence-BERT
- Baseline models utilize pre-trained weights from ImageNet, CLIP (OpenAI), and other public sources

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact [lyl_arch@seu.edu.cn].

---

**Note**: Large data files and trained model weights are excluded from this repository due to size constraints. Please follow the data preparation guide to set up your own datasets.
