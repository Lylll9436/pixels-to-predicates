# Installation and Setup Guide

This guide provides detailed instructions for setting up the development environment for the Graph-based Urban Perception Prediction project.

## 📋 Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows 10/11
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended for training)
  - Minimum 8GB VRAM for baseline models
  - 12GB+ VRAM recommended for larger datasets
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB+ free space for data and models

### Required Software

1. **Python 3.8+**
   - Download from [python.org](https://www.python.org/downloads/)
   - Verify installation: `python --version`

2. **Git**
   - Download from [git-scm.com](https://git-scm.com/)
   - Verify installation: `git --version`

3. **CUDA Toolkit** (for GPU support)
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
   - Recommended: CUDA 11.7 or 11.8
   - Verify installation: `nvcc --version`

## 🔧 Installation Steps

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/structure_image.git
cd structure_image
```

### Step 2: Create Virtual Environment

Using **venv** (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

Using **conda** (alternative):

```bash
# Create conda environment
conda create -n urban_perception python=3.8
conda activate urban_perception
```

### Step 3: Install PyTorch

Install PyTorch with appropriate CUDA version:

**For CUDA 11.8:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 11.7:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

**For CPU only:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Verify PyTorch installation:

```python
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 4: Install PyTorch Geometric

```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu118.html
```

### Step 5: Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- sentence-transformers (text embeddings)
- scikit-learn (evaluation metrics)
- matplotlib, seaborn (visualization)
- pandas, numpy (data processing)
- Pillow (image handling)
- tqdm (progress bars)
- requests (API calls)
- networkx (graph utilities)

### Step 6: Install Optional Dependencies

For baseline models:

```bash
# For CLIP baseline
pip install open-clip-torch

# For ViT baseline (if using timm)
pip install timm
```

## ⚙️ Configuration

### Step 1: Create Configuration File

```bash
cp config/.env.example config/.env
```

### Step 2: Edit Configuration

Open `config/.env` and fill in your API credentials:

```env
# API Base URL (for scene description generation)
API_BASE=https://your-api-endpoint.com

# Model name
MODEL=gemini-2.5-flash

# API Keys (comma-separated for multiple keys)
API_KEYS=your_api_key_1,your_api_key_2,your_api_key_3

# Parallel workers per key
PER_KEY_WORKERS=1
```

### API Key Setup

To obtain API keys for scene description generation:

1. **Google Gemini API**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create or sign in to your account
   - Generate API key
   - Copy the key to your `.env` file

2. **Alternative APIs**:
   - You can modify `src/01_describe_pic.py` to use other vision-language models
   - Options include: OpenAI GPT-4V, Azure Vision, Claude, etc.

## 📁 Directory Setup

Create necessary directories:

```bash
mkdir -p output/01
mkdir -p output/scene_graphs
mkdir -p output/scene_graphs_pytorch
mkdir -p result
mkdir -p logs
mkdir -p packed
```

## ✅ Verify Installation

Run the verification script:

```python
# test_installation.py
import sys
import torch
import torch_geometric
from sentence_transformers import SentenceTransformer

print("=" * 50)
print("Installation Verification")
print("=" * 50)

print(f"\n✓ Python version: {sys.version.split()[0]}")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ PyTorch Geometric version: {torch_geometric.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("\nTesting Sentence-BERT...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("test sentence")
print(f"✓ Sentence embedding shape: {embedding.shape}")

print("\n" + "=" * 50)
print("All checks passed! ✓")
print("=" * 50)
```

Run it:

```bash
python test_installation.py
```

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution**: Reduce batch size in training scripts:

```bash
python src/06_comparison_trainer.py --batch-size 16  # Instead of 32
```

#### 2. PyTorch Geometric Installation Fails

**Solution**: Install wheels manually:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

#### 3. Import Error: "No module named 'torchvision'"

**Solution**: Reinstall torchvision:

```bash
pip install --upgrade torchvision
```

#### 4. SSL Certificate Error

**Solution**: Use trusted host flag:

```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

#### 5. Permission Denied on Linux

**Solution**: Add user to docker group or use `--user` flag:

```bash
pip install --user -r requirements.txt
```

## 🔄 Updating

To update the project:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## 🧹 Cleanup

To remove the environment:

```bash
# Deactivate virtual environment
deactivate

# Remove venv directory
rm -rf venv

# Or for conda
conda remove -n urban_perception --all
```

## 📚 Next Steps

Once installation is complete:

1. Prepare your data following [`data/README.md`](../data/README.md)
2. Read the usage guide in [`docs/USAGE.md`](USAGE.md)
3. Run the example pipeline to verify everything works

## 🆘 Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Search existing [GitHub issues](https://github.com/yourusername/structure_image/issues)
3. Create a new issue with:
   - Your OS and Python version
   - Complete error message
   - Steps to reproduce

---

**Last Updated**: October 2024

