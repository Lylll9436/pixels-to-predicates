# Data Directory

This directory contains the image datasets used for urban perception prediction. Due to size and licensing constraints, the actual data files are not included in this repository.

## 📁 Expected Directory Structure

```
data/
├── PP2/                              # Photo Preference Dataset
│   ├── final_photo_dataset/          # Image files
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   └── metadata/                     # Metadata files
│       ├── final_data.csv            # Image metadata
│       └── votes.csv                 # Pairwise comparison votes
│
├── SVI/                              # Street View Imagery Dataset
│   ├── train_val/                    # Training and validation data
│   │   ├── amsterdam/
│   │   ├── budapest/
│   │   ├── goa/
│   │   ├── moscow/
│   │   ├── paris/
│   │   ├── tokyo/
│   │   └── zurich/
│   ├── test/                         # Test data
│   │   ├── athens/
│   │   ├── bengaluru/
│   │   ├── kampala/
│   │   └── stockholm/
│   └── metadata/                     # CSV metadata files
│       ├── train_val/
│       └── test/
│
└── sample/                           # Small sample dataset for testing
    ├── image_001.jpg
    └── ...
```

## 📊 Dataset Descriptions

### PP2 (Photo Preference Dataset)

A dataset of urban street scene images with pairwise comparison annotations for perceptual quality assessment.

**Metadata Format** (`final_data.csv`):

| Column | Description |
|--------|-------------|
| `image_id` | Unique identifier for each image |
| `filename` | Image filename |
| `category` | Perceptual category (e.g., "safety", "lively", "beautiful") |
| `score` | Aggregated quality score |
| Additional columns | Context-specific metadata |

**Votes Format** (`votes.csv`):

| Column | Description |
|--------|-------------|
| `left_id` | ID of the first image |
| `right_id` | ID of the second image |
| `winner` | Which image was preferred ("left" or "right") |
| `category` | Perceptual dimension being compared |
| `voter_id` | Anonymous voter identifier |

### SVI (Street View Imagery Dataset)

Larger-scale dataset collected from multiple cities worldwide, organized by location.

**Metadata Format** (CSV files in `metadata/`):

| Column | Description |
|--------|-------------|
| `image_id` | Unique identifier |
| `city` | City name |
| `latitude` | GPS latitude |
| `longitude` | GPS longitude |
| `category` | Perceptual category |
| Other fields | Context and annotations |

## 🔍 Data Requirements

### Image Format

- **Format**: JPEG (.jpg)
- **Color**: RGB
- **Recommended size**: 512x512 or larger
- **Quality**: High resolution street-level photography

### Metadata Requirements

All CSV files should:
- Use UTF-8 encoding
- Include headers in the first row
- Use comma (`,`) as delimiter
- Quote text fields containing commas

## 📥 Data Preparation

### Option 1: Use Your Own Data

1. Organize images into the directory structure shown above
2. Create corresponding metadata CSV files with the required columns
3. Ensure image filenames match the IDs in metadata files

### Option 2: Public Datasets

You can use publicly available urban imagery datasets such as:

- **MIT Places2**: Large-scale scene recognition dataset
- **Street View datasets**: Various city-specific datasets
- **Mapillary Street-level Imagery**: Open dataset of street photos

**Note**: Ensure you have appropriate permissions and follow licensing terms when using public datasets.

## 🚀 Quick Start with Sample Data

For testing the pipeline without full datasets:

1. Place a small collection of images in `data/sample/`
2. Create minimal CSV files with required metadata
3. Run the pipeline with `--max-samples` flag to limit data usage:

```bash
python src/01_describe_pic.py --max-samples 100
```

## ⚠️ Important Notes

### Privacy and Ethics

- Remove or blur any personally identifiable information (faces, license plates)
- Ensure compliance with local data protection regulations (GDPR, CCPA, etc.)
- Obtain necessary permissions for using and sharing image data

### Storage Considerations

- Full datasets can be **very large** (tens of GBs)
- Use `.gitignore` to prevent accidentally committing data files
- Consider using external storage or data versioning tools (DVC, Git LFS)

## 📋 Data Preprocessing

Before running the pipeline, ensure:

1. ✅ Images are in JPEG format
2. ✅ Filenames don't contain special characters
3. ✅ Metadata CSV files are properly formatted
4. ✅ Image IDs in CSV match actual filenames
5. ✅ No duplicate image IDs

## 🔗 Related Scripts

- `src/01_describe_pic.py` - Reads images and generates descriptions
- `src/02_merge_data.py` - Merges image data with metadata
- `src/03_build_scene_graphs.py` - Processes data into scene graphs

## 📞 Questions?

If you encounter data-related issues, please check:

1. File paths are correct relative to project root
2. CSV encodings are UTF-8
3. Image files are readable and not corrupted
4. Metadata columns match expected format

For additional help, please open an issue on GitHub.

---

**Last Updated**: October 2024

