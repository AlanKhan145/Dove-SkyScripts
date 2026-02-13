# 🕊️ DOVE-SkyScripts

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/AlanKhan145/Dove-SkyScripts)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)

**Direction-Oriented Visual–Semantic Embedding for Remote Sensing Image–Text Retrieval**

Official implementation of DOVE methodology for remote sensing image-text retrieval tasks.

---

## 📖 Table of Contents

- [Introduction](#-introduction)
- [Problem Statement](#-problem-statement)
- [Method Overview](#-method-overview)
- [Project Structure](#-project-structure)
- [Dataset: SkyScript](#-dataset-skyscript)
- [Installation](#️-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Training](#️-training)
- [Evaluation](#-evaluation)
- [Hardware Requirements](#-hardware-requirements)
- [Experimental Insights](#-experimental-insights)
- [Future Work](#-future-work)
- [Citation](#-citation)
- [Author](#-author)

---

## 📖 Introduction

This project implements **DOVE** (Direction-Oriented Visual-Semantic Embedding) for remote sensing image–text retrieval.

The implementation is inspired by:
- **Direction-Oriented Visual-semantic Embedding Model (DOVE)** for Remote Sensing Image-text Retrieval
- **SkyScript**: A Large-Scale Remote Sensing Image-Text Dataset

**Official Repository:** [https://github.com/AlanKhan145/Dove-SkyScripts](https://github.com/AlanKhan145/Dove-SkyScripts)

---

## 🎯 Problem Statement

Remote sensing image–text retrieval is significantly more challenging than natural image retrieval due to:

- ❌ **Visual–semantic redundancy** (irrelevant background regions)
- ❌ **Inter-class similarity** (different scenes look visually similar)
- ❌ **Small objects with complex layouts**
- ❌ **Multi-scale semantics**

### Traditional Limitations

Traditional global or purely local alignment strategies often fail because they either:
1. Over-focus on a single object
2. Encode too much irrelevant background information

### ✅ DOVE Solution

DOVE solves this by introducing **region-oriented alignment guidance**.

---

## 🧠 Method Overview

DOVE reduces visual–semantic imbalance through three key components:

### 🔹 1️⃣ DTGA – Digging Text Genome Assistant

Enhances textual semantic representation via:
- **BiGRU** dual-direction encoding
- **Dual-flow gating**
- **Gated self-attention**
- Richer **word-to-word interaction** modeling

### 🔹 2️⃣ ROAM – Regional-Oriented Attention Module

ROAM uses regional visual features as directional anchors.

**Components:**
- **IFA (Intra-modal Fusion Attention)**
  - Fuse multiscale global + regional visual features
  
- **IGA (Inter-modal Guidance Attention)**
  - Regional features guide textual alignment

### 🔹 3️⃣ Global Visual–Semantic Constraint

Besides the main triplet ranking loss, DOVE introduces:
- A **global constraint**
- Reduces single-region dominance
- Stabilizes embedding geometry

**Recommended hyperparameter:** `λg ≈ 10.0`

---

## 📂 Project Structure

```
Dove-SkyScripts/
│
├── train_dove.py              # Main training script
├── download_skyscript.sh      # Dataset download script
├── unzip_skyscript.sh         # Dataset extraction script
├── requirements.txt           # Python dependencies
│
├── data/
│   ├── images/               # Image files
│   └── dataframe/            # CSV metadata files
│
└── runs/                     # Training outputs and checkpoints
```

---

## 📦 Dataset: SkyScript

**SkyScript** is a large-scale remote sensing image–text dataset built by:
- Aligning satellite imagery (Google Earth Engine)
- With OpenStreetMap semantic tags
- Using geographic coordinate matching

### 📊 Dataset Statistics

| Item | Value |
|------|-------|
| Image–Text pairs (filtered) | ~2.6M |
| Semantic tags | ~29K |
| Resolution range | 0.1m – 30m |
| Caption types | Single-object & Multi-object |

**After extraction:**
```
data/
├── images/
└── dataframe/
```

---

## ⚙️ Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/AlanKhan145/Dove-SkyScripts.git
cd Dove-SkyScripts
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset Preparation

### Step 1: Make Scripts Executable

```bash
chmod +x /home/hungvu/code/khanh/SkyScript/download_skyscript.sh
```

### Step 2: Download Dataset

```bash
/home/hungvu/code/khanh/SkyScript/download_skyscript.sh
```

### Step 3: Unzip Dataset

**Normal extraction:**
```bash
bash /home/hungvu/code/khanh/SkyScript/unzip_skyscript.sh
```

**Force overwrite:**
```bash
FORCE=1 bash /home/hungvu/code/khanh/SkyScript/unzip_skyscript.sh
```

---

## 🏋️ Training

Run the following command to start training:

```bash
python train_dove.py \
  --data_root "data/images" \
  --train_csv "data/dataframe/SkyScript_train_top50pct_filtered_by_CLIP_openai.csv" \
  --val_csv "data/dataframe/SkyScript_val_5K_filtered_by_CLIP_openai.csv" \
  --caption_field "title_multi_objects" \
  --image_size 256 \
  --center_box_policy "none" \
  --batch_size 64 \
  --epochs 20 \
  --lr 2e-4 \
  --weight_decay 0.05 \
  --embed_dim 512 \
  --lambda_g 10.0 \
  --margin 0.2 \
  --random_rotation 1 \
  --num_workers 4 \
  --amp 1 \
  --out_dir "runs/dove_skyscript"
```

### 🔧 Parameter Explanation

| Parameter | Description |
|-----------|-------------|
| `--data_root` | Path to image directory |
| `--train_csv` | Training CSV file |
| `--val_csv` | Validation CSV file |
| `--caption_field` | Which caption to use |
| `--image_size` | Input image size |
| `--batch_size` | Training batch size |
| `--epochs` | Total training epochs |
| `--lr` | Learning rate |
| `--weight_decay` | Weight decay |
| `--embed_dim` | Embedding dimension |
| `--lambda_g` | Global constraint weight |
| `--margin` | Triplet ranking margin |
| `--random_rotation` | Data augmentation |
| `--amp` | Mixed precision training |

---

## 📊 Evaluation

### Metrics

- **Recall@1 / Recall@5 / Recall@10**
- **Mean Recall (mR)**
- **Image-to-Text Retrieval (i2t)**
- **Text-to-Image Retrieval (t2i)**

### Benchmarks

Commonly used benchmarks:
- **RSICD**
- **RSITMD**
- **UCM-Captions**

---

## 💻 Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | ≥ 24GB VRAM |
| Recommended | A100 / RTX 3090 |
| Memory Optimization | Enable AMP (`--amp 1`) |

---

## 🔬 Experimental Insights

Key findings from experiments:

✅ **Regional guidance** significantly improves alignment stability  
✅ **Global constraint** prevents embedding collapse  
✅ **Multi-object captions** increase robustness  
✅ **λg around 10.0** provides best trade-off  

---

## 🔮 Future Work

Planned improvements and extensions:

- [ ] SkyCLIP continual pre-training support
- [ ] Zero-shot evaluation pipeline
- [ ] Fine-grained attribute classification
- [ ] LLM-enhanced natural caption generation
- [ ] Multi-spectral extension

---

## 📜 Citation

If you use this repository, please cite:

```bibtex
@article{DOVE2023,
  title={Direction-Oriented Visual-semantic Embedding Model for Remote Sensing Image-text Retrieval}
}

@article{SkyScript2024,
  title={SkyScript: Large-Scale Remote Sensing Image-Text Dataset}
}
```

---

## 👨‍💻 Author

**GitHub:** [https://github.com/AlanKhan145](https://github.com/AlanKhan145)

---

## 📝 License

Please check the repository for license information.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

<div align="center">
  <strong>⭐ If you find this project useful, please consider giving it a star! ⭐</strong>
</div>
