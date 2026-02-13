# DOVE-SkyScripts

<div align="center">

**Direction-Oriented Visual-semantic Embedding Model for Remote Sensing Image-Text Retrieval**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2310.08276-b31b1b.svg)](https://arxiv.org/abs/2310.08276)
[![Dataset](https://img.shields.io/badge/Dataset-SkyScript-green.svg)](https://github.com/wangzhecheng/SkyScript)

</div>

---

## 📖 Tổng quan

DOVE (Direction-Oriented Visual-semantic Embedding) là một framework đột phá cho **remote sensing image-text retrieval**, được huấn luyện trên **SkyScript dataset** - bộ dataset vision-language lớn nhất và đa dạng nhất cho ảnh viễn thám với **2.6M image-text pairs** và **29K semantic tags** riêng biệt.

### 🎯 Vấn đề chính

Remote sensing image-text retrieval đối mặt với thử thách **visual-semantic imbalance**:

- **Visual-semantic redundancy**: Các vật thể nhỏ dễ bị nhiễu từ background và irrelevant objects
- **Inter-class similarity**: Ảnh của các scene khác nhau có thể rất giống nhau

<div align="center">
<img src="docs/images/visual_semantic_imbalance.png" width="800">
<p><i>Hình 1: Visual-semantic imbalance trong remote sensing</i></p>
</div>

### 💡 Giải pháp của DOVE

DOVE giải quyết vấn đề này bằng cách:

1. **Regional-Oriented Attention Module (ROAM)**: Điều chỉnh khoảng cách giữa visual và textual embeddings trong latent space
2. **Digging Text Genome Assistant (DTGA)**: Tăng cường textual representation với global word-level semantic connections
3. **Global Visual-Semantic Constraint**: Giảm single visual dependency và constraint cho final embeddings

<div align="center">
<img src="docs/images/dove_architecture.png" width="900">
<p><i>Hình 2: Kiến trúc tổng thể của DOVE model</i></p>
</div>

---

## ✨ Tính năng chính

### 🔥 Performance

- ✅ **+6.2% accuracy** so với baseline CLIP trên zero-shot scene classification
- ✅ **SOTA results** trên RSICD và RSITMD datasets
- ✅ **Zero-shot transfer** cho fine-grained object attribute classification
- ✅ **Cross-modal retrieval** với mean recall vượt trội

### 📊 Dataset: SkyScript

- 📸 **2.6M image-text pairs** (5.2M unfiltered)
- 🏷️ **29K distinct semantic tags** (44K unfiltered)
- 🌍 **Global coverage** từ multiple satellite sources
- 🎯 **Multi-resolution**: 0.1m - 30m GSD
- 🔗 **Multi-source**: SWISSIMAGE, NAIP, Sentinel-2, Landsat, Planet SkySat

<div align="center">
<img src="docs/images/skyscript_coverage.png" width="800">
<p><i>Hình 3: Geographic coverage của SkyScript dataset</i></p>
</div>

---

## 🏗️ Kiến trúc

### DOVE Model Components

```
┌─────────────────────────────────────────────────────┐
│                  Input Representation                │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ MSV Encoder  │  │  RoI Encoder │  │    DTGA    │ │
│  │  (ResNet-50) │  │  (ResNet-50) │  │ (BiGRU +   │ │
│  │              │  │              │  │  Gated SA) │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────┐
│              Modality Interaction (ROAM)             │
│  ┌──────────────────────┐  ┌─────────────────────┐  │
│  │ Intra-modal Fusion   │  │ Inter-modal Guidance│  │
│  │   Attention (IFA)    │  │   Attention (IGA)   │  │
│  └──────────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────┐
│            Similarity Measurement & Loss             │
│  • Ranking Loss: L(V_MR, T_RG)                      │
│  • Global Constraint: λ_g × L(V_M, T_G)             │
└─────────────────────────────────────────────────────┘
```

### Key Modules

#### 1. **DTGA (Digging Text Genome Assistant)**

```python
# Dual-branch structure để enhance textual features
H_f = GRU_forward(text_embeddings)
H_b = GRU_backward(text_embeddings)

# Gated self-attention
H_f_tilde = GatedSelfAttention(H_f)
H_b_tilde = GatedSelfAttention(H_b)

# Interactive features
T_f⊙b = T_f ⊙ Probability(H_b)
T_b⊙f = T_b ⊙ Probability(H_f)

# Final textual features
F_G = MLP(T_f⊙b + T_b⊙f)
```

#### 2. **ROAM (Regional-Oriented Attention Module)**

- **IFA (Intra-modal Fusion Attention)**: Fuse multiscale và regional visual features
- **IGA (Inter-modal Guidance Attention)**: Guide textual features bằng regional visual features

---

## 🚀 Cài đặt

### Requirements

- Python >= 3.8
- CUDA >= 11.8 (recommended)
- 16GB+ RAM
- NVIDIA GPU với 8GB+ VRAM (16GB+ recommended)

### 1. Clone Repository

```bash
git clone https://github.com/AlanKhan145/Dove-SkyScripts.git
cd Dove-SkyScripts
```

### 2. Tạo môi trường

```bash
# Sử dụng conda (recommended)
conda create -n dove python=3.9
conda activate dove

# Hoặc sử dụng venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cài đặt PyTorch

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 📊 Dataset Setup

### Download SkyScript Dataset

```bash
# Download script
bash download_skyscript.sh

# Hoặc download thủ công từ các nguồn sau:
# - Training data (top 50%): SkyScript_train_top50pct_filtered_by_CLIP_openai.csv
# - Validation data: SkyScript_val_5K_filtered_by_CLIP_openai.csv
# - Test data: SkyScript_test_30K_filtered_by_CLIP_openai.csv
```

### Cấu trúc thư mục

```
Dove-SkyScripts/
├── data/
│   ├── images/
│   │   ├── images2/
│   │   ├── images3/
│   │   ├── ... 
│   │   └── images7/
│   └── dataframe/
│       ├── SkyScript_train_top50pct_filtered_by_CLIP_openai.csv
│       ├── SkyScript_val_5K_filtered_by_CLIP_openai.csv
│       └── SkyScript_test_30K_filtered_by_CLIP_openai.csv
├── src/
│   └── dove/
│       ├── models/
│       ├── datasets/
│       └── utils/
├── runs/
├── train_dove.py
├── eval_retrieval.py
└── demo_dove_retrieval.ipynb
```

### Unzip images

```bash
bash unzip_skyscript.sh
```

---

## 🎓 Training

### Basic Training

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

### Distributed Training (Multi-GPU)

```bash
torchrun --nproc_per_node=4 train_dove.py \
    --data_root "data/images" \
    --train_csv "data/dataframe/SkyScript_train_top50pct_filtered_by_CLIP_openai.csv" \
    --val_csv "data/dataframe/SkyScript_val_5K_filtered_by_CLIP_openai.csv" \
    --caption_field "title_multi_objects" \
    --batch_size 128 \
    --epochs 20 \
    --out_dir "runs/dove_skyscript_ddp"
```

### Training Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--data_root` | Đường dẫn đến thư mục chứa images | - | `data/images` |
| `--train_csv` | CSV file cho training | - | Required |
| `--val_csv` | CSV file cho validation | - | Required |
| `--caption_field` | Trường caption trong CSV | `title` | `title_multi_objects` |
| `--image_size` | Kích thước ảnh input | 256 | 256 |
| `--batch_size` | Batch size | 64 | 64-128 |
| `--epochs` | Số epochs | 20 | 20-50 |
| `--lr` | Learning rate | 2e-4 | 2e-4 |
| `--embed_dim` | Embedding dimension | 512 | 512 |
| `--lambda_g` | Global constraint weight | 10.0 | 10.0 |
| `--margin` | Triplet loss margin | 0.2 | 0.2 |
| `--amp` | Mixed precision training | 0 | 1 |

---

## 📈 Evaluation

### Demo Notebook

Sử dụng Jupyter notebook để test retrieval:

```bash
jupyter notebook demo_dove_retrieval.ipynb
```

<div align="center">
<img src="docs/images/retrieval_demo.png" width="800">
<p><i>Hình 4: Demo cross-modal retrieval</i></p>
</div>

---

---

](https://github.com/AlanKhan145/Dove-SkyScripts)
