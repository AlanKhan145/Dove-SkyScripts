# 🕊️ DOVE-SkyScripts

<div align="center">

**Direction-Oriented Visual-semantic Embedding Model for Remote Sensing Image-Text Retrieval**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2310.08276-b31b1b.svg)](https://arxiv.org/abs/2310.08276)
[![Dataset](https://img.shields.io/badge/Dataset-SkyScript-green.svg)](https://github.com/wangzhecheng/SkyScript)

[English](README.md) | [中文](README_zh.md) | [Tiếng Việt](README_vi.md)

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

### Cross-Modal Retrieval

```bash
python eval_retrieval.py \
    --data_root "data/images" \
    --csv "data/dataframe/SkyScript_test_30K_filtered_by_CLIP_openai.csv" \
    --ckpt "runs/dove_skyscript/best.pt" \
    --caption_field "title_multi_objects" \
    --center_box_policy "none" \
    --image_size 256 \
    --batch_size 128
```

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

## 📊 Results

### Zero-Shot Scene Classification

| Model | AID | EuroSAT | fMoW | MillionAID | PatternNet | RESISC45 | RSI-CB | **Avg** |
|-------|-----|---------|------|------------|------------|----------|--------|---------|
| CLIP-original | 55.06 | 69.25 | 41.89 | 26.19 | 57.88 | 71.39 | 66.70 | 53.76 |
| RemoteCLIP | 34.40 | 70.85 | 27.81 | 16.77 | 47.20 | 61.91 | 74.31 | 49.95 |
| CLIP-laion-RS | 58.81 | 71.70 | 54.30 | 27.21 | 60.77 | 72.68 | 71.21 | 57.87 |
| **DOVE-50** | **70.89** | **71.70** | **51.33** | **27.12** | **67.45** | **80.88** | **70.94** | **59.93** |

### Cross-Modal Retrieval (Mean Recall %)

| Model | RSICD | RSITMD | UCM-Captions |
|-------|-------|--------|--------------|
|       | i2t / t2i | i2t / t2i | i2t / t2i |
| AMFMN* | 14.62 / 18.21 | 25.74 / 33.69 | 43.65 / 48.51 |
| GaLR* | 19.16 / 18.77 | 29.65 / 33.17 | - / - |
| CLIP-original | 19.67 / 13.84 | 27.51 / 24.10 | 68.41 / 56.76 |
| **DOVE** | **23.70 / 19.97** | **30.75 / 30.58** | **72.22 / 59.33** |

*Supervised models (seen benchmark datasets during training)

### Fine-Grained Classification (Top-1 Accuracy %)

| Model | Roof Shape | Road Smoothness | Road Surface |
|-------|------------|-----------------|--------------|
| CLIP-original | 37.50 | 25.40 | 42.73 |
| **DOVE** | **46.83** | **35.80** | **67.50** |

---

## 🎯 Use Cases

### 1. Zero-Shot Classification

```python
from dove import DOVE
import torch
from PIL import Image

# Load model
model = DOVE.from_pretrained("runs/dove_skyscript/best.pt")
model.eval()

# Load image
image = Image.open("example.jpg")

# Define classes
classes = ["airport", "beach", "bridge", "farmland", "forest"]

# Predict
with torch.no_grad():
    probs = model.classify(image, classes)
    pred_class = classes[probs.argmax()]
    
print(f"Predicted: {pred_class} (confidence: {probs.max():.2%})")
```

### 2. Image-Text Retrieval

```python
# Image to Text
image = Image.open("satellite.jpg")
top_texts = model.image_to_text(image, text_database, top_k=5)

# Text to Image  
query = "airport with multiple runways"
top_images = model.text_to_image(query, image_database, top_k=5)
```

### 3. Feature Extraction

```python
# Extract visual features
image_features = model.encode_image(image)

# Extract text features
text_features = model.encode_text("residential area with roads")

# Compute similarity
similarity = torch.cosine_similarity(image_features, text_features)
```

---

## 📁 Code Structure

```
src/dove/
├── models/
│   ├── dove.py              # Main DOVE model
│   ├── encoders.py          # MSV & RoI encoders
│   ├── roam.py              # ROAM module
│   └── dtga.py              # DTGA module
├── datasets/
│   ├── skyscript.py         # SkyScript dataset loader
│   └── transforms.py        # Data augmentation
├── utils/
│   ├── metrics.py           # Evaluation metrics
│   ├── losses.py            # Loss functions
│   └── visualization.py     # Visualization utilities
└── config.py                # Configuration
```

---

## 🔬 Ablation Studies

### Ảnh hưởng của λ_g (Global Constraint Weight)

<div align="center">
<img src="docs/images/lambda_g_ablation.png" width="600">
<p><i>Hình 5: Retrieval performance với các giá trị λ_g khác nhau</i></p>
</div>

### Ảnh hưởng của DTGA Module

| Input Combination | Sentence Retrieval | Image Retrieval | mR |
|-------------------|-------------------|-----------------|-----|
| H^f, H^f | 15.93 / 33.19 / 46.68 | 13.50 / 44.20 / 64.51 | 36.33 |
| H^b, H^b | 15.27 / 36.06 / 50.44 | 13.89 / 44.82 / 66.11 | 37.77 |
| **H^f, H^b** | **17.04 / 39.60 / 50.88** | **13.63 / 45.27 / 66.11** | **38.75** |

---

## 🤝 Contributing

Chúng tôi welcome contributions! Vui lòng:

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📝 Citation

Nếu bạn sử dụng DOVE hoặc SkyScript trong nghiên cứu, vui lòng cite:

### DOVE Model

```bibtex
@article{ma2024dove,
  title={Direction-Oriented Visual-semantic Embedding Model for Remote Sensing Image-text Retrieval},
  author={Ma, Qing and Pan, Jiancheng and Bai, Cong},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={62},
  pages={1--14},
  year={2024},
  publisher={IEEE}
}
```

### SkyScript Dataset

```bibtex
@article{wang2023skyscript,
  title={SkyScript: A Large and Semantically Diverse Vision-Language Dataset for Remote Sensing},
  author={Wang, Zhecheng and Prabha, Rajanie and Huang, Tianyuan and Wu, Jiajun and Rajagopal, Ram},
  journal={arXiv preprint arXiv:2312.12856},
  year={2023}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **DOVE Model**: Based on research by Ma et al. (IEEE TGRS 2024)
- **SkyScript Dataset**: Developed by Wang et al. (AAAI 2024)
- **Pretrained Models**: ResNet-50 pretrained on AID dataset
- **Framework**: Built on PyTorch and OpenCLIP

---

## 📮 Contact

- **Issues**: [GitHub Issues](https://github.com/AlanKhan145/Dove-SkyScripts/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AlanKhan145/Dove-SkyScripts/discussions)
- **Email**: [your-email@example.com](mailto:your-email@example.com)

---

## 🔗 Related Projects

- [SkyScript Official](https://github.com/wangzhecheng/SkyScript) - Official SkyScript dataset repository
- [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP) - Vision Language Foundation Model for RS
- [RSICD](https://github.com/201528014227051/RSICD_optimal) - Remote Sensing Image Captioning Dataset
- [RSITMD](https://github.com/xiaoyuan1996/AMFMN) - Fine-grained Remote Sensing Dataset

---

<div align="center">

**⭐ Nếu project này hữu ích, hãy cho chúng tôi một star! ⭐**

Made with ❤️ by Remote Sensing Community

</div>
