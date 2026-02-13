# Hướng dẫn cài đặt Dove-SkyScripts

## Giới thiệu
Dove-SkyScripts là framework multimodal vision-language cho satellite image captioning và cross-modal retrieval, được thiết kế cho high-precision aerial scene understanding và large-scale remote sensing research.

## Cài đặt

### 1. Tạo môi trường ảo (khuyến nghị)

#### Sử dụng conda:
```bash
conda create -n dove-skyscripts python=3.9
conda activate dove-skyscripts
```

#### Hoặc sử dụng venv:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Cài đặt PyTorch (nếu chưa có)

Tùy thuộc vào hệ thống của bạn, bạn có thể cần cài đặt PyTorch với CUDA support:

#### Với CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Với CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Chỉ CPU (không cần GPU):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Cài đặt các thư viện tùy chọn (nếu cần)

#### Cho geospatial data processing:
```bash
pip install rasterio geopandas earthengine-api
```

#### Cho data augmentation nâng cao:
```bash
pip install albumentations
```

#### Cho distributed training:
```bash
pip install accelerate deepspeed
```

#### Cho experiment tracking:
```bash
pip install wandb
```

## Kiểm tra cài đặt

```python
import torch
import open_clip
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"OpenCLIP installed successfully")
```

## Download dataset

Tham khảo file `download_skyscript.sh` trong repository để tải xuống SkyScript dataset.

## Sử dụng

### Training:
```bash
python train_dove.py
```

### Demo retrieval:
```bash
jupyter notebook demo_dove_retrieval.ipynb
```

## Yêu cầu hệ thống

- Python: 3.8 - 3.11
- GPU: NVIDIA GPU với ít nhất 8GB VRAM (khuyến nghị 16GB+)
- RAM: Ít nhất 16GB
- Storage: Ít nhất 100GB cho dataset

## Ghi chú

- Dự án này dựa trên [SkyScript dataset](https://github.com/wangzhecheng/SkyScript)
- Sử dụng OpenCLIP framework cho CLIP model implementation
- Hỗ trợ cả ViT-B/32 và ViT-L/14 architectures

## License

MIT License (tương tự SkyScript)
