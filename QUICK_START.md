# 🚀 Quick Start Guide

Hướng dẫn nhanh để bắt đầu với DOVE-SkyScripts trong 5 phút.

---

## ⚡ Setup nhanh (5 phút)

### 1. Clone và cài đặt (2 phút)

```bash
# Clone repository
git clone https://github.com/AlanKhan145/Dove-SkyScripts.git
cd Dove-SkyScripts

# Tạo môi trường
conda create -n dove python=3.9 -y
conda activate dove

# Cài đặt dependencies
pip install -r requirements.txt

# Cài PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Download dataset (2 phút)

```bash
# Download và unzip SkyScript dataset
bash download_skyscript.sh
bash unzip_skyscript.sh
```

### 3. Test installation (1 phút)

```python
# test_install.py
import torch
import torchvision
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
print(f"✅ Installation successful!")
```

```bash
python test_install.py
```

---

## 🎯 Use Cases

### Case 1: Training từ đầu

```bash
# Training with default settings
python train_dove.py \
    --data_root "data/images" \
    --train_csv "data/dataframe/SkyScript_train_top50pct_filtered_by_CLIP_openai.csv" \
    --val_csv "data/dataframe/SkyScript_val_5K_filtered_by_CLIP_openai.csv" \
    --out_dir "runs/my_first_model"
```

**Thời gian**: ~8-10 giờ trên 1 GPU A100 (20 epochs)

### Case 2: Evaluation với pretrained model

```bash
# Download pretrained checkpoint
wget https://example.com/dove_pretrained.pt -O checkpoints/dove_pretrained.pt

# Evaluate
python eval_retrieval.py \
    --data_root "data/images" \
    --csv "data/dataframe/SkyScript_test_30K_filtered_by_CLIP_openai.csv" \
    --ckpt "checkpoints/dove_pretrained.pt"
```

**Thời gian**: ~30 phút

### Case 3: Demo với Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook demo_dove_retrieval.ipynb
```

Trong notebook:
1. Load model
2. Upload ảnh của bạn
3. Xem top-k similar images/texts
4. Visualize attention maps

---

## 🎓 Training Examples

### Example 1: Fast training (cho testing)

```bash
python train_dove.py \
    --data_root "data/images" \
    --train_csv "data/dataframe/SkyScript_train_top50pct_filtered_by_CLIP_openai.csv" \
    --val_csv "data/dataframe/SkyScript_val_5K_filtered_by_CLIP_openai.csv" \
    --batch_size 128 \
    --epochs 5 \
    --lr 5e-4 \
    --out_dir "runs/fast_test"
```

**Ưu điểm**: Nhanh, phù hợp để test code
**Nhược điểm**: Accuracy thấp hơn

### Example 2: Production training (recommended)

```bash
python train_dove.py \
    --data_root "data/images" \
    --train_csv "data/dataframe/SkyScript_train_top50pct_filtered_by_CLIP_openai.csv" \
    --val_csv "data/dataframe/SkyScript_val_5K_filtered_by_CLIP_openai.csv" \
    --caption_field "title_multi_objects" \
    --batch_size 64 \
    --epochs 20 \
    --lr 2e-4 \
    --embed_dim 512 \
    --lambda_g 10.0 \
    --amp 1 \
    --out_dir "runs/production"
```

**Ưu điểm**: SOTA results
**Nhược điểm**: Mất thời gian

### Example 3: Multi-GPU training

```bash
# 4 GPUs
torchrun --nproc_per_node=4 train_dove.py \
    --data_root "data/images" \
    --train_csv "data/dataframe/SkyScript_train_top50pct_filtered_by_CLIP_openai.csv" \
    --val_csv "data/dataframe/SkyScript_val_5K_filtered_by_CLIP_openai.csv" \
    --batch_size 128 \
    --epochs 20 \
    --out_dir "runs/distributed"
```

**Ưu điểm**: Nhanh gấp 4 lần
**Yêu cầu**: 4 GPUs

---

## 📊 Expected Results

### Training Progress

Sau 20 epochs, bạn sẽ thấy:

```
Epoch 20/20
├── Train Loss: 0.15
├── Val Loss: 0.18
├── Image→Text R@1: 17.04%
├── Image→Text R@5: 39.60%
├── Text→Image R@1: 13.63%
└── Text→Image R@5: 45.27%
```

### Evaluation Metrics

```
Cross-Modal Retrieval Results:
┌─────────────────────┬────────────────┐
│ Metric              │ Value          │
├─────────────────────┼────────────────┤
│ Image→Text R@1      │ 17.04%         │
│ Image→Text R@5      │ 39.60%         │
│ Image→Text R@10     │ 50.88%         │
│ Text→Image R@1      │ 13.63%         │
│ Text→Image R@5      │ 45.27%         │
│ Text→Image R@10     │ 66.11%         │
│ Mean Recall (mR)    │ 38.75%         │
└─────────────────────┴────────────────┘
```

---

## 🔧 Common Issues

### Issue 1: CUDA Out of Memory

**Triệu chứng**: `RuntimeError: CUDA out of memory`

**Giải pháp**:
```bash
# Giảm batch size
python train_dove.py --batch_size 32  # thay vì 64

# Hoặc giảm image size
python train_dove.py --image_size 224  # thay vì 256

# Hoặc sử dụng gradient accumulation
python train_dove.py --batch_size 32 --accumulation_steps 2
```

### Issue 2: Slow training

**Triệu chứng**: Quá chậm, 1 epoch mất > 2 giờ

**Giải pháp**:
```bash
# Bật mixed precision training
python train_dove.py --amp 1

# Tăng num_workers
python train_dove.py --num_workers 8

# Sử dụng smaller dataset
python train_dove.py --train_csv "data/dataframe/SkyScript_train_top30pct_filtered_by_CLIP_openai.csv"
```

### Issue 3: Low accuracy

**Triệu chứng**: Accuracy < 50% sau training

**Checklist**:
- [ ] Đã dùng đúng pretrained ResNet-50 on AID?
- [ ] Đã set `caption_field = "title_multi_objects"`?
- [ ] Đã set `lambda_g = 10.0`?
- [ ] Đã train đủ 20 epochs?
- [ ] Learning rate phù hợp (2e-4)?

---

## 💡 Tips & Tricks

### Tip 1: Sử dụng wandb để track experiments

```python
# Trong train_dove.py, thêm:
import wandb

wandb.init(
    project="dove-skyscripts",
    config={
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }
)

# Log metrics
wandb.log({"train_loss": loss, "val_loss": val_loss})
```

### Tip 2: Early stopping

```python
# Thêm vào training loop:
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(epochs):
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint()
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print("Early stopping!")
        break
```

### Tip 3: Learning rate scheduling

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

for epoch in range(epochs):
    train()
    validate()
    scheduler.step()
```

---

## 📚 Next Steps

Sau khi hoàn thành Quick Start:

1. **Đọc full README**: [README.md](README.md)
2. **Xem chi tiết Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
3. **Tìm hiểu Dataset**: [docs/DATASET.md](docs/DATASET.md)
4. **Advanced Training**: [docs/TRAINING.md](docs/TRAINING.md)
5. **Deploy model**: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

---

## 🆘 Getting Help

- **Quick questions**: [GitHub Discussions](https://github.com/AlanKhan145/Dove-SkyScripts/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/AlanKhan145/Dove-SkyScripts/issues)
- **Email**: your-email@example.com

---

**Happy coding! 🚀**
