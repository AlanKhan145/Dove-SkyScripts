# Changelog

Tất cả thay đổi quan trọng của project sẽ được ghi chép tại đây.

Format dựa theo [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
và project tuân theo [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- [ ] Support for additional image sources (Maxar, Airbus)
- [ ] Multi-language caption support
- [ ] Interactive web demo
- [ ] Model quantization for deployment
- [ ] Docker container for easy setup
- [ ] Pre-trained model zoo

---

## [1.0.0] - 2024-02-13

### 🎉 Initial Release

#### Added
- **Core DOVE Model Implementation**
  - Direction-Oriented Visual-semantic Embedding (DOVE) architecture
  - Regional-Oriented Attention Module (ROAM)
    - Intra-modal Fusion Attention (IFA)
    - Inter-modal Guidance Attention (IGA)
  - Digging Text Genome Assistant (DTGA)
  - Global visual-semantic constraint

- **Training Pipeline**
  - Single-GPU training support
  - Multi-GPU distributed training (DDP)
  - Mixed precision training (AMP)
  - Automatic checkpointing
  - TensorBoard logging
  - Learning rate scheduling

- **Evaluation Tools**
  - Cross-modal retrieval evaluation
  - Zero-shot scene classification
  - Fine-grained attribute classification
  - Visualization utilities

- **Dataset Integration**
  - SkyScript dataset loader
  - Data augmentation pipeline
  - Multi-object caption support
  - Metadata handling

- **Documentation**
  - Comprehensive README
  - Quick start guide
  - Dataset documentation
  - Installation guide
  - API reference

#### Performance
- **Zero-Shot Scene Classification**:
  - Average accuracy: 59.93% across 7 benchmark datasets
  - +6.2% improvement over baseline CLIP

- **Cross-Modal Retrieval**:
  - RSICD: 23.70% (i2t) / 19.97% (t2i) mean recall
  - RSITMD: 30.75% (i2t) / 30.58% (t2i) mean recall
  - UCM-Captions: 72.22% (i2t) / 59.33% (t2i) mean recall

- **Fine-Grained Classification**:
  - Roof shape: 46.83% top-1 accuracy
  - Road smoothness: 35.80% top-1 accuracy
  - Road surface: 67.50% top-1 accuracy

#### Dependencies
- Python >= 3.8
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- open_clip_torch >= 2.20.0
- timm >= 0.9.0

---

## [0.5.0] - 2024-01-15 (Beta)

### Added
- Beta release for testing
- Basic DOVE model architecture
- Simple training script
- Preliminary evaluation tools

### Changed
- Refactored model structure
- Improved data loading efficiency

### Fixed
- Memory leak in data loader
- Incorrect loss calculation
- CUDA compatibility issues

---

## [0.2.0] - 2023-12-20 (Alpha)

### Added
- Initial DOVE model prototype
- Basic SkyScript dataset integration
- Experimental training script

### Known Issues
- Low training stability
- Limited documentation
- No multi-GPU support

---

## [0.1.0] - 2023-12-01 (Pre-alpha)

### Added
- Project initialization
- Literature review
- Dataset exploration
- Proof of concept

---

## Version Naming Convention

- **Major version (X.0.0)**: Incompatible API changes
- **Minor version (0.X.0)**: New features, backward compatible
- **Patch version (0.0.X)**: Bug fixes, backward compatible

---

## Upgrade Guide

### From 0.5.0 to 1.0.0

1. **Update dependencies**:
```bash
pip install -r requirements.txt --upgrade
```

2. **Update config files**:
```python
# Old config (0.5.0)
config = {
    'visual_encoder': 'resnet50',
    'text_encoder': 'gru'
}

# New config (1.0.0)
config = {
    'msv_encoder': 'resnet50',
    'roi_encoder': 'resnet50',
    'text_encoder': 'bigru_dtga'
}
```

3. **Update training command**:
```bash
# Old (0.5.0)
python train.py --data data/ --epochs 20

# New (1.0.0)
python train_dove.py \
    --data_root "data/images" \
    --train_csv "data/dataframe/SkyScript_train_top50pct_filtered_by_CLIP_openai.csv" \
    --val_csv "data/dataframe/SkyScript_val_5K_filtered_by_CLIP_openai.csv" \
    --epochs 20
```

4. **Load checkpoint**:
```python
# Old (0.5.0)
model = DOVE()
model.load_state_dict(torch.load('checkpoint.pth'))

# New (1.0.0)
from dove import DOVE
model = DOVE.from_pretrained('checkpoint.pth')
```

---

## Breaking Changes

### Version 1.0.0

- **Renamed modules**:
  - `visual_encoder` → `msv_encoder` (Multiscale Visual Encoder)
  - `text_encoder` → `text_encoder_dtga` (with DTGA module)

- **Changed config structure**:
  ```python
  # Old
  {'embed_dim': 512}
  
  # New
  {'visual_embed_dim': 512, 'text_embed_dim': 512}
  ```

- **Updated loss function**:
  - Added `lambda_g` parameter for global constraint
  - Default value changed from 5.0 to 10.0

---

## Deprecation Warnings

### To be removed in version 2.0.0

- `center_box_policy='crop'` (use 'none' instead)
- `old_data_format=True` (legacy CSV format)
- Single caption mode (always use multi-object captions)

---

## Migration Help

Need help upgrading? 

- Check [UPGRADE.md](docs/UPGRADE.md) for detailed instructions
- Open an issue: [GitHub Issues](https://github.com/AlanKhan145/Dove-SkyScripts/issues)
- Ask in discussions: [GitHub Discussions](https://github.com/AlanKhan145/Dove-SkyScripts/discussions)

---

**Full Changelog**: https://github.com/AlanKhan145/Dove-SkyScripts/compare/v0.5.0...v1.0.0
