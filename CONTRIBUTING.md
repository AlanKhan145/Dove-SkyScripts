# 🤝 Contributing to DOVE-SkyScripts

Cảm ơn bạn đã quan tâm đến việc đóng góp cho DOVE-SkyScripts! Chúng tôi hoan nghênh mọi đóng góp từ cộng đồng.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Guidelines](#coding-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)
- [Questions](#questions)

---

## 📜 Code of Conduct

Project này tuân theo [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Bằng cách tham gia, bạn đồng ý tuân theo các quy tắc này.

### Nguyên tắc cơ bản:

- ✅ Tôn trọng và thân thiện
- ✅ Constructive feedback
- ✅ Focus on what is best for the community
- ✅ Show empathy towards other contributors
- ❌ Harassment hoặc discriminatory language
- ❌ Personal attacks
- ❌ Trolling hoặc insulting comments

---

## 🚀 Getting Started

### Prerequisites

Trước khi contribute, hãy chắc chắn bạn có:

- [ ] Git installed
- [ ] Python 3.8+ installed
- [ ] GitHub account
- [ ] Familiarity với PyTorch
- [ ] Understanding của remote sensing basics

### First Time Contributors

Nếu bạn là lần đầu contribute to open source:

1. **Tìm "good first issue"**: 
   - Look for issues labeled with `good first issue`
   - These are typically easier tasks suitable for beginners

2. **Read the codebase**:
   - Explore the repository structure
   - Read existing code and documentation
   - Understand the architecture

3. **Ask questions**:
   - Don't hesitate to ask in [Discussions](https://github.com/AlanKhan145/Dove-SkyScripts/discussions)
   - We're here to help!

---

## 🛠️ How to Contribute

### Types of Contributions

Chúng tôi chấp nhận các loại contributions sau:

#### 1. 🐛 Bug Fixes
- Fix known bugs
- Improve error handling
- Fix documentation errors

#### 2. ✨ New Features
- Implement new models
- Add new datasets
- Improve training pipeline
- Add visualization tools

#### 3. 📚 Documentation
- Improve README
- Add tutorials
- Write API documentation
- Translate documentation

#### 4. 🧪 Tests
- Add unit tests
- Add integration tests
- Improve test coverage

#### 5. 🎨 Code Quality
- Refactor code
- Improve performance
- Optimize memory usage

#### 6. 📊 Benchmarks
- Add new benchmark datasets
- Improve evaluation metrics
- Compare with SOTA methods

---

## 💻 Development Setup

### 1. Fork và Clone

```bash
# Fork repository trên GitHub, sau đó clone
git clone https://github.com/YOUR-USERNAME/Dove-SkyScripts.git
cd Dove-SkyScripts

# Add upstream remote
git remote add upstream https://github.com/AlanKhan145/Dove-SkyScripts.git
```

### 2. Create Development Environment

```bash
# Create virtual environment
conda create -n dove-dev python=3.9
conda activate dove-dev

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Setup

```bash
# Run tests
pytest tests/

# Check code style
flake8 src/
black --check src/

# Run type checking
mypy src/
```

---

## 📝 Coding Guidelines

### Python Style Guide

Chúng tôi tuân theo [PEP 8](https://www.python.org/dev/peps/pep-0008/) với một số modifications:

```python
# ✅ Good
def train_model(data_loader, model, optimizer, device):
    """Train the model for one epoch.
    
    Args:
        data_loader: DataLoader for training data
        model: DOVE model instance
        optimizer: PyTorch optimizer
        device: Device to train on (cpu/cuda)
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    
    for batch in data_loader:
        images = batch['image'].to(device)
        captions = batch['caption']
        
        # Forward pass
        loss = model(images, captions)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

# ❌ Bad
def train(dl,m,opt,dev):  # Unclear variable names
    m.train()
    l=0  # Single letter variable
    for b in dl:
        i=b['image'].to(dev)
        c=b['caption']
        loss=m(i,c)  # No space around =
        opt.zero_grad()
        loss.backward()
        opt.step()
        l+=loss.item()  # No space around +=
    return l/len(dl)
```

### Code Formatting

**Sử dụng Black formatter**:

```bash
# Format code
black src/ tests/

# Check formatting
black --check src/ tests/
```

**Line length**: 88 characters (Black default)

### Import Organization

```python
# Standard library imports
import os
import sys
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Local imports
from dove.models import DOVE, ROAM, DTGA
from dove.datasets import SkyScriptDataset
from dove.utils import compute_metrics
```

### Documentation

**Docstrings** theo Google style:

```python
def regional_oriented_attention(
    visual_features: torch.Tensor,
    text_features: torch.Tensor,
    regional_features: torch.Tensor
) -> torch.Tensor:
    """Apply regional-oriented attention mechanism.
    
    This function implements the ROAM (Regional-Oriented Attention Module)
    which adaptively adjusts the distance between visual and textual 
    embeddings using regional visual features as orientation.
    
    Args:
        visual_features: Multiscale visual features of shape (B, N_m, D)
        text_features: Word-level text features of shape (B, N_c, D)
        regional_features: RoI features of shape (B, N_r, D)
        
    Returns:
        Fused features of shape (B, D)
        
    Example:
        >>> visual_feat = torch.randn(32, 4, 512)
        >>> text_feat = torch.randn(32, 20, 512)
        >>> regional_feat = torch.randn(32, 36, 512)
        >>> output = regional_oriented_attention(visual_feat, text_feat, regional_feat)
        >>> output.shape
        torch.Size([32, 512])
    """
    # Implementation...
    pass
```

### Type Hints

**Luôn sử dụng type hints**:

```python
from typing import Dict, List, Optional, Tuple, Union

def encode_image(
    self,
    image: torch.Tensor,
    return_features: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Encode image into embedding space."""
    # Implementation...
```

### Testing

**Mỗi new feature cần có tests**:

```python
# tests/test_roam.py
import pytest
import torch
from dove.models import ROAM

class TestROAM:
    @pytest.fixture
    def roam_module(self):
        return ROAM(embed_dim=512, num_heads=8)
    
    def test_forward_shape(self, roam_module):
        """Test output shape of ROAM module."""
        batch_size = 16
        visual_feat = torch.randn(batch_size, 4, 512)
        text_feat = torch.randn(batch_size, 20, 512)
        regional_feat = torch.randn(batch_size, 36, 512)
        
        output = roam_module(visual_feat, text_feat, regional_feat)
        
        assert output.shape == (batch_size, 512)
    
    def test_forward_values(self, roam_module):
        """Test output values are finite."""
        visual_feat = torch.randn(8, 4, 512)
        text_feat = torch.randn(8, 20, 512)
        regional_feat = torch.randn(8, 36, 512)
        
        output = roam_module(visual_feat, text_feat, regional_feat)
        
        assert torch.isfinite(output).all()
```

---

## 🔄 Pull Request Process

### 1. Create a Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/amazing-feature
# hoặc
git checkout -b bugfix/fix-something
# hoặc
git checkout -b docs/improve-readme
```

### 2. Make Changes

- Write clean, well-documented code
- Follow coding guidelines
- Add tests for new features
- Update documentation

### 3. Commit Changes

**Commit message format**:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples**:

```bash
# Good commit messages
git commit -m "feat(roam): add inter-modal guidance attention"
git commit -m "fix(training): resolve CUDA out of memory issue"
git commit -m "docs(readme): add installation guide for Windows"

# Bad commit messages
git commit -m "update"
git commit -m "fix bug"
git commit -m "changes"
```

### 4. Push Changes

```bash
git push origin feature/amazing-feature
```

### 5. Create Pull Request

1. Go to your fork on GitHub
2. Click "Compare & pull request"
3. Fill in the PR template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes

## Testing
How has this been tested?

## Screenshots (if applicable)
Add screenshots here
```

### 6. Review Process

- **Automated checks**: CI/CD will run tests automatically
- **Code review**: Maintainers will review your code
- **Revisions**: Address feedback and update PR
- **Merge**: Once approved, we'll merge your PR!

---

## 🐛 Reporting Bugs

### Before Reporting

- [ ] Search existing issues
- [ ] Check if it's fixed in latest version
- [ ] Try to reproduce with minimal example

### Bug Report Template

```markdown
**Bug Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Run '....'
3. See error

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9]
- PyTorch version: [e.g., 2.0.0]
- CUDA version: [e.g., 11.8]
- GPU: [e.g., NVIDIA A100]

**Additional Context**
Add any other context, screenshots, or logs
```

---

## ✨ Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Clear description of the feature

**Motivation**
Why is this feature needed?

**Proposed Solution**
How should it work?

**Alternatives**
Any alternative solutions?

**Additional Context**
Any other relevant information
```

---

## ❓ Questions

### Where to Ask

- **General questions**: [GitHub Discussions](https://github.com/AlanKhan145/Dove-SkyScripts/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/AlanKhan145/Dove-SkyScripts/issues)
- **Feature requests**: [GitHub Issues](https://github.com/AlanKhan145/Dove-SkyScripts/issues)
- **Private concerns**: Email maintainers

---

## 🏆 Recognition

Contributors will be recognized in:

- `CONTRIBUTORS.md` file
- Release notes
- Project README
- Special thanks in publications (for significant contributions)

---

## 📞 Contact

- **Email**: your-email@example.com
- **Twitter**: @YourHandle
- **Discord**: [Join our server](https://discord.gg/...)

---

Thank you for contributing to DOVE-SkyScripts! 🎉
