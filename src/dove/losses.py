"""
Loss functions for DOVE (Direction-Oriented Visual-semantic Embedding).

Combines:
1. OpenCLIP-style InfoNCE Loss (supports Distributed Data Parallel).
2. DOVE-specific Global Constraint Loss (regional alignment).

Reference: Adapted from https://github.com/mlfoundations/open_clip
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


# -----------------------------------------------------------------------------
# Distributed Helpers (From OpenCLIP)
# -----------------------------------------------------------------------------

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    """
    Gathers embeddings from all GPUs to compute the contrastive loss over the global batch.
    """
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    
    # CASE 1: Horovod
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    
    # CASE 2: Torch Distributed
    else:
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


# -----------------------------------------------------------------------------
# DOVE Specific Logic
# -----------------------------------------------------------------------------

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """FP16-safe L2 normalization."""
    if x.numel() == 0:
        return x
    if x.dtype in (torch.float16, torch.bfloat16):
        denom = (
            torch.linalg.norm(x.float(), ord=2, dim=dim, keepdim=True)
            .clamp_min(eps)
            .to(dtype=x.dtype)
        )
    else:
        denom = torch.linalg.norm(x, ord=2, dim=dim, keepdim=True).clamp_min(eps)
    return x / denom


def global_constraint_loss(
    global_emb: torch.Tensor,
    region_emb: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    DOVE Global Constraint:
    Encourage global embedding to align with the mean of regional/token embeddings.
    """
    if region_emb is None or global_emb is None:
        return torch.tensor(0.0, device=global_emb.device if global_emb is not None else 'cpu')
        
    # Region emb shape: (B, R, D) -> Mean over regions -> (B, D)
    anchor = region_emb.mean(dim=1) 
    
    # Normalize
    global_emb = l2norm(global_emb, eps=eps)
    anchor = l2norm(anchor, eps=eps)

    # Cosine distance (1 - similarity)
    cos = (global_emb * anchor).sum(dim=-1)  # (B,)
    loss = 1.0 - cos
    return loss.mean()


# -----------------------------------------------------------------------------
# Main Loss Module
# -----------------------------------------------------------------------------

class DOVELoss(nn.Module):
    """
    Unified DOVE Loss: OpenCLIP InfoNCE + DOVE Global Constraint.
    """
    def __init__(
            self,
            lambda_g: float = 10.0,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.lambda_g = lambda_g
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor, 
        logit_scale: torch.Tensor, 
        image_regions: torch.Tensor = None,
        text_regions: torch.Tensor = None,
        output_dict: bool = False
    ):
        """
        Args:
            image_features: (B, D) Global image embeddings
            text_features: (B, D) Global text embeddings
            logit_scale: Learnable temperature parameter
            image_regions: (B, R, D) Optional regional image features for DOVE constraint
            text_regions: (B, L, D) Optional token text features for DOVE constraint
        """
        device = image_features.device
        
        # 1. InfoNCE (CLIP) Loss
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        # 2. DOVE Global Constraint Loss
        global_loss = torch.tensor(0.0, device=device)
        if self.lambda_g > 0.0:
            loss_g_img = global_constraint_loss(image_features, image_regions)
            loss_g_txt = global_constraint_loss(text_features, text_regions)
            
            # Combine available constraints
            terms = []
            if image_regions is not None: terms.append(loss_g_img)
            if text_regions is not None: terms.append(loss_g_txt)
            
            if terms:
                global_loss = sum(terms) / len(terms)

        # 3. Total Loss
        total_loss = contrastive_loss + self.lambda_g * global_loss

        if output_dict:
            return {
                "loss": total_loss,
                "contrastive_loss": contrastive_loss,
                "global_constraint_loss": global_loss
            }
        
        return total_loss