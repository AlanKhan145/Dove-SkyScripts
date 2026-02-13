# src/dove/models/visual.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

try:
    from torchvision.ops import roi_align
    HAS_ROI_ALIGN = True
except ImportError:
    HAS_ROI_ALIGN = False

@dataclass
class VisualOut:
    global_emb: torch.Tensor
    region_emb: torch.Tensor
    scale_emb: torch.Tensor

class ResNetMSV(nn.Module):
    """
    Multi-Scale Visual Encoder.
    """
    def __init__(
        self,
        embed_dim: int = 512,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        roi_out: int = 3,
        roi_grid: int = 6,
        add_center_box: bool = True,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        backbone = resnet50(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.proj2 = self._build_proj(256, embed_dim, dropout)
        self.proj3 = self._build_proj(512, embed_dim, dropout)
        self.proj4 = self._build_proj(1024, embed_dim, dropout)
        self.proj5 = self._build_proj(2048, embed_dim, dropout)
        self.roi_proj = self._build_proj(2048, embed_dim, dropout)

        self.scale_gate = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4)
        )

        self.roi_out = roi_out
        self.roi_grid = roi_grid
        self.add_center_box = add_center_box

        if freeze_backbone:
            for m in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
                for p in m.parameters():
                    p.requires_grad = False

    def _build_proj(self, in_dim, out_dim, dropout):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Flatten(2)
        )

    def _gap_proj(self, x, proj_layer):
        x_gap = F.adaptive_avg_pool2d(x, (1, 1))
        return proj_layer(x_gap).squeeze(-1)

    def _get_grid_regions(self, c5):
        regions = F.adaptive_avg_pool2d(c5, (self.roi_grid, self.roi_grid))
        regions = self.roi_proj(regions)
        return regions.transpose(1, 2)

    def forward(self, x: torch.Tensor, image_size: int = 256, center_boxes: Optional[torch.Tensor] = None) -> VisualOut:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Global
        e2 = self._gap_proj(c2, self.proj2)
        e3 = self._gap_proj(c3, self.proj3)
        e4 = self._gap_proj(c4, self.proj4)
        e5 = self._gap_proj(c5, self.proj5)
        scale_emb = torch.stack([e2, e3, e4, e5], dim=1)
        
        concat = scale_emb.view(x.shape[0], -1)
        weights = F.softmax(self.scale_gate(concat), dim=-1).unsqueeze(-1)
        global_emb = (scale_emb * weights).sum(dim=1)

        # Regional
        B = x.shape[0]
        if HAS_ROI_ALIGN and self.add_center_box and center_boxes is not None:
            device = x.device
            step = image_size / self.roi_grid
            shifts = torch.arange(0, self.roi_grid, dtype=torch.float32, device=device) * step
            yy, xx = torch.meshgrid(shifts, shifts, indexing='ij')
            x1 = xx.reshape(-1); y1 = yy.reshape(-1)
            x2 = x1 + step; y2 = y1 + step
            grid_boxes = torch.stack([x1, y1, x2, y2], dim=1).unsqueeze(0).repeat(B, 1, 1)
            
            center_boxes_abs = center_boxes * image_size
            all_boxes = torch.cat([center_boxes_abs.unsqueeze(1), grid_boxes], dim=1)
            
            batch_inds = torch.arange(B, device=device).view(B, 1, 1).expand(-1, all_boxes.size(1), -1)
            rois_fmt = torch.cat([batch_inds, all_boxes], dim=2).reshape(-1, 5)
            
            spatial_scale = c5.shape[-1] / image_size
            regions = roi_align(c5, rois_fmt, output_size=(self.roi_out, self.roi_out), spatial_scale=spatial_scale)
            
            regions = F.adaptive_avg_pool2d(regions, (1, 1))
            regions = self.roi_proj(regions).squeeze(-1)
            region_emb = regions.view(B, -1, self.roi_proj[0].out_channels)
        else:
            region_emb = self._get_grid_regions(c5)

        return VisualOut(global_emb, region_emb, scale_emb)