# src/dove/models/dove.py
from __future__ import annotations
import math
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .text import TextEncoderBiGRU_DTGA, TextOut
from .visual import ResNetMSV, VisualOut

class SelfAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*4, d_model), nn.Dropout(dropout))
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, x, key_padding_mask=None):
        r = x
        x = self.ln1(x)
        a, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = r + a
        r = x
        x = self.ln2(x)
        return r + self.mlp(x)

class CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*4, d_model), nn.Dropout(dropout))
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, q, k, v):
        r = q
        q = self.ln1(q)
        a, _ = self.attn(q, k, v, need_weights=False)
        q = r + a
        r = q
        q = self.ln2(q)
        return r + self.mlp(q)

class DOVEModel(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int = 0, *, embed_dim: int = 512, image_size: int = 256, 
                 pretrained_backbone: bool = True, freeze_backbone: bool = True, roi_out: int = 3, roi_grid: int = 6, 
                 add_center_box: bool = True, word_emb_dim: int = 300, gru_hidden: int = 256, txt_attn_dim: int = 256,
                 enable_roam: bool = True, roam_layers: int = 1, n_heads: int = 8, roam_dropout: float = 0.1,
                 learnable_logit_scale: bool = False, initial_temperature: float = 0.07, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.enable_roam = enable_roam
        self.image_size = image_size

        self.visual = ResNetMSV(embed_dim=embed_dim, pretrained_backbone=pretrained_backbone, freeze_backbone=freeze_backbone,
                                roi_out=roi_out, roi_grid=roi_grid, add_center_box=add_center_box, dropout=roam_dropout)
        self.text = TextEncoderBiGRU_DTGA(vocab_size=vocab_size, word_emb_dim=word_emb_dim, gru_hidden=gru_hidden,
                                          out_dim=embed_dim, pad_id=pad_id, attn_hidden=txt_attn_dim)

        if self.enable_roam:
            self.v_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.v_self_attn = nn.ModuleList([SelfAttnBlock(embed_dim, n_heads, roam_dropout) for _ in range(roam_layers)])
            self.t_self_attn = nn.ModuleList([SelfAttnBlock(embed_dim, n_heads, roam_dropout) for _ in range(roam_layers)])
            self.t_cross_attn = nn.ModuleList([CrossAttnBlock(embed_dim, n_heads, roam_dropout) for _ in range(roam_layers)])
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.t_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        else:
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.t_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / initial_temperature))
        if not learnable_logit_scale: self.logit_scale.requires_grad = False

    def forward(self, images: torch.Tensor, input_ids: Optional[torch.Tensor] = None, 
                lengths: Optional[torch.Tensor] = None, *, center_boxes: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        
        v_out: VisualOut = self.visual(images, image_size=self.image_size, center_boxes=center_boxes)
        if input_ids is None:
            v_feat = self.v_proj(v_out.global_emb) if not self.enable_roam else v_out.global_emb 
            return {"image_features": F.normalize(v_feat, dim=-1)}

        t_out: TextOut = self.text(input_ids, lengths)

        if self.enable_roam:
            B = images.shape[0]
            cls_token = self.v_cls_token.expand(B, -1, -1)
            v_seq = torch.cat([cls_token, v_out.scale_emb, v_out.region_emb], dim=1)
            for blk in self.v_self_attn: v_seq = blk(v_seq)
            v_global = self.v_proj(v_seq[:, 0])

            t_seq = t_out.token_feats
            t_mask = ~t_out.mask
            for blk in self.t_self_attn: t_seq = blk(t_seq, key_padding_mask=t_mask)
            
            v_kv = torch.cat([v_out.scale_emb, v_out.region_emb], dim=1)
            for blk in self.t_cross_attn: t_seq = blk(t_seq, v_kv, v_kv)
            
            mask_float = t_out.mask.unsqueeze(-1).float()
            t_sum = (t_seq * mask_float).sum(dim=1)
            t_len = mask_float.sum(dim=1).clamp_min(1.0)
            t_global = self.t_proj(t_sum / t_len)
        else:
            v_global = self.v_proj(v_out.global_emb)
            t_global = self.t_proj(t_out.global_feat)

        return {
            "image_features": F.normalize(v_global, dim=-1),
            "text_features": F.normalize(t_global, dim=-1),
            "logit_scale": self.logit_scale.exp(),
            "image_regions": v_out.region_emb,
            "text_regions": t_out.token_feats,
        }

    @torch.no_grad()
    def encode_image(self, images, center_boxes=None):
        return self.forward(images, center_boxes=center_boxes)["image_features"]

    @torch.no_grad()
    def encode_text(self, input_ids, lengths=None):
        t_out = self.text(input_ids, lengths)
        return F.normalize(self.t_proj(t_out.global_feat), dim=-1)