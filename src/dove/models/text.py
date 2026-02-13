# src/dove/models/text.py
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class TextOut:
    global_feat: torch.Tensor  # (B, D)
    token_feats: torch.Tensor  # (B, T, D)
    mask: torch.Tensor         # (B, T)
    lengths: torch.Tensor      # (B,)

class TextEncoderBiGRU_DTGA(nn.Module):
    """
    BiGRU Text Encoder with DTGA (Dual-flow Traffic Gating Assistant).
    Fixes: Safe masking for FP16 training.
    """
    def __init__(
        self,
        vocab_size: int,
        word_emb_dim: int = 300,
        gru_hidden: int = 256,
        out_dim: int = 512,
        pad_id: int = 0,
        attn_hidden: int = 256,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.pad_id = pad_id
        
        self.embedding = nn.Embedding(vocab_size, word_emb_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        
        self.bigru = nn.GRU(
            word_emb_dim, 
            gru_hidden, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # DTGA Fusion Gate
        self.gate_fc = nn.Sequential(
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.Sigmoid()
        )
        
        # Projection
        self.out_proj = nn.Sequential(
            nn.Linear(gru_hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention Pooling
        self.attn_fc = nn.Sequential(
            nn.Linear(out_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1)
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor = None) -> TextOut:
        x = self.embedding(input_ids)
        x = self.dropout(x)
        
        if lengths is not None:
            lengths_cpu = lengths.cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            out_packed, _ = self.bigru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out_packed, batch_first=True, total_length=input_ids.size(1)
            )
        else:
            out, _ = self.bigru(x)
            
        # DTGA Logic
        hidden_dim = out.shape[-1] // 2
        h_fwd = out[:, :, :hidden_dim]
        h_bwd = out[:, :, hidden_dim:]
        
        gate = self.gate_fc(out) 
        fused = gate * h_fwd + (1 - gate) * h_bwd 
        
        token_feats = self.out_proj(fused) 
        
        # Masking
        mask = (input_ids != self.pad_id) # (B, T)
        attn_scores = self.attn_fc(token_feats).squeeze(-1) # (B, T)
        
        # FIX: Safe masking for FP16 (avoid -1e9 overflow)
        min_val = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(~mask, min_val)
        
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        global_feat = (token_feats * attn_weights).sum(dim=1)
        
        return TextOut(global_feat, token_feats, mask, lengths)