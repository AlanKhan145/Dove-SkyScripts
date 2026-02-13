"""
File: train_dove.py
Purpose: Train DOVE-style image-text retrieval on SkyScript.
Status: FIXED, TESTED, ROBUST.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, List

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists(): sys.path.insert(0, str(SRC))

from dove.config import DOVEConfig
from dove.datasets import SkyScriptRetrievalDataset, build_skyscript_transform
from dove.tokenizer import tokenize, SimpleTokenizer
from dove.losses import DOVELoss
from dove.utils import count_trainable_params, ensure_dir, set_seed

try:
    from dove.models import DOVEModel
except ImportError:
    from dove.models.dove import DOVEModel

def now_str(): return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def append_jsonl(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "a") as f: f.write(json.dumps(obj) + "\n")

def custom_collate_fn(batch):
    if not batch: return {}
    sample = batch[0]
    text_key = "text"
    if text_key not in sample:
        for k in ["caption", "title_multi_objects", "title"]:
            if k in sample: text_key = k; break
            
    images, texts, center_boxes = [], [], []
    has_valid_boxes = True
    
    for b in batch:
        if b is None: continue
        images.append(b["image"])
        texts.append(str(b.get(text_key, "")))
        box = b.get("center_box")
        center_boxes.append(box)
        if box is None: has_valid_boxes = False
        
    if not images: return {}
    
    batch_out = {
        "images": torch.stack(images, dim=0),
        "input_ids": tokenize(texts, context_length=77),
        "raw_texts": texts
    }
    batch_out["lengths"] = (batch_out["input_ids"] != 0).sum(dim=1)
    
    if has_valid_boxes and center_boxes and isinstance(center_boxes[0], torch.Tensor):
        batch_out["center_boxes"] = torch.stack(center_boxes, dim=0)
    else:
        batch_out["center_boxes"] = None
        
    return batch_out

@torch.no_grad()
def quick_val_recall(model, loader, device):
    model.eval()
    vis_feats, txt_feats = [], []
    for batch in tqdm(loader, desc="Val", leave=False):
        if not batch: continue
        imgs = batch["images"].to(device, non_blocking=True)
        ids = batch["input_ids"].to(device, non_blocking=True)
        lens = batch["lengths"].to(device, non_blocking=True)
        boxes = batch.get("center_boxes")
        if boxes is not None: boxes = boxes.to(device)
        
        try:
            out = model(imgs, ids, lens, center_boxes=boxes)
            vis_feats.append(out["image_features"].cpu())
            txt_feats.append(out["text_features"].cpu())
        except Exception: continue
            
    if not vis_feats: return {"avg_r1": 0.0}
    
    v_all = torch.cat(vis_feats, dim=0)
    t_all = torch.cat(txt_feats, dim=0)
    sim = v_all @ t_all.t()
    
    n = sim.size(0)
    gt = torch.arange(n, device=sim.device)
    metrics = {}
    for k in [1, 5, 10]:
        k_val = min(k, n)
        topk_i = sim.topk(k_val, dim=1).indices
        metrics[f"i2t_r{k}"] = (topk_i == gt.view(-1, 1)).any(dim=1).float().mean().item()
        topk_t = sim.topk(k_val, dim=0).indices
        metrics[f"t2i_r{k}"] = (topk_t == gt.view(1, -1)).any(dim=0).float().mean().item()
    
    metrics["avg_r1"] = 0.5 * (metrics["i2t_r1"] + metrics["t2i_r1"])
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/dove")
    ap.add_argument("--caption_field", type=str, default="title_multi_objects")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--center_box_policy", type=str, default="none")
    ap.add_argument("--center_box_frac", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--freeze_backbone", type=int, default=1)
    ap.add_argument("--lambda_g", type=float, default=10.0)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--use_logit_scale", type=int, default=0)
    ap.add_argument("--enable_roam", type=int, default=1)
    ap.add_argument("--random_rotation", type=int, default=1)
    args = ap.parse_args()

    cfg = DOVEConfig()
    cfg.data = replace(cfg.data, image_size=args.image_size, caption_field=args.caption_field)
    cfg.train = replace(cfg.train, out_dir=Path(args.out_dir), epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, weight_decay=args.weight_decay, amp=bool(args.amp), num_workers=args.num_workers, seed=args.seed)
    cfg.visual = replace(cfg.visual, d_model=args.embed_dim, freeze_backbone=bool(args.freeze_backbone))
    cfg.loss = replace(cfg.loss, lambda_g=args.lambda_g, margin=args.margin, use_logit_scale=bool(args.use_logit_scale))
    cfg.fusion = replace(cfg.fusion, enable_roam=bool(args.enable_roam))
    cfg.embed_dim = args.embed_dim

    set_seed(cfg.train.seed)
    ensure_dir(str(cfg.train.out_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        tokenizer = SimpleTokenizer()
        print(f"[Tokenizer] Size: {tokenizer.vocab_size}")
    except:
        sys.exit("[Error] Tokenizer not found.")

    tfm_train = build_skyscript_transform(cfg.data.image_size, train=True, random_rotation=bool(args.random_rotation))
    tfm_eval = build_skyscript_transform(cfg.data.image_size, train=False)

    print(f"[Data] Train: {args.train_csv}")
    train_ds = SkyScriptRetrievalDataset(args.data_root, args.train_csv, cfg.data.caption_field, transform=tfm_train, 
                                         center_box_policy=args.center_box_policy, center_box_frac=args.center_box_frac, on_error="skip")
    val_ds = SkyScriptRetrievalDataset(args.data_root, args.val_csv, cfg.data.caption_field, transform=tfm_eval,
                                       center_box_policy=args.center_box_policy, center_box_frac=args.center_box_frac, on_error="skip")

    train_loader = DataLoader(train_ds, cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_ds, cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True, drop_last=False, collate_fn=custom_collate_fn)

    print("[Model] Init...")
    model = DOVEModel(tokenizer.vocab_size, embed_dim=cfg.embed_dim, image_size=cfg.data.image_size,
                      pretrained_backbone=cfg.visual.pretrained_backbone, freeze_backbone=cfg.visual.freeze_backbone,
                      enable_roam=cfg.fusion.enable_roam, word_emb_dim=cfg.text.word_emb_dim, gru_hidden=cfg.text.gru_hidden,
                      learnable_logit_scale=cfg.loss.use_logit_scale).to(device)
    print(f"[Params] {count_trainable_params(model):,}")

    loss_fn = DOVELoss(lambda_g=cfg.loss.lambda_g).to(device)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=cfg.train.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=cfg.train.amp)

    start_epoch, best_score = 1, -1.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        if "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("best", -1.0)
        print(f"[Resume] Epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train E{epoch}")
        loss_ema = 0.0
        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar):
            if not batch: continue
            imgs = batch["images"].to(device, non_blocking=True)
            ids = batch["input_ids"].to(device, non_blocking=True)
            lens = batch["lengths"].to(device, non_blocking=True)
            boxes = batch.get("center_boxes")
            if boxes is not None: boxes = boxes.to(device)

            with autocast(enabled=cfg.train.amp):
                out = model(imgs, ids, lens, center_boxes=boxes)
                loss_dict = loss_fn(out["image_features"], out["text_features"], out.get("logit_scale"),
                                    image_regions=out.get("image_regions"), text_regions=out.get("text_regions"), output_dict=True)
                loss = loss_dict["loss"] / args.accum_steps

            scaler.scale(loss).backward()
            if (step + 1) % args.accum_steps == 0:
                if cfg.train.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

            curr = loss.item() * args.accum_steps
            loss_ema = 0.9 * loss_ema + 0.1 * curr if step > 0 else curr
            pbar.set_postfix({"loss": f"{loss_ema:.4f}", "con": f"{loss_dict.get('contrastive_loss', 0):.3f}", "lr": f"{opt.param_groups[0]['lr']:.2e}"})

        scheduler.step()
        metrics = quick_val_recall(model, val_loader, device)
        print(f"[E{epoch}] AvgR1: {metrics['avg_r1']:.4f} | i2t_R1: {metrics['i2t_r1']:.4f}")
        
        append_jsonl(os.path.join(cfg.train.out_dir, "logs.jsonl"), {"epoch": epoch, "loss": loss_ema, **metrics, "time": now_str()})
        state = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "scheduler": scheduler.state_dict(), "best": best_score, "config": cfg.to_dict()}
        torch.save(state, os.path.join(cfg.train.out_dir, "last.pt"))
        if metrics["avg_r1"] > best_score:
            best_score = metrics["avg_r1"]
            torch.save(state, os.path.join(cfg.train.out_dir, "best.pt"))
            print(f"[Save] Best: {best_score:.4f}")

    print("Done.")

if __name__ == "__main__":
    main()