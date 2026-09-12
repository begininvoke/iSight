"""Fully finetune timm UNI2-h + 43 binary target-cell heads.

One binary head per class in `meta/classes_43.csv` (tissue x cell-type): given a cell crop,
its class head decides target vs non-target. Labels are the cluster labels from the
extraction step, which already writes `class_idx` in the 0..42 space.

DDP launch (3 GPUs):
    CUDA_VISIBLE_DEVICES=4,5,6 torchrun --nproc_per_node=3 train_target_head.py \
        --crops_h5 $DATA/crops_target.h5 \
        --uni2_ckpt $UNI2_CKPT_PATH \
        --epochs 10 --bs_per_gpu 256 --backbone_lr 1e-5 --head_lr 1e-3 \
        --out_dir $RUNS/target_cell_43cls

Per-cell schema in crops_target.h5:
    crops: (N, 64, 64, 3) uint8 RGB
    class_idx: (N,) int8 (0..42, row order of meta/classes_43.csv)
    label: (N,) int8 (0=non_target, 1=target)
    split: (N,) S5 (b'train' / b'val')
"""
import os, sys, time, json, argparse
from pathlib import Path

# CPU thread limits (must be set BEFORE importing torch/numpy)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np, pandas as pd, h5py
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.checkpoint import checkpoint_sequential
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import timm

# Explicit torch thread caps
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# ============= args =============
ap = argparse.ArgumentParser()
ap.add_argument("--crops_h5", required=True)
ap.add_argument("--uni2_ckpt", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--bs_per_gpu", type=int, default=256)
ap.add_argument("--backbone_lr", type=float, default=1e-5)
ap.add_argument("--head_lr", type=float, default=1e-3)
ap.add_argument("--weight_decay", type=float, default=1e-4)
ap.add_argument("--warmup_epochs", type=int, default=1)
ap.add_argument("--num_workers", type=int, default=2,
                help="DataLoader workers per rank (low to reduce RSS due to COW)")
ap.add_argument("--patience", type=int, default=2,
                help="early-stop patience (n epochs without F1 improvement)")
ap.add_argument("--seed", type=int, default=42)
# Dropout config
ap.add_argument("--drop_path", type=float, default=0.2,
                help="stochastic depth (DropPath) rate across transformer blocks")
ap.add_argument("--backbone_drop_rate", type=float, default=0.2,
                help="proj dropout (attention output + MLP fc1/fc2 input) in backbone")
ap.add_argument("--attn_drop_rate", type=float, default=0.0,
                help="attention weights dropout (usually 0 for ViT)")
ap.add_argument("--head_dropout", type=float, default=0.3,
                help="dropout before binary head (regularization at decision boundary)")

args = ap.parse_args()


# ============= DDP setup =============
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
dist.init_process_group(backend="nccl")
torch.cuda.set_device(LOCAL_RANK)
DEV = f"cuda:{LOCAL_RANK}"
IS_MASTER = (RANK == 0)
torch.manual_seed(args.seed + RANK)
np.random.seed(args.seed + RANK)

OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
def mprint(*a, **k):
    if IS_MASTER: print(*a, **k, flush=True)

mprint(f"[init] DDP world_size={WORLD_SIZE}, rank={RANK}, device={DEV}")


# ============= data =============
# Preload crops + put in shared memory (so all DataLoader workers share the same physical pages,
# no COW penalty across forks). Only rank 0 loads from disk; other ranks read from /dev/shm via
# torch shared tensor.
crops_t = class_t = label_t = split_t = None
if LOCAL_RANK == 0:
    mprint(f"[data] preload {args.crops_h5} → torch shared memory...")
with h5py.File(args.crops_h5, "r") as f:
    # Each rank loads (one OS-level file, fast buff/cache after first rank)
    crops_np = f["crops"][:]    # (N, 64, 64, 3) uint8
    class_np = f["class_idx"][:].astype(np.int64)
    label_np = f["label"][:].astype(np.int64)
    split_np = f["split"][:]    # S5 bytes
crops_t = torch.from_numpy(crops_np)
crops_t.share_memory_()    # storage goes to /dev/shm — workers fork w/o COW
del crops_np
class_t = torch.from_numpy(class_np)
class_t.share_memory_()
del class_np
label_t = torch.from_numpy(label_np)
label_t.share_memory_()
del label_np
# split_np is S5 bytes (object), keep as numpy
split_all = split_np

n = len(crops_t)
mprint(f"[data] {n:,} cells, crops {crops_t.element_size()*crops_t.numel()/1e9:.1f} GB (shared /dev/shm)")
# class_idx is already the 0..42 class index (the extraction step applied the 43-class table).
cls_np_all = class_t.numpy()
N_CLASSES = 43
assert cls_np_all.min() >= 0 and cls_np_all.max() < N_CLASSES, (
    f"class_idx must be 0..{N_CLASSES-1}; got {cls_np_all.min()}..{cls_np_all.max()}. "
    "Re-run extract_target_crops.py, which applies meta/classes_43.csv.")
class_idx_43 = class_t
mprint(f"[data] {N_CLASSES} classes, {len(np.unique(cls_np_all))} present in this file")

is_train = (split_all == b"train")
is_val = (split_all == b"val")
n_train = int(is_train.sum()); n_val = int(is_val.sum())
mprint(f"[data] train: {n_train:,}, val: {n_val:,}")

NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1,3,1,1)
NORM_STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1,3,1,1)


class CropDataset(Dataset):
    """Shared-memory dataset. Outputs (crop tensor, class_idx_remapped, label)."""
    def __init__(self, mask):
        self.idx = torch.from_numpy(np.where(mask)[0])
        self.idx.share_memory_()    # index also shared
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = int(self.idx[i])
        return crops_t[j], int(class_idx_43[j]), int(label_t[j])


def collate(batch):
    crops = torch.stack([b[0] for b in batch])     # (B, 64, 64, 3) uint8 — views of shared mem
    cls = torch.tensor([b[1] for b in batch], dtype=torch.int64)
    lab = torch.tensor([b[2] for b in batch], dtype=torch.int64)
    return crops, cls, lab


ds_tr = CropDataset(is_train)
ds_va = CropDataset(is_val)
samp_tr = DistributedSampler(ds_tr, shuffle=True, seed=args.seed)
samp_va = DistributedSampler(ds_va, shuffle=False)
dl_tr = DataLoader(ds_tr, batch_size=args.bs_per_gpu, sampler=samp_tr,
                    num_workers=args.num_workers, collate_fn=collate, drop_last=True,
                    persistent_workers=False, pin_memory=False)
dl_va = DataLoader(ds_va, batch_size=args.bs_per_gpu * 2, sampler=samp_va,
                    num_workers=args.num_workers, collate_fn=collate,
                    persistent_workers=False, pin_memory=False)
mprint(f"[data] loaders ready (tr {len(dl_tr)} steps/ep, va {len(dl_va)} steps/ep)")


# ============= model =============
mprint(f"[model] loading timm UNI2-h from {args.uni2_ckpt}...")
backbone = timm.create_model(
    "vit_giant_patch14_224",
    img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
    embed_dim=1536, mlp_ratio=2.66667*2, num_classes=0, no_embed_class=True,
    mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
    reg_tokens=8, dynamic_img_size=True,
    # Dropout config
    drop_path_rate=args.drop_path,        # stochastic depth (linear ramp 0 → drop_path across 24 blocks)
    proj_drop_rate=args.backbone_drop_rate,  # MLP/proj dropout in each block
    attn_drop_rate=args.attn_drop_rate,   # attention weight dropout
)
sd = torch.load(args.uni2_ckpt, map_location="cpu", weights_only=True)
miss, unexpected = backbone.load_state_dict(sd, strict=False)
# Strict=False here because dropout config doesn't change params, but
# safer to check no critical key is missing (DropPath is non-param module)
if miss or unexpected:
    mprint(f"  [warn] missing={len(miss)} unexpected={len(unexpected)}; first miss: {miss[:3]}, first unexpected: {unexpected[:3]}")
else:
    mprint(f"  ✅ load_state_dict OK (all 343 keys matched)")
backbone.set_grad_checkpointing(enable=True)
mprint(f"[model] backbone params: {sum(p.numel() for p in backbone.parameters())/1e6:.0f} M")
mprint(f"[model] dropout: drop_path={args.drop_path}, proj_drop={args.backbone_drop_rate}, attn_drop={args.attn_drop_rate}, head_dropout={args.head_dropout}")


class MultiHead(nn.Module):
    def __init__(self, n_classes, feat_dim=1536, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.W = nn.Parameter(torch.randn(n_classes, feat_dim, 2) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_classes, 2))
    def forward(self, feat, class_id):
        feat = self.dropout(feat)
        W_b = self.W[class_id]
        b_b = self.b[class_id]
        return torch.einsum("bf,bfo->bo", feat, W_b) + b_b


class Model(nn.Module):
    def __init__(self, backbone, n_classes, head_dropout=0.0):
        super().__init__()
        self.backbone = backbone
        self.head = MultiHead(n_classes, dropout=head_dropout)
    def forward(self, x, cls):
        feat = self.backbone(x)
        return self.head(feat, cls)


model = Model(backbone, N_CLASSES, head_dropout=args.head_dropout).to(DEV).to(memory_format=torch.channels_last)
model = DDP(model, device_ids=[LOCAL_RANK], find_unused_parameters=False)

# Differential LR
param_groups = [
    {"params": model.module.backbone.parameters(), "lr": args.backbone_lr, "name": "backbone"},
    {"params": model.module.head.parameters(), "lr": args.head_lr, "name": "head"},
]
opt = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay, fused=True)

total_steps = len(dl_tr) * args.epochs
warmup_steps = len(dl_tr) * args.warmup_epochs
def lr_at(step):
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    # cosine
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + np.cos(np.pi * progress))
mprint(f"[opt] AdamW fused, backbone_lr={args.backbone_lr}, head_lr={args.head_lr}, "
       f"warmup={warmup_steps} steps, total={total_steps} steps")


# ============= eval =============
def evaluate(model, dl):
    model.eval()
    all_pred, all_lab, all_cls = [], [], []
    with torch.inference_mode():
        for crops, cls, lab in dl:
            x = crops.to(DEV, non_blocking=True).permute(0,3,1,2).float() / 255.0
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False, antialias=True)
            x = (x - NORM_MEAN) / NORM_STD
            x = x.to(memory_format=torch.channels_last)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, cls.to(DEV))
            pred = logits.argmax(-1)
            all_pred.append(pred.cpu()); all_lab.append(lab); all_cls.append(cls)
    pred = torch.cat(all_pred).numpy()
    lab = torch.cat(all_lab).numpy()
    cls = torch.cat(all_cls).numpy()
    # gather across ranks
    pred_t = torch.from_numpy(pred).to(DEV)
    lab_t = torch.from_numpy(lab).to(DEV)
    cls_t = torch.from_numpy(cls).to(DEV)
    sizes = [torch.tensor([len(pred_t)], device=DEV) for _ in range(WORLD_SIZE)]
    dist.all_gather(sizes, torch.tensor([len(pred_t)], device=DEV))
    max_size = max(int(s.item()) for s in sizes)
    def pad(t):
        if len(t) < max_size:
            pad_t = torch.zeros(max_size - len(t), dtype=t.dtype, device=DEV)
            return torch.cat([t, pad_t])
        return t
    pred_p, lab_p, cls_p = pad(pred_t), pad(lab_t), pad(cls_t)
    g_pred = [torch.zeros(max_size, dtype=pred_p.dtype, device=DEV) for _ in range(WORLD_SIZE)]
    g_lab = [torch.zeros(max_size, dtype=lab_p.dtype, device=DEV) for _ in range(WORLD_SIZE)]
    g_cls = [torch.zeros(max_size, dtype=cls_p.dtype, device=DEV) for _ in range(WORLD_SIZE)]
    dist.all_gather(g_pred, pred_p); dist.all_gather(g_lab, lab_p); dist.all_gather(g_cls, cls_p)
    if IS_MASTER:
        all_pred = []; all_lab = []; all_cls = []
        for i, s in enumerate(sizes):
            n_i = int(s.item())
            all_pred.append(g_pred[i][:n_i].cpu().numpy())
            all_lab.append(g_lab[i][:n_i].cpu().numpy())
            all_cls.append(g_cls[i][:n_i].cpu().numpy())
        pred = np.concatenate(all_pred)
        lab = np.concatenate(all_lab)
        cls = np.concatenate(all_cls)
        # Overall + per-class F1
        f1 = f1_score(lab, pred, average="macro", zero_division=0)
        acc = accuracy_score(lab, pred)
        # Per-class
        rows = []
        for c in range(N_CLASSES):
            m = cls == c
            if m.sum() < 5: continue
            rows.append({
                "cls43": int(c),
                "n": int(m.sum()),
                "f1": float(f1_score(lab[m], pred[m], average="macro", zero_division=0)),
                "prec": float(precision_score(lab[m], pred[m], pos_label=1, zero_division=0)),
                "rec": float(recall_score(lab[m], pred[m], pos_label=1, zero_division=0)),
                "acc": float(accuracy_score(lab[m], pred[m])),
            })
        return {"f1_overall": float(f1), "acc_overall": float(acc), "per_class": rows}
    return None


# ============= train loop =============
best_f1 = 0
no_imp = 0
history = []
mprint(f"\n[train] {args.epochs} epochs, early-stop patience={args.patience}")
global_step = 0
for ep in range(1, args.epochs + 1):
    model.train()
    samp_tr.set_epoch(ep)
    t0 = time.time()
    losses = []
    for step, (crops, cls, lab) in enumerate(dl_tr):
        x = crops.to(DEV, non_blocking=True).permute(0,3,1,2).float() / 255.0
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False, antialias=True)
        x = (x - NORM_MEAN) / NORM_STD
        x = x.to(memory_format=torch.channels_last)
        cls = cls.to(DEV, non_blocking=True)
        lab = lab.to(DEV, non_blocking=True)

        lr_mult = lr_at(global_step)
        for pg in opt.param_groups:
            base = args.backbone_lr if pg["name"] == "backbone" else args.head_lr
            pg["lr"] = base * lr_mult

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, cls)
            loss = F.cross_entropy(logits, lab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        global_step += 1

        if step % 200 == 0 and IS_MASTER:
            print(f"  ep{ep} step {step}/{len(dl_tr)} loss={loss.item():.3f} lr_mult={lr_mult:.3f}", flush=True)

    if IS_MASTER:
        elapsed = time.time() - t0
        mprint(f"\nep{ep} train_loss={np.mean(losses):.3f} t={elapsed:.0f}s")

    # Eval
    metrics = evaluate(model, dl_va)
    if metrics and IS_MASTER:
        mprint(f"ep{ep} val: f1_overall={metrics['f1_overall']:.4f} acc={metrics['acc_overall']:.4f}")
        mprint(f"  per-class mean F1: {np.mean([r['f1'] for r in metrics['per_class']]):.4f}")
        history.append({"ep": ep, "train_loss": float(np.mean(losses)), **{k: v for k, v in metrics.items() if k != 'per_class'}})
        pd.DataFrame(history).to_csv(OUT / "history.csv", index=False)
        pd.DataFrame(metrics["per_class"]).to_csv(OUT / f"per_class_val_ep{ep:02d}.csv", index=False)

        # Save best
        if metrics["f1_overall"] > best_f1:
            best_f1 = metrics["f1_overall"]
            no_imp = 0
            torch.save({"model": model.module.state_dict(),
                         "ep": ep, "f1_overall": best_f1,
                         "args": vars(args), "n_heads": N_CLASSES},
                        OUT / "best.pt")
            mprint(f"  ★ new best F1={best_f1:.4f}, saved")
        else:
            no_imp += 1
            mprint(f"  (no improvement for {no_imp}/{args.patience} ep, best still {best_f1:.4f})")
        # Always save last
        torch.save({"model": model.module.state_dict(), "ep": ep, "args": vars(args), "n_heads": N_CLASSES},
                    OUT / "last.pt")

        # Early stop signal (broadcast from rank 0)
        stop_signal = torch.tensor([1 if no_imp >= args.patience else 0], device=DEV)
    else:
        stop_signal = torch.tensor([0], device=DEV)
    dist.broadcast(stop_signal, src=0)
    if stop_signal.item() == 1:
        mprint(f"[early-stop] no improvement for {args.patience} ep, stopping at ep{ep} (best={best_f1:.4f})")
        break

dist.destroy_process_group()
if IS_MASTER:
    mprint(f"\n[done] best f1_overall={best_f1:.4f}")
