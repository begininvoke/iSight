"""Round-1 dual-head cell-level foundation training (STREAMING, DDP).

Scale: ~118M target cells (1.4 TB crops) across 83,391 imgs -> cannot load to RAM like
deps/train_finetune.py. Instead stream per-image "bags": each train step
reads one file's crops (sequential 24MB read, page-cache friendly: 1.4TB fits in 2TB RAM
cache so epoch 2+ is ~in-RAM speed) and samples K random cells (weak image-level label
broadcast to its cells).

Heads: intensity(4) + location(4). Loss = mean of two CEs. Backbone = timm UNI2-h
(fully fine-tuned, drop_path 0.2 / backbone_drop 0.2 / head dropout 0.3). Differential LR
(backbone 1e-5, head 1e-3), bf16, grad checkpointing.

Split: meta/split.csv (image-level, no leakage). Labels from crop_master (or LABEL_CSV
override for round-2 refined labels). Reuses Net / crops_to_input / aggregate_image from
train_finetune.
"""
import os, sys, time
from datetime import timedelta
from pathlib import Path
import numpy as np, h5py, torch, torch.nn as nn, torch.nn.functional as F, pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import f1_score

ISIGHT = Path(os.environ.get("ISIGHT_ROOT", ""))   # data root; see README
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deps"))  # train_finetune, backbones
import train_finetune as T
from backbones.vision_encoders.uni2 import build_uni2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from earlystop import EarlyStopper

HF = ISIGHT / "iSight_ihc_foundation"
TARGET = Path(os.environ.get("TARGET_OUT", HF / "data/target_cells"))
SPLIT = HF / "meta/split.csv"; CROPM = HF / "meta/crop_master.csv"
RUN = Path(os.environ.get("RUN_DIR", HF / "runs/round1"))
K_TRAIN = int(os.environ.get("K_TRAIN", "64"))
FILES_PER_BATCH = int(os.environ.get("FILES_PER_BATCH", "4"))
EPOCHS = int(os.environ.get("EPOCHS", "15"))
BACKBONE_LR = float(os.environ.get("BACKBONE_LR", "1e-5"))
HEAD_LR = float(os.environ.get("HEAD_LR", "1e-3"))
WD = float(os.environ.get("WD", "1e-4")); WARMUP = int(os.environ.get("WARMUP", "1"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
EVAL_BS = int(os.environ.get("EVAL_BS", "1024"))
EVAL_K = int(os.environ.get("EVAL_K", "500"))   # cap cells/img in val eval (frac estimate) -> fast eval, avoids NCCL barrier timeout
SMOKE = int(os.environ.get("SMOKE", "0"))          # >0: only this many train steps, no DDP, throughput probe
LABEL_OVERRIDE = os.environ.get("LABEL_CSV", "")


REFINE_DIR = os.environ.get("REFINE_DIR", "")   # Round-2: train from refined (agreeing) cells; val/test stay original
BALANCE = int(os.environ.get("BALANCE_INTENSITY", "0"))  # oversample train to ~balance the 4 intensity classes
# --- early stopping ---
EARLY_STOP = int(os.environ.get("EARLY_STOP", "1"))              # 0 disables, run the full budget
ES_METRIC = os.environ.get("ES_METRIC", "val_cell_avg_f1")       # validation macro-F1
ES_MIN_DELTA = float(os.environ.get("ES_MIN_DELTA", "0.001"))
ES_PATIENCE = int(os.environ.get("ES_PATIENCE", "2"))


def build_index(rank):
    cm = pd.read_csv(CROPM, usecols=["flat", "intensity", "staining_location"])
    cm["yi"] = cm.intensity.map(T.IMAP); cm["yl"] = cm.staining_location.map(T.LMAP)
    cm = cm.merge(pd.read_csv(SPLIT), on="flat")
    if LABEL_OVERRIDE:
        ov = pd.read_csv(LABEL_OVERRIDE)
        cm = cm.drop(columns=["yi", "yl"]).merge(ov, on="flat", how="inner")
    # train dir = REFINE_DIR (if set), eval dir = original TARGET (test/val unchanged)
    rdir = Path(REFINE_DIR) if REFINE_DIR else TARGET
    rexist = set(os.listdir(rdir)); texist = set(os.listdir(TARGET))
    def keep(row):
        d = rexist if (REFINE_DIR and row.split == "train") else texist
        return f"{row.flat}.h5" in d
    cm = cm[cm.apply(keep, axis=1)].reset_index(drop=True)
    def path(row):
        base = rdir if (REFINE_DIR and row.split == "train") else TARGET
        return str(base / f"{row.flat}.h5")
    cm["path"] = cm.apply(path, axis=1)
    if rank == 0:
        print(f"[index] {len(cm)} files | split {cm.split.value_counts().to_dict()} | "
              f"REFINE_DIR={'yes' if REFINE_DIR else 'no'} BALANCE={BALANCE}", flush=True)
    return cm


def balance_intensity(tr, rank, target=None):
    """Oversample (with replacement) minority intensity classes to ~equal counts."""
    cnt = tr.yi.value_counts().to_dict()
    if target is None:                       # ~equal: cap big, lift small -> total ~= original
        target = int(np.median(list(cnt.values())))
        target = max(target, int(0.6 * max(cnt.values())))
    parts = []
    rs = np.random.RandomState(0)
    for yi, g in tr.groupby("yi"):
        idx = rs.choice(len(g), target, replace=len(g) < target)
        parts.append(g.iloc[idx])
    out = pd.concat(parts, ignore_index=True)
    if rank == 0:
        print(f"[balance] target {target}/class | before {cnt} -> after {out.yi.value_counts().to_dict()} "
              f"({len(tr)}->{len(out)} imgs)", flush=True)
    return out


class BagDS(Dataset):
    def __init__(self, df, K):
        self.path = df.path.values; self.yi = df.yi.values.astype(np.int64)
        self.yl = df.yl.values.astype(np.int64); self.K = K
    def __len__(self): return len(self.path)
    def __getitem__(self, i):
        # read a CONTIGUOUS random block of K cells (h5py reads only needed chunks ~1.5MB)
        # instead of the whole 24MB file -> stays within the job's page-cache limit (OOM otherwise).
        with h5py.File(self.path[i], "r") as f:
            n = int(f.attrs["n_cells"])
            if self.K and n > self.K:
                r = np.random.randint(0, n - self.K + 1)
                crops = f["crops"][r:r + self.K]
            else:
                crops = f["crops"][:]
        return crops, self.yi[i], self.yl[i]


def bag_collate(batch):
    crops = np.concatenate([b[0] for b in batch])
    yi = np.concatenate([np.full(len(b[0]), b[1], np.int64) for b in batch])
    yl = np.concatenate([np.full(len(b[0]), b[2], np.int64) for b in batch])
    return crops, torch.from_numpy(yi), torch.from_numpy(yl)


@torch.inference_mode()
def evaluate(core, df, device, m_, s_, rank=0, world=1, ddp=False):
    """DISTRIBUTED val eval: each rank evaluates its 1/world shard of files (val split by
    image -> per-image aggregation complete per-rank), all_gather to rank 0 for F1.
    Returns (cell,img) on rank 0, (None,None) elsewhere."""
    from torch.amp import autocast
    core.eval()
    sub = df.iloc[rank::world] if ddp else df           # shard files across ranks
    cyi, cyl, cpi, cpl = [], [], [], []                  # cell-level true/pred
    iy, ip, ly, lp = [], [], [], []                      # image-level true/pred
    for p, yi, yl, flat in zip(sub.path.values, sub.yi.values, sub.yl.values, sub.flat.values):
        with h5py.File(p, "r") as f:
            n = int(f.attrs["n_cells"])
            if n == 0: continue
            crops = f["crops"][:]                         # full cells (distributed -> fast)
        PI, PL = [], []
        for i in range(0, n, EVAL_BS):
            x = T.crops_to_input(crops[i:i + EVAL_BS], device, False, m_, s_)
            with autocast("cuda", dtype=torch.bfloat16):
                li, ll = core(x)
            PI.append(li.argmax(-1).cpu().numpy()); PL.append(ll.argmax(-1).cpu().numpy())
        pi = np.concatenate(PI); pl = np.concatenate(PL)
        cyi.append(np.full(n, yi)); cyl.append(np.full(n, yl)); cpi.append(pi); cpl.append(pl)
        frac = (pi != 0).mean()
        ii = 0 if frac < 0.05 or (pi != 0).sum() == 0 else int(np.bincount(pi[pi != 0]).argmax())
        iy.append(yi); ip.append(ii); ly.append(yl); lp.append(int(np.bincount(pl, minlength=4).argmax()))
    local = dict(cyi=np.concatenate(cyi) if cyi else np.array([], int), cyl=np.concatenate(cyl) if cyl else np.array([], int),
                 cpi=np.concatenate(cpi) if cpi else np.array([], int), cpl=np.concatenate(cpl) if cpl else np.array([], int),
                 iy=np.array(iy, int), ip=np.array(ip, int), ly=np.array(ly, int), lp=np.array(lp, int))
    if ddp:
        gath = [None] * world
        torch.distributed.all_gather_object(gath, local)
        if rank != 0: return None, None
        local = {k: np.concatenate([g[k] for g in gath]) for k in local}
    cyi, cyl, cpi, cpl = local["cyi"], local["cyl"], local["cpi"], local["cpl"]
    cell = {"int_f1": f1_score(cyi, cpi, average="macro"), "loc_f1": f1_score(cyl, cpl, average="macro"),
            "int_acc": float((cyi == cpi).mean())}
    cell["avg_f1"] = (cell["int_f1"] + cell["loc_f1"]) / 2
    img = {"int_f1": f1_score(local["iy"], local["ip"], average="macro"),
           "loc_f1": f1_score(local["ly"], local["lp"], average="macro"), "n_img": len(local["iy"])}
    return cell, img


def main():
    rank = int(os.environ.get("RANK", "0")); local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1")); ddp = world > 1 and not SMOKE
    if ddp:
        torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=60)); torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}"); is_main = rank == 0
    if is_main: RUN.mkdir(parents=True, exist_ok=True)

    idx = build_index(rank)
    tr = idx[idx.split == "train"].reset_index(drop=True)
    va = idx[idx.split == "val"].reset_index(drop=True)
    te = idx[idx.split == "test"].reset_index(drop=True)
    if BALANCE:
        tr = balance_intensity(tr, rank, target=int(os.environ.get("BALANCE_TARGET", "0")) or None).reset_index(drop=True)

    backbone = build_uni2(drop_path_rate=0.2, drop_rate=0.2, attn_drop_rate=0.0)
    if int(os.environ.get("GRAD_CKPT", "1")) and hasattr(backbone, "set_grad_checkpointing"):
        backbone.set_grad_checkpointing(True)
    model = T.Net(backbone, mlp_head=True, dropout=0.3).to(device).to(memory_format=torch.channels_last)
    if ddp: model = nn.parallel.DistributedDataParallel(model, device_ids=[local])
    core = model.module if ddp else model

    opt = torch.optim.AdamW(
        [{"params": list(core.head_i.parameters()) + list(core.head_l.parameters()), "lr": HEAD_LR},
         {"params": core.backbone.parameters(), "lr": BACKBONE_LR}], weight_decay=WD, fused=True)

    ds = BagDS(tr, K_TRAIN)
    sampler = DistributedSampler(ds, shuffle=True) if ddp else None
    dl = DataLoader(ds, batch_size=FILES_PER_BATCH, shuffle=(sampler is None), sampler=sampler,
                    num_workers=NUM_WORKERS, collate_fn=bag_collate, pin_memory=True,
                    drop_last=True, persistent_workers=NUM_WORKERS > 0)
    steps = max(len(dl) * EPOCHS, 1); warm = max(len(dl) * WARMUP, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: st / warm if st < warm else 0.5 * (1 + np.cos(np.pi * (st - warm) / max(steps - warm, 1))))
    m_ = T.NORM_MEAN.to(device); s_ = T.NORM_STD.to(device)
    from torch.amp import autocast
    if is_main:
        print(f"[train] {len(tr)} imgs | {len(dl)} steps/ep x {EPOCHS} ep | K={K_TRAIN} FPB={FILES_PER_BATCH} "
              f"-> {K_TRAIN*FILES_PER_BATCH} cells/GPU/step", flush=True)

    if SMOKE:  # throughput probe: time first SMOKE steps
        model.train(); t0 = time.time(); nc = 0
        for st, (crops, yi, yl) in enumerate(dl):
            yi = yi.to(device); yl = yl.to(device); nc += len(crops)
            x = T.crops_to_input(crops, device, True, m_, s_)
            with autocast("cuda", dtype=torch.bfloat16):
                li, ll = model(x); loss = (F.cross_entropy(li, yi) + F.cross_entropy(ll, yl)) / 2
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            if st == 0: torch.cuda.synchronize(); t0 = time.time(); nc = 0  # skip warmup step
            if st >= SMOKE: break
        torch.cuda.synchronize(); dt = time.time() - t0
        print(f"[SMOKE] {SMOKE} steps in {dt:.1f}s | {SMOKE/dt:.2f} step/s | {nc/dt:.0f} cells/s (1 GPU) | "
              f"est epoch ({len(dl)} steps) ~ {len(dl)/(SMOKE/dt)/60:.1f} min/GPU", flush=True)
        return

    best = -1; hist = []; es = EarlyStopper(ES_MIN_DELTA, ES_PATIENCE); sel_ep = None
    for ep in range(1, EPOCHS + 1):
        if ddp: sampler.set_epoch(ep)
        model.train(); t0 = time.time(); losses = []; st0 = time.time(); ncs = 0
        for si, (crops, yi, yl) in enumerate(dl):
            yi = yi.to(device); yl = yl.to(device); ncs += len(crops)
            x = T.crops_to_input(crops, device, True, m_, s_)
            with autocast("cuda", dtype=torch.bfloat16):
                li, ll = model(x); loss = (F.cross_entropy(li, yi) + F.cross_entropy(ll, yl)) / 2
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            losses.append(float(loss))
            if is_main and (si + 1) % 50 == 0:
                dt = time.time() - st0
                print(f"  [ep{ep} {si+1}/{len(dl)}] {ncs/dt:.0f} cells/s/GPU loss={np.mean(losses[-50:]):.4f}", flush=True)
                st0 = time.time(); ncs = 0
        cell, img = evaluate(core, va, device, m_, s_, rank, world, ddp)  # all ranks (all_gather)
        if is_main:
            rec = {"ep": ep, "loss": float(np.mean(losses)), "t": round(time.time() - t0),
                   "val_cell_int_f1": cell["int_f1"], "val_cell_loc_f1": cell["loc_f1"],
                   "val_cell_avg_f1": cell["avg_f1"], "val_img_int_f1": img["int_f1"], "val_img_loc_f1": img["loc_f1"]}
            hist.append(rec); pd.DataFrame(hist).to_csv(RUN / "history.csv", index=False)
            print(f"[ep{ep:02d}] loss={rec['loss']:.4f} {rec['t']}s | val cell int_f1={cell['int_f1']:.3f} "
                  f"loc_f1={cell['loc_f1']:.3f} avg={cell['avg_f1']:.3f} | img int_f1={img['int_f1']:.3f} "
                  f"loc_f1={img['loc_f1']:.3f}", flush=True)
            if cell["avg_f1"] > best:
                best = cell["avg_f1"]
                torch.save({"model": core.state_dict(), "ep": ep, "val_avg_f1": best}, RUN / "best_model.pt")
                print(f"  -> saved best (val avg_f1={best:.4f})", flush=True)
        # --- early stopping ---
        stop = torch.zeros(1, device=device)
        if is_main and EARLY_STOP:
            if es.update(rec[ES_METRIC]): stop[0] = 1
            print(f"  [early-stop] {ES_METRIC}={rec[ES_METRIC]:.6f} {es.status()}", flush=True)
        if ddp:
            torch.distributed.barrier()
            torch.distributed.broadcast(stop, src=0)
        if stop.item():
            sel_ep = ep                                   # the stopping epoch is the model
            if is_main:
                torch.save({"model": core.state_dict(), "ep": ep, ES_METRIC: rec[ES_METRIC]}, RUN / "selected_model.pt")
                print(f"  -> early stop at ep{ep:02d}; selected_model.pt = ep{ep:02d}", flush=True)
            break
    # final test on best (all ranks load best + participate)
    if ddp: torch.distributed.barrier()
    sel = RUN / "selected_model.pt"                       # written only if early stopping fired
    final_ckpt = sel if (EARLY_STOP and sel.exists()) else RUN / "best_model.pt"
    core.load_state_dict(torch.load(final_ckpt, map_location=device)["model"])
    if is_main: print(f"[final] test on {final_ckpt.name} (ep{sel_ep if sel_ep else 'best'})", flush=True)
    cell, img = evaluate(core, te, device, m_, s_, rank, world, ddp)
    if is_main:
        print(f"\n[TEST] cell int_f1={cell['int_f1']:.3f} loc_f1={cell['loc_f1']:.3f} int_acc={cell['int_acc']:.3f} "
              f"| img int_f1={img['int_f1']:.3f} loc_f1={img['loc_f1']:.3f}\nDONE", flush=True)
    if ddp: torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
