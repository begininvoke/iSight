"""Round-3 per-cell trainer that RE-SAMPLES a fresh balanced 16M each epoch (no pre-built shards).

Each epoch e: draw CELLS_PER_CLASS cells/intensity-class (4 classes) from the refined_both scan
(.percell_scan_*.npz: files, n, yi, yl) with seed=e -> globally-random balanced sample. Each
(rank,worker) owns files by (fi % gn == gid), reads its sampled crops ONCE into a RAM buffer,
shuffles, yields -> per-cell diversity like pre-shuffled shards, but a NEW 16M every epoch.
VAL/TEST = data/target_cells (identical to train_shards). Env: RUN_DIR SCAN_NPZ TARGET EPOCHS BS
CELLS_PER_CLASS BACKBONE_LR HEAD_LR WD WARMUP NUM_WORKERS GRAD_CKPT
"""
import os, sys, time
from datetime import timedelta
from pathlib import Path
import numpy as np, pandas as pd, h5py, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from sklearn.metrics import f1_score
ISIGHT = Path(os.environ.get("ISIGHT_ROOT", ""))   # data root; see README
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deps"))  # train_finetune, backbones
import train_finetune as T
from backbones.vision_encoders.uni2 import build_uni2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from earlystop import EarlyStopper

HF = ISIGHT / "iSight_ihc_foundation_final"
RUN = Path(os.environ.get("RUN_DIR", HF / "runs/round3_resample"))
SCAN_NPZ = Path(os.environ["SCAN_NPZ"])                       # .percell_scan_<refined_both>.npz
TARGET = Path(os.environ.get("TARGET", ISIGHT / "iSight_ihc_foundation/data/target_cells"))  # val/test
CROPM = Path(os.environ.get("CROPM", ISIGHT / "iSight_ihc_foundation/meta/crop_master.csv"))
SPLIT = Path(os.environ.get("SPLIT", ISIGHT / "iSight_ihc_foundation/meta/split.csv"))
EPOCHS = int(os.environ.get("EPOCHS", "20")); BS = int(os.environ.get("BS", "1024"))
CELLS_PER_CLASS = int(os.environ.get("CELLS_PER_CLASS", "4000000"))
BACKBONE_LR = float(os.environ.get("BACKBONE_LR", "1e-5")); HEAD_LR = float(os.environ.get("HEAD_LR", "1e-3"))
WD = float(os.environ.get("WD", "1e-4")); WARMUP = int(os.environ.get("WARMUP", "1"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8")); EVAL_BS = 2048
SEED = int(os.environ.get("SEED", "0"))                       # varies resampling + model-init RNG (0 = legacy)
RESUME = os.environ.get("RESUME", "")                        # path to ckpt_epNN.pt to continue from
SAVE_EVERY_EPOCH = int(os.environ.get("SAVE_EVERY_EPOCH", "1"))  # write per-epoch full ckpt (model+opt+sched) for resume
# --- early stopping (paper protocol): val macro-F1, min_delta 0.1%, patience 2, keep-current ---
EARLY_STOP = int(os.environ.get("EARLY_STOP", "1"))              # 0 disables, run the full budget
ES_METRIC = os.environ.get("ES_METRIC", "val_cell_avg_f1")       # validation macro-F1
ES_MIN_DELTA = float(os.environ.get("ES_MIN_DELTA", "0.001"))
ES_PATIENCE = int(os.environ.get("ES_PATIENCE", "2"))


class ResampleIDS(IterableDataset):
    """Fresh balanced 16M sample each epoch from the scan; per-worker file ownership + RAM buffer shuffle."""
    def __init__(self, files, ns, yis, yls, rank, world, cells_per_class, seed=0):
        self.files = files; self.ns = ns.astype(np.int64); self.yis = yis.astype(np.int8)
        self.yls = yls.astype(np.int8); self.rank = rank; self.world = world
        self.cpc = cells_per_class; self.epoch = 0; self.seed = int(seed)
    def set_epoch(self, e): self.epoch = e
    def _sample(self, rng):
        FI, LI, YI = [], [], []
        for c in range(4):
            fc = np.where(self.yis == c)[0]
            if len(fc) == 0: continue
            csum = np.cumsum(self.ns[fc]); tot = int(csum[-1]); prev = np.concatenate([[0], csum[:-1]])
            g = rng.integers(0, tot, self.cpc); fpos = np.searchsorted(csum, g, side="right"); sel = fc[fpos]
            FI.append(sel.astype(np.int64)); LI.append((g - prev[fpos]).astype(np.int64)); YI.append(np.full(self.cpc, c, np.int8))
        return np.concatenate(FI), np.concatenate(LI), np.concatenate(YI)
    def __iter__(self):
        wi = get_worker_info(); wid = wi.id if wi else 0; nw = wi.num_workers if wi else 1
        gid = self.rank * nw + wid; gn = self.world * nw
        ep = self.epoch
        while True:                                          # infinite -> train loop bounds steps
            s = self.seed * 10_007 + ep                      # SEED offsets the per-epoch draw
            FI, LI, YI = self._sample(np.random.default_rng(s))      # SAME 16M on every worker (seed=s)
            own = (FI % gn) == gid                            # this worker owns files fi%gn==gid
            fi, li, yi = FI[own], LI[own], YI[own]
            order = np.argsort(fi, kind="stable"); fi, li, yi = fi[order], li[order], yi[order]
            buf_c = np.empty((len(fi), 64, 64, 3), np.uint8); buf_i = yi.copy(); buf_l = np.empty(len(fi), np.int8)
            i = 0
            while i < len(fi):
                f = fi[i]; j = i
                while j < len(fi) and fi[j] == f: j += 1
                try:
                    with h5py.File(self.files[f], "r") as h:
                        n = int(h.attrs["n_cells"]); loc = np.clip(li[i:j], 0, max(n - 1, 0))
                        uniq, inv = np.unique(loc, return_inverse=True)
                        buf_c[i:j] = h["crops"][uniq][inv]
                    buf_l[i:j] = self.yls[f]
                except Exception:
                    buf_c[i:j] = 0; buf_l[i:j] = 0
                i = j
            perm = np.random.default_rng(s * 100003 + gid).permutation(len(buf_c))   # shuffle worker buffer
            for k in perm:
                yield buf_c[k], int(buf_i[k]), int(buf_l[k])
            ep += 1


def cell_collate(batch):
    return np.stack([b[0] for b in batch]), torch.tensor([b[1] for b in batch]), torch.tensor([b[2] for b in batch])


def build_eval_df(split):
    cm = pd.read_csv(CROPM, usecols=["flat", "intensity", "staining_location"])
    cm["yi"] = cm.intensity.map(T.IMAP); cm["yl"] = cm.staining_location.map(T.LMAP)
    cm = cm.dropna(subset=["yi", "yl"]).merge(pd.read_csv(SPLIT), on="flat")
    cm = cm[cm.split == split].copy()
    exist = set(os.listdir(TARGET))
    cm = cm[cm.flat.map(lambda f: f"{f}.h5" in exist)].reset_index(drop=True)
    cm["path"] = cm.flat.map(lambda f: str(TARGET / f"{f}.h5"))
    cm["yi"] = cm.yi.astype(int); cm["yl"] = cm.yl.astype(int)
    return cm


@torch.inference_mode()
def evaluate(core, df, device, m_, s_, rank, world, ddp):
    from torch.amp import autocast
    core.eval(); sub = df.iloc[rank::world] if ddp else df
    cyi, cyl, cpi, cpl, iy, ip, ly, lp = [], [], [], [], [], [], [], []
    for p, yi, yl in zip(sub.path.values, sub.yi.values, sub.yl.values):
        with h5py.File(p, "r") as f:
            n = int(f.attrs["n_cells"])
            if n == 0: continue
            crops = f["crops"][:]
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
        lf = (pl != 0).mean(); ll_ = 0 if lf < 0.05 else int(np.bincount(pl[pl != 0], minlength=4).argmax())
        iy.append(yi); ip.append(ii); ly.append(yl); lp.append(ll_)
    local = dict(cyi=np.concatenate(cyi) if cyi else np.array([], int), cyl=np.concatenate(cyl) if cyl else np.array([], int),
                 cpi=np.concatenate(cpi) if cpi else np.array([], int), cpl=np.concatenate(cpl) if cpl else np.array([], int),
                 iy=np.array(iy, int), ip=np.array(ip, int), ly=np.array(ly, int), lp=np.array(lp, int))
    if ddp:
        gath = [None] * world; torch.distributed.all_gather_object(gath, local)
        if rank != 0: return None, None
        local = {k: np.concatenate([g[k] for g in gath]) for k in local}
    cyi, cyl, cpi, cpl = local["cyi"], local["cyl"], local["cpi"], local["cpl"]
    cell = {"int_f1": f1_score(cyi, cpi, average="macro"), "loc_f1": f1_score(cyl, cpl, average="macro"), "int_acc": float((cyi == cpi).mean())}
    cell["avg_f1"] = (cell["int_f1"] + cell["loc_f1"]) / 2
    img = {"int_f1": f1_score(local["iy"], local["ip"], average="macro"), "loc_f1": f1_score(local["ly"], local["lp"], average="macro"), "n_img": len(local["iy"])}
    return cell, img


def main():
    rank = int(os.environ.get("RANK", "0")); local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1")); ddp = world > 1
    if ddp:
        torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=60)); torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}"); is_main = rank == 0
    torch.manual_seed(SEED); np.random.seed(SEED)            # SEED varies head init + dropout draw
    if is_main: RUN.mkdir(parents=True, exist_ok=True)

    z = np.load(SCAN_NPZ, allow_pickle=True); files = z["files"]; ns = z["n"]; yis = z["yi"]; yls = z["yl"]
    total_cells = CELLS_PER_CLASS * 4; steps_per_ep = total_cells // (BS * world)
    va = build_eval_df("val"); te = build_eval_df("test")
    if is_main:
        print(f"[resample] {len(files)} src imgs | {total_cells/1e6:.0f}M cells/ep (RE-SAMPLED each epoch) | "
              f"{steps_per_ep} steps/ep x {EPOCHS} ep | BS={BS}/GPU x {world} | val {len(va)} test {len(te)}", flush=True)

    backbone = build_uni2(drop_path_rate=0.2, drop_rate=0.2, attn_drop_rate=0.0)
    if int(os.environ.get("GRAD_CKPT", "1")) and hasattr(backbone, "set_grad_checkpointing"):
        backbone.set_grad_checkpointing(True)
    model = T.Net(backbone, mlp_head=True, dropout=0.3).to(device).to(memory_format=torch.channels_last)
    if ddp: model = nn.parallel.DistributedDataParallel(model, device_ids=[local])
    core = model.module if ddp else model
    opt = torch.optim.AdamW(
        [{"params": list(core.head_i.parameters()) + list(core.head_l.parameters()), "lr": HEAD_LR},
         {"params": core.backbone.parameters(), "lr": BACKBONE_LR}], weight_decay=WD, fused=True)
    m_ = T.NORM_MEAN.to(device); s_ = T.NORM_STD.to(device)
    from torch.amp import autocast
    total_steps = max(steps_per_ep * EPOCHS, 1); warm = max(steps_per_ep * WARMUP, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: st / warm if st < warm else 0.5 * (1 + np.cos(np.pi * (st - warm) / max(total_steps - warm, 1))))

    ds = ResampleIDS(files, ns, yis, yls, rank, world, CELLS_PER_CLASS, seed=SEED)
    dl = DataLoader(ds, batch_size=BS, num_workers=NUM_WORKERS, collate_fn=cell_collate, pin_memory=True, drop_last=True, persistent_workers=False)
    best = -1; hist = []; es = EarlyStopper(ES_MIN_DELTA, ES_PATIENCE); sel_ep = None; start_ep = 1
    if RESUME:                                               # continue from a saved per-epoch ckpt
        ck = torch.load(RESUME, map_location=device)
        core.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep = int(ck["ep"]) + 1; best = float(ck.get("best", -1))
        hp = RUN / "history.csv"
        if hp.exists(): hist = pd.read_csv(hp)[lambda d: d.ep < start_ep].to_dict("records")
        if EARLY_STOP and hist: es.replay([r[ES_METRIC] for r in hist])   # counter survives resume
        if is_main: print(f"[resume] loaded {RESUME} -> start ep{start_ep}, best so far {best:.4f}", flush=True)
    for ep in range(start_ep, EPOCHS + 1):
        ds.set_epoch(ep); model.train(); t0 = time.time(); losses = []; st0 = time.time(); ncs = 0; it = iter(dl)
        for si in range(steps_per_ep):
            crops, yi, yl = next(it); yi = yi.to(device); yl = yl.to(device); ncs += len(crops)
            x = T.crops_to_input(crops, device, True, m_, s_)
            with autocast("cuda", dtype=torch.bfloat16):
                lpi, lpl = model(x); loss = (F.cross_entropy(lpi, yi) + F.cross_entropy(lpl, yl)) / 2
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); losses.append(loss.item())
            if is_main and (si + 1) % 100 == 0:
                dt = time.time() - st0; print(f"  [ep{ep} {si+1}/{steps_per_ep}] {ncs/dt:.0f} cells/s/GPU loss={np.mean(losses[-100:]):.4f}", flush=True); st0 = time.time(); ncs = 0
        cell, img = evaluate(core, va, device, m_, s_, rank, world, ddp)
        if is_main:
            rec = {"ep": ep, "loss": float(np.mean(losses)), "t": round(time.time() - t0), "val_cell_int_f1": cell["int_f1"], "val_cell_loc_f1": cell["loc_f1"], "val_cell_avg_f1": cell["avg_f1"], "val_img_int_f1": img["int_f1"], "val_img_loc_f1": img["loc_f1"]}
            hist.append(rec); pd.DataFrame(hist).to_csv(RUN / "history.csv", index=False)
            print(f"[ep{ep:02d}] loss={rec['loss']:.4f} {rec['t']}s | val cell int_f1={cell['int_f1']:.3f} loc_f1={cell['loc_f1']:.3f} avg={cell['avg_f1']:.3f} | img int_f1={img['int_f1']:.3f} loc_f1={img['loc_f1']:.3f}", flush=True)
            if cell["avg_f1"] > best:
                best = cell["avg_f1"]; torch.save({"model": core.state_dict(), "ep": ep, "val_avg_f1": best}, RUN / "best_model.pt"); print(f"  -> saved best (val avg_f1={best:.4f})", flush=True)
            if SAVE_EVERY_EPOCH:                              # full ckpt (model+opt+sched) -> resume from any epoch
                torch.save({"model": core.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "ep": ep, "best": best, "val_avg_f1": cell["avg_f1"], "seed": SEED},
                           RUN / f"ckpt_ep{ep:02d}.pt")
        # --- early stopping: validation macro-F1, min_delta 0.1%, patience 2, keep-current ---
        stop = torch.zeros(1, device=device)
        if is_main and EARLY_STOP:
            if es.update(rec[ES_METRIC]): stop[0] = 1
            print(f"  [early-stop] {ES_METRIC}={rec[ES_METRIC]:.6f} {es.status()}", flush=True)
        if ddp:
            torch.distributed.barrier()
            torch.distributed.broadcast(stop, src=0)
        if stop.item():
            sel_ep = ep                                   # keep-current: THIS epoch is the model
            if is_main:
                torch.save({"model": core.state_dict(), "ep": ep, ES_METRIC: rec[ES_METRIC]}, RUN / "selected_model.pt")
                print(f"  -> early stop at ep{ep:02d}; selected_model.pt = ep{ep:02d} (keep-current, NOT best-so-far)", flush=True)
            break
    if ddp: torch.distributed.barrier()
    sel = RUN / "selected_model.pt"                       # written only if early stopping fired
    final_ckpt = sel if (EARLY_STOP and sel.exists()) else RUN / "best_model.pt"
    core.load_state_dict(torch.load(final_ckpt, map_location=device)["model"])
    if is_main: print(f"[final] test on {final_ckpt.name} (ep{sel_ep if sel_ep else 'best'})", flush=True)
    cell, img = evaluate(core, te, device, m_, s_, rank, world, ddp)
    if is_main: print(f"\n[TEST] cell int_f1={cell['int_f1']:.3f} loc_f1={cell['loc_f1']:.3f} int_acc={cell['int_acc']:.3f} | img int_f1={img['int_f1']:.3f} loc_f1={img['loc_f1']:.3f}\nDONE", flush=True)
    if ddp: torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
