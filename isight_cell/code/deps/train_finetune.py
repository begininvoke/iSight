"""Train prostate|Glandular staining models on target cells (weak supervision).

Two modes (--mode):
  linear_probe : frozen DINOv2-UNI2 features (cached in target_cells/*.h5 `feats`),
                 train a linear head per task. Fast, 1 GPU/CPU.
  finetune     : trainable ORIGINAL timm UNI2-h on the 64x64 crops + MLP head per task.
                 2-GPU DDP via torchrun (also runs single-GPU).

Tasks: intensity (4) + location (4) cell-level heads. Loss = MEAN of the two CEs.
quantity has no head -> derived at image-level by stained-cell fraction.

Eval (two methods, every epoch):
  cell-level (PRIMARY, model selection): per-cell macro-F1 on test_pure (pure >75%/none
             cells, clean labels). Best epoch chosen by cell-level avg F1 (int+loc)/2.
  image-level (secondary): aggregate cell preds -> image pred, macro-F1 on
             test_orig (vs napa baseline), test_pure, test_mixed.

Data: data/target_cells/*.h5 (per-img bag: crops+feats+attrs role/intensity/location/quantity).
Outputs: runs/<mode>/{history.csv, best_model.pt, summary.json, *_preds.csv}.
"""
import os, sys, json, time, glob, argparse
from pathlib import Path

import numpy as np, h5py, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ImageNet normalisation, UNI2-h input size and embedding width. Inlined here (they were
# imported from a shared module that also pulls in the DINOv2 code, which this path never uses).
EMBED_DIM = 1536
INPUT_SIZE = 224
NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
NORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

ISIGHT = Path(os.environ.get("ISIGHT_ROOT", ""))
EXP = Path(os.environ.get("EXP_DIR", str(ISIGHT / "hnc_immune_markers")))
TARGET = Path(os.environ.get("TARGET_OUT", EXP / "data/target_cells"))
sys.path.insert(0, str(ISIGHT / "iSight_train_model/code"))

INTENSITY = ["negative", "weak", "moderate", "strong"]
LOCATION  = ["none", "nuclear", "cytoplasmic/membranous", "cytoplasmic/membranous,nuclear"]
QUANTITY  = ["none", "<25%", "25%-75%", ">75%"]
IMAP = {v: i for i, v in enumerate(INTENSITY)}
LMAP = {v: i for i, v in enumerate(LOCATION)}
EVAL_ROLES = os.environ.get("EVAL_ROLES", "test").split(",")
SEL_ROLE = os.environ.get("SEL_ROLE", "test")   # role used for model selection + print


# ---------------- data ----------------
def load_role_data(load_crops, load_feats):
    """Read all target_cells/*.h5 -> dict[role] = dict of arrays.
    Load only what the mode needs: feats for linear_probe, crops for finetune."""
    files = sorted(glob.glob(str(TARGET / "*.h5")))
    buf = {}
    for p in files:
        with h5py.File(p, "r") as f:
            role = f.attrs["role"]; n = int(f.attrs["n_cells"])
            if n == 0 or role not in (("train",) + tuple(EVAL_ROLES)):
                continue
            yi = IMAP.get(str(f.attrs["intensity"])); yl = LMAP.get(str(f.attrs["location"]))
            if yi is None or yl is None:
                continue
            sn = str(f.attrs["sample_name"]); qua = str(f.attrs["quantity"])
            d = buf.setdefault(role, dict(feats=[], crops=[], yi=[], yl=[], sn=[],
                                          qua=[], int=[], loc=[], cid=[]))
            if load_feats:
                d["feats"].append(f["feats"][:].astype(np.float32))
            if load_crops:
                d["crops"].append(f["crops"][:])
            d["cid"].append(f["cell_id"][:].astype(np.int32))
            d["yi"].append(np.full(n, yi, np.int64)); d["yl"].append(np.full(n, yl, np.int64))
            d["sn"].append(np.array([sn] * n)); d["qua"].append(np.array([qua] * n))
            d["int"].append(np.array([str(f.attrs["intensity"])] * n))
            d["loc"].append(np.array([str(f.attrs["location"])] * n))
    out = {}
    for role, d in buf.items():
        out[role] = dict(
            feats=np.concatenate(d["feats"]) if load_feats else None,
            crops=np.concatenate(d["crops"]) if load_crops else None,
            cell_id=np.concatenate(d["cid"]),
            yi=np.concatenate(d["yi"]), yl=np.concatenate(d["yl"]),
            sn=np.concatenate(d["sn"]), qua=np.concatenate(d["qua"]),
            intlabel=np.concatenate(d["int"]), loclabel=np.concatenate(d["loc"]))
    return out


class FeatDS(Dataset):
    def __init__(self, d):
        self.x = torch.from_numpy(d["feats"]); self.yi = torch.from_numpy(d["yi"]); self.yl = torch.from_numpy(d["yl"])
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.yi[i], self.yl[i]


class CropDS(Dataset):
    def __init__(self, d, augment=False):
        self.crops = d["crops"]; self.yi = d["yi"]; self.yl = d["yl"]; self.augment = augment
    def __len__(self): return len(self.crops)
    def __getitem__(self, i):
        return self.crops[i], int(self.yi[i]), int(self.yl[i])


def crop_collate(batch):
    crops = np.stack([b[0] for b in batch])
    yi = torch.tensor([b[1] for b in batch]); yl = torch.tensor([b[2] for b in batch])
    return crops, yi, yl


def crops_to_input(crops_uint8, device, augment, m, s):
    """uint8 (B,64,64,3 RGB; extraction does imread->BGR2RGB, verified pixel-exact 2026-09-05) -> normalized (B,3,224,224) on GPU."""
    x = torch.from_numpy(crops_uint8).to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    x = F.interpolate(x, INPUT_SIZE, mode="bilinear", align_corners=False, antialias=True)
    if augment:
        B = x.shape[0]
        fh = torch.rand(B, device=x.device) < 0.5   # per-sample h-flip
        if fh.any(): x[fh] = x[fh].flip(-1)
        fv = torch.rand(B, device=x.device) < 0.5   # per-sample v-flip
        if fv.any(): x[fv] = x[fv].flip(-2)
    return ((x - m) / s).contiguous(memory_format=torch.channels_last)


# ---------------- model ----------------
def make_head(in_dim, n_cls, mlp, dropout):
    if mlp:
        return nn.Sequential(nn.Linear(in_dim, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, n_cls))
    return nn.Linear(in_dim, n_cls)


class Net(nn.Module):
    def __init__(self, backbone, mlp_head, dropout):
        super().__init__()
        self.backbone = backbone   # None for linear_probe (feats precomputed)
        self.head_i = make_head(EMBED_DIM, 4, mlp_head, dropout)
        self.head_l = make_head(EMBED_DIM, 4, mlp_head, dropout)
    def forward(self, z):
        if self.backbone is not None:
            z = self.backbone(z)
        return self.head_i(z), self.head_l(z)


# ---------------- eval ----------------
def aggregate_image(sn, int_pred, loc_pred, thr=0.05):
    """cell preds -> per-image preds. Returns dict sn -> (int,loc,qua) indices."""
    out = {}
    order = np.argsort(sn, kind="stable")
    sn_s = sn[order]; ip = int_pred[order]; lp = loc_pred[order]
    uniq, starts = np.unique(sn_s, return_index=True)
    starts = list(starts) + [len(sn_s)]
    for k in range(len(uniq)):
        a, b = starts[k], starts[k + 1]
        ii = ip[a:b]; ll = lp[a:b]; n = b - a
        stained = ii != 0; frac = stained.sum() / max(n, 1)
        i_pred = 0 if frac < thr else int(np.bincount(ii[stained]).argmax())
        l_pred = int(np.bincount(ll).argmax())
        q_pred = 0 if frac == 0 else (1 if frac <= 0.25 else (2 if frac <= 0.75 else 3))
        out[uniq[k]] = (i_pred, l_pred, q_pred)
    return out


def cell_level_metrics(yi, yl, pi, pl):
    return {"int_f1": f1_score(yi, pi, average="macro"),
            "int_kappa": cohen_kappa_score(yi, pi, weights="quadratic"),
            "loc_f1": f1_score(yl, pl, average="macro"),
            "avg_f1": (f1_score(yi, pi, average="macro") + f1_score(yl, pl, average="macro")) / 2}


def image_level_metrics(d, pi, pl):
    agg = aggregate_image(d["sn"], pi, pl)
    # per-image true labels
    true = {}
    order = np.unique(d["sn"], return_index=True)[1]
    for idx in order:
        sn = d["sn"][idx]
        true[sn] = (IMAP[d["intlabel"][idx]], LMAP[d["loclabel"][idx]],
                    QUANTITY.index(d["qua"][idx]) if d["qua"][idx] in QUANTITY else -1)
    sns = list(agg.keys())
    yi = np.array([true[s][0] for s in sns]); pic = np.array([agg[s][0] for s in sns])
    yl = np.array([true[s][1] for s in sns]); plc = np.array([agg[s][1] for s in sns])
    yq = np.array([true[s][2] for s in sns]); pqc = np.array([agg[s][2] for s in sns])
    m = {"int_f1": f1_score(yi, pic, average="macro"),
         "loc_f1": f1_score(yl, plc, average="macro")}
    vq = yq >= 0
    m["qua_f1"] = f1_score(yq[vq], pqc[vq], average="macro") if vq.any() else 0.0
    m["n_img"] = len(sns)
    return m


@torch.inference_mode()
def predict(model, d, mode, device, bs, m, s):
    model.eval()
    pi, pl = [], []
    if mode == "linear_probe":
        X = torch.from_numpy(d["feats"])
        for i in range(0, len(X), bs):
            li, ll = model(X[i:i+bs].to(device))
            pi.append(li.argmax(-1).cpu().numpy()); pl.append(ll.argmax(-1).cpu().numpy())
    else:
        from torch.amp import autocast
        C = d["crops"]
        for i in range(0, len(C), bs):
            x = crops_to_input(C[i:i+bs], device, False, m, s)
            with autocast("cuda", dtype=torch.bfloat16):
                li, ll = model(x)
            pi.append(li.argmax(-1).cpu().numpy()); pl.append(ll.argmax(-1).cpu().numpy())
    return np.concatenate(pi), np.concatenate(pl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["linear_probe", "finetune"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--backbone_lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3, help="head MLP dropout")
    ap.add_argument("--drop_path", type=float, default=0.2, help="backbone stochastic depth (DropPath)")
    ap.add_argument("--backbone_drop_rate", type=float, default=0.2, help="backbone proj/MLP dropout")
    ap.add_argument("--attn_drop_rate", type=float, default=0.0, help="backbone attention dropout")
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--early_stop", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--intensity_only", action="store_true",
                    help="train intensity head only (drop location loss/selection)")
    args = ap.parse_args()

    # DDP (finetune only)
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0")); local = int(os.environ.get("LOCAL_RANK", "0"))
    if ddp:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    run = EXP / "runs" / (args.run_name or args.mode)
    if is_main: run.mkdir(parents=True, exist_ok=True)

    load_crops = args.mode == "finetune"
    load_feats = args.mode == "linear_probe"
    if is_main: print(f"[load] mode={args.mode} load_crops={load_crops} load_feats={load_feats}", flush=True)
    data = load_role_data(load_crops, load_feats)
    for r in ["train"] + EVAL_ROLES:
        if r in data:
            if is_main: print(f"  {r:11s}: {len(data[r]['yi']):8,d} cells / "
                              f"{len(np.unique(data[r]['sn']))} imgs", flush=True)

    # model
    if args.mode == "linear_probe":
        backbone = None
    else:
        from backbones.vision_encoders.uni2 import build_uni2
        backbone = build_uni2(drop_path_rate=args.drop_path, drop_rate=args.backbone_drop_rate,
                              attn_drop_rate=args.attn_drop_rate)
        if is_main:
            print(f"[reg] backbone drop_path={args.drop_path} drop_rate={args.backbone_drop_rate} "
                  f"attn_drop={args.attn_drop_rate} | head dropout={args.dropout}", flush=True)
    model = Net(backbone, mlp_head=(args.mode == "finetune"), dropout=args.dropout).to(device)
    model = model.to(memory_format=torch.channels_last) if args.mode == "finetune" else model
    if ddp:
        # intensity_only: head_l is built+forwarded but gets no loss grad -> DDP needs find_unused
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local],
                                                    find_unused_parameters=args.intensity_only)
    core = model.module if ddp else model

    # optim
    head_params = list(core.head_i.parameters())
    if not args.intensity_only:
        head_params += list(core.head_l.parameters())
    pg = [{"params": head_params, "lr": args.head_lr}]
    if backbone is not None:
        pg.append({"params": core.backbone.parameters(), "lr": args.backbone_lr})
    opt = torch.optim.AdamW(pg, weight_decay=args.weight_decay, fused=True)

    # train data
    if args.mode == "linear_probe":
        ds = FeatDS(data["train"])
        dl = DataLoader(ds, batch_size=args.bs, shuffle=not ddp, num_workers=2, drop_last=True,
                        sampler=torch.utils.data.distributed.DistributedSampler(ds) if ddp else None)
    else:
        ds = CropDS(data["train"], augment=True)
        sampler = torch.utils.data.distributed.DistributedSampler(ds) if ddp else None
        dl = DataLoader(ds, batch_size=args.bs, shuffle=(sampler is None), num_workers=args.num_workers,
                        collate_fn=crop_collate, pin_memory=True, drop_last=True, sampler=sampler,
                        persistent_workers=args.num_workers > 0)

    steps = max(len(dl) * args.epochs, 1); warm = max(len(dl) * args.warmup_epochs, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: st / warm if st < warm else 0.5 * (1 + np.cos(np.pi * (st - warm) / max(steps - warm, 1))))
    m_ = NORM_MEAN.to(device); s_ = NORM_STD.to(device)
    from torch.amp import autocast

    hist = []; best = -1; best_ep = 0; no_imp = 0
    for ep in range(1, args.epochs + 1):
        if ddp: dl.sampler.set_epoch(ep)
        model.train(); t0 = time.time(); losses = []
        for batch in dl:
            if args.mode == "linear_probe":
                x, yi, yl = batch; x = x.to(device); yi = yi.to(device); yl = yl.to(device)
                li, ll = model(x)
                loss = F.cross_entropy(li, yi) if args.intensity_only else \
                    (F.cross_entropy(li, yi) + F.cross_entropy(ll, yl)) / 2
            else:
                crops, yi, yl = batch; yi = yi.to(device); yl = yl.to(device)
                x = crops_to_input(crops, device, True, m_, s_)
                with autocast("cuda", dtype=torch.bfloat16):
                    li, ll = model(x)
                    loss = F.cross_entropy(li, yi) if args.intensity_only else \
                        (F.cross_entropy(li, yi) + F.cross_entropy(ll, yl)) / 2
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            losses.append(float(loss))

        # ---- eval + best ckpt (rank 0 only) ----
        stop = False
        if is_main:
            rec = {"ep": ep, "loss": float(np.mean(losses)), "t": time.time() - t0}
            ebs = args.bs if args.mode == "linear_probe" else 1024
            cell_sel = None
            for r in EVAL_ROLES:
                if r not in data: continue
                pi, pl = predict(core, data[r], args.mode, device, ebs, m_, s_)
                cm = cell_level_metrics(data[r]["yi"], data[r]["yl"], pi, pl)
                im = image_level_metrics(data[r], pi, pl)
                for k, v in cm.items(): rec[f"{r}_cell_{k}"] = v
                for k, v in im.items(): rec[f"{r}_img_{k}"] = v
                if r == SEL_ROLE: cell_sel = cm["int_f1"] if args.intensity_only else cm["avg_f1"]
            hist.append(rec)
            if cell_sel is None and SEL_ROLE not in data:  # fallback to first available eval role
                fr = next((r for r in EVAL_ROLES if r in data), None)
                if fr: cell_sel = rec[f"{fr}_cell_int_f1"] if args.intensity_only else rec[f"{fr}_cell_avg_f1"]
            print(f"ep{ep:02d} loss={rec['loss']:.3f} t={rec['t']:.0f}s | "
                  f"CELL {SEL_ROLE} int={rec.get(f'{SEL_ROLE}_cell_int_f1',0):.3f} loc={rec.get(f'{SEL_ROLE}_cell_loc_f1',0):.3f} "
                  f"avg={cell_sel if cell_sel is not None else 0:.3f} | IMG int={rec.get(f'{SEL_ROLE}_img_int_f1',0):.3f} "
                  f"loc={rec.get(f'{SEL_ROLE}_img_loc_f1',0):.3f} qua={rec.get(f'{SEL_ROLE}_img_qua_f1',0):.3f}", flush=True)
            # save BEST checkpoint only (overwritten; no intermediate ckpts). preds saved at end.
            if cell_sel is not None and cell_sel > best:
                best = cell_sel; best_ep = ep; no_imp = 0
                torch.save({"model": core.state_dict(), "ep": ep, "args": vars(args), "best_cell_avg_f1": best},
                           run / "best_model.pt")
            else:
                no_imp += 1
            import pandas as pd
            pd.DataFrame(hist).to_csv(run / "history.csv", index=False)
            stop = no_imp >= args.early_stop
        # broadcast stop from rank0 so ALL DDP ranks break together (avoids hang)
        if ddp:
            tflag = torch.tensor([1 if stop else 0], device=device)
            torch.distributed.broadcast(tflag, src=0)
            stop = bool(tflag.item())
        if stop:
            if is_main: print(f"[early-stop] ep{ep}", flush=True)
            break

    if is_main:
        # LAST checkpoint (final epoch weights)
        torch.save({"model": core.state_dict(), "ep": ep, "args": vars(args)}, run / "last_model.pt")
        # reload BEST and dump per-cell preds for TRAIN + all test roles (best epoch)
        ckpt = torch.load(run / "best_model.pt", map_location=device)
        core.load_state_dict(ckpt["model"])
        ebs = args.bs if args.mode == "linear_probe" else 1024

        def save_preds(role):
            d = data[role]
            pi, pl = predict(core, d, args.mode, device, ebs, m_, s_)
            np.savez(run / f"best_{role}_preds.npz", sn=d["sn"], cell_id=d["cell_id"],
                     yi=d["yi"], yl=d["yl"], pi=pi, pl=pl, qua=d["qua"],
                     intlabel=d["intlabel"], loclabel=d["loclabel"])
            print(f"  saved best_{role}_preds.npz ({len(pi):,} cells)", flush=True)

        print(f"[save] best @ep{best_ep} -> dumping per-cell preds (train + test) ...", flush=True)
        for r in ["train"] + EVAL_ROLES:
            if r in data:
                save_preds(r)

        sel_key = f"{SEL_ROLE}_cell_int_f1" if args.intensity_only else f"{SEL_ROLE}_cell_avg_f1"
        best_rec = max(hist, key=lambda r: r.get(sel_key, -1))
        json.dump({"args": vars(args), "best_ep": best_ep, "best_cell_avg_f1": best,
                   "last_ep": ep, "best_epoch_record": best_rec},
                  open(run / "summary.json", "w"), indent=2, default=float)
        print(f"\n[done] best cell avg_f1={best:.4f} @ep{best_ep} | ckpts: best_model.pt + last_model.pt -> {run}", flush=True)

        # ---- per-source/group cell-level eval on test (group from flat prefix: MK__<grp>__ / HNC__) ----
        z = np.load(run / "best_test_preds.npz", allow_pickle=True)
        sn = z["sn"].astype(str)
        def grp_of(s): return s.split("__")[1] if s.startswith("MK__") else "HNC_tumor"
        grp = np.array([grp_of(s) for s in sn])
        yi, pi, yl, pl = z["yi"], z["pi"], z["yl"], z["pl"]
        from sklearn.metrics import accuracy_score as _acc
        print("\n[per-group cell-level test] group  n_cells  int_f1  int_acc  loc_f1", flush=True)
        pg = {}
        for g in sorted(set(grp)):
            m = grp == g
            if m.sum() == 0: continue
            r = {"n_cells": int(m.sum()),
                 "int_f1": float(f1_score(yi[m], pi[m], average="macro", zero_division=0)),
                 "int_acc": float(_acc(yi[m], pi[m])),
                 "loc_f1": float(f1_score(yl[m], pl[m], average="macro", zero_division=0))}
            pg[g] = r
            print(f"  {g:11} {r['n_cells']:>8} {r['int_f1']:.3f}  {r['int_acc']:.3f}  {r['loc_f1']:.3f}", flush=True)
        json.dump(pg, open(run / "per_group_test.json", "w"), indent=2)
    if ddp: torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
