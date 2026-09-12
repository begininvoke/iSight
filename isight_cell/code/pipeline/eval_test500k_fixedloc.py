"""FIXED-LOCATION eval (location = mode over STAINED cells pi!=0). Evaluate cell model on the clean 500K test set (held out from BOTH models).
Reads ALL data/test500k/target_cells, true image labels from meta/test500k_master.csv.
Per cell -> intensity/location; aggregate per image (frac->quantity bin) -> image-level
int/loc/quantity F1. Also cell-level int/loc F1 vs weak label. For head-to-head vs MAIN model.
"""
import os, sys, glob
from pathlib import Path
import numpy as np, h5py, torch, pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score
ISIGHT = Path(os.environ.get("ISIGHT_ROOT", ""))   # data root; see README
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deps"))  # train_finetune, backbones
import train_finetune as T
from backbones.vision_encoders.uni2 import build_uni2
HF = ISIGHT / "iSight_ihc_foundation"
CKPT = Path(os.environ.get("CKPT", HF / "runs/round1/best_model.pt"))
TC = Path(os.environ.get("TC", str(HF / "data/test500k/target_cells")))
M = pd.read_csv(HF / "meta/test500k_master.csv").set_index("flat")
BS = 2048
QUA = ["none", "<25%", "25%-75%", ">75%"]; QMAP = {q: i for i, q in enumerate(QUA)}


@torch.inference_mode()
def main():
    rank = int(os.environ.get("RANK", "0")); local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1")); ddp = world > 1
    if ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl"); torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}"); is_main = rank == 0
    model = T.Net(build_uni2(drop_path_rate=0, drop_rate=0, attn_drop_rate=0), mlp_head=True, dropout=0.3).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["model"]); model.eval()
    m_, s_ = T.NORM_MEAN.to(dev), T.NORM_STD.to(dev)
    from torch.amp import autocast
    files = sorted(glob.glob(str(TC / "*.h5")))[rank::world]    # shard images across GPUs
    if is_main: print(f"[eval test500k] {len(files)}/shard x {world} GPU | CKPT={CKPT.name}", flush=True)
    cyi, cyl, cpi, cpl = [], [], [], []
    rows = []
    for k, p in enumerate(files):
        flat = Path(p).stem
        if flat not in M.index: continue
        r = M.loc[flat]
        yi = T.IMAP.get(str(r.staining_intensity)); yl = T.LMAP.get(str(r.staining_location)); yq = QMAP.get(str(r.staining_quantity))
        if yi is None or yl is None or yq is None: continue
        with h5py.File(p, "r") as f:
            n = int(f.attrs["n_cells"])
            if n == 0: continue
            n = min(n, int(os.environ.get("EVAL_K", "800")))   # cap cells/img -> faster, frac/F1 stable
            crops = f["crops"][:n]
        PI, PL = [], []
        for i in range(0, n, BS):
            x = T.crops_to_input(crops[i:i+BS], dev, False, m_, s_)
            with autocast("cuda", dtype=torch.bfloat16):
                li, ll = model(x)
            PI.append(li.argmax(-1).cpu().numpy()); PL.append(ll.argmax(-1).cpu().numpy())
        pi = np.concatenate(PI); pl = np.concatenate(PL)
        cyi.append(np.full(n, yi)); cyl.append(np.full(n, yl)); cpi.append(pi); cpl.append(pl)
        frac = (pi != 0).mean()
        ii = 0 if frac < 0.05 or (pi != 0).sum() == 0 else int(np.bincount(pi[pi != 0]).argmax())
        _lf = (pl != 0).mean()
        ll_ = 0 if _lf < 0.05 else int(np.bincount(pl[pl != 0], minlength=4).argmax())  # FIXED-D: loc-stained (pl!=0) + 5% gate
        qq = 0 if frac == 0 else (1 if frac <= .25 else (2 if frac <= .75 else 3))
        nc = len(pi); bc = np.bincount(pi, minlength=4)
        rows.append(dict(flat=flat, true_int=yi, true_loc=yl, true_qua=yq, pred_int=ii, pred_loc=ll_, pred_qua=qq,
                         frac=float(frac), n_cells=nc, n_neg=int(bc[0]), n_weak=int(bc[1]), n_mod=int(bc[2]), n_strong=int(bc[3])))
        if is_main and (k+1) % 500 == 0: print(f"  {k+1}/{len(files)} (rank0)", flush=True)
    local = dict(rows=rows, cyi=np.concatenate(cyi) if cyi else np.array([],int),
                 cyl=np.concatenate(cyl) if cyl else np.array([],int),
                 cpi=np.concatenate(cpi) if cpi else np.array([],int),
                 cpl=np.concatenate(cpl) if cpl else np.array([],int))
    if ddp:
        import torch.distributed as dist
        gath = [None]*world; dist.all_gather_object(gath, local)
        if rank != 0: dist.destroy_process_group(); return
        rows = [r for g in gath for r in g["rows"]]
        cyi = np.concatenate([g["cyi"] for g in gath]); cyl = np.concatenate([g["cyl"] for g in gath])
        cpi = np.concatenate([g["cpi"] for g in gath]); cpl = np.concatenate([g["cpl"] for g in gath])
    else:
        cyi, cyl, cpi, cpl = local["cyi"], local["cyl"], local["cpi"], local["cpl"]
    R = pd.DataFrame(rows)
    print(f"\n===== CELL-LEVEL ({len(cyi):,} cells, vs weak image label) =====")
    print(f"  intensity F1={f1_score(cyi,cpi,average='macro'):.4f} acc={(cyi==cpi).mean():.4f}")
    print(f"  location  F1={f1_score(cyl,cpl,average='macro'):.4f} acc={(cyl==cpl).mean():.4f}")
    print(f"\n===== IMAGE-LEVEL ({len(R)} imgs) =====")
    for name, tc, pc in [("intensity", R.true_int, R.pred_int), ("location", R.true_loc, R.pred_loc), ("quantity", R.true_qua, R.pred_qua)]:
        print(f"  {name:10} F1={f1_score(tc,pc,average='macro'):.4f} bal_acc={balanced_accuracy_score(tc,pc):.4f} acc={(tc==pc).mean():.4f}")
    print("\n  quantity confusion (rows true none/<25/25-75/>75):")
    print(confusion_matrix(R.true_qua, R.pred_qua, labels=[0, 1, 2, 3]))
    print("\n  per-bin:")
    for q in range(4):
        g = R[R.true_qua == q]
        if len(g): print(f"    {QUA[q]:8} n={len(g):4d} qua_acc={(g.pred_qua==q).mean():.3f} int_acc={(g.true_int==g.pred_int).mean():.3f}")
    R.to_csv(os.environ.get("OUT", str(HF / "runs/round1/eval_test500k_preds.csv")), index=False)
    print("DONE")
    if ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
