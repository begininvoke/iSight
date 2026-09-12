"""Predict the Round-N model on ALL target cells of every image, recording per-cell
prediction + probability. Writes results BACK into each target_cells/<flat>.h5 as new
datasets (keeps per-image organization for downstream pseudo-label refinement):

  pred_intensity  (N,)   int8    argmax of intensity head (0=neg 1=weak 2=mod 3=strong)
  prob_intensity  (N,4)  float16 softmax over intensity classes
  pred_location   (N,)   int8    argmax of location head (0=none 1=nuc 2=cyto/mem 3=mixed)
  prob_location   (N,4)  float16 softmax over location classes

Sharded across GPUs (N_SHARDS/SHARD_ID). Pure GPU inference on stored crops.
"""
import os, sys, glob, time
from pathlib import Path
import numpy as np, h5py, torch
ISIGHT = Path(os.environ.get("ISIGHT_ROOT", ""))   # data root; see README
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deps"))  # train_finetune, backbones
import train_finetune as T
from backbones.vision_encoders.uni2 import build_uni2

HF = ISIGHT / "iSight_ihc_foundation"
TARGET = Path(os.environ.get("TARGET_OUT", HF / "data/target_cells"))
CKPT = Path(os.environ.get("CKPT", HF / "runs/round1/best_model.pt"))
BS = int(os.environ.get("BS", "2048"))
N_SHARDS = int(os.environ.get("N_SHARDS", "1")); SHARD_ID = int(os.environ.get("SHARD_ID", "0"))
OVERWRITE = int(os.environ.get("OVERWRITE", "0"))


@torch.inference_mode()
def main():
    dev = torch.device("cuda")
    model = T.Net(build_uni2(drop_path_rate=0, drop_rate=0, attn_drop_rate=0), mlp_head=True, dropout=0.3).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"]); model.eval()
    model = model.to(memory_format=torch.channels_last)
    m_, s_ = T.NORM_MEAN.to(dev), T.NORM_STD.to(dev)
    from torch.amp import autocast
    files = sorted(glob.glob(str(TARGET / "*.h5")))
    sf = os.environ.get("SPLIT_FILTER", "")           # e.g. "val,test" -> only those split flats
    if sf:
        import pandas as pd
        keep = set(pd.read_csv(HF / "meta/split.csv").query("split in @sf.split(',')").flat)
        files = [p for p in files if Path(p).stem in keep]
    files = files[SHARD_ID::N_SHARDS]
    print(f"[predict_all shard {SHARD_ID}/{N_SHARDS}] {len(files)} imgs | CKPT={CKPT.name} | split={sf or 'all'}", flush=True)
    t0 = time.time(); c = {"ok": 0, "skip": 0, "empty": 0, "err": 0}; ncell = 0
    for k, p in enumerate(files):
        try:
            with h5py.File(p, "r") as f:
                if not OVERWRITE and "pred_intensity" in f: c["skip"] += 1; continue
                n = int(f.attrs["n_cells"])
                if n == 0: c["empty"] += 1; continue
                crops = f["crops"][:]
            PIp, PLp = [], []
            for i in range(0, n, BS):
                x = T.crops_to_input(crops[i:i+BS], dev, False, m_, s_)
                with autocast("cuda", dtype=torch.bfloat16):
                    li, ll = model(x)
                PIp.append(torch.softmax(li.float(), -1).cpu().numpy())
                PLp.append(torch.softmax(ll.float(), -1).cpu().numpy())
            pi = np.concatenate(PIp); pl = np.concatenate(PLp)        # (n,4) each
            with h5py.File(p, "a") as f:
                for name in ("pred_intensity", "prob_intensity", "pred_location", "prob_location"):
                    if name in f: del f[name]
                f.create_dataset("pred_intensity", data=pi.argmax(1).astype(np.int8))
                f.create_dataset("prob_intensity", data=pi.astype(np.float16))
                f.create_dataset("pred_location", data=pl.argmax(1).astype(np.int8))
                f.create_dataset("prob_location", data=pl.astype(np.float16))
                f.attrs["pred_ckpt"] = CKPT.name
            c["ok"] += 1; ncell += n
        except Exception as e:
            c["err"] += 1
            if c["err"] <= 5: print(f"  ERR {Path(p).stem[:40]}: {type(e).__name__} {e}", flush=True)
        if (k+1) % 1000 == 0:
            print(f"  [{SHARD_ID}] {k+1}/{len(files)} {c} | {ncell/1e6:.1f}M cells {(k+1)/(time.time()-t0):.1f} img/s", flush=True)
    print(f"[shard {SHARD_ID}] DONE {c} cells={ncell:,} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
