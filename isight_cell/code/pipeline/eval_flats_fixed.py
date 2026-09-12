"""Generic per-image cell-model eval with the SAME fixed aggregation as test500k:
  intensity = mode of stained cells (pred!=0), gated by stained-fraction < 0.05 -> negative
  location  = def-D: mode of location-stained cells (pl!=0), gated by loc-fraction < 0.05 -> none
  quantity  = stained-fraction binned at 0.05 / 0.25 / 0.75
Lets Stage-1 (round1) and Stage-2 (ep09) be compared on ANY flat list under identical rules.

Env: CSV (flat + staining_* columns) TC (target-cells dir) CKPT OUT [SHARD_ID N_SHARDS]
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, h5py, torch
ISIGHT = Path(os.environ.get("ISIGHT_ROOT", ""))   # data root; see README
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deps"))  # train_finetune, backbones
import train_finetune as T
from backbones.vision_encoders.uni2 import build_uni2

CSV = os.environ["CSV"]; TC = Path(os.environ["TC"]); CKPT = os.environ["CKPT"]; OUT = os.environ["OUT"]
BS = int(os.environ.get("BS", "2048"))
IMAP, LMAP = T.IMAP, T.LMAP                       # training-consistent label maps
QUA = ["none", "<25%", "25%-75%", ">75%"]; QMAP = {q: i for i, q in enumerate(QUA)}


def main():
    dev = torch.device("cuda")
    model = T.Net(build_uni2(drop_path_rate=0, drop_rate=0, attn_drop_rate=0),
                  mlp_head=True, dropout=0.3).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["model"])
    model.eval().to(memory_format=torch.channels_last)
    m_, s_ = T.NORM_MEAN.to(dev), T.NORM_STD.to(dev)
    from torch.amp import autocast

    df = pd.read_csv(CSV)
    if "flat" not in df.columns: df["flat"] = df["sample_name"]
    rows = []
    with torch.inference_mode():
        for k, r in df.iterrows():
            flat = r["flat"]; p = TC / f"{flat}.h5"
            if not p.exists(): continue
            yi = IMAP.get(str(r["staining_intensity"])); yl = LMAP.get(str(r["staining_location"]))
            yq = QMAP.get(str(r["staining_quantity"]))
            if yi is None or yl is None or yq is None: continue
            with h5py.File(p, "r") as f:
                n = int(f.attrs["n_cells"])
                if n == 0: continue
                crops = f["crops"][:]
            PI, PL = [], []
            for i in range(0, n, BS):
                x = T.crops_to_input(crops[i:i + BS], dev, False, m_, s_)
                with autocast("cuda", dtype=torch.bfloat16):
                    li, ll = model(x)
                PI.append(li.argmax(-1).cpu().numpy()); PL.append(ll.argmax(-1).cpu().numpy())
            pi = np.concatenate(PI); pl = np.concatenate(PL)
            frac = float((pi != 0).mean())
            ii = 0 if frac < 0.05 or (pi != 0).sum() == 0 else int(np.bincount(pi[pi != 0]).argmax())
            lf = (pl != 0).mean()
            ll_ = 0 if lf < 0.05 else int(np.bincount(pl[pl != 0], minlength=4).argmax())
            qq = 0 if frac < 0.05 else (1 if frac <= .25 else (2 if frac <= .75 else 3))
            rows.append(dict(flat=flat, true_int=yi, true_loc=yl, true_qua=yq,
                             pred_int=ii, pred_loc=ll_, pred_qua=qq, frac=frac, n_cells=n))
            if (k + 1) % 200 == 0: print(f"  {k+1}/{len(df)}", flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT, index=False)
    from sklearn.metrics import cohen_kappa_score
    q = lambda t, p_: cohen_kappa_score(t, p_, weights="quadratic", labels=[0, 1, 2, 3])
    print(f"\nn={len(R)}  CKPT={Path(CKPT).name}  TC={TC.name}")
    print(f"  int acc={(R.true_int==R.pred_int).mean():.4f} QWK={q(R.true_int,R.pred_int):.4f}")
    print(f"  loc acc={(R.true_loc==R.pred_loc).mean():.4f}")
    print(f"  qua acc={(R.true_qua==R.pred_qua).mean():.4f} QWK={q(R.true_qua,R.pred_qua):.4f}")
    print(f"-> {OUT}\nEVAL_FLATS_DONE", flush=True)


if __name__ == "__main__":
    main()
