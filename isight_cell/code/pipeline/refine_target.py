"""Round-2 pseudo-label refinement: keep only cells whose R1 predicted intensity AGREES
with the image-level true intensity label, then train Round-2 on those denoised cells.

E.g. a true-weak image has 2000 target cells, R1 predicted ~200 of them weak (rest negative/
moderate); we keep ONLY those ~200 weak cells (the real weak signal) and drop the broadcast
noise. Writes data/target_cells_refined/<flat>.h5 (compact: only agreeing crops + labels).

Only processes TRAIN-split images (val/test stay full/original for honest eval).
"""
import os, sys, glob, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, h5py, pandas as pd
ISIGHT = Path(os.environ.get("ISIGHT_ROOT", ""))   # data root; see README
HF = ISIGHT / "iSight_ihc_foundation"
SRC = Path(os.environ.get("SRC", HF / "data/target_cells"))
OUT = Path(os.environ.get("OUT", HF / "data/target_cells_refined")); OUT.mkdir(parents=True, exist_ok=True)
INTENSITY = ["negative", "weak", "moderate", "strong"]
IMAP = {v: i for i, v in enumerate(INTENSITY)}
LOCATION = ["none", "nuclear", "cytoplasmic/membranous", "cytoplasmic/membranous,nuclear"]  # MUST match T.LMAP
LMAP = {v: i for i, v in enumerate(LOCATION)}
MODE = os.environ.get("REFINE_MODE", "intensity")            # "intensity" | "both" (intensity AND location agree)
KEEP_DS = ("crops", "center_xy", "bbox", "area", "cell_id")
KEEP_ATTR = ("sample_name", "role", "intensity", "location", "quantity", "source", "group", "head_idx",
             "is_target", "head_idx")


def one(flat):
    sp = SRC / f"{flat}.h5"; op = OUT / f"{flat}.h5"
    if op.exists(): return "skip"
    try:
        with h5py.File(sp, "r") as f:
            n = int(f.attrs["n_cells"])
            if n == 0 or "pred_intensity" not in f: return "no_pred"
            yi = IMAP.get(str(f.attrs["intensity"]))
            if yi is None: return "bad_label"
            pi = f["pred_intensity"][:]
            keep = (pi == yi)                                  # intensity agreement
            if MODE == "both":                                # ALSO require location agreement
                yl = LMAP.get(str(f.attrs["location"]))
                if yl is None or "pred_location" not in f: return "bad_label"
                keep = keep & (f["pred_location"][:] == yl)
            agree = np.where(keep)[0]
            if len(agree) == 0: return "no_agree"
            attrs = {k: f.attrs[k] for k in set(KEEP_ATTR) if k in f.attrs}
            data = {k: f[k][:][agree] for k in KEEP_DS if k in f}
        tmp = str(op) + ".tmp"
        with h5py.File(tmp, "w") as g:
            g.attrs.update(n_cells=len(agree), n_orig=n, refine=f"{MODE}_agree", **attrs)
            for k, v in data.items():
                g.create_dataset(k, data=v, chunks=(min(128, len(agree)),) + v.shape[1:] if v.ndim > 1 else None)
        os.rename(tmp, op)
        return f"ok"
    except Exception as e:
        return f"err:{type(e).__name__}"


if __name__ == "__main__":
    sp = pd.read_csv(HF / "meta/split.csv")
    train = set(sp[sp.split == "train"].flat)
    files = [Path(p).stem for p in glob.glob(str(SRC / "*.h5"))]
    todo = [f for f in files if f in train]
    print(f"refine {len(todo)} train imgs (keep cells with pred_intensity==image label) -> {OUT}", flush=True)
    t0 = time.time(); c = {}; kept = 0; orig = 0
    with Pool(int(os.environ.get("WORKERS", "64"))) as pool:
        for i, st in enumerate(pool.imap_unordered(one, todo, chunksize=16)):
            c[st] = c.get(st, 0) + 1
            if (i + 1) % 5000 == 0: print(f"  {i+1}/{len(todo)} {c}", flush=True)
    # quick stats on kept fraction
    print(f"[done] {c} in {time.time()-t0:.0f}s", flush=True)
