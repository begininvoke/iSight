"""Build the per-cell scan cache (.percell_scan_<dir>.npz) for a refined cell dir, using the
CORRECT location LMAP (matches train_finetune.T.LMAP). Output consumed by train_shards_resample.py ($SCAN_NPZ).

The earlier inline scan used a wrong class-3 location string ("nuclear and cytoplasmic/membranous")
that does NOT exist in the data — the real value is "cytoplasmic/membranous,nuclear" — which silently
DROPPED all class-3 images, so shards had 0 mixed-location cells and location macro-F1 collapsed.
"""
import os, sys, glob
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, h5py
ISIGHT = Path(os.environ.get("ISIGHT_ROOT", ""))   # data root; see README
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deps"))  # train_finetune, backbones
import train_finetune as T                              # authoritative IMAP / LMAP
HF = ISIGHT / "iSight_ihc_foundation"
RD = Path(os.environ.get("RD", HF / "data/target_cells_refined_both"))
WORKERS = int(os.environ.get("WORKERS", "48"))

cm = pd.read_csv(HF / "meta/crop_master.csv", usecols=["flat", "intensity", "staining_location"])
cm["yi"] = cm.intensity.map(T.IMAP); cm["yl"] = cm.staining_location.map(T.LMAP)
bad = cm.staining_location.notna() & cm.yl.isna()
if bad.any():
    print(f"WARN {bad.sum()} rows have location not in T.LMAP: {sorted(cm[bad].staining_location.unique())[:5]}", flush=True)
lab = cm.dropna(subset=["yi", "yl"]).set_index("flat")[["yi", "yl"]].astype(int)
train = set(pd.read_csv(HF / "meta/split.csv").query("split=='train'").flat)


def rd(p):
    flat = Path(p).stem
    if flat not in train or flat not in lab.index: return None
    try:
        with h5py.File(p, "r") as f:
            n = int(f.attrs["n_cells"])
    except Exception:
        return None
    return (p, n, int(lab.loc[flat, "yi"]), int(lab.loc[flat, "yl"])) if n > 0 else None


if __name__ == "__main__":
    files = sorted(glob.glob(str(RD / "*.h5")))
    with Pool(WORKERS) as pool:
        res = [r for r in pool.map(rd, files, chunksize=64) if r]
    fs = [r[0] for r in res]; ns = np.array([r[1] for r in res], np.int64)
    yi = np.array([r[2] for r in res], np.int8); yl = np.array([r[3] for r in res], np.int8)
    cache = RD.parent / f".percell_scan_{RD.name}.npz"
    np.savez(cache, files=np.array(fs), n=ns, yi=yi, yl=yl)
    print(f"[scan] {len(fs)} imgs, {ns.sum()/1e6:.1f}M cells -> {cache}", flush=True)
    print(f"  intensity dist (imgs): {np.bincount(yi, minlength=4)}", flush=True)
    print(f"  location  dist (imgs): {np.bincount(yl, minlength=4)}  <- class-3 should be NONZERO now", flush=True)
    print(f"  cells/class: {[int(ns[yi==c].sum()) for c in range(4)]}", flush=True)
