"""Extract 64x64 RGB crops for the cluster-labeled cells of the 43 target classes.

The 43 classes are fixed by `meta/classes_43.csv` (tissue x cell-type). Cells whose class is
not in that table are dropped here, and `class_idx` is written as the 0..42 class index, so
everything downstream works in the 43-class space with no further remapping.

Reads each bundle's meta/cluster_labeled_cells.csv → group by sample_name → for each img:
  1. Read JPG, cvtColor BGR→RGB, pad 32
  2. Read zarr, get center_xy for the labeled cells
  3. Crop 64×64 around each center
Output single big h5:
  crops_target.h5
    crops: (N, 64, 64, 3) uint8
    class_idx: (N,) int8 (0..42, row order of meta/classes_43.csv)
    label: (N,) int8 (0=non_target, 1=target)
    split: (N,) S5 (b'train' or b'val')
    sample_name: (N,) variable-length str (for debugging)
"""
import os, re, time, sys
from pathlib import Path
import numpy as np, pandas as pd, h5py, cv2, zarr
from zarr.storage import ZipStore
from multiprocessing import Pool

ROOT = Path(os.environ.get("ISIGHT_EXPERIMENT_ROOT", ""))
IMG_DIR = ROOT / "hpa_500k_sample/images"
ZARR_DIR = ROOT / "hpa_500k_sample/cells_zarr_v2_masked"
DST = ROOT / "iSight_pro/iSight_cell_foundation/data"
DST.mkdir(parents=True, exist_ok=True)
OUT = DST / "crops_target.h5"
PAD = 32

# Load aggregate cluster_labeled across the bundles
BUNDLES = Path(os.environ.get("ISIGHT_BUNDLES", "")) if os.environ.get("ISIGHT_BUNDLES") else ROOT / "transfer_to_parcc"
print("[load] gathering cluster_labeled_cells.csv from bundles...", flush=True)
all_cells = []
for b in sorted(BUNDLES.iterdir()):
    if not (b.is_dir() and re.match(r"^\d+_", b.name)): continue
    idx = int(b.name.split("_")[0])
    if idx < 10: continue
    csv = b / "meta" / "cluster_labeled_cells.csv"
    if not csv.exists(): continue
    df = pd.read_csv(csv)
    all_cells.append(df)
samples = pd.concat(all_cells, ignore_index=True).drop_duplicates(["sample_name","cell_idx","class_idx"])

# Restrict to the 43 target classes and re-index class_idx to 0..42 (done once, here).
CLASSES_43 = Path(os.environ.get("CLASSES_43", Path(__file__).resolve().parents[2] / "meta/classes_43.csv"))
_c43 = pd.read_csv(CLASSES_43)
KEY_TO_CLS43 = dict(zip(_c43.head_idx.astype(int), _c43.cls43.astype(int)))
assert sorted(KEY_TO_CLS43.values()) == list(range(43)), f"{CLASSES_43}: cls43 must be 0..42"
_n0 = len(samples)
samples = samples[samples.class_idx.astype(int).isin(KEY_TO_CLS43)].copy()
samples["class_idx"] = samples.class_idx.astype(int).map(KEY_TO_CLS43).astype(int)
print(f"  43-class filter: kept {len(samples):,}/{_n0:,} cells", flush=True)

print(f"  total: {len(samples):,} cells across {samples.sample_name.nunique()} imgs", flush=True)
print(f"  target: {(samples.label==1).sum():,}, non_target: {(samples.label==0).sum():,}", flush=True)
print(f"  classes: {sorted(samples.class_idx.unique())}", flush=True)
print(f"  splits: {samples.split.value_counts().to_dict()}", flush=True)


def process_img(sample_name_group):
    """Crop all cells for one image; returns list of (crops, class_idx, label, split, sample_name)."""
    sample_name, group = sample_name_group
    img_p = IMG_DIR / f"{sample_name}.jpg"
    zarr_p = ZARR_DIR / f"{sample_name}_cells.zarr.zip"
    if not img_p.exists() or not zarr_p.exists():
        return None
    try:
        img = cv2.imread(str(img_p))
        if img is None: return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        img_pad = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(255,255,255))

        with ZipStore(str(zarr_p), mode="r") as store:
            g = zarr.open_group(store=store, mode="r")
            center_xy = g["center_xy"][:]
        n_z = len(center_xy)
        # Filter to in-range cell_idx
        group = group[(group.cell_idx >= 0) & (group.cell_idx < n_z)]
        if len(group) == 0: return None

        crops = np.empty((len(group), 64, 64, 3), dtype=np.uint8)
        for i, ci in enumerate(group.cell_idx.values):
            cx, cy = center_xy[ci]
            crops[i] = img_pad[int(cy):int(cy)+64, int(cx):int(cx)+64]
        return (crops,
                 group.class_idx.values.astype(np.int8),
                 group.label.values.astype(np.int8),
                 group.split.values.astype("S5"),
                 [sample_name] * len(group))
    except Exception as e:
        return None


if __name__ == "__main__":
    print(f"[extract] starting (16 workers)...", flush=True)
    t0 = time.time()
    grouped = list(samples.groupby("sample_name"))
    print(f"  unique imgs: {len(grouped)}", flush=True)

    # Pre-allocate big arrays (overshoot, will trim)
    N_max = len(samples) + 1000
    all_crops = np.empty((N_max, 64, 64, 3), dtype=np.uint8)
    all_class = np.empty((N_max,), dtype=np.int8)
    all_label = np.empty((N_max,), dtype=np.int8)
    all_split = np.empty((N_max,), dtype="S5")
    all_sn = []
    cursor = 0
    n_done = 0; n_skip = 0

    with Pool(16) as pool:
        for res in pool.imap_unordered(process_img, grouped, chunksize=4):
            n_done += 1
            if res is None:
                n_skip += 1
                continue
            crops, cls, lab, spl, sns = res
            k = len(crops)
            all_crops[cursor:cursor+k] = crops
            all_class[cursor:cursor+k] = cls
            all_label[cursor:cursor+k] = lab
            all_split[cursor:cursor+k] = spl
            all_sn.extend(sns)
            cursor += k
            if n_done % 100 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (len(grouped) - n_done) / rate
                print(f"  [{n_done:>4d}/{len(grouped)}] {cursor:,} cells | "
                      f"{rate:.1f} img/sec | ETA {eta/60:.0f} min", flush=True)

    print(f"\n[done] {n_done} imgs processed ({n_skip} skipped), {cursor:,} cells", flush=True)

    # Trim + save
    all_crops = all_crops[:cursor]
    all_class = all_class[:cursor]
    all_label = all_label[:cursor]
    all_split = all_split[:cursor]
    sn_bytes = np.array(all_sn, dtype=object)

    print(f"[save] writing {OUT} ({all_crops.nbytes/1e9:.1f} GB)...", flush=True)
    with h5py.File(OUT, "w") as f:
        f.create_dataset("crops", data=all_crops, chunks=(min(1024, cursor), 64, 64, 3),
                          compression=None, shuffle=False)
        f.create_dataset("class_idx", data=all_class)
        f.create_dataset("label", data=all_label)
        f.create_dataset("split", data=all_split)
        sn_dt = h5py.special_dtype(vlen=str)
        f.create_dataset("sample_name", data=sn_bytes, dtype=sn_dt)
        f.attrs["n_cells"] = cursor
        f.attrs["n_classes_subset"] = int(np.unique(all_class).size)
        f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M")

    elapsed = time.time() - t0
    print(f"[done] {elapsed/60:.0f} min total, output {OUT}", flush=True)
