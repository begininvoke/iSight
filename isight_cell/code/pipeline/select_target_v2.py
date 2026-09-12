"""Select target cells per image with the 43-class target-cell classifier.

The classifier carries one binary head per class in `meta/classes_43.csv` (tissue x cell-type).
Each image's h5 stores `head_idx`, the key for its class; `meta/classes_43.csv` maps that key to
the 0..42 head column. That table is the same mapping the released checkpoint was trained with,
so no conversion happens at run time.

For each all_cells/{flat}.h5: forward its crops through the backbone -> feat(1536), take the
image's class head -> p_target = softmax(feat @ W[c] + b[c])[1]. Keep p > P_THR, sample
N_TARGET, write target_cells/{flat}.h5 (crops + geometry + p_target + labels). GPU; sharded.

Set $CKPT to the target-cell checkpoint (`target_cell_43cls`, val mean F1 0.9954 at ep4) --
this is the selector behind the released staining model and everything downstream of it.

crops are RGB uint8 (crop_all_master does BGR->RGB), matching training (extract_target_crops
also BGR->RGB). Preprocess = resize 64->224 bilinear antialias + ImageNet norm.
"""
import os, sys, glob, time
from pathlib import Path
import numpy as np, h5py, torch, torch.nn.functional as F, timm, cv2

ISIGHT = Path(os.environ.get("ISIGHT_ROOT", ""))   # data root; see README
HF = ISIGHT / "iSight_ihc_foundation"
ALL = Path(os.environ.get("ALL_CELLS_OUT", HF / "data/all_cells"))
OUT = Path(os.environ.get("TARGET_OUT", HF / "data/target_cells")); OUT.mkdir(parents=True, exist_ok=True)
TISSUE_MASKS = os.environ.get("TISSUE_MASKS", "")   # REQUIRED: dir of <flat>.png -> drop cells outside tissue mask
CKPT = Path(os.environ["CKPT"])                      # target-cell classifier (43 heads)
CLASSES_43 = Path(os.environ.get("CLASSES_43", Path(__file__).resolve().parents[2] / "meta/classes_43.csv"))
N_TARGET = int(os.environ.get("N_TARGET", "1000")); P_THR = float(os.environ.get("P_THR", "0.5"))
EMBED_CAP = int(os.environ.get("EMBED_CAP", "4000"))     # cap candidate cells/img before forward
BATCH = int(os.environ.get("BATCH", "1024"))
N_SHARDS = int(os.environ.get("N_SHARDS", "1")); SHARD_ID = int(os.environ.get("SHARD_ID", "0"))
SEED = 42

# --- class table: key stored in each h5 -> head column (precomputed, not derived here) ---
def _load_classes():
    import csv as _csv
    with open(CLASSES_43) as fh:
        rows = list(_csv.DictReader(fh))
    m = {int(r["head_idx"]): int(r["cls43"]) for r in rows}
    assert sorted(m.values()) == list(range(43)), f"{CLASSES_43}: cls43 must be 0..42"
    return m

# --- optional external head override -------------------------------------------------
# The h5 files carry head_idx from HPA metadata. A cohort without compatible metadata
# (constant cell_type, free-text clinical site) needs it supplied: HEAD_MAP lets a
# corrected flat->head_idx table override the stored attribute without touching the data.
HEAD_MAP = os.environ.get("HEAD_MAP", "")
_HEAD_OVERRIDE = {}
if HEAD_MAP:
    import csv as _csv
    with open(HEAD_MAP) as _fh:
        for _r in _csv.DictReader(_fh):
            if _r.get("head_idx", "").strip() not in ("", "nan"):
                _HEAD_OVERRIDE[_r["flat"]] = int(float(_r["head_idx"]))
    print(f"[head-map] {len(_HEAD_OVERRIDE)} overrides from {HEAD_MAP}", flush=True)

INPUT_SIZE = 224
NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
NORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
ATTRS = ("sample_name", "role", "intensity", "location", "quantity", "source", "group", "head_idx")


def build_v2():
    dev = torch.device("cuda")
    model = timm.create_model(
        "vit_giant_patch14_224", img_size=224, patch_size=14, depth=24, num_heads=24,
        init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0,
        no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
        reg_tokens=8, dynamic_img_size=True)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    bb = {k[len("backbone."):]: v for k, v in ck["model"].items() if k.startswith("backbone.")}
    model.load_state_dict(bb, strict=True)
    model = model.eval().to(dev).to(memory_format=torch.channels_last)
    W = ck["model"]["head.W"].float().to(dev)   # (43,1536,2)
    b = ck["model"]["head.b"].float().to(dev)   # (43,2)
    key2col = _load_classes()                    # {head_idx stored in the h5: 0..42}
    assert W.shape[0] == 43, f"expected 43 heads, checkpoint has {W.shape[0]}"
    if "old_to_new" in ck:                       # older checkpoints carry the table inline
        inline = {int(k): int(v) for k, v in ck["old_to_new"].items()}
        assert inline == key2col, f"{CLASSES_43} disagrees with the mapping stored in {CKPT}"
    return model, W, b, key2col, dev


@torch.inference_mode()
def main():
    model, W, b, key2col, dev = build_v2()
    m_, s_ = NORM_MEAN.to(dev), NORM_STD.to(dev)
    from torch.amp import autocast
    # balance shards by cell count: read precomputed sizes cache (sorted big-first),
    # round-robin. Avoids each shard opening all 83k h5 (filesystem metadata storm).
    cache = ALL.parent / "all_cells_sizes.csv"
    FLATS = os.environ.get("FLATS", "")          # optional: newline-separated flat list to restrict to
    keep_flats = set(l.strip() for l in open(FLATS) if l.strip()) if FLATS else None
    if cache.exists():
        import pandas as pd
        sdf = pd.read_csv(cache).sort_values("n_cells", ascending=False)
        todo = list(sdf["path"])
        if keep_flats is not None:
            todo = [p for p in todo if Path(p).stem in keep_flats]
        todo = todo[SHARD_ID::N_SHARDS]
    else:
        files = sorted(glob.glob(str(ALL / "*.h5")))
        sizes = []
        for p in files:
            try:
                with h5py.File(p, "r") as f: sizes.append(int(f.attrs["n_cells"]))
            except Exception: sizes.append(0)
        order = np.argsort(-np.array(sizes))
        todo = [files[i] for i in order]
        if keep_flats is not None:
            todo = [p for p in todo if Path(p).stem in keep_flats]
        todo = todo[SHARD_ID::N_SHARDS]
    print(f"[v2-select shard {SHARD_ID}/{N_SHARDS}] {len(todo)} imgs | N_TARGET={N_TARGET} "
          f"P_THR={P_THR} EMBED_CAP={EMBED_CAP} -> {OUT}", flush=True)
    t0 = time.time(); c = {}; ncell = 0
    for k, p in enumerate(todo):
        flat = Path(p).stem; out = OUT / f"{flat}.h5"
        if out.exists(): c["skip"] = c.get("skip", 0) + 1; continue
        try:
            with h5py.File(p, "r") as f:
                n = int(f.attrs["n_cells"])
                attrs = {kk: f.attrs[kk] for kk in ATTRS}
                if n == 0: c["no_cells"] = c.get("no_cells", 0) + 1; continue
                hi = int(attrs["head_idx"])
                hi = _HEAD_OVERRIDE.get(flat, hi)
                if hi not in key2col: c["no_map"] = c.get("no_map", 0) + 1; continue
                crops = f["crops"][:]; cxy = f["center_xy"][:]; bbox = f["bbox"][:]
                area = f["area"][:]; cid = f["cell_id"][:]
                img_h = int(f.attrs.get("image_h", 0)); img_w = int(f.attrs.get("image_w", 0))
            # TISSUE MASK filter: drop cells whose center is outside the tissue mask (required)
            if TISSUE_MASKS:
                mp = os.path.join(TISSUE_MASKS, f"{flat}.png")
                tm = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if os.path.exists(mp) else None
                if tm is None: c["no_mask"] = c.get("no_mask", 0) + 1; continue
                Hm, Wm = tm.shape
                sx = Wm / img_w if img_w else 1.0; sy = Hm / img_h if img_h else 1.0
                yy = np.clip((cxy[:, 1] * sy).astype(int), 0, Hm - 1)
                xx = np.clip((cxy[:, 0] * sx).astype(int), 0, Wm - 1)
                keepm = tm[yy, xx] > 0
                crops, cxy, bbox, area, cid = crops[keepm], cxy[keepm], bbox[keepm], area[keepm], cid[keepm]
                n = int(keepm.sum())
                if n == 0: c["mask_empty"] = c.get("mask_empty", 0) + 1; continue
            # cap candidates per image
            if EMBED_CAP and n > EMBED_CAP:
                rs = np.random.RandomState(SEED)
                ci = np.sort(rs.choice(n, EMBED_CAP, replace=False))
                crops = crops[ci]; cxy = cxy[ci]; bbox = bbox[ci]; area = area[ci]; cid = cid[ci]
            nc = len(crops)
            col = key2col[hi]; Wc = W[col]; bc = b[col]        # (1536,2),(2,)
            P = np.empty(nc, np.float32)
            for i in range(0, nc, BATCH):
                x = torch.from_numpy(crops[i:i + BATCH]).to(dev).permute(0, 3, 1, 2).float().div_(255.0)
                x = F.interpolate(x, INPUT_SIZE, mode="bilinear", align_corners=False, antialias=True)
                x = ((x - m_) / s_).contiguous(memory_format=torch.channels_last)
                with autocast("cuda", dtype=torch.bfloat16):
                    feat = model(x).float()                    # (B,1536)
                logit = feat @ Wc + bc                         # (B,2)
                P[i:i + len(x)] = torch.softmax(logit, -1)[:, 1].cpu().numpy()
            tgt = np.where(P > P_THR)[0]
            if len(tgt) == 0: c["no_sel"] = c.get("no_sel", 0) + 1; continue
            rs = np.random.RandomState(SEED)
            keep = np.sort(rs.choice(tgt, N_TARGET, replace=False) if len(tgt) > N_TARGET else tgt)
            nk = len(keep)
            tmp = str(out) + ".tmp"
            with h5py.File(tmp, "w") as f:
                f.attrs.update(n_cells=nk, n_total_cells=n, n_candidates=int(nc),
                               n_target_total=int(len(tgt)), p_thr=P_THR, is_target=1,
                               model=CKPT.name, ckpt_path=str(CKPT),
                               n_heads=int(W.shape[0]), **attrs)
                ck_ = (min(128, nk),)
                f.create_dataset("crops", data=crops[keep], chunks=ck_ + crops.shape[1:])
                f.create_dataset("center_xy", data=cxy[keep]); f.create_dataset("bbox", data=bbox[keep])
                f.create_dataset("area", data=area[keep]); f.create_dataset("cell_id", data=cid[keep])
                f.create_dataset("p_target", data=P[keep].astype(np.float16))
            os.rename(tmp, out); c["ok"] = c.get("ok", 0) + 1; ncell += nk
        except Exception as e:
            c[f"err:{type(e).__name__}"] = c.get(f"err:{type(e).__name__}", 0) + 1
        if (k + 1) % 500 == 0:
            r = (k + 1) / (time.time() - t0)
            print(f"  [{SHARD_ID}] {k+1}/{len(todo)} {c} | {r:.1f} img/s tgt_cells={ncell}", flush=True)
    print(f"[shard {SHARD_ID}] DONE {c} cells={ncell} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
