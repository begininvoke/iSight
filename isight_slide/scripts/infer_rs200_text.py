"""iSight-slide on the 200 reader-study images, with the text query optionally injected.

Same model/patching logic as infer_test2000.py: the text branch in forward() is gated on
`phase == "train"`, so the published reader-study numbers never used the text at all.
`model.force_query = True` turns it on at test time without retraining.

Queries come from readerstudy200_manifest_query.csv (see build_rs200_queries.py) — HPA rows
carry the full training string including SNOMED, Stanford rows have no SNOMED to carry.

Env: OUT [USE_QUERY=0|1] [LIMIT=n] [CKPT]
"""
import os, sys, time
import numpy as np, pandas as pd, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "isight_slide"))        # model/
sys.path.insert(0, os.path.join(_REPO, "isight_cell", "code"))  # tissue.py
from model.patch_encoder_with_clam import create_clam_vit
from tissue import tissue_mask_px

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.environ.get("MANIFEST", f"{HERE}/readerstudy200_manifest_query.csv")
OUT = os.environ["OUT"]
USE_QUERY = int(os.environ.get("USE_QUERY", "0"))
LIMIT = int(os.environ.get("LIMIT", "0"))
CKPT = os.environ.get("CKPT", "checkpoint/checkpoints/iSight_model_checkpoint.pth")
PATCH = 336; MIN_TISSUE = 0.10

CELL_TYPES = ['Glandular cells','Exocrine glandular cells','Tumor cells','Cholangiocytes','Adipocytes',
 'Squamous epithelial cells','Glial cells','Cells in endometrial stroma','Cells in red pulp','Alveolar cells',
 'Respiratory epithelial cells','Cells in granular layer','Endothelial cells','Fibroblasts','Decidual cells',
 'Cells in glomeruli','Myocytes','Cells in seminiferous ducts','Hematopoietic cells','Germinal center cells',
 'Cardiomyocytes','Urothelial cells','Trophoblastic cells','Smooth muscle cells','Ovarian stroma cells',
 'Follicle cells','Epidermal cells','Chondrocytes','Hepatocytes','Lymphoid tissue','Non-germinal center cells',
 'Cells in molecular layer','Keratinocytes','Peripheral nerve','Cells in tubules','Neuronal cells','Leydig cells',
 'Cells in white pulp','Langerhans']
INT = ["negative","weak","moderate","strong"]
LOC = ["none","cytoplasmic/membranous","nuclear","cytoplasmic/membranous,nuclear"]
QUA = ["none","<25%","25%-75%",">75%"]


class Cfg:
    base_model_name = "openai/clip-vit-large-patch14-336"
    model_version = "v3_all_tokens"
    learning_rate = 1e-6; weight_decay = 1e-5
    freeze_vit = False; freeze_query_features_encoder = False
    use_cell_type_embedding = True; use_cell_type = "All"
    batch_size = 1; num_workers = 8; amp = False


def crop_patches(img, mask):
    W, H = img.size
    out = []
    for i in range(int(np.ceil(H / PATCH))):
        for j in range(int(np.ceil(W / PATCH))):
            l, t = j * PATCH, i * PATCH
            r, b = min(l + PATCH, W), min(t + PATCH, H)
            if r <= l or b <= t: continue
            if mask[t:b, l:r].mean() < MIN_TISSUE: continue
            p = img.crop((l, t, r, b))
            if p.size != (PATCH, PATCH):
                canvas = Image.new("RGB", (PATCH, PATCH), "white"); canvas.paste(p, (0, 0)); p = canvas
            out.append(p)
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Cfg()
    model, _, _ = create_clam_vit(base_model_name=cfg.base_model_name, training_config=cfg, device=dev,
                                  lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                                  freeze_vit=cfg.freeze_vit,
                                  freeze_query_features_encoder=cfg.freeze_query_features_encoder,
                                  use_cell_type_embedding=cfg.use_cell_type_embedding)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    miss, unexp = model.load_state_dict(ck["model_state_dict"], strict=False)
    print(f"loaded ckpt (epoch {ck.get('epoch')}) | missing {len(miss)} unexpected {len(unexp)}", flush=True)
    model.to(dev).eval()
    model.force_query = bool(USE_QUERY)
    print(f"USE_QUERY={USE_QUERY} -> text branch {'ON' if USE_QUERY else 'OFF (original behaviour)'}", flush=True)
    proc = model.patch_processor

    df = pd.read_csv(MANIFEST)
    if LIMIT: df = df.head(LIMIT)
    rows = []; t0 = time.time()
    with torch.inference_mode():
        for k, r in df.reset_index(drop=True).iterrows():
            img = Image.open(r["image_path"]).convert("RGB")
            patches = crop_patches(img, tissue_mask_px(np.array(img)).astype(np.uint8))
            if not patches:
                print(f"  !! no patches: {r['image_path']}", flush=True); continue
            px = torch.stack([proc(images=p, return_tensors="pt")["pixel_values"].squeeze(0)
                              for p in patches]).to(dev)
            ct = torch.zeros(1, 39, device=dev)
            ci = CELL_TYPES.index(r["cell_type"]) if r["cell_type"] in CELL_TYPES else CELL_TYPES.index("Tumor cells")
            ct[0, ci] = 1.0
            with torch.cuda.amp.autocast():
                oi, ol, oq, ot, om, _ = model([px], [r["query"]], ct, phase="test")
            import torch.nn.functional as _F
            pi_ = _F.softmax(oi.float(), -1)[0].cpu().numpy()
            pl_ = _F.softmax(ol.float(), -1)[0].cpu().numpy()
            pq_ = _F.softmax(oq.float(), -1)[0].cpu().numpy()
            rec = dict(image_url=r["image_url"], image_source=r["image_source"], n_patches=len(patches),
                       pred_intensity=INT[int(oi.argmax(-1))], pred_location=LOC[int(ol.argmax(-1))],
                       pred_quantity=QUA[int(oq.argmax(-1))],
                       gene=r["gene"], tissue=r["tissue"], cell_type_used=CELL_TYPES[ci], query=r["query"])
            for j in range(4):
                rec[f"prob_intensity_{j}"] = float(pi_[j])
                rec[f"prob_location_{j}"] = float(pl_[j])
                rec[f"prob_quantity_{j}"] = float(pq_[j])
            rows.append(rec)
            if (k + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {k+1}/{len(df)}  {el/(k+1):.2f}s/img", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"saved {len(rows)} rows -> {OUT}\nINFER_DONE", flush=True)


if __name__ == "__main__":
    main()
