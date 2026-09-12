"""Generate tissue masks for all IHC foundation images (pixel-level edge, code/tissue.py).
Independent of cellpose (only needs images). Parallel; writes data/tissue_masks/<flat>.png."""
import os,sys
from pathlib import Path
import numpy as np, cv2, pandas as pd
from multiprocessing import Pool
cv2.setNumThreads(1)
sys.path.insert(0,str(Path(__file__).parent)); from tissue import tissue_mask_px
HF = Path(os.environ.get("ISIGHT_ROOT", "")) / "iSight_ihc_foundation"
OUT=Path(os.environ.get("MASK_OUT", str(HF/"data/tissue_masks"))); OUT.mkdir(parents=True,exist_ok=True)
def work(args):
    flat,ip=args; op=OUT/f"{flat}.png"
    if op.exists(): return (flat,-2)
    bgr=cv2.imread(ip)
    if bgr is None: return (flat,-1)
    m=tissue_mask_px(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cv2.imwrite(str(op),(m*255).astype(np.uint8),[cv2.IMWRITE_PNG_COMPRESSION,5])
    return (flat,float(m.mean()))
def main():
    M=pd.read_csv(os.environ.get("MASTER_CSV", str(HF/"meta/master_manifest.csv")))
    todo=list(zip(M.flat,M.out_jpg))
    print(f"{len(todo)} images, {os.environ.get('WORKERS','60')} workers",flush=True)
    rows=[]
    with Pool(int(os.environ.get("WORKERS","60"))) as p:
        for i,(flat,fr) in enumerate(p.imap_unordered(work,todo,chunksize=16)):
            rows.append((flat,fr))
            if (i+1)%5000==0: print(f"  {i+1}/{len(todo)}",flush=True)
    pd.DataFrame(rows,columns=["flat","mask_frac"]).to_csv(HF/"data/mask_frac.csv",index=False)
    print("DONE",len(rows),flush=True)
if __name__=="__main__": main()
