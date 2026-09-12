"""Image-level metrics from ALL target cells (no 1000-cap). Same fixed aggregation as before."""
import h5py, glob, os, sys, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import cohen_kappa_score
ISIGHT = os.environ.get("ISIGHT_ROOT", "")   # data root; see README
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deps"))  # train_finetune, backbones
import train_finetune as T
from pathlib import Path
QUA=["none","<25%","25%-75%",">75%"]; QMAP={q:i for i,q in enumerate(QUA)}
def agg(p):
    try:
        with h5py.File(p,"r") as f:
            a=f.attrs
            yi=T.IMAP.get(str(a["intensity"])); yl=T.LMAP.get(str(a["location"])); yq=QMAP.get(str(a.get("quantity")))
            if yi is None or yl is None or yq is None or "pred_intensity" not in f: return None
            pi=f["pred_intensity"][:]; pl=f["pred_location"][:]
    except: return None
    if len(pi)==0: return None
    fr=(pi!=0).mean()
    ii=0 if fr<0.05 or (pi!=0).sum()==0 else int(np.bincount(pi[pi!=0]).argmax())
    lf=(pl!=0).mean(); ll=0 if lf<0.05 else int(np.bincount(pl[pl!=0],minlength=4).argmax())
    qq=0 if fr<0.05 else (1 if fr<=.25 else (2 if fr<=.75 else 3))
    return dict(flat=os.path.basename(p)[:-3],true_int=yi,true_loc=yl,true_qua=yq,
                pred_int=ii,pred_loc=ll,pred_qua=qq,frac=fr,n_cells=len(pi))
q=lambda t,p: cohen_kappa_score(t,p,weights="quadratic",labels=[0,1,2,3])
rows=[]
for s in ["val","indist","test500k"]:
    for st,nm in [("s1","Stage 1"),("s2ep09","Stage 2 (ep09)")]:
        fs=sorted(glob.glob(f"data/uncap_{s}_{st}/*.h5"))
        with ThreadPoolExecutor(48) as ex: R=[x for x in ex.map(agg,fs) if x]
        d=pd.DataFrame(R); d.to_csv(f"runs/stage_compare/uncap_{s}_{st}.csv",index=False)
        rows.append(dict(test=s,stage=nm,n=len(d),cells=d.n_cells.sum(),
            int_acc=(d.true_int==d.pred_int).mean(), loc_acc=(d.true_loc==d.pred_loc).mean(),
            qua_acc=(d.true_qua==d.pred_qua).mean(),
            int_qwk=q(d.true_int,d.pred_int), qua_qwk=q(d.true_qua,d.pred_qua)))
M=pd.DataFrame(rows); M.to_csv("runs/stage_compare/uncap_summary.csv",index=False)
print(M.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
