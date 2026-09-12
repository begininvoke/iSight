"""Rich VAL-set evaluation per epoch: image-level intensity/location/quantity accuracy AND QWK,
using the SAME aggregation as reader study / test500k (int=mode of stained+5% gate, loc def-D,
qua=frac bins). GT from val target-cell h5 attrs. Lets us pick an epoch by a validation-only,
image-level criterion (no test-set peeking). Env: EPS (space-sep epochs), CKPTDIR, OUT."""
import os, sys, glob, time
from pathlib import Path
import numpy as np, pandas as pd, h5py, torch
from sklearn.metrics import cohen_kappa_score, f1_score
ISIGHT = Path(os.environ.get("ISIGHT_ROOT", ""))   # data root; see README
for p in ("hnc_immune_markers/code","iSight_prostate_finetune/code","iSight_train_model/code"):
    sys.path.insert(0, str(ISIGHT / p))
import train_finetune as T
from backbones.vision_encoders.uni2 import build_uni2

FINAL = ISIGHT / "iSight_ihc_foundation_final"
TC = FINAL / "data/target_cells"
CKPTDIR = Path(os.environ.get("CKPTDIR", FINAL / "runs/round3_resample_s1_lr2e5"))
OUT = os.environ.get("OUT", str(FINAL / "runs/round3_resample_s1_lr2e5_valrich.csv"))
EPS = os.environ["EPS"].split()
IMAP=T.IMAP                    # use the exact TRAINING label maps (avoid 1<->2 loc swap bug)
LMAP=T.LMAP                    # 0=none 1=nuclear 2=cytoplasmic/membranous 3=cyto/mem+nuclear
def qmap(s):
    s=str(s)
    if s in ("none","0","negative"): return 0
    if s in ("<25%","25%",): return 1
    if s in ("25%-75%","25-75%",">25%"): return 2
    if s in (">75%","75%"): return 3
    return None
def qbin(fr): return 0 if fr<0.05 else (1 if fr<=.25 else (2 if fr<=.75 else 3))
def qwk(t,p): return cohen_kappa_score(t,p,weights="quadratic",labels=[0,1,2,3]) if len(set(t))>1 else float("nan")

sp=pd.read_csv(ISIGHT/"iSight_ihc_foundation/meta/split.csv"); val=set(sp[sp.split=="val"].flat)
files=[p for p in sorted(glob.glob(str(TC/"*.h5"))) if Path(p).stem in val]
dev=torch.device("cuda"); m_,s_=T.NORM_MEAN.to(dev),T.NORM_STD.to(dev)
from torch.amp import autocast
rows=[]
for ep in EPS:
    ck=CKPTDIR/f"ckpt_ep{ep}.pt"
    model=T.Net(build_uni2(drop_path_rate=0,drop_rate=0,attn_drop_rate=0),mlp_head=True,dropout=0.3).to(dev)
    model.load_state_dict(torch.load(ck,map_location="cpu",weights_only=False)["model"]); model.eval()
    model=model.to(memory_format=torch.channels_last)
    gi,pi_,gl,pl_,gq,pq_=[],[],[],[],[],[]; t0=time.time()
    with torch.inference_mode():
        for k,p in enumerate(files):
            flat=Path(p).stem
            with h5py.File(p,"r") as f:
                n=int(f.attrs["n_cells"])
                if n==0: continue
                yi=IMAP.get(str(f.attrs["intensity"])); yl=LMAP.get(str(f.attrs["location"])); yq=qmap(f.attrs.get("quantity"))
                if yi is None or yl is None or yq is None: continue
                crops=f["crops"][:]
            PI,PL=[],[]
            for i in range(0,n,2048):
                x=T.crops_to_input(crops[i:i+2048],dev,False,m_,s_)
                with autocast("cuda",dtype=torch.bfloat16): li,ll=model(x)
                PI.append(li.argmax(-1).cpu().numpy()); PL.append(ll.argmax(-1).cpu().numpy())
            pi=np.concatenate(PI); pl=np.concatenate(PL)
            fr=(pi!=0).mean()
            ii=0 if fr<0.05 or (pi!=0).sum()==0 else int(np.bincount(pi[pi!=0]).argmax())
            lf=(pl!=0).mean(); ll_=0 if lf<0.05 else int(np.bincount(pl[pl!=0],minlength=4).argmax())
            qq=qbin(fr)
            gi.append(yi); pi_.append(ii); gl.append(yl); pl_.append(ll_); gq.append(yq); pq_.append(qq)
    gi,pi_,gl,pl_,gq,pq_=map(np.array,(gi,pi_,gl,pl_,gq,pq_))
    r=dict(ep=int(ep), n=len(gi),
           int_acc=float((gi==pi_).mean()), loc_acc=float((gl==pl_).mean()), qua_acc=float((gq==pq_).mean()),
           int_qwk=qwk(gi,pi_), loc_qwk=qwk(gl,pl_), qua_qwk=qwk(gq,pq_),
           int_f1=f1_score(gi,pi_,average="macro"), loc_f1=f1_score(gl,pl_,average="macro"), qua_f1=f1_score(gq,pq_,average="macro"))
    r["img_acc_avg"]=(r["int_acc"]+r["loc_acc"]+r["qua_acc"])/3
    r["img_qwk_avg"]=np.nanmean([r["int_qwk"],r["loc_qwk"],r["qua_qwk"]])
    rows.append(r); print(f"ep{ep} n={r['n']} | acc i/l/q {r['int_acc']:.3f}/{r['loc_acc']:.3f}/{r['qua_acc']:.3f} | QWK i/l/q {r['int_qwk']:.3f}/{r['loc_qwk']:.3f}/{r['qua_qwk']:.3f} | {time.time()-t0:.0f}s",flush=True)
pd.DataFrame(rows).to_csv(OUT,index=False); print("WROTE",OUT)
