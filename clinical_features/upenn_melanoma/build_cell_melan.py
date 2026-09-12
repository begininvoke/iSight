#!/usr/bin/env python3
"""MelanDx cell features = the USABLE subset of HANCOCK's neat154 descriptors, computed on
2-class cells (tumor=iSight target / non-tumor) with the single S100 marker (iSight intensity/location).
Kept EXACTLY as HANCOCK defines them; dropped only what needs a 3rd (immune) class (is_edge) or
was removed as non-interpretable (mix_entropy/assort/enrich). 21 descriptors, patient-mean. -> cell_melan.npz
  base(12): t_neg/weak/mod/strong, s_neg/weak/mod/strong, loc_nuc/cyto/mixed/none
  ti(7):    infil_pos/strong/mod/weak, intra_dens, stromal_dens, log_stained
  graph(2): ts_edge, stained_frac"""
import os, csv, numpy as np, zarr, collections
from scipy.spatial import cKDTree
PC = os.environ.get("PENN_IHC_DIR", "")   # per-slide .zarr cell predictions
LINK = os.environ.get("MELAN_LINKAGE_DIR", "")   # slide <-> patient linkage tables
A = os.environ.get("MELAN_ANALYSIS_DIR", "")     # cohort analysis outputs
K=8; MAXN=60000
NAMES=["t_neg","t_weak","t_mod","t_strong","s_neg","s_weak","s_mod","s_strong",
       "loc_nuc","loc_cyto","loc_mixed","loc_none",
       "infil_pos","infil_strong","infil_mod","infil_weak","intra_dens","stromal_dens","log_stained"]
def load(zp):
    z=zarr.open_group(zp,mode="r"); c=z["SegmentationNode"]["centroids"][:].astype(np.float32); n=len(c)
    cp=z["CellPred"]; ki=cp["keep_index"][:]; tgt=np.zeros(n,bool); inten=np.full(n,-1,np.int8); loc=np.full(n,-1,np.int8)
    tgt[ki[cp["is_target"][:]]]=True; inten[ki]=cp["pred_intensity"][:]; loc[ki]=cp["pred_location"][:]
    return c,tgt,inten,loc
def slide_feat(zp):
    try: c,tgt,inten,loc=load(zp)
    except Exception: return None
    n=len(c)
    if n>MAXN:
        sel=np.sort(np.random.RandomState(0).choice(n,MAXN,replace=False)); c,tgt,inten,loc=c[sel],tgt[sel],inten[sel],loc[sel]; n=MAXN
    tum=tgt
    if tum.sum()<20 or n<40: return None
    tpi=inten[tum]; npi=inten[~tum]                     # tumor / non-tumor S100 intensity
    f=np.full(len(NAMES),np.nan,np.float32)
    tt=tpi[tpi>=0]; ntt=max(len(tt),1); ss=npi[npi>=0]; nss=max(len(ss),1)
    for i,lv in enumerate([0,1,2,3]): f[i]=(tt==lv).sum()/ntt           # t_*
    for i,lv in enumerate([0,1,2,3]): f[4+i]=(ss==lv).sum()/nss         # s_*
    stl=loc[(inten>0)]; stl=stl[stl>=0]; nstl=max(len(stl),1)           # location on S100+ cells
    f[8]=(stl==1).sum()/nstl; f[9]=(stl==2).sum()/nstl; f[10]=(stl==3).sum()/nstl; f[11]=max(0,1-f[8]-f[9]-f[10])
    # ti (HANCOCK ti_feats): tpi=tumor, npi=non-tumor
    nt=int((tpi>0).sum()); nn=int((npi>0).sum())
    f[12]=nt/(nt+nn) if (nt+nn)>0 else np.nan                            # infil_pos
    for k,lv in [(13,3),(14,2),(15,1)]:
        a=int((tpi==lv).sum()); b=int((npi==lv).sum()); f[k]=a/(a+b) if (a+b)>0 else np.nan
    f[16]=float((tpi>0).mean()) if len(tpi) else np.nan                  # intra_dens
    f[17]=float((npi>0).mean()) if len(npi) else np.nan                  # stromal_dens
    f[18]=float(np.log1p(nt+nn))                                         # log_stained
    return f, int(tum.sum())
rows=list(csv.DictReader(open(f"{LINK}/master_slide_table.csv")))
pat=collections.defaultdict(list)
for r in rows:
    if r['modality']!='IHC' or not r['plgid'] or r['plgid_flag']=='control': continue
    res=slide_feat(f"{PC}/{r['sub']}/{r['slide']}.zarr")
    if res: pat[r['plgid']].append(res)
plg=[]; feat=[]
for p,lst in pat.items():
    F=np.array([f for f,_ in lst]); w=np.array([nn for _,nn in lst],float); w=w/w.sum()
    v=np.nansum(np.where(np.isfinite(F),F,0)*w[:,None],0); plg.append(p); feat.append(v)
np.savez(f"{A}/cell_melan.npz",plgids=np.array(plg),feat=np.array(feat,np.float32),cols=np.array(NAMES))
print(f"cell_melan.npz: {len(plg)} patients x {len(NAMES)} (HANCOCK-usable descriptors)")
F=np.array(feat)
for i,nm in enumerate(NAMES): print(f"  {nm:14s} mean={np.nanmean(F[:,i]):.3f}")
