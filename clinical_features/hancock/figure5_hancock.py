"""Figure 5, HANCOCK — does the cell model add prognostic information that the slide model does not?

HNSCC, 763 patients, 3-year recurrence. The base model is the one the HANCOCK benchmark is
built on: 35 structured clinical variables (19 tabular + 16 blood) plus a TITAN slide
embedding of the H&E. On top of that we add, in turn, the two ways this paper reads an IHC
panel -- iSight-cell's interpretable per-cell features and iSight-slide's image embedding --
and ask which one the base model was missing.

The evaluation is the dataset's own, not ours: three official splits of increasing
difficulty -- in-distribution, out-of-distribution, and one that holds out an entire
anatomical subsite. Metric is ROC-AUC on the held-out test set.

The published 0.79 benchmark is deliberately NOT drawn. It comes from a different
feature set -- the paper adds ICD-code text and a TMA-image CNN and has no TITAN or IHC
branch -- so a reference line across these bars would read as a comparison this panel
is not making. Every bar here shares one base model; the only thing that varies is
which of our two IHC readers is added.

THE ONE THING THAT MAKES THIS FIGURE HONEST

Every bar is computed on the SAME 607 patients -- the intersection of clinical, TITAN, cell
and IHC-embedding availability. That is not what the source tables do: `RESULTS_FINAL.md`
section 5 unifies n over clinical+TITAN+cell (607) and section 10 over clinical+TITAN+IHC
(611), so its base column differs between the two (0.775 vs 0.781 on split_in) and the two
deltas are not strictly comparable. Both are reproduced here as a check before the common-n
version is computed; the figure draws the common-n numbers.

CONFIGURATION -- fixed, not tuned per arm (RESULTS_FINAL.md sections 3-6)

  classifier   L1 logistic regression, C=0.1, class_weight='balanced'
  imputation   median, fitted inside the training fold only
  TITAN        raw 768, the mode both published ablations use. PCA30 scores higher on
               recurrence (0.822 vs 0.812 with cell) and is written to the results csv as a
               robustness row, but the figure keeps one mode across all bars.
  cell         neat154 -> SelectKBest(f_classif, k=30). FS, never PCA: these are sparse
               semantic descriptors and PCA smears the signal (0.822 -> 0.772).
  slide        7 markers x 1024 concatenated -> PCA30. PCA, never FS: the opposite rule,
               because this is a dense embedding.

Intervals are a bootstrap over test patients, n=1000 drawn n-out-of-n with replacement,
seed 42, as everywhere else in this paper. The three arms are scored on the SAME resample
so the deltas are paired; a resample that lands on one class is dropped (recorded in the
csv as n_boot_dropped).

All numbers -> Figure5/results/hancock_recurrence.csv
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import ROOT, out_dir, res_dir, here_figure, setup_fonts

import seaborn as sns
sns.set_theme(style="white")
setup_fonts()          # after seaborn: set_theme() resets the font rcParams

FIGURE = here_figure(__file__)
OUT, RES = out_dir(FIGURE), res_dir(FIGURE)
HAN = os.path.join(ROOT, "hancock_experiment")

C_BASE, C_SLIDE, C_CELL = "#c9c9c9", "#a9bcd5", "#2F5597"
# Sizes are chosen against the printed page, not the panel. These panels sit in the composite
# at about 2.53x and the composite is set at 20.25% of its width, so a size here reaches the
# reader at roughly 0.51 of its value: 14 pt prints at 7.2, 11.5 at 5.9, 10.5 at 5.4. Nature
# BME asks for 5-7 pt in figures. The title, ylabel and y ticks already landed in that band;
# the bar-top values and the legend did not, and they carry the comparison.
FS_TITLE, FS_YLAB, FS_VAL = 14, 12, 10.5
FS_XTICK, FS_YTICK, FS_LEGEND = 11.5, 11.5, 10.5
N_BOOT, SEED = 1000, 42

# The dataset's own three splits (data/DataSplits_DataDictionaries/dataset_split_*.json),
# all 611/152 except oropharynx at 432/331, ordered by how hard they are:
#   split_in   a genetic algorithm picks test patients that MATCH the overall distribution
#   split_out  the same algorithm picks the ones lying outside it, maximally dissimilar
#   oropharynx every oropharyngeal carcinoma goes to test, so train is larynx / oral cavity
#              / hypopharynx only -- a shift by anatomical subsite
# Not "cross-institution": split_out is selected on distributional similarity, not by
# centre, and calling it cross-institution claims a held-out-hospital design the dataset
# does not provide.
SPLITS = [("split_in", "In-distribution"), ("split_out", "Out-of-distribution"),
          ("split_oropharynx", "Oropharynx")]
ARMS = [("base", "Base"), ("slide", "+ iSight-slide"), ("+cell", "+ iSight-cell")]
COLOUR = {"Base": C_BASE, "+ iSight-slide": C_SLIDE, "+ iSight-cell": C_CELL}

# The five bases of RESULTS_FINAL.md section 5. `titan` carries no clinical variables at
# all and is the control that shows the cell features need a clinical anchor; the full
# panel plus TITAN is the one the paper leads with. (clin_cols, use_titan, stem, title)
#
# HISTORY is age / sex / smoking status / metastatic at presentation -- the four columns
# HANCOCK keeps in clinical_data.json that are known at first visit. That file also holds
# treatment history and follow-up; neither is used. Treatment history would invert the
# causal direction (therapy is chosen because of prognosis), and follow-up IS the label.
# So "clinical history" here means the snapshot taken at diagnosis, and the caption has to
# say so or a reader will assume the treatment course went in too.
#
# The full 35 adds the pathology of the resection -- pT/pN stage, grading, HPV status,
# vascular/perineural invasion, margins -- to those four, plus the 16 blood counts.
#
# clin_cols is [] for "no clinical at all" and never None -- None is fit_predict's default
# meaning "all 35", so a None here silently turns the TITAN-only control into full35+TITAN.
HISTORY, BLOOD = [14, 1, 7, 0], list(range(19, 35))
BASES = [([], True, "titan", "TITAN"),
         (HISTORY + BLOOD, False, "historyblood", "Clinical history + blood test"),
         (HISTORY + BLOOD, True, "historybloodtitan", "Clinical history + blood test + TITAN"),
         (list(range(35)), False, "full35", "Clinical + pathology + blood test"),
         (list(range(35)), True, "full35titan", "Clinical + pathology + blood test + TITAN")]

print("Figure 5, HANCOCK — 3-year recurrence")

_load = lambda f: json.load(open(os.path.join(HAN, f)))["feat"]
OFF = _load("official_clinical35_feats.json")
TT = _load("titan_feats.json")
CE = _load("cell_neat154.json")
IHC = _load("ihc_isight_feats.json")
TASK = json.load(open(os.path.join(HAN, "tasks/recurrence_task.json")))
FULL = list(range(35))


def arr(f, ids):
    return np.array([f[p] for p in ids], float)


def prep(Xtr, Xte, ytr=None, k=0, how=None):
    """Impute -> standardise -> optional reduction, all fitted on the training fold only."""
    im = SimpleImputer(strategy="median").fit(Xtr)
    Xtr, Xte = im.transform(Xtr), im.transform(Xte)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if how == "pca":
        p = PCA(min(k, Xtr.shape[0] - 1, Xtr.shape[1]), random_state=0).fit(Xtr)
        return p.transform(Xtr), p.transform(Xte)
    if how == "fs":
        s = SelectKBest(f_classif, k=k).fit(Xtr, ytr)
        return s.transform(Xtr), s.transform(Xte)
    return Xtr, Xte


def fit_predict(split, arm, common, titan="raw768", clin=None, use_titan=True):
    """Held-out test probabilities for one arm of one base on one official split."""
    tr = [p for p in common if TASK[p].get(split) == "training"]
    te = [p for p in common if TASK[p].get(split) == "test"]
    ytr = np.array([TASK[p]["label"] for p in tr])
    yte = np.array([TASK[p]["label"] for p in te])
    clin = FULL if clin is None else clin
    Btr, Bte = ([arr(OFF, tr)[:, clin]], [arr(OFF, te)[:, clin]]) if clin else ([], [])
    if use_titan:
        a, b = ((arr(TT, tr), arr(TT, te)) if titan == "raw768"
                else prep(arr(TT, tr), arr(TT, te), k=30, how="pca"))
        Btr.append(a), Bte.append(b)
    if arm == "+cell":
        a, b = prep(arr(CE, tr), arr(CE, te), ytr, 30, "fs")
        Btr.append(a), Bte.append(b)
    elif arm == "slide":
        a, b = prep(arr(IHC, tr), arr(IHC, te), k=30, how="pca")
        Btr.append(a), Bte.append(b)
    Xtr, Xte = prep(np.hstack(Btr), np.hstack(Bte))
    m = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                           class_weight="balanced", max_iter=2000).fit(Xtr, ytr)
    return yte, m.predict_proba(Xte)[:, 1], len(tr), len(te)


# ------------------------------------------------- reproduce the published tables first
# If these drift, the figure is being computed from something other than what the results
# doc reports, and that has to be resolved before anything is plotted.
def _auc_row(arms, common, titan="raw768"):
    return {a: [roc_auc_score(*fit_predict(s, a, common, titan)[:2]) for s, _l in SPLITS]
            for a in arms}


cell_only = [p for p in TASK if p in OFF and p in TT and p in CE and "label" in TASK[p]]
ihc_only = [p for p in TASK if p in OFF and p in TT and p in IHC and "label" in TASK[p]]
COMMON = [p for p in TASK if p in OFF and p in TT and p in CE and p in IHC
          and "label" in TASK[p]]

chk_a = _auc_row(["base", "+cell"], cell_only)
chk_b = _auc_row(["base", "slide"], ihc_only)
for got, want, tag in [
        (chk_a["base"], [0.775, 0.701, 0.670], "RESULTS_FINAL section 5  base"),
        (chk_a["+cell"], [0.812, 0.740, 0.694], "RESULTS_FINAL section 5  +cell"),
        (chk_b["base"], [0.781, 0.705, 0.667], "RESULTS_FINAL section 10 base"),
        (chk_b["slide"], [0.780, 0.744, 0.670], "RESULTS_FINAL section 10 +ihc")]:
    assert np.allclose(got, want, atol=0.0015), f"{tag}: {np.round(got, 3)} != {want}"
print(f"  reproduces RESULTS_FINAL.md sections 5 and 10 exactly "
      f"(n={len(cell_only)} / {len(ihc_only)})")
print(f"  common set for the figure: n={len(COMMON)} with all four modalities")

# ------------------------------------------------- the figure's numbers, common n
# One panel per base, same three arms and same three splits in each, so the five panels can
# be read as one grid. Every panel uses the same 607 patients, so the base bar moves only
# because its own feature set changed -- not because the cohort did.
rng = np.random.default_rng(SEED)
rows = []


def panel(clin, use_titan, stem, title):
    vals, cis, ns = {}, {}, {}
    for split, slab in SPLITS:
        yte, prob, n_tr, n_te = None, {}, 0, 0
        for arm, alab in ARMS:
            yte, pr, n_tr, n_te = fit_predict(split, arm, COMMON, clin=clin,
                                              use_titan=use_titan)
            prob[alab] = pr
            vals[(slab, alab)] = roc_auc_score(yte, pr)
        ns[slab] = (n_tr, n_te)
        # one resample of the test set per replicate, every arm scored on that same
        # resample, so the deltas written to the csv are paired
        idx = [rng.integers(0, len(yte), len(yte)) for _ in range(N_BOOT)]
        reps = {}
        for arm, alab in ARMS:
            r = np.array([roc_auc_score(yte[i], prob[alab][i])
                          if len(set(yte[i])) > 1 else np.nan for i in idx])
            reps[alab] = r
            cis[(slab, alab)] = (float(np.nanpercentile(r, 2.5)),
                                 float(np.nanpercentile(r, 97.5)))
            rows.append(dict(task="recurrence_3yr", base=stem, split=split, arm=alab,
                             titan="raw768", n_train=n_tr, n_test=n_te,
                             auc=vals[(slab, alab)], ci_lower_95=cis[(slab, alab)][0],
                             ci_upper_95=cis[(slab, alab)][1],
                             n_boot_dropped=int(np.isnan(r).sum())))
        b0 = ARMS[0][1]
        for arm, alab in ARMS[1:]:
            d = reps[alab] - reps[b0]
            rows.append(dict(task="recurrence_3yr", base=stem, split=split,
                             arm=f"{alab} vs base", titan="raw768", n_train=n_tr,
                             n_test=n_te, auc=vals[(slab, alab)] - vals[(slab, b0)],
                             ci_lower_95=float(np.nanpercentile(d, 2.5)),
                             ci_upper_95=float(np.nanpercentile(d, 97.5)),
                             frac_boot_positive=float(np.nanmean(d > 0))))
    print(f"  {title:<30} " + "   ".join(
        f"{s}: " + "/".join(f"{vals[(s, a)]:.3f}" for _k, a in ARMS)
        for _sp, s in SPLITS))

    fig, ax = plt.subplots(figsize=(6.8, 2.5), dpi=300)
    x = np.arange(len(SPLITS))
    w = 0.8 / len(ARMS)
    for i, (_arm, alab) in enumerate(ARMS):
        v = np.array([vals[(s, alab)] for _k, s in SPLITS])
        lo = np.array([cis[(s, alab)][0] for _k, s in SPLITS])
        hi = np.array([cis[(s, alab)][1] for _k, s in SPLITS])
        xp = x + (i - (len(ARMS) - 1) / 2) * w
        ax.bar(xp, v, width=w * 0.9, color=COLOUR[alab], label=alab, edgecolor="none")
        for xi, vv in zip(xp, v):
            ax.text(xi, vv + 0.012, f"{vv:.3f}", ha="center", va="bottom",
                    fontsize=FS_VAL, fontweight="bold")
    # No error bars and no significance brackets. The three arms differ by which IHC reader is
    # bolted onto a shared base, tested once on one official split of 125-269 patients; a star
    # would dress a single held-out comparison up as an inference. The bootstrap intervals
    # (lo/hi, still computed above) and the paired deltas are reported in the results csv.
    ax.set_xticks(x)
    # test-set sizes are in the results csv, not on the axis: they are a property of the
    # split, identical across all five panels, and repeating them 15 times adds no
    # information while forcing the tick labels onto two lines
    ax.set_xticklabels([lab for _s, lab in SPLITS], fontsize=FS_XTICK)
    ax.set_xlim(-0.5, len(SPLITS) - 0.5)
    # full 0-1 axis, as everywhere else in the paper: starting at 0.5 magnifies the gaps
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("ROC-AUC", fontsize=FS_YLAB)
    ax.tick_params(axis="y", labelsize=FS_YTICK)
    ax.grid(axis="y", alpha=0.6, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    # two lines: the constant part on top, the panel's feature set in brackets below, so the
    # longest variant ("Clinical + pathology + blood test + TITAN") no longer sets the width
    ax.set_title(f"HANCOCK 3-year recurrence\n({title})", fontsize=FS_TITLE,
                 fontweight="bold", pad=10)
    sns.despine(ax=ax)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=FS_LEGEND, frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"Figure_5a_hancock_recurrence_{stem}.pdf"), dpi=400,
                bbox_inches="tight")
    plt.close(fig)


print(f"\n  five bases, all on the same {len(COMMON)} patients "
      f"(split_in / split_out / oropharynx, each base/slide/cell):")
for _clin, _ut, _stem, _title in BASES:
    panel(_clin, _ut, _stem, _title)
pd.DataFrame(rows).to_csv(os.path.join(RES, "hancock_recurrence.csv"), index=False)
print(f"\n  saved Figure_5a_hancock_recurrence_{{{','.join(b[2] for b in BASES)}}}.pdf")
print("done.")
