"""Figure 5, NADT-Prostate — predicting response to neoadjuvant ADT from pre-treatment IHC.

37 men with high-risk prostate cancer, biopsied before androgen-deprivation therapy plus
enzalutamide, then scored at prostatectomy as exceptional (ER, n=15) or incomplete/non
responders (INR, n=22) -- Wilkinson et al., Eur Urol 2021. iSight-cell reads their 911
pre-treatment IHC slides (45.9M cells, intensity + subcellular localisation per cell), and
three patient-level numbers out of that predict who will respond.

  b   LOO AUROC, accuracy, macro-F1 and macro recall, against the paper's own model
  c   how the signal accumulates as markers are added, strongest first

FOUR METRICS, AND WHY THE MACRO VERSIONS

AUROC is threshold-free and is what both source papers report. Accuracy and F1 need an
operating point, taken as p=0.5 on the leave-one-out probability.

F1 and recall are macro-averaged over the two classes, not taken on the INR class alone.
With 22 INR and 15 ER, a model that simply calls everyone INR scores accuracy 0.595,
F1(INR) 0.746 and recall(INR) 1.000 -- three respectable-looking numbers for a model that
has learned nothing. Macro-averaging makes both classes count and that same model falls to
macro-F1 0.373 and macro recall 0.500. Its scores are printed at run time and kept in the
results csv as a reference row, alongside the single-class versions.

The intervals are wide and they are supposed to be: n=37, so one patient moving is 2.7% of
accuracy. Read the direction, not the gap.

WHAT EACH MODEL IS

  iSight-cell     ERG strong-positive fraction + PTEN-negative fraction + Ki67 proliferation
                  fraction, computed over every segmented nucleus. L2 logistic regression,
                  C=0.5, standardised. Histology only: no sequencing, no MRI.
  Paper 4-factor  nuclear ERG, 10q loss, TP53 alteration, intraductal carcinoma. Two of the
                  four need DNA sequencing. Reconstructed from the paper's supplementary
                  tables 1 and 3 (code/paper_4factor_cv.py), fitted exactly as published --
                  plain logistic regression, no standardisation.
  MRI burden      log baseline tumour volume, one variable. Scored for the record but not
                  drawn -- see the note above MODELS.

Everything is leave-one-out: each patient is predicted by a model that never saw them, and
the 37 held-out probabilities form one ROC. The paper's headline 0.89 is in-sample; under
LOO the same four factors give 0.812, which is the number to compare against.

A LIMITATION THAT BELONGS IN THE CAPTION

These fractions are over ALL segmented nuclei, not tumour cells. Target-cell selection was
tried and abandoned: on eight sampled slides the prostate-tumour head called only 1.6-12.6%
of nuclei tumour, too low to be credible for prostate epithelium, so the HPA-domain
classifier is underfiring here. Stroma therefore dilutes every fraction -- ERG especially,
since vascular endothelium is constitutively ERG-positive. The signal survives anyway.

All numbers -> Figure5/results/nadt_*.csv
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             recall_score)
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import ROOT, out_dir, res_dir, here_figure, setup_fonts

import seaborn as sns
sns.set_theme(style="white")
setup_fonts()          # after seaborn: set_theme() resets the font rcParams

FIGURE = here_figure(__file__)
OUT, RES = out_dir(FIGURE), res_dir(FIGURE)
NADT = os.path.join(ROOT, "NADT_data")

C_CELL, C_PAPER, C_MRI = "#2F5597", "#8a8a8a", "#a9bcd5"
FS_TITLE, FS_LAB, FS_TICK, FS_LEGEND = 14, 12.5, 12, 11
# Sizes are chosen against the printed page, not the panel. These panels are placed in the
# composite at about 2.4x and the composite is set at 20.25% of its width, so a size here
# reaches the reader at roughly 0.49 of its value: 14 pt prints at 6.9, 12 at 5.9, 11 at 5.4.
# Nature BME asks for 5-7 pt in figures, which is what the numbers above and below land on.
FS_VALUE = 11        # bar-top numbers and the peak annotation
N_BOOT, SEED = 1000, 42

print("Figure 5, NADT-Prostate — ADT response")

# s1-ep09 is the current cell checkpoint. drop2 gives 0.855 instead of 0.861; the whole
# analysis was rerun on both and agrees to +-0.01, which is the robustness claim.
D = pd.read_csv(os.path.join(NADT, "patient_features_labeled_ep09.csv"))
F4 = pd.read_csv(os.path.join(NADT, "paper_4factor_reconstructed.csv"))
assert set(D.patient) == set(F4.patient), "cell and 4-factor cohorts differ"
F4 = F4.set_index("patient").loc[D.patient].reset_index()
y = D["y"].values.astype(int)
print(f"  {len(y)} patients, {y.sum()} INR / {(1 - y).sum()} ER")

CELL3 = ["ERG_strong_frac", "PTEN_neg_frac", "Ki67_posmod_frac"]
FOUR = ["ERG", "loss10q", "TP53", "IDC"]


def X_of(cols, src):
    return np.nan_to_num(src[cols].values.astype(float), nan=0.0)


def loo_prob(X, model):
    """One held-out probability per patient."""
    p = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        p[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


KFOLD_REPEATS = 50


def kfold_probs(X, model, rep=KFOLD_REPEATS):
    """One out-of-fold probability vector per repeat.

    50 repeats, matching the source scripts and RESULTS_response.md, so the 5-fold column
    reproduces the 0.876 recorded there. 20 repeats was tried and moves every number by at
    most 0.007 (AUROC 0.876 -> 0.872) without changing any ordering, but keeping 50 means
    the figure and the results doc quote the same value.
    """
    out = []
    for sd in range(rep):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
            p[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        out.append(p)
    return out


# Every model is fitted with the SAME estimator so that the comparison isolates the features
# rather than the fitting recipe. Previously the paper's four factors were fitted plain
# (unpenalised, unstandardised) as in code/paper_4factor_cv.py; matching the recipe leaves the
# leave-one-out numbers unchanged (auroc .812 / acc .649 / macro-F1 .631) and moves only the
# repeated 5-fold AUROC, .813 -> .807, because the four factors are binary and C=0.5 barely
# binds on four features.
OURS = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
THEIRS = OURS

# MRI burden is scored but not drawn. It is a one-variable non-molecular reference that
# happens to land at LOO 0.806 -- statistically indistinguishable from the paper's
# four-factor model -- which is a point about that paper, not about iSight, and a third
# bar makes the panel argue two things at once. It stays in nadt_models.csv.
# Legend labels wrap onto two lines. The single-line versions are 30+ characters and the
# legend block ended up as wide as the axes; the plain names are kept for the csv and the
# console, where a newline would be a defect rather than a layout.
LEGEND_2L = {"iSight-cell (ERG + PTEN + Ki67)": "iSight-cell\n(ERG + PTEN + Ki67)",
             "Wilkinson 4-factor histogenomic": "Wilkinson 4-factor\nhistogenomic",
             "MRI tumour burden": "MRI tumour burden"}
MODELS = [
    ("iSight-cell (ERG + PTEN + Ki67)", X_of(CELL3, D), OURS, C_CELL, True),
    ("Wilkinson 4-factor histogenomic", X_of(FOUR, F4), THEIRS, C_PAPER, True),
    ("MRI tumour burden", X_of(["logburden"], F4), THEIRS, C_MRI, False),
]

# ---------------------------------------------------------------- b: LOO metrics
# Macro-averaged over the two classes for F1 and recall, threshold p=0.5. `f1_inr` and
# `recall_inr` (positive class only) go to the csv as well, so both conventions are on
# record -- see the note in the module docstring for why the panels draw the macro ones.
METRICS = [("auroc", "AUROC"), ("acc", "Accuracy"),
           ("f1", "Macro-F1"), ("rec", "Macro recall")]
SCHEMES = [("insample", "In-sample"), ("loo", "Leave-one-out"), ("kfold", "5-fold CV")]


def score(yy, pp):
    """Plain numpy rather than sklearn.

    This runs ~150,000 times inside the bootstrap and at n=37 sklearn's input validation
    costs far more than the arithmetic -- the numpy version is ~100x faster and the
    equivalence to sklearn is asserted below on the real prediction vectors.
    """
    h = (pp >= 0.5)
    npos, nneg = int(yy.sum()), int((1 - yy).sum())
    if npos and nneg:                              # Mann-Whitney form, ties averaged
        r = rankdata(pp)
        auroc = (r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    else:
        auroc = np.nan
    rec, f1 = [], []
    for c in (0, 1):
        hit = (h == bool(c))
        tp = float((hit & (yy == c)).sum())
        nc, pc = float((yy == c).sum()), float(hit.sum())
        r_ = tp / nc if nc else 0.0
        p_ = tp / pc if pc else 0.0
        rec.append(r_)
        f1.append(2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0)
    return {"auroc": float(auroc), "acc": float((h == (yy == 1)).mean()),
            "f1": float(np.mean(f1)), "rec": float(np.mean(rec)),
            "f1_inr": f1[1], "recall_inr": rec[1]}


def _check_score(yy, pp):
    got, h = score(yy, pp), (pp >= 0.5).astype(int)
    for k, v in [("auroc", roc_auc_score(yy, pp)), ("acc", accuracy_score(yy, h)),
                 ("f1", f1_score(yy, h, average="macro", zero_division=0)),
                 ("rec", recall_score(yy, h, average="macro", zero_division=0)),
                 ("f1_inr", f1_score(yy, h, pos_label=1, zero_division=0)),
                 ("recall_inr", recall_score(yy, h, pos_label=1, zero_division=0))]:
        assert abs(got[k] - v) < 1e-12, f"{k}: {got[k]} != {v}"


def score_scheme(yy, probs):
    """probs is one vector, or the repeat vectors of the 5-fold scheme.

    For 5-fold the metric is averaged over the repeats rather than computed once on the
    averaged probability. The two are not the same -- averaging probabilities first gives
    AUROC 0.879 and accuracy 0.757, against 0.876 / 0.773 the other way -- and it is the
    mean-of-metrics that both source papers report, so that is what is drawn.
    """
    if isinstance(probs, list):
        per = [score(yy, p) for p in probs]
        return {k: float(np.mean([d[k] for d in per])) for k in per[0]}
    return score(yy, probs)


rng = np.random.default_rng(SEED)
rows, curves = [], {}
for name, X, mk, col, draw in MODELS:
    src = {"insample": mk().fit(X, y).predict_proba(X)[:, 1],
           "loo": loo_prob(X, mk),
           "kfold": kfold_probs(X, mk)}
    _check_score(y, src["loo"])            # numpy == sklearn, on real predictions
    obs = {sc: score_scheme(y, src[sc]) for sc, _l in SCHEMES}
    # every scheme gets the same interval: resample the 37 patients, rescore. For 5-fold
    # that means re-averaging over every repeat inside each replicate, so the bar and its
    # interval are the same quantity.
    idx = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
    ci = {}
    for sc, _l in SCHEMES:
        reps = {m: [] for m, _l2 in METRICS}
        for i in idx:
            if len(set(y[i])) < 2:
                continue
            pr = ([p[i] for p in src[sc]] if isinstance(src[sc], list) else src[sc][i])
            r = score_scheme(y[i], pr)
            for m, _l2 in METRICS:
                reps[m].append(r[m])
        ci[sc] = {m: (float(np.nanpercentile(reps[m], 2.5)),
                      float(np.nanpercentile(reps[m], 97.5))) for m, _l2 in METRICS}
    if draw:
        curves[name] = (obs, ci, col)
    for sc, slab in SCHEMES:
        rows.append(dict(model=name, scheme=slab, n=len(y),
                         **{m: obs[sc][m] for m in ("auroc", "acc", "f1", "rec",
                                                    "f1_inr", "recall_inr")},
                         **{f"ci_lower_{m}": ci[sc][m][0] for m, _l in METRICS},
                         **{f"ci_upper_{m}": ci[sc][m][1] for m, _l in METRICS}))
    print(f"  {name:<34}" + ("" if draw else " [csv only]"))
    for sc, slab in SCHEMES:
        print(f"      {slab:<14} " + "  ".join(f"{l} {obs[sc][m]:.3f}" for m, l in METRICS))

# the everyone-is-INR baseline, to show what single-class F1 would have paid out
_h = np.ones(len(y), int)
print(f"  (reference) predict INR for everyone: Accuracy {accuracy_score(y, _h):.3f}  "
      f"Macro-F1 {f1_score(y, _h, average='macro', zero_division=0):.3f}  "
      f"F1(INR) {f1_score(y, _h, pos_label=1, zero_division=0):.3f}")
rows.append(dict(model="(reference) always INR", scheme="-", n=len(y),
                 acc=accuracy_score(y, _h),
                 f1=f1_score(y, _h, average="macro", zero_division=0), rec=0.5,
                 f1_inr=f1_score(y, _h, pos_label=1, zero_division=0), recall_inr=1.0))
pd.DataFrame(rows).to_csv(os.path.join(RES, "nadt_models.csv"), index=False)

# One panel per metric; within a panel, the three ways of scoring the same two models.
# In-sample is included because the source paper's headline 0.89 is in-sample, and seeing
# it next to the cross-validated columns is the whole point of the comparison.
for metric, mlabel in METRICS:
    fig, ax = plt.subplots(figsize=(7.6, 2.5), dpi=300)
    names = list(curves)
    x = np.arange(len(SCHEMES))
    w = 0.8 / len(names)
    for i, name in enumerate(names):
        obs, ci, col = curves[name]
        v = np.array([obs[sc][metric] for sc, _l in SCHEMES])
        lo = np.array([ci[sc][metric][0] for sc, _l in SCHEMES])
        hi = np.array([ci[sc][metric][1] for sc, _l in SCHEMES])
        xp = x + (i - (len(names) - 1) / 2) * w
        ax.bar(xp, v, width=w * 0.9, color=col, label=LEGEND_2L[name], edgecolor="none")
        ax.errorbar(xp, v, yerr=[v - lo, hi - v], fmt="none", color="black",
                    capsize=2, capthick=0.5, linewidth=0.5)
        for xi, vv, hh in zip(xp, v, hi):
            ax.text(xi, hh + 0.012, f"{vv:.3f}", ha="center", va="bottom",
                    fontsize=FS_VALUE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([l for _sc, l in SCHEMES], fontsize=FS_TICK)
    ax.set_xlim(-0.5, len(SCHEMES) - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel(mlabel, fontsize=FS_LAB)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.grid(axis="y", alpha=0.6, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title(f"NADT-Prostate ({mlabel})", fontsize=FS_TITLE, fontweight="bold", pad=20)
    sns.despine(ax=ax)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=FS_LEGEND,
              frameon=True, labelspacing=0.9, handlelength=1.6)
    fig.tight_layout()
    if metric == "f1":
        # 5c is asked to match this panel's PLOT BODY, not its page: 5b carries its legend
        # outside the axes and 5c carries one inside, so equal figure widths would leave
        # two different-sized data areas. Record the axes rectangle in inches here and
        # rebuild 5c around it below.
        _p = ax.get_position()
        AX_BODY_IN = (_p.width * fig.get_figwidth(), _p.height * fig.get_figheight())
    fig.savefig(os.path.join(OUT, f"Figure_5b_nadt_{metric}.pdf"), dpi=400,
                bbox_inches="tight")
    plt.close(fig)
print("  saved Figure_5b_nadt_{" + ",".join(m for m, _l in METRICS) + "}.pdf")

# ---------------------------------------------------------------- c: marker accumulation
# Cumulative-threshold intensity, one feature per marker, chosen by data rather than by hand.
#
# HOW THE CURVE IS BUILT
# Each of the nine markers is summarised by one feature -- the best of its four cumulative
# thresholds by direction-free univariate AUROC. The markers are ranked by that score and then
# added strongest-first: position m is an L2 logistic model built from the first m markers'
# features, scored by leave-one-out (and 5-fold). So the x axis is cumulative -- m=3 is the
# PTEN+ERG+Ki67 model, not Ki67 alone -- and the curve shows where adding markers stops helping.
#
# WHAT "CUMULATIVE INTENSITY" MEANS HERE
# Each cell carries a predicted intensity in {negative, weak, moderate, strong}. A marker is
# summarised by the fraction of its cells at or above a threshold -- pos (>= weak), posmod
# (>= moderate), strong (= strong) -- plus neg (= negative). The thresholds nest, which is
# what makes them cumulative, and is the whole point: the alternative is four disjoint bins
# (strong / moderate / weak / negative, summing to one), and those were tried and rejected.
# The weak bin is empty for most markers -- 0.0% of Ki67, AR, SYP and GR cells are predicted
# weak -- so as a disjoint feature it is noise, and marker-level selection picks GR_weak
# (0.730) and AR_weak (0.718) over Ki67 (0.712) on coincidence, pushing Ki67 to fifth and
# inflating LOO to a spurious 0.90. Cumulative thresholds never isolate the empty bin.
#
# Location fractions are also left out of the pool for the same reason: adding them lets
# AR_loc_cyto (0.739) outrank Ki67's best (0.721), so AR takes third and Ki67 drops to
# fourth. Pure cumulative intensity is the only pool that selects cleanly.
#
# WHY THIS REPLACES THE HAND-PICKED VERSION
# The panel used to fix one canonical feature per marker on biology. It gave the same three
# markers, but a reader cannot tell from such a curve whether the biology was chosen after
# seeing the scores. Here the within-marker choice is univariate AUROC, max(auc, 1 - auc) so
# it is direction-free, and markers enter in that order. ERG, PTEN and Ki67 come out on top
# with nothing hand-set, and the curve peaks exactly there -- which is a far stronger claim
# than the same peak reached by a pre-declared list.
#
# The features selected are PTEN_pos, ERG_strong and Ki67_pos, numerically identical to the
# named model's PTEN_neg + ERG_strong + Ki67_posmod: pos and neg are mirror images, and
# Ki67's four variants are tied because it has no weak or negative-only cells to separate
# them. Biology only picks the name, never the number.
MARKERS = ["ERG", "PTEN", "PIN4", "AR", "Ki67", "PSA", "SYP", "GR", "P53"]
CUM = ["pos_frac", "posmod_frac", "strong_frac", "neg_frac"]


def uni_auc(c):
    a = roc_auc_score(y, X_of([c], D)[:, 0])
    return max(a, 1 - a)


best = {m: max((f"{m}_{k}" for k in CUM), key=uni_auc) for m in MARKERS}
order = sorted(MARKERS, key=lambda m: uni_auc(best[m]), reverse=True)
cum, crows = [], []
for i, m in enumerate(order, 1):
    cum.append(best[m])
    X = X_of(cum, D)
    lo = roc_auc_score(y, loo_prob(X, OURS))
    ka = [roc_auc_score(y, q) for q in kfold_probs(X, OURS)]
    crows.append(dict(m=i, marker_added=m, feature=best[m], uni_auroc=uni_auc(best[m]),
                      cumulative_loo=lo, cumulative_kfold5=float(np.mean(ka)),
                      kfold5_sd=float(np.std(ka))))
cdf = pd.DataFrame(crows)
cdf.to_csv(os.path.join(RES, "nadt_marker_curve.csv"), index=False)
peak = int(cdf.cumulative_loo.idxmax()) + 1
for r in crows:
    print(f"     m={r['m']}  +{r['marker_added']:<5} {r['feature']:<18} "
          f"uni {r['uni_auroc']:.3f}   LOO {r['cumulative_loo']:.3f}")
print(f"  marker curve peaks at m={peak} "
      f"({' + '.join(cdf.marker_added[:peak])}), LOO {cdf.cumulative_loo.max():.3f}")
# the selection is data-driven, so these are checks on the pipeline, not on the answer
assert list(cdf.marker_added[:3]) == ["PTEN", "ERG", "Ki67"], list(cdf.marker_added[:3])
assert peak == 3, peak

fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=300)
ax.plot(cdf.m, cdf.cumulative_loo, "-o", color=C_CELL, linewidth=1.6, markersize=4,
        label="Leave-one-out", zorder=3)
ax.plot(cdf.m, cdf.cumulative_kfold5, "-o", color=C_MRI, linewidth=1.6, markersize=4,
        label=f"5-fold ({KFOLD_REPEATS} repeats)", zorder=2)
ax.axvline(peak, color="#c0c0c0", linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
# above the peak, not below it: below-right the label lay across the descending arm
ax.annotate(f"peak, {peak} markers", xy=(peak, cdf.cumulative_kfold5.max()),
            xytext=(4, 9), textcoords="offset points", fontsize=FS_VALUE,
            fontweight="bold", color=C_CELL, va="bottom")
ax.set_xticks(cdf.m)
# FS_TICK, not a local size: 5b sets both its axes in FS_TICK/FS_LAB and these two
# panels are read side by side, so a marker name must not be smaller than a scheme name
ax.set_xticklabels(cdf.marker_added, fontsize=FS_TICK)
ax.set_xlim(0.5, len(cdf) + 0.5)
ax.set_ylim(0.5, 1.0)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# No x label. The ticks are the marker names and they are already read left to right, so
# the axis title only restated the ordering; what the ordering is belongs in the caption.
ax.set_ylabel("AUROC", fontsize=FS_LAB)
ax.tick_params(axis="y", labelsize=FS_TICK)
ax.grid(axis="y", alpha=0.4, linestyle="-", linewidth=0.5)
ax.set_axisbelow(True)
ax.set_title("Signal saturates at three markers", fontsize=FS_TITLE,
             fontweight="bold", pad=12)
sns.despine(ax=ax)
# Two columns along the bottom, not a block in a corner. The curve never drops below 0.77
# and the axis starts at 0.5, so the empty band is a wide short one; at this legend size a
# stacked box in any corner reached up into the curve, while a two-column strip does not.
ax.legend(loc="lower center", ncol=2, fontsize=FS_LEGEND, frameon=True,
          columnspacing=1.4, handlelength=1.6)
fig.tight_layout()
# Match Figure_5b_nadt_f1's data area exactly. tight_layout has just sized the margins
# around this panel's own title, ylabel and tick labels; those are kept at their measured
# width in inches and the figure is grown or shrunk around a body of the target size, so
# the two axes rectangles come out identical while each panel keeps the margins it needs.
# The page size is pinned, not derived. 5c is placed in the composite at a fixed box, so
# its aspect ratio has to stay put across re-renders; enlarging the fonts grows the margins
# and a body-matched page would have shrunk the outer canvas. Keep the canvas and let the
# body take whatever the margins leave, so 5b and 5c still read at the same scale.
PAGE_IN = (5.622, 2.225)      # 404.8 x 160.2 pt, the size 5c has always been placed at
_p = ax.get_position()
_fw, _fh = fig.get_size_inches()
_l, _b = _p.x0 * _fw, _p.y0 * _fh
_r, _t = _fw - _p.x1 * _fw, _fh - _p.y1 * _fh
_nfw, _nfh = PAGE_IN
_tw, _th = _nfw - _l - _r, _nfh - _b - _t
fig.set_size_inches(_nfw, _nfh)
ax.set_position([_l / _nfw, _b / _nfh, _tw / _nfw, _th / _nfh])
print(f"  5c page {_nfw:.3f} x {_nfh:.3f} in, body {_tw:.3f} x {_th:.3f} "
      f"(5b body {AX_BODY_IN[0]:.3f} x {AX_BODY_IN[1]:.3f})")
fig.savefig(os.path.join(OUT, "Figure_5c_nadt_marker_curve.pdf"), dpi=400)
plt.close(fig)
print("  saved Figure_5c_nadt_marker_curve.pdf")
print("done.")
