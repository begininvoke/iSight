"""Supplementary — UPenn melanoma, univariate discrimination of iSight cell readouts
against clinical microstaging.

Paper build of the cohort's `marker_review/heatmap_cindex.py` (the internal analysis
script, not shipped), which is the
reference for the layout, the colours and the statistics. This is a port, not a reanalysis:
same cohort, same estimator, same RdBu_r scale, same three-level stars on the raw Cox p, same
proportions, same footnote. Every number here has been checked against
`figures/heatmap_cindex_all.png` cell by cell and matches.

WHAT A CELL IS

Rows are the two endpoints, columns are single variables. Each cell is the **univariate
concordance index** with the direction fixed rather than maximised:

    C = concordance_index(follow-up years, -x, event)

so the variable is always read as a risk score. C > 0.5 means a higher value goes with
shorter survival (harmful, red); C < 0.5 means protective (blue). This is why mirror markers
land at exactly 1 - C of each other. Note this is NOT the max(a, 1-a) convention used in
Figure 5c's NADT marker curve, where the direction of each feature is chosen by the data --
here the direction carries the meaning, so it must not be folded away.

Stars are the univariate Cox **likelihood-ratio** p on the z-scored variable,
* .05 ** .01 *** .001. Upstream stars the Wald p; see `cell()` for why LR is used instead,
and note that both are written to the results csv. FDR-adjusted q for the iSight markers (BH over the 19-marker panel) is in
source_docs/marker_status.csv and is carried into results/upenn_cindex_heatmap.csv next to
the raw p, so the caption can state both. Under q the iSight block loses every star -- the
best four sit at exactly 0.050 -- which is the README's "everything here is exploratory"
stated in a different way; the panel shows p because the reference panel shows p.

THE ONE STATISTICAL FIX: THE BRESLOW SENTINEL

Not a style choice, so it is kept even though everything else follows upstream.

One patient in the 150 carries breslow = 88.88 mm. Across the full 383-patient clinical table
the same field also holds 99.99 -- registry sentinels for "unknown", not thicknesses; the
largest real value in the cohort is 8.86 mm.

Left in, that single point breaks the DSS Cox fit outright: Newton-Raphson fails to converge
and returns HR 0.35, p = 0.28. The upstream figure therefore prints Breslow / DSS as 0.78
with no stars -- the strongest DSS discriminator in the whole panel, shown as
non-significant. With the sentinels set to missing the same fit converges.

    breslow, DSS     as shipped   n=148  C=0.776  HR 0.35  p=0.277   [convergence FAILED]
                     fixed        n=147  C=0.770  HR 1.73  p=3.5e-07 [ok]
    breslow, OS      as shipped   n=148  C=0.720  HR 1.35  p=3.0e-05
                     fixed        n=147  C=0.716  HR 1.48  p=1.0e-06

The C-index barely moves -- it is rank-based, and one point cannot shift it much -- but the
p-value moves by six orders of magnitude, and it moves in the direction that flatters us:
Breslow is the clinical benchmark the iSight markers are measured against. So Breslow / DSS
is the ONE cell that differs from the reference png: 0.78 with *** here, 0.78 bare there.
The script raises on non-convergence rather than swallowing it, which is what let this
through upstream (`except: pv=nan`).

DEPARTURES from the napa script, all of them small

  1. 's_' labelled "Non-tumor", not "Stroma"    s_ is every cell the target head did not call
                                                tumour -- lymphocytes, endothelium,
                                                fibroblasts. "Stroma" names one of the three.
                                                Requested change.
  2. breslow sentinels 88.88 / 99.99 -> missing (see above)
  3. Arial, and PDF not PNG                     paper-wide, applies to every figure here
  4. title uses parentheses, not an em-dash     paper-wide title rule
  5. one panel per PDF                          no compositing

Everything else -- RdBu_r, TwoSlopeNorm(0.30, 0.5, 0.70), 0.82*nc+2 by 3.4 in, font sizes
8 / 8.5 / 10 / 11, the white minor grid, the black block divider, the two block headers, the
footnote sentence -- is upstream's and is deliberately unchanged.

Cohort entry is the README's rule and is asserted, not assumed: label_crosscheck category in
{CONFIRMED, PLGID_ONLY} with survival and resolved, present in panel19 and in
patient_multimodal, fu_years > 0. N=150, OS 76 events, DSS 38.

    python upenn_cindex_heatmap.py

output/FigS_upenn_cindex_all12.pdf, output/FigS_upenn_cindex_filtered.pdf
results/upenn_cindex_heatmap.csv  (C, raw p, FDR q, HR per SD, n per cell)
"""
import os
import sys
import csv
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

HERE = os.path.dirname(os.path.abspath(__file__))
# walk up to the paper root rather than counting directories: this folder has moved once
# already (supple_figure/upenn_melanoma -> figures/Figure5/upenn_melanoma), and a fixed
# dirname() chain silently resolves to the wrong place when the depth changes
ROOTFIG = HERE
while not os.path.exists(os.path.join(ROOTFIG, "figures", "common.py")):
    _up = os.path.dirname(ROOTFIG)
    assert _up != ROOTFIG, "figures/common.py not found above " + HERE
    ROOTFIG = _up
sys.path.insert(0, os.path.join(ROOTFIG, "figures"))
from common import setup_fonts                                       # noqa: E402

setup_fonts()          # Arial; no seaborn theme here, upstream draws on plain matplotlib

BASE = os.path.dirname(HERE)
SRC = os.path.join(BASE, "source_docs")
OUT, RES = os.path.join(BASE, "output"), os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# upstream's colour scale, unchanged
CMAP, VMIN, VMAX = "RdBu_r", 0.30, 0.70

# GEOMETRY AND TYPE, retuned from upstream's 0.82 in/column and 8-11 pt.
#
# Upstream's panel is 15.9 x 3.4 in for 17 columns: each cell is ~65 x ~105 pt carrying 8 pt
# type, so the numbers occupy about a tenth of the box they sit in and the figure has to be
# scaled down hard to fit a page, which shrinks them again. Enlarging the type alone does not
# fix that -- the fix is to take the surplus width out of the cells at the same time, so the
# type grows twice over: once absolutely, once relative to a narrower figure.
#
# Column width is bounded below by the rotated labels, not by the cells. At 40 degrees the
# clear distance between neighbouring baselines is w*sin(40) = 0.64*w, which has to exceed
# roughly 1.2 * the label size; at 11.5 pt that is 0.31 in, so 0.60 in leaves ~2x headroom.
WIDTH_PER_COL, WIDTH_PAD, FIG_H = 0.60, 2.2, 3.6
# FS_HDR carries the two block names. They label the panel's main division, so they sit
# with the row labels and the title rather than with the column labels under them.
# This panel is placed across the full width of the composite at about 2.54x, and the
# composite is set at 20.25% of its width, so a size here prints at roughly half its value.
# The cell values were already the largest data label in the figure at 6.2 pt; 13 is the
# most they can take before passing the 7 pt Nature BME asks figures to stay inside.
FS_CELL, FS_STAR, FS_COL, FS_HDR, FS_ROW, FS_TITLE, FS_CB = 13, 16, 11.5, 15.5, 20, 19, 11

# registry sentinels for "unknown" in the breslow field, not thicknesses
BRESLOW_SENTINELS = (88.88, 99.99)

# upstream's labels, with Stroma -> Non-tumor
OUR = [("t_positive", "Tumor positive"), ("t_neg", "Tumor neg"),
       ("s_positive", "Non-tumor positive"), ("s_neg", "Non-tumor neg"),
       ("g_pos", "Global positive"), ("g_neg", "Global neg"),
       ("t_strong", "Tumor strong"), ("s_strong", "Non-tumor strong"),
       ("g_strong", "Global strong"),
       ("loc_cyto", "Loc: cytoplasmic"), ("loc_mixed", "Loc: nuc+cyto")]
# The whole panel, in ladder order within each group (neg -> weak -> mod -> strong ->
# positive) rather than the selected-marker order used above: with the seven non-selected
# bins back in, the group structure is what makes the block readable, and the ladder shows
# directly that the signal sits at the two ends and not in the middle bins. The seven that
# the review did not select -- t_weak, t_mod, s_weak, s_mod, g_weak, g_mod, loc_nuc -- are
# the only difference from ALL12.
ALL18 = [("t_neg", "Tumor neg"), ("t_weak", "Tumor weak"), ("t_mod", "Tumor mod"),
         ("t_strong", "Tumor strong"), ("t_positive", "Tumor positive"),
         ("s_neg", "Non-tumor neg"), ("s_weak", "Non-tumor weak"),
         ("s_mod", "Non-tumor mod"), ("s_strong", "Non-tumor strong"),
         ("s_positive", "Non-tumor positive"),
         ("g_neg", "Global neg"), ("g_weak", "Global weak"), ("g_mod", "Global mod"),
         ("g_strong", "Global strong"), ("g_pos", "Global positive"),
         ("loc_nuc", "Loc: nuclear"), ("loc_cyto", "Loc: cytoplasmic"),
         ("loc_mixed", "Loc: nuc+cyto")]
FILT = [("t_positive", "Tumor positive"), ("g_pos", "Global positive"),
        ("t_strong", "Tumor strong"), ("loc_cyto", "Loc: cytoplasmic"),
        ("loc_mixed", "Loc: nuc+cyto")]
CLN = [("breslow", "Breslow"), ("ulceration", "Ulceration"), ("mitrate", "Mitotic rate"),
       ("tils", "TILs"), ("lvi", "LVI")]
ENDPOINTS = [("os", "OS"), ("dss", "DSS")]

print("Supplementary — UPenn melanoma, univariate C-index heatmap")

# ------------------------------------------------------------------ inputs and cohort
d = np.load(os.path.join(SRC, "cell_melan_panel19.npz"), allow_pickle=True)
CELL = {str(p): v for p, v in zip(d["plgids"], d["feat"])}
COLS = [str(c) for c in d["cols"]]
cl = np.load(os.path.join(SRC, "clinical_feats.npz"), allow_pickle=True)
CLIN = {str(p): v for p, v in zip(cl["plgids"], cl["feat"])}
CLC = [str(c) for c in cl["cols"]]
mm = {r["plgid"]: r for r in csv.DictReader(open(os.path.join(SRC, "patient_multimodal.csv")))}
cc = [r for r in csv.DictReader(open(os.path.join(SRC, "label_crosscheck.csv")))
      if r["category"] in ("CONFIRMED", "PLGID_ONLY") and r["survival"] and r["resolved"]]
STATUS = pd.read_csv(os.path.join(SRC, "marker_status.csv")).set_index("marker")


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


P = []
for p in sorted({r["resolved"] for r in cc}):
    if p not in CELL or p not in mm:
        continue
    fu = fl(mm[p]["fu_years"])
    if not fu > 0:
        continue
    P.append(dict(p=p, t=fu, os=int(mm[p]["os_event"] or 0),
                  dss=int(mm[p]["dss_event"] or 0)))
T = np.array([q["t"] for q in P])
EV = {k: np.array([q[k] for q in P]) for k, _l in ENDPOINTS}
N_OS, N_DSS = int(EV["os"].sum()), int(EV["dss"].sum())
print(f"  N={len(P)}  OS events={N_OS}  DSS events={N_DSS}")
assert len(P) == 150, f"expected the 150-patient cohort, got {len(P)}"
assert (N_OS, N_DSS) == (76, 38), "event counts differ from the README"


# LOCATION IS RENORMALISED OVER {nuclear, cytoplasmic, cyto+nuc}, AND loc_none IS DROPPED.
#
# The four loc_* features share one denominator: cells the intensity head called POSITIVE
# (build_cell_melan.py: `stl = loc[inten > 0]`). Conditioned on that, "none" is definitionally
# empty -- a stained cell has a staining shape. It is non-zero anyway for all 194 patients
# (median 0.19% of positive cells, max 2.96%) because the intensity and location heads are
# independent softmaxes with nothing tying them together, so one can say positive while the
# other says none. `loc_none` is not counted but taken as the residual
# `max(0, 1 - nuc - cyto - mixed)`, so it silently absorbs exactly that disagreement.
#
# What it therefore measures is head disagreement, not biology, and it tracks negativity:
# corr(loc_none, g_neg) = +0.755, R^2 = 0.571 against g_neg alone. marker_status.csv already
# flags it REDUNDANT(=t_neg) with rho_glob = -0.82. Reporting it as a marker would be
# reporting the model arguing with itself.
#
# So those cells leave the denominator instead of forming a class. The three real classes are
# divided by (1 - loc_none), which is exact because all four came from the same denominator.
# The markers barely move -- corr(new, old) >= 0.9996, max change 2.3 pts -- which is the
# point: the fix removes a spurious column without disturbing the real ones.
_LOC3 = ("loc_nuc", "loc_cyto", "loc_mixed")


def getval(pid, nm):
    if nm in _LOC3 and pid in CELL:
        v = CELL[pid][COLS.index(nm)]
        none = CELL[pid][COLS.index("loc_none")]
        return v / (1.0 - none) if none < 1.0 else np.nan
    if nm in COLS and pid in CELL:
        return CELL[pid][COLS.index(nm)]
    if nm in CLC and pid in CLIN:
        v = CLIN[pid][CLC.index(nm)]
        if nm == "breslow" and any(np.isclose(v, s, atol=0.02) for s in BRESLOW_SENTINELS):
            return np.nan          # sentinel, not a thickness -- see the docstring
        return v
    return np.nan


_nsent = sum(1 for q in P if q["p"] in CLIN
             and any(np.isclose(CLIN[q["p"]][CLC.index("breslow")], s, atol=0.02)
                     for s in BRESLOW_SENTINELS))
print(f"  breslow sentinels dropped: {_nsent} patient(s)")
assert _nsent == 1, f"expected the one 88.88 mm record, found {_nsent}"


def cell(nm, ev):
    """C-index, Cox LR p, Cox Wald p, HR per 1 SD, n.

    THE STARS COME FROM THE LIKELIHOOD-RATIO TEST, not the Wald test upstream used.

    Both test the same single-covariate Cox model, and they are asymptotically equivalent,
    but they part company at this sample size (N=150, and only 38 DSS events). Wald divides
    the coefficient by its standard error, which is a local approximation taken at the
    maximum-likelihood point; LR compares the log-likelihood of the fitted model against the
    null across the whole curve. LR is the one to prefer for small samples, and the standard
    caution against Wald -- the Hauck-Donner effect, where a large effect inflates the
    standard error and *loses* power -- is visible in this panel rather than theoretical:

        median Wald/LR p ratio, weak effects (HR within 1.35)   0.92   Wald anti-conservative
        median Wald/LR p ratio, strong effects                  1.65   Wald conservative

    Five of 34 cells change star level, in both directions -- t_positive/DSS and t_neg/DSS
    gain a star, loc_none/OS gains one, loc_mixed/OS drops from ** to *, lvi/DSS drops from
    *** to **. The choice is not made for the direction it moves our markers; it is made
    because 38 events is squarely where Wald is known to be unreliable.

    Non-convergence is raised, not swallowed: the upstream `except: pv=nan` is what let the
    Breslow failure through as a blank cell."""
    x = np.array([getval(q["p"], nm) for q in P], float)
    m = np.isfinite(x)
    xx, tt, ee = x[m], T[m], EV[ev][m]
    if len(np.unique(xx)) < 2 or ee.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan, int(m.sum())
    c = concordance_index(tt, -xx, ee)
    z = (xx - xx.mean()) / (xx.std() + 1e-9)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f = CoxPHFitter().fit(pd.DataFrame({"x": z, "t": tt, "e": ee}), "t", "e")
        if any("converge" in str(q.message) for q in w):
            raise RuntimeError(f"Cox did not converge for {nm}/{ev} -- check for sentinels")
    return (c, float(f.log_likelihood_ratio_test().p_value),
            float(f.summary.loc["x", "p"]),
            float(f.summary.loc["x", "exp(coef)"]), int(m.sum()))


def star(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def bh(p):
    """Benjamini-Hochberg, NaN-safe, input order. Used only for the results csv."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    if not ok.any():
        return q
    v = p[ok]
    o = np.argsort(v)
    n = len(v)
    adj = np.minimum.accumulate((v[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(adj, 1.0)
    q[ok] = out
    return q


# ------------------------------------------------------------------ compute every cell once
stats = {(nm, ev): cell(nm, ev) for nm, _l in ALL18 + CLN for ev, _e in ENDPOINTS}

# q is not drawn, only recorded. iSight q comes from the panel review (BH over all 19
# markers) rather than being recomputed over the 12 shown, which were selected because they
# were significant; the clinical five were not screened, so they get their own BH family.
CLINQ = {}
for ev, _e in ENDPOINTS:
    for (nm, _l), q in zip(CLN, bh([stats[(nm, ev)][1] for nm, _l in CLN])):
        CLINQ[(nm, ev)] = float(q)

rows = []
for nm, lab in ALL18 + CLN:
    for ev, elab in ENDPOINTS:
        c, p_lr, p_wald, hr, n = stats[(nm, ev)]
        isc = nm in STATUS.index
        rows.append(dict(variable=nm, label=lab,
                         block="iSight-cell" if isc else "clinical", endpoint=elab, n=n,
                         c_index=c, hr_per_sd=hr, cox_p_lr=p_lr, cox_p_wald=p_wald,
                         star_shown=star(p_lr),
                         fdr_q=(float(STATUS.loc[nm, f"{elab}_qcox"]) if isc
                                else CLINQ[(nm, ev)]),
                         q_source=("marker_status.csv, BH over 19 markers" if isc
                                   else "BH over 5 clinical variables")))
pd.DataFrame(rows).to_csv(os.path.join(RES, "upenn_cindex_heatmap.csv"), index=False)

_clip = [(r["label"], r["endpoint"], round(r["c_index"], 3)) for r in rows
         if np.isfinite(r["c_index"]) and not VMIN <= r["c_index"] <= VMAX]
print(f"  {len(_clip)} cell(s) clipped by the {VMIN}-{VMAX} colour range: {_clip}")


# ------------------------------------------------------------------ draw, upstream's layout
GAP = "__gap__"


def draw(ourset, fname, subtitle, with_clinical=True):
    # An EMPTY COLUMN between the blocks, not a rule laid over them. A thick white line
    # drawn at the boundary is centred on it, so half its width is taken out of the last
    # iSight cell and half out of Breslow -- the two cells the separation is supposed to
    # keep whole end up clipped. Giving the gap a column of its own leaves every cell at
    # full width and puts the rule in space that belongs to no cell.
    # with_clinical=False drops the right-hand block entirely -- and with it the gap column,
    # the divider and the second header, which would otherwise be a rule with nothing on the
    # far side of it. The title also loses "and clinical microstaging", because the panel no
    # longer makes that comparison.
    allv = ourset + ([(GAP, "")] + CLN if with_clinical else [])
    nc, split = len(allv), len(ourset)          # split is the index OF the gap column
    C = np.array([[np.nan if nm == GAP else stats[(nm, ev)][0] for nm, _l in allv]
                  for ev, _e in ENDPOINTS])
    Pv = np.array([[np.nan if nm == GAP else stats[(nm, ev)][1] for nm, _l in allv]
                   for ev, _e in ENDPOINTS])

    fig, ax = plt.subplots(figsize=(WIDTH_PER_COL * nc + WIDTH_PAD, FIG_H), dpi=300)
    # aspect="auto", as upstream. Square cells (aspect="equal") were tried and reverted:
    # they pull the band down to two 45 pt rows on an 800 pt width, which reads as a thin
    # strip rather than a panel. The cells are meant to be landscape here -- the numbers
    # and their stars stack two lines deep inside them.
    im = ax.imshow(C, cmap=CMAP, norm=TwoSlopeNorm(vmin=VMIN, vcenter=0.5, vmax=VMAX),
                   aspect="auto")
    for i in range(len(ENDPOINTS)):
        for j in range(nc):
            if not np.isfinite(C[i, j]):
                continue
            s = star(Pv[i, j])
            col = "white" if abs(C[i, j] - 0.5) > 0.13 else "#222"
            # number and stars are drawn separately so the stars can be larger. As one
            # "0.61\n*" string they share a size, and an asterisk set at the number's size
            # is a fraction of its ink -- at 12 pt it all but disappears against a
            # saturated cell, which matters more now that the footnote key is gone.
            ax.text(j, i - (0.16 if s else 0.0), f"{C[i, j]:.2f}", ha="center",
                    va="center", fontsize=FS_CELL, color=col)
            if s:
                ax.text(j, i + 0.20, s, ha="center", va="center", fontsize=FS_STAR,
                        color=col, fontweight="bold")

    ax.set_xticks(range(nc))
    ax.set_xticklabels([lab for _n, lab in allv], rotation=40, ha="right", fontsize=FS_COL)
    ax.set_yticks(range(len(ENDPOINTS)))
    ax.set_yticklabels([e for _k, e in ENDPOINTS], fontsize=FS_ROW)
    # Block separation, three cues rather than one. A 2.5 pt rule between two saturated
    # cells is the weakest possible boundary -- it reads as a heavy gridline, and the eye
    # runs straight across it from "Loc: none" to "Breslow", which is exactly the comparison
    # the panel is making and exactly the one that must not look like a continuation. So:
    # a white gutter cut through the row band, the rule inside it, and the rule carried up
    # past the top of the heatmap to meet the two block headers.
    # the rule sits at the centre of the empty column and runs up to the block headers.
    # ax.plot updates the data limits even with clip_on=False, so drawing past the top of
    # the heatmap pulls the y-limit -- and the top spine with it -- up over those headers.
    # Snapshot the imshow limits and put them back.
    _yl = ax.get_ylim()
    if not with_clinical:
        # single block: centre one header over the whole grid, no rule
        ax.text((nc - 1) / 2, -0.66, "iSight-cell markers", ha="center", va="bottom",
                fontsize=FS_HDR, fontweight="bold", color="#4a4a4a")
        ax.set_title("Univariate discrimination (C-index): iSight-cell markers",
                     fontsize=FS_TITLE, pad=44, fontweight="bold")
    # equal overhang top and bottom: the heatmap spans -0.5 to nrows-0.5, so the rule runs
    # OVER beyond each edge rather than stopping flush at the bottom
    OVER = 0.30
    if with_clinical:
        ax.plot([split, split], [-0.5 - OVER, len(ENDPOINTS) - 0.5 + OVER], color="k", lw=2.2,
                zorder=4, clip_on=False, solid_capstyle="butt")
    ax.set_ylim(_yl)
    # "iSight-cell", not upstream's "iSight_cell": the model is hyphenated everywhere else
    # in this paper, and an underscore in a figure reads as a variable name
    if with_clinical:
        ax.text((split - 1) / 2, -0.66, "iSight-cell markers", ha="center", va="bottom",
                fontsize=FS_HDR, fontweight="bold", color="#4a4a4a")
        ax.text(split + 1 + (nc - split - 2) / 2, -0.66, "Clinical microstaging",
                ha="center", va="bottom", fontsize=FS_HDR, fontweight="bold", color="#4a4a4a")
    ax.set_xticks(np.arange(-.5, nc, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ENDPOINTS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=[0.3, 0.4, 0.5, 0.6, 0.7])
    cb.set_label("C-index", fontsize=FS_CB)
    cb.ax.tick_params(labelsize=FS_CB - 1)

    # The title names the comparison, not the marker count. Upstream's "... : all selected
    # iSight markers (12)" says which columns are present but not what the panel is for --
    # and what it is for is putting the two blocks against each other on one scale. Which
    # subset is drawn moves to the second line, next to the cohort, where it belongs as a
    # qualifier. Colon rather than the upstream em-dash, per the paper title rule.
    # One line. The second line carried the marker subset and the cohort (N=150, 76 OS /
    # 38 DSS events); both are removed from the panel and BOTH NOW HAVE TO BE IN THE
    # CAPTION -- the cohort in particular, since nothing on the panel states how many
    # patients or events any of these C-indices rest on. `subtitle` survives only to
    # distinguish the three variants in the build log.
    # "and", not "vs". The two blocks are not competing measurements of the same thing --
    # the clinical variables are the established staging axis and the iSight readouts are a
    # different kind of quantity put on the same scale beside them. "vs" frames the panel as
    # a contest one side has to win, which is not the claim.
    # pad 26 -> 44: with the second title line gone there is nothing between the 22 pt title
    # and the 15.5 pt block headers, and two bold greys that close together read as one
    # three-line heading
    if with_clinical:
        ax.set_title("Univariate discrimination (C-index): "
                     "iSight-cell markers and clinical microstaging",
                     fontsize=FS_TITLE, pad=44, fontweight="bold")
    # No footnote. Upstream carried the colour and star key under the panel; in this paper
    # that belongs in the caption, like every other figure here. The caption must therefore
    # state both: cell = univariate C-index, direction fixed so >0.5 is harmful (red) and
    # <0.5 protective (blue); stars = univariate Cox likelihood-ratio p, unadjusted,
    # * p<.05 ** p<.01 *** p<.001.
    plt.tight_layout()
    # aspect="equal" shrinks the drawn heatmap inside the axes slot but leaves the slot --
    # and therefore the colourbar, which was sized from it -- at full height, so the bar
    # ran well past the band at both ends. get_position() still reports the slot; the drawn
    # box has to come from get_window_extent, after a draw.
    fig.canvas.draw()
    bb = ax.get_window_extent().transformed(fig.transFigure.inverted())
    pc = cb.ax.get_position()
    cb.ax.set_position([pc.x0, bb.y0, pc.width, bb.height])
    fig.savefig(os.path.join(OUT, fname), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved output/{fname}   ({subtitle})")


draw(ALL18, "FigS_upenn_cindex_all18.pdf", "full 18-marker iSight panel")
draw(OUR, "FigS_upenn_cindex_all11.pdf", "all 11 selected iSight markers")
draw(FILT, "FigS_upenn_cindex_filtered.pdf", "representative iSight markers, 1 per axis")
draw(OUR, "FigS_upenn_cindex_all11_only.pdf", "all 11 selected iSight markers, no clinical",
     with_clinical=False)
draw(ALL18, "FigS_upenn_cindex_all18_only.pdf", "full 18-marker iSight panel, no clinical",
     with_clinical=False)
print("done.")
