"""Supplementary — UPenn melanoma, survival by iSight cell-level S100 readouts.

150 patients from the Penn melanoma cohort with both survival and iSight cell features,
median follow-up 17.5 years (0.6-33.5). S100 + D2-40 double-brown Aperio slides; iSight
segments every nucleus, calls staining intensity and subcellular location, and the
per-patient features are fractions over those calls. Tumour vs non-tumour comes from
iSight's own target-cell head.

WHICH MARKERS ARE DRAWN, AND WHY ONLY THESE

The 19-marker panel collapses to far fewer independent signals, and plotting all of them
would be plotting the same thing eight times. From `source_docs/marker_status.csv`:

  positivity / tumour-burden axis   t_neg, t_positive, s_neg, s_positive, g_neg, g_pos,
                                    loc_none (|rho| 0.9-1.0 with each other) -- eight
                                    markers, one signal: how widely S100 stains at all.
  localisation axis                 loc_cyto <-> loc_mixed (rho = -0.93, mirrors), and
                                    orthogonal to the positivity axis (rho_glob = -0.12).
  strong-staining group             t_strong, s_strong, g_strong.

So: one panel per axis, plus the mirror of the main marker as corroboration.

  a  loc_cyto, OS      the only marker that is selected, not a staining proxy, and not
                       redundant -- the headline
  b  loc_mixed, OS     its mirror; should reverse direction, and does
  c  g_pos, OS         representative of the burden axis
  d  g_pos, DSS        the same marker on the clean endpoint

WHY g_pos AND NOT t_positive REPRESENTS THE BURDEN AXIS

t_positive -- the fraction of *tumour* cells staining for S100 -- is the more appealing
quantity, because it sounds like a tumour-intrinsic phenotype. On this cohort it is not one,
for three reasons that were checked rather than assumed:

1. It is saturated. Median 0.951, IQR 0.084 (against 0.212 for g_pos). S100 marks 95-100% of
   melanoma cells, so a median of 95.1% is the model working correctly -- but it leaves
   almost no dynamic range, and most of what range there is comes from measurement rather
   than biology: nucleus-centred crops under-sample a largely cytoplasmic stain, and any
   lymphocyte or fibroblast inside a tumour nest that the target head calls tumour is
   counted as an S100-negative tumour cell.

2. It does not survive adjustment. Per 1 SD on OS, t_positive alone gives HR 1.40 (p=0.033);
   put s_positive in the same model and t_positive collapses to HR 1.05 (p=0.85) while
   s_positive holds at 1.38. Same on DSS (1.05, p=0.90).

3. The non-tumour compartment predicts better than the tumour one. Univariately s_positive
   beats t_positive on both endpoints (OS p=0.009 vs 0.033; DSS p=0.020 vs 0.066). If the
   signal were tumour-intrinsic S100 that ordering should not happen.

g_pos -- every cell on the section -- is the honest version of the same axis: one simple
definition, no dependence on the target-cell head being right, and rho=0.985 with
s_positive, so nothing is lost. What it must NOT be called is a tumour phenotype; it is how
widely S100 stains, which tracks tumour burden and Breslow thickness. The caption has to say
so. Note also that these are S100 + D2-40 double-brown slides -- both chromogens are brown
and cannot be separated by colour -- so lymphatic endothelium contributes to positivity too.

loc_cyto sidesteps the whole problem: it is computed among positive cells only, so it is a
shape-of-staining measure rather than an amount-of-staining one, which is why it comes out
orthogonal to this axis (rho_glob = -0.12).

TWO ENDPOINTS, NEITHER OF THEM FREE

OS has 76 events but includes 38 deaths that are not melanoma (survstatus: NED 74 / DOD 38 /
DOC 20 / DUC 13 / DWD 5). DSS has only 38 events and is underpowered. Every marker that
clears DSS sits on the burden axis and correlates with Breslow thickness -- adjust for
Breslow and it collapses -- so the honest reading is that DSS-significant here means tumour
burden restated. loc_cyto is clean and orthogonal but clears OS only.

EVERYTHING HERE IS EXPLORATORY

q-values cluster at 0.05-0.09 after BH-FDR over 19 markers. Nothing reaches q<0.05 on the
clean endpoint. loc_cyto is the most defensible because it passes both tests on OS, but it
depends on iSight's `pred_location` head and has not yet been verified against images. The
panels are annotated with the FDR q from marker_status.csv rather than a raw p, and the
titles say exploratory.

All numbers -> results/upenn_km_stats.csv
"""
import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
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

import seaborn as sns                                                # noqa: E402
sns.set_theme(style="white")
setup_fonts()          # after seaborn: set_theme() resets the font rcParams

BASE = os.path.dirname(HERE)
SRC = os.path.join(BASE, "source_docs")
OUT, RES = os.path.join(BASE, "output"), os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)

# Structure follows zhi_analysis/univariate_survival.py -- confidence bands, censor ticks,
# the at-risk table, the median-split-plus-continuous-Cox stat block, the per-endpoint y
# floor. Typography and palette are this paper's: Arial, the same tick and label sizes as
# every other figure here, and the deep blue / brick red pair in place of zhi's
# #2a78d6 / #eb6834.
C_LOW, C_HIGH = "#2F5597", "#c0504d"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
# Raised together from upstream's 13/12/11.5/10. With the stat line and the at-risk table
# gone the panel is all curve, so the type that is left has to hold the panel on its own,
# and the landscape box below gives it the room.
# These panels go into the composite at 1:1 and the composite is set at 20.25% of its
# width, so a size here reaches the reader at almost exactly a fifth of its value. At the
# old 22/20/19/15 the legend printed at 3.0 pt and the axis labels at 4.1, well under the
# 5-7 pt Nature BME asks for. 28 is the ceiling for the title: the longest of the three
# measures 561 pt there, and the plot box is 580 pt wide.
FS_TITLE, FS_LAB, FS_TICK, FS_LEGEND = 25, 25, 25, 22

# Titles name the readout, not the antibody -- every panel here is S100, so repeating it
# four times says nothing while the marker is what distinguishes them. The endpoint is not
# in the title either: the y-axis label already carries it. Machine-readable short names
# stay in the filenames and the results csv.
# No "Survival (...)" wrapper any more. The y axis already reads "Overall survival
# probability" / "Melanoma-specific survival probability", so the word was said twice, and
# at 22 pt the longest of the three ran wider than the plot box and pushed the page out to
# 650 pt to fit a centred title. What distinguishes the panels is the marker; that is what
# the title now carries, and it fits.
# Worded as the caption words them, and spelled as the rest of the paper spells them.
# The size each panel is placed at in the composite, in inches (607.5 x 360.9 / 366.2 pt).
PAGE_IN = {"os": (8.438, 5.013), "dss": (8.438, 5.086)}

NAMES = {"loc_cyto": "Cytoplasmic localization ratio",
         "loc_mixed": "Nuclear and cytoplasmic localization ratio",
         "g_pos": "Global positivity ratio"}

# (marker, endpoint, panel letter, what it is)
PANELS = [("loc_cyto", "os", "a", "cytoplasmic S100, among positive cells"),
          ("loc_mixed", "os", "b", "nuclear + cytoplasmic S100"),
          ("g_pos", "os", "c", "S100-positive cells, whole section"),
          ("g_pos", "dss", "d", "S100-positive cells, whole section"),
          # the localisation axis on the clean endpoint too, so every marker drawn here is
          # shown on both. NEITHER CLEARS FDR ON DSS -- loc_cyto q=0.172, loc_mixed q=0.161,
          # against 0.050 for both on OS -- which is the README's point that the only
          # orthogonal, non-staining-proxy signal we have is OS-only and underpowered on the
          # 38-event endpoint. The panels are worth showing precisely because they are the
          # negative half of that claim.
          ("loc_cyto", "dss", "e", "cytoplasmic S100, among positive cells"),
          ("loc_mixed", "dss", "f", "nuclear + cytoplasmic S100")]
ENDPOINT = {"os": ("Overall survival", "os_event"),
            "dss": ("Melanoma-specific survival", "dss_event")}

# The y label is set over two lines and broken at the same place on both endpoints, so the
# second line reads "survival probability" either way. One line does not fit: at this size
# the melanoma-specific string is longer than the page is tall.
YLAB = {"os": "Overall\nsurvival probability",
        "dss": "Melanoma-specific\nsurvival probability"}

print("Supplementary — UPenn melanoma survival")

# ------------------------------------------------- cohort, per the README's entry rule
d = np.load(os.path.join(SRC, "cell_melan_panel19.npz"), allow_pickle=True)
CELL = {str(p): v for p, v in zip(d["plgids"], d["feat"])}
COLS = [str(c) for c in d["cols"]]
mm = {r["plgid"]: r for r in csv.DictReader(open(os.path.join(SRC, "patient_multimodal.csv")))}
cc = [r for r in csv.DictReader(open(os.path.join(SRC, "label_crosscheck.csv")))
      if r["category"] in ("CONFIRMED", "PLGID_ONLY") and r["survival"] and r["resolved"]]


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


pids, feats, T, E = [], [], {"os": [], "dss": []}, {"os": [], "dss": []}
years = []
for p in sorted({r["resolved"] for r in cc}):
    if p not in CELL or p not in mm:
        continue
    r = mm[p]
    if not r["fu_years"] or fl(r["fu_years"]) <= 0:
        continue
    pids.append(p)
    feats.append(CELL[p])
    years.append(fl(r["fu_years"]))
    for k, (_lab, col) in ENDPOINT.items():
        E[k].append(int(r[col] or 0))
F = np.array(feats)
years = np.array(years)
E = {k: np.array(v) for k, v in E.items()}
print(f"  N={len(pids)}  OS events={E['os'].sum()}  DSS events={E['dss'].sum()}  "
      f"median follow-up {np.median(years):.1f} y")
assert len(pids) == 150, f"expected the 150-patient cohort, got {len(pids)}"
assert E["os"].sum() == 76 and E["dss"].sum() == 38, "event counts differ from the README"

# FDR q-values are not recomputed here -- they are BH over all 19 markers and belong to the
# panel review, so they are read from its output rather than re-derived from 4 of them.
#
# TWO DIFFERENT HAZARD RATIOS, DO NOT MIX THEM. marker_status.csv reports the CONTINUOUS
# Cox HR per 1 SD (usable_markers.py fits on zs(x)); loc_cyto is 0.77 there. The curves
# below are a median split, whose High-vs-Low HR is 0.55. Both are right and they answer
# different questions, so the panel annotates the median-split HR -- the one the two drawn
# curves actually compare -- next to q_log-rank, which is the test computed on that same
# split. q_Cox belongs to the continuous model and is labelled as such.
STATUS = pd.read_csv(os.path.join(SRC, "marker_status.csv")).set_index("marker")


# ------------------------------------------------- statistics, then draw
# Panel contents follow zhi_analysis/univariate_survival.py: confidence bands, censor
# ticks, an at-risk table, and the full stat block -- median cut, high-vs-low HR with its
# interval and log-rank p, then the continuous per-SD Cox with interval, q and Harrell C.
# Only the typography and palette are this paper's.
def stats_for(marker, ep):
    # LOCATION IS RENORMALISED OVER {nuclear, cytoplasmic, cyto+nuc}. The four loc_* features
    # share one denominator -- cells the intensity head called POSITIVE -- so conditioned on
    # that, "none" is definitionally empty: a stained cell has a staining shape. It is
    # non-zero anyway (median 0.19% of positive cells, max 2.96%) because the intensity and
    # location heads are independent softmaxes that can disagree, and build_cell_melan.py
    # takes loc_none as the RESIDUAL `max(0, 1 - nuc - cyto - mixed)`, so it silently absorbs
    # that disagreement. Those cells leave the denominator instead of forming a class.
    #
    # Dividing by (1 - loc_none) is exact -- same denominator for all four. It matters here
    # in a way it did not for the C-index: the median split moves 4 of 150 patients on
    # loc_cyto, and OS goes HR 0.553 -> 0.510, log-rank p 0.011 -> 0.0038. THAT IS IN OUR
    # FAVOUR, so it is recorded rather than quietly banked. loc_mixed moves nobody and is
    # unchanged to four decimals.
    x = F[:, COLS.index(marker)]
    if marker in ("loc_nuc", "loc_cyto", "loc_mixed"):
        _none = F[:, COLS.index("loc_none")]
        x = np.where(_none < 1.0, x / (1.0 - _none), np.nan)
    med = float(np.median(x))
    hi = x > med
    if hi.sum() in (0, len(x)):                 # ties at the median would empty a group
        hi = x >= med
    ev = E[ep]
    cph = CoxPHFitter().fit(pd.DataFrame({"t": years, "e": ev, "g": hi.astype(int)}),
                            "t", "e")
    hr = float(np.exp(cph.params_["g"]))
    lo, up = np.exp(cph.confidence_intervals_.loc["g"].values)
    # continuous, per 1 SD -- the model marker_status.csv reports
    z = (x - x.mean()) / (x.std() + 1e-9)
    cz = CoxPHFitter().fit(pd.DataFrame({"t": years, "e": ev, "x": z}), "t", "e")
    hr_sd = float(np.exp(cz.params_["x"]))
    lo_sd, up_sd = np.exp(cz.confidence_intervals_.loc["x"].values)
    return dict(x=x, med=med, hi=hi, ev=ev, hr=hr, lo=float(lo), up=float(up),
                p_lr=float(logrank_test(years[~hi], years[hi], ev[~hi], ev[hi]).p_value),
                hr_sd=hr_sd, lo_sd=float(lo_sd), up_sd=float(up_sd),
                p_cox=float(cz.summary.loc["x", "p"]),
                c_index=float(concordance_index(years, -z, ev)))


S = {(m, e): stats_for(m, e) for m, e, _l, _w in PANELS}

# One y-floor per endpoint, as in zhi's y_floor(): the next 0.1 step below the lowest point
# any drawn curve reaches. Without it the DSS panel sits squashed in the top third while the
# OS panel uses the full axis, and the two stop being comparable by eye.
FLOOR = {}
for _m, ep, _l, _w in PANELS:
    lo = FLOOR.get(ep, 1.0)
    st = S[(_m, ep)]
    for mask in (~st["hi"], st["hi"]):
        lo = min(lo, float(KaplanMeierFitter().fit(years[mask], st["ev"][mask])
                           .survival_function_.min().iloc[0]))
    FLOOR[ep] = lo
FLOOR = {k: float(np.clip(np.floor(v * 10) / 10, 0.0, 0.8)) for k, v in FLOOR.items()}
print("  y-axis floor per endpoint:", FLOOR)


def fmt_p(p):
    return "p<0.001" if p < 1e-3 else f"p={p:.3f}"


rows = []
for marker, ep, letter, what in PANELS:
    st = S[(marker, ep)]
    hi, ev = st["hi"], st["ev"]
    eplabel, _col = ENDPOINT[ep]
    q_cox = float(STATUS.loc[marker, f"{ep.upper()}_qcox"])
    q_lr = float(STATUS.loc[marker, f"{ep.upper()}_qlr"])

    rows.append(dict(panel=letter, marker=marker, endpoint=ep.upper(), n=len(pids),
                     n_events=int(ev.sum()), median_cut=st["med"],
                     n_low=int((~hi).sum()), n_high=int(hi.sum()),
                     events_low=int(ev[~hi].sum()), events_high=int(ev[hi].sum()),
                     hr_median_split=st["hr"], hr_ci_lower=st["lo"], hr_ci_upper=st["up"],
                     p_logrank=st["p_lr"],
                     hr_continuous_per_sd=st["hr_sd"], hr_sd_ci_lower=st["lo_sd"],
                     hr_sd_ci_upper=st["up_sd"], p_cox_continuous=st["p_cox"],
                     c_index=st["c_index"], q_cox_bh19=q_cox, q_logrank_bh19=q_lr,
                     hr_continuous_in_marker_status=float(STATUS.loc[marker,
                                                                     f"{ep.upper()}_HR"]),
                     verdict=STATUS.loc[marker, "verdict"],
                     rho_with_global_positivity=STATUS.loc[marker, "rho_glob"]))
    print(f"  {letter}  {marker:<11} {ep.upper():<4} cut={st['med']:.3f}  "
          f"HR={st['hr']:.2f}[{st['lo']:.2f}-{st['up']:.2f}] {fmt_p(st['p_lr'])}  "
          f"perSD={st['hr_sd']:.2f} q={q_cox:.3f} C={st['c_index']:.3f}")

    # Raising the point size on a large canvas does not make the type look bigger: the
    # panel is scaled to a column width when it is placed and everything shrinks with it.
    # What matters is the ratio of type to plot box, so the canvas comes down as the sizes
    # go up. 6.0 x 4.0 with 27/25/23/21 was tried and was too far -- the y label ran into
    # the tick labels, the legend sat on the curves and the title was wider than the whole
    # figure. This lands the tick digits at about 1/21 of the box width, against 1/28
    # before and 1/13 at the overshoot.
    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=300)
    # Landscape, and the ratio is set on the AXES not the canvas: the title sits above the
    # axes and the x label below it while only the y label eats width, so an 8.4 x 5.6
    # canvas alone would not give the intended plot box. set_box_aspect pins the box and
    # lets the canvas be whatever the labels need. 0.50 (was 1.0, then 0.62) also suits the
    # data: the curves run to 33 years across a 0-1 probability range, so width is where
    # the two arms separate visibly and height buys little.
    ax.set_box_aspect(0.50)
    fits = []
    for lab, mask, colour in (("Low", ~hi, C_LOW), ("High", hi, C_HIGH)):
        k = KaplanMeierFitter(label=f"{lab} (n={int(mask.sum())}, "
                                    f"{int(ev[mask].sum())} events)")
        k.fit(years[mask], ev[mask])
        k.plot_survival_function(ax=ax, color=colour, lw=2, ci_show=True,
                                 ci_alpha=0.12, show_censors=True,
                                 censor_styles={"marker": "|", "ms": 5, "alpha": .8})
        fits.append(k)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("black")
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors="black", labelsize=FS_TICK, length=3)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(FLOOR[ep], 1.02)
    # five ticks spanning whatever floor this endpoint got, so OS reads 0/.25/.5/.75/1 and
    # DSS reads .6/.7/.8/.9/1 rather than matplotlib's three
    ax.set_yticks(np.linspace(FLOOR[ep], 1.0, 5))
    # the axis runs a little past the last tick so the spine does not stop dead on "35";
    # the data end at 33.5 y, so nothing is hidden by the extra room
    ax.set_xlim(0, 36.5)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
    # Both axes start at the same corner, so "0" under the origin and the y floor label
    # beside it are two numbers a few points apart naming the same point. The y ladder is
    # the one that has to stay whole -- dropping its bottom rung breaks the 0/.25/.5/.75/1
    # (or .6/.7/.8/.9/1) reading -- so the x zero is the one that goes. The tick mark stays;
    # only its label is blanked, and the axis still obviously starts at zero.
    ax.set_xticklabels([""] + [str(t) for t in [5, 10, 15, 20, 25, 30, 35]])
    ax.set_xlabel("Years from definitive therapy", fontsize=FS_LAB, color=INK2)
    ax.set_ylabel(YLAB[ep], fontsize=FS_LAB, color=INK2)
    # Which corner is free depends on the endpoint. Overall survival is drawn on the full
    # 0-1 axis and the curves stay in its upper half, leaving the bottom left empty;
    # melanoma-specific survival is drawn on 0.6-1.0, where the curves fall into the bottom
    # left and the top right is the empty one. The box is also set a little smaller than
    # the rest of the panel type so that it clears the curves either way.
    ax.legend(fontsize=FS_LEGEND, frameon=False, labelcolor=INK2,
              loc="lower left" if ep == "os" else "upper right",
              borderaxespad=0.15 if ep == "os" else 0.5, borderpad=0.2)

    # Title only. The statistics line under it and the at-risk / censored / events table
    # under the axes were both removed: every number they carried -- median split point,
    # median-split HR and CI, log-rank p, Cox HR per SD and CI, Cox p, FDR q, C-index, and
    # the per-interval at-risk counts -- is in results/upenn_km_stats.csv, and the two that
    # a reader needs while looking at the curves (group n and event count) are already in
    # the legend. On a 4 x 4 in panel the table was taking a third of the height and the
    # stat line was setting the width, both in type smaller than anything else in the
    # figure. THE CAPTION NOW HAS TO CARRY THE STATISTICS -- they are no longer on the panel.
    ax.set_title(NAMES[marker], fontsize=FS_TITLE, fontweight="bold", color=INK,
                 pad=20)
    fig.tight_layout()
    # The title is centred over the plot box, and the plot box sits right of centre because
    # the y label is on the left, so the longest of the three names runs off the right edge
    # of a page this width. Slide it left until it is inside, and only shrink it if sliding
    # is not enough, so all six panels keep one title size.
    _ttl = ax.title
    _rd = fig.canvas.get_renderer()
    _pw = fig.get_figwidth() * fig.dpi
    _axw = ax.get_window_extent(renderer=_rd).width
    _marg = 12 / 72 * fig.dpi     # Agg metrics run a little short of the PDF backend's
    _over = _ttl.get_window_extent(renderer=_rd).x1 - (_pw - _marg)
    if _over > 0:
        _ttl.set_x(_ttl.get_position()[0] - _over / _axw)
        print(f"     {letter}: title slid {_over / fig.dpi * 72:.0f} pt left to fit the page")
    for _fs in [FS_TITLE - 0.5 * k for k in range(0, 13)]:
        _ttl.set_fontsize(_fs)
        _e = _ttl.get_window_extent(renderer=_rd)
        if _e.x1 <= _pw - _marg and _e.x0 >= _marg:
            break
    if _fs != FS_TITLE:
        print(f"     {letter}: title {FS_TITLE} -> {_fs} pt to fit the page")
    # The page is pinned, not cropped. These six panels are placed side by side in the
    # composite, so they have to come out at one size; letting bbox_inches="tight" set it
    # made the two long-titled panels 40 pt wider than the rest. Keep the page and let the
    # plot box take what the margins leave.
    _pg = PAGE_IN[ep]
    _p = ax.get_position()
    _fw, _fh = fig.get_size_inches()
    _l, _b = _p.x0 * _fw, _p.y0 * _fh
    _r, _t = _fw - _p.x1 * _fw, _fh - _p.y1 * _fh
    fig.set_size_inches(*_pg)
    ax.set_position([_l / _pg[0], _b / _pg[1],
                     (_pg[0] - _l - _r) / _pg[0], (_pg[1] - _b - _t) / _pg[1]])
    fig.savefig(os.path.join(OUT, f"FigS_upenn_{letter}_{marker}_{ep}.pdf"), dpi=400,
                facecolor="white")
    plt.close(fig)

pd.DataFrame(rows).to_csv(os.path.join(RES, "upenn_km_stats.csv"), index=False)
print(f"\n  saved {len(PANELS)} panels to output/")
print("done.")
