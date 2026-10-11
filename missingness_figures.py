"""
Figures for the DETECT missingness report.

Reads Missingness_Report/DETECT_Missingness_Report_<tag>.xlsx (run
missingness_report.py first) and the raw streams, then writes PNGs to
Missingness_Report/figures/ plus one PDF containing every figure.

  01_stream_overview          subjects with data and overall % complete, per stream
  02_subject_stream_heatmap   % complete for every subject x stream, plus clinical data
  03_completeness_spread      spread of subject % complete within each stream
  04_streams_per_subject      how many streams each subject has
  05_stream_overlap           subjects who have both stream A and stream B
  06_subjects_per_month       subjects contributing to each stream over time
  07_timeline_<stream>        each subject's weekly coverage across the study
  08_clinical_by_visit        yearly clinical assessments done at each UDS3 visit
  09_clinical_one_time        amyloid PET, QUEST, APOE, and UDS3 visits per subject

Usage:
    python missingness_figures.py
"""

import importlib.util
import io
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

ROOT = Path(__file__).resolve().parent
TAG = "2025-07-16"
REPORT = ROOT / "Missingness_Report" / f"DETECT_Missingness_Report_{TAG}.xlsx"
FIG_DIR = ROOT / "Missingness_Report" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"
NO_DATA = "#ebeae5"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

DAILY = ["Watch_Sleep", "Watch_Steps", "Emfit", "Scale", "NYCE", "Worktime"]
SURVEYS = ["SMART", "HUF"]
MEASURED = DAILY + SURVEYS
ALL = MEASURED + ["UDS3", "Amyloid", "APOE"]
LABEL = {"Watch_Sleep": "Watch sleep", "Watch_Steps": "Watch steps", "Emfit": "Emfit bed mat",
         "Scale": "Scale", "NYCE": "NYCE motion", "Worktime": "Worktime",
         "SMART": "SMART (monthly)", "HUF": "HUF (weekly)", "UDS3": "UDS3 clinical",
         "Amyloid": "Amyloid", "APOE": "APOE"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 26,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
})


def load_report_module():
    spec = importlib.util.spec_from_file_location("missingness_report", ROOT / "missingness_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def subtitle(ax, text):
    ax.annotate(text, (0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
                color=INK_2, fontsize=9.5, va="bottom")


def hgrid(ax, axis="x"):
    ax.grid(True, axis=axis)
    ax.set_axisbelow(True)


def fig_title(fig, title, sub):
    """Title and subtitle pinned to the top-left of the figure, a fixed distance in inches."""
    h = fig.get_figheight()
    fig.text(0.02, 1 - 0.3 / h, title, fontsize=13, fontweight="semibold", color=INK, va="top")
    fig.text(0.02, 1 - 0.62 / h, sub, fontsize=9.5, color=INK_2, va="top")


# ---------------------------------------------------------------- figures

def fig_overview(overview, n_subjects):
    ov = overview.set_index("stream").reindex(ALL)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True,
                                 gridspec_kw={"wspace": 0.08})
    y = np.arange(len(ALL))[::-1]
    labels = [LABEL[s] for s in ALL]

    n = ov["subjects_with_data"].values
    a1.barh(y, n, height=0.62, color=SERIES)
    for yi, v in zip(y, n):
        a1.text(v + n_subjects * 0.01, yi, f"{v}", va="center", fontsize=9, color=INK_2)
    a1.set_xlim(0, n_subjects * 1.12)
    a1.axvline(n_subjects, color=AXIS, lw=0.8)
    a1.text(n_subjects + 1.5, y.min() - 0.1, f"all {n_subjects}", ha="left", va="center",
            fontsize=8.5, color=MUTED)
    a1.set_yticks(y, labels)
    a1.set_title("Subjects with any data")
    subtitle(a1, f"Out of {n_subjects} enrolled subjects")
    hgrid(a1)

    pct = ov["overall_pct_complete"].values
    a2.barh(y, np.nan_to_num(pct), height=0.62, color=SERIES)
    for yi, v in zip(y, pct):
        txt = "no expected window" if np.isnan(v) else f"{v:.0f}%"
        a2.text((0 if np.isnan(v) else v) + 1.2, yi, txt, va="center", fontsize=9,
                color=MUTED if np.isnan(v) else INK_2)
    a2.set_xlim(0, 108)
    a2.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    a2.set_title("Overall % complete")
    subtitle(a2, "Observed ÷ expected, first to last observed date")
    hgrid(a2)
    a2.tick_params(left=False)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.08)
    return fig


def fig_subject_heatmap(pct, avail):
    # Sort: most streams first, then highest average completeness
    order = (avail.assign(_mean=pct[MEASURED].mean(axis=1))
             .sort_values(["n_streams", "_mean"], ascending=[False, False]).index)
    p = pct.loc[order, MEASURED]
    n = len(p)
    fig = plt.figure(figsize=(9.5, max(8, n * 0.13 + 2)))
    gs = fig.add_gridspec(1, 2, width_ratios=[len(MEASURED), 3.2], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    axc = fig.add_subplot(gs[1], sharey=ax)

    vals = p.values / 100
    ax.imshow(np.where(np.isnan(vals), 1, np.nan), cmap=ListedColormap([NO_DATA]),
              aspect="auto", interpolation="nearest")
    im = ax.imshow(vals, cmap=SEQ, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    # 2px surface gaps between cells
    ax.set_xticks(np.arange(-0.5, len(MEASURED)), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(len(MEASURED)), [LABEL[s] for s in MEASURED], rotation=40, ha="left")
    ax.xaxis.tick_top()
    ax.set_yticks(range(n), p.index, fontsize=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig_title(fig, "% complete by subject and stream",
              "Gray = no data in that stream. Sorted by number of streams, then completeness.")

    # Clinical panel: UDS3 visit count and amyloid availability
    uds = avail.loc[order, "uds3_n_visits"].fillna(0).astype(int).values
    amy = avail.loc[order, "Amyloid"].eq("Y").values
    apo = avail.loc[order, "APOE"].eq("Y").values
    for i, (u, a, g) in enumerate(zip(uds, amy, apo)):
        if u:
            axc.text(0, i, str(u), ha="center", va="center", fontsize=6.5, color=INK)
        else:
            axc.text(0, i, "–", ha="center", va="center", fontsize=6.5, color=MUTED)
        axc.scatter(1, i, s=14, color=SERIES if a else NO_DATA, linewidths=0)
        axc.scatter(2, i, s=14, color=SERIES if g else NO_DATA, linewidths=0)
    axc.set_xlim(-0.6, 2.6)
    axc.set_xticks([0, 1, 2], ["UDS3\nvisits", "Amyloid", "APOE"], rotation=40, ha="left")
    axc.xaxis.tick_top()
    axc.tick_params(which="both", left=False, labelleft=False)
    for sp in axc.spines.values():
        sp.set_visible(False)

    cax = fig.add_axes([0.13, 0.01, 0.45, 0.01])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)
    h = fig.get_figheight()
    fig.subplots_adjust(left=0.08, right=0.98, top=1 - 1.75 / h, bottom=0.6 / h)
    cax.set_position([0.13, 0.22 / h, 0.45, 0.1 / h])
    return fig


def fig_spread(detail):
    d = detail[detail["stream"].isin(MEASURED) & detail["pct_complete"].notna()]
    order = d.groupby("stream")["pct_complete"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    rng = np.random.default_rng(7)
    for i, s in enumerate(order):
        v = d.loc[d["stream"] == s, "pct_complete"].values
        jitter = rng.uniform(-0.22, 0.22, len(v))
        ax.scatter(v, i + jitter, s=12, color=SERIES, alpha=0.45, linewidths=0)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([q1, q3], [i, i], color=INK, lw=2, solid_capstyle="round")
        ax.plot([med, med], [i - 0.3, i + 0.3], color=INK, lw=2.4)
        ax.text(102, i, f"median {med:.0f}%  ·  n={len(v)}", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(order)), [LABEL[s] for s in order])
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.spines["bottom"].set_bounds(0, 100)
    hgrid(ax)
    ax.set_title("How complete is each subject's data?")
    subtitle(ax, "Each dot is one subject. Black bar = middle 50% of subjects, tick = median.")
    fig.subplots_adjust(left=0.15, right=0.97, top=0.86, bottom=0.08)
    return fig


def fig_streams_per_subject(avail):
    counts = avail["n_streams"].value_counts().reindex(range(len(ALL) + 1), fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(counts.index, counts.values, width=0.7, color=SERIES)
    for x, v in counts.items():
        if v:
            ax.text(x, v + counts.max() * 0.015, str(v), ha="center", va="bottom", fontsize=9, color=INK_2)
    ax.set_xticks(counts.index)
    ax.set_xlabel(f"Number of streams with any data (out of {len(ALL)})")
    ax.set_ylabel("Subjects")
    hgrid(ax, "y")
    ax.set_title("How many data streams does each subject have?")
    subtitle(ax, f"{len(avail)} enrolled subjects")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.85, bottom=0.13)
    return fig


def fig_overlap(overlap):
    o = overlap.reindex(index=ALL, columns=ALL)
    k = len(ALL)
    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    im = ax.imshow(o.values, cmap=SEQ, vmin=0, vmax=o.values.max(), interpolation="nearest")
    for i in range(k):
        for j in range(k):
            v = o.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if v > o.values.max() * 0.55 else INK,
                    fontweight="semibold" if i == j else "normal")
    ax.set_xticks(range(k), [LABEL[s] for s in ALL], rotation=40, ha="left")
    ax.xaxis.tick_top()
    ax.set_yticks(range(k), [LABEL[s] for s in ALL])
    ax.set_xticks(np.arange(-0.5, k), minor=True)
    ax.set_yticks(np.arange(-0.5, k), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig_title(fig, "Subjects with both streams", "Diagonal = subjects with that stream at all")
    fig.subplots_adjust(left=0.17, right=0.97, top=0.78, bottom=0.03)
    return fig


def fig_monthly(monthly):
    m = monthly.copy()
    m.index = pd.PeriodIndex(m.index, freq="M").to_timestamp()
    streams = [s for s in MEASURED if s in m.columns]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2), sharex=True, sharey=True)
    ymax = m[streams].values.max()
    for ax, s in zip(axes.flat, streams):
        v = m[s].astype(float)
        active = v[v > 0].index
        v[(v.index < active.min()) | (v.index > active.max())] = np.nan
        ax.plot(m.index, v, color=SERIES, lw=2)
        ax.fill_between(m.index, v, color=SERIES, alpha=0.08, linewidth=0)
        ax.annotate(f"ends {active.max():%b %Y}", (active.max(), v[active.max()]),
                    xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=MUTED, va="center")
        ax.set_title(LABEL[s], fontsize=11, pad=6)
        hgrid(ax, "y")
        peak = m[s].idxmax()
        ax.annotate(f"peak {m[s].max()}", (peak, m[s].max()), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=8, color=INK_2)
        ax.set_ylim(0, ymax * 1.18)
        ax.set_xlim(m.index.min(), m.index.max() + pd.Timedelta(days=330))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes[:, 0]:
        ax.set_ylabel("Subjects")
    fig.suptitle("Subjects contributing data each month", x=0.06, ha="left",
                 fontsize=13, fontweight="semibold", color=INK)
    fig.text(0.06, 0.915, "Same scale on every panel. Each line stops at the stream's last month of data.",
             color=INK_2, fontsize=9.5)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.07, hspace=0.35, wspace=0.08)
    return fig


def fig_timeline(stream, days, unit_freq, unit_label):
    """Subject x period grid, blank outside the subject's span.

    unit_freq "D": daily stream in weekly columns, shaded by days covered.
    unit_freq "W" / "M": survey stream in weekly / monthly columns, survey or not.
    """
    d = days.copy()
    col_freq = "M" if unit_freq == "M" else "W-SUN"
    col_name = "month" if unit_freq == "M" else "week"
    d["col"] = d["date"].dt.to_period(col_freq)
    if unit_freq == "D":
        cov = d.groupby(["subid", "col"])["date"].nunique() / 7
    else:
        cov = d.groupby(["subid", "col"]).size().clip(upper=1).astype(float)
    cov = cov.unstack()
    periods = pd.period_range(cov.columns.min(), cov.columns.max(), freq=col_freq)
    cov = cov.reindex(columns=periods)
    cov.columns = periods.start_time
    all_weeks = cov.columns
    first = cov.notna().idxmax(axis=1)
    last = cov.notna().iloc[:, ::-1].idxmax(axis=1)
    # Inside a subject's span, weeks without data are 0, not blank
    in_span = (cov.columns.values[None, :] >= first.values[:, None]) & \
              (cov.columns.values[None, :] <= last.values[:, None])
    cov = cov.where(~(in_span & cov.isna()), 0)
    cov = cov.loc[first.sort_values().index]

    n = len(cov)
    fig, ax = plt.subplots(figsize=(13, max(4.5, n * 0.085 + 1.8)))
    x0, x1 = mdates.date2num(all_weeks[0]), mdates.date2num(periods[-1].end_time)
    cmap = SEQ if unit_freq == "D" else ListedColormap([BLUE_RAMP[0], BLUE_RAMP[-1]])
    im = ax.imshow(cov.values, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                   extent=[x0, x1, n - 0.5, -0.5])
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_yticks(range(n), cov.index, fontsize=5 if n > 80 else 6.5)
    ax.set_ylabel("Subject (sorted by first observed date)")
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig_title(fig, f"{LABEL[stream]}: coverage over time",
              f"Each row is a subject, each column a {col_name}. {unit_label} "
              "Blank = before first or after last observed date.")
    cax = fig.add_axes([0.80, 1 - 0.5 / fig.get_figheight(), 0.17, 0.12 / fig.get_figheight()])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    if unit_freq == "D":
        cb.set_ticks([0, 0.5, 1], labels=["0 days", "3–4", "7 days"])
    else:
        cb.set_ticks([0.25, 0.75], labels=["no survey", "survey"])
    cb.ax.tick_params(labelsize=8, length=0)
    fig.subplots_adjust(left=0.07, right=0.99, top=1 - 1.0 / fig.get_figheight(),
                        bottom=0.55 / fig.get_figheight())
    return fig


def fig_clinical_by_visit(clin):
    """Assessment x visit grid: share of attended visits where the assessment was done."""
    y = clin[clin["type"] != "one-time"].set_index("assessment")
    cols = ["Initial", "FU1", "FU2"]
    done = y[[f"subjects_at_{c}" for c in cols]].values.astype(int)
    attended = done + y[[f"missing_at_{c}" for c in cols]].values.astype(int)
    frac = done / attended
    k = len(y)
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * k + 2.2))
    ax.imshow(frac, cmap=SEQ, vmin=0.9, vmax=1, aspect="auto", interpolation="nearest")
    for i in range(k):
        for j in range(3):
            miss = attended[i, j] - done[i, j]
            txt = f"{done[i, j]}/{attended[i, j]}" + (f"  ({miss} missing)" if miss else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="#ffffff" if frac[i, j] > 0.97 else INK)
    ax.set_xticks(range(3), ["Initial visit", "Follow-up 1", "Follow-up 2"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(k), y.index)
    ax.set_xticks(np.arange(-0.5, 3), minor=True)
    ax.set_yticks(np.arange(-0.5, k), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig_title(fig, "Yearly clinical assessments by visit",
              "Subjects with the assessment / subjects who attended that visit. "
              "Lighter = more missing.")
    h = fig.get_figheight()
    fig.subplots_adjust(left=0.3, right=0.98, top=1 - 1.35 / h, bottom=0.2 / h)
    return fig


def fig_clinical_one_time(clin, visit_counts, n_subjects):
    one = clin[clin["type"] == "one-time"].set_index("assessment")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"wspace": 0.45})

    y = np.arange(len(one))[::-1]
    have = one["subjects_with_any"].astype(int).values
    a1.barh(y, have, height=0.6, color=SERIES)
    for yi, v, (name, r) in zip(y, have, one.iterrows()):
        a1.text(v + n_subjects * 0.01, yi, f"{v}  ({n_subjects - v} without)", va="center",
                fontsize=9, color=INK_2)
        if isinstance(r.get("result_breakdown"), str) and r["result_breakdown"]:
            a1.text(0, yi - 0.42, r["result_breakdown"], fontsize=8, color=MUTED, va="center")
    a1.set_yticks(y, one.index)
    a1.set_xlim(0, n_subjects * 1.3)
    a1.set_ylim(-0.8, len(one) - 0.4)
    a1.set_title("One-time clinical measures")
    subtitle(a1, f"Subjects with each measure, out of {n_subjects}")
    hgrid(a1)

    vc = visit_counts.set_index("uds3_visits_attended")["n_subjects"].reindex(range(4), fill_value=0)
    a2.bar(vc.index, vc.values, width=0.65, color=SERIES)
    for xv, v in vc.items():
        a2.text(xv, v + vc.max() * 0.02, str(v), ha="center", va="bottom", fontsize=9, color=INK_2)
    a2.set_xticks(range(4), ["None", "1", "2", "3"])
    a2.set_xlabel("UDS3 visits attended (Initial, FU 1, FU 2)")
    a2.set_ylabel("Subjects")
    a2.set_title("Yearly visits per subject")
    subtitle(a2, "Subjects with no visit have none of the yearly assessments")
    hgrid(a2, "y")
    fig.subplots_adjust(left=0.13, right=0.98, top=0.84, bottom=0.13)
    return fig


# ---------------------------------------------------------------- main

def save_png(fig, path, attempts=5):
    """Render to memory, then write. Retry briefly if another process (sync, viewer,
    antivirus) is holding the file."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    for i in range(attempts):
        try:
            path.write_bytes(buf.getvalue())
            return
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(1 + i)


def main():
    if not REPORT.exists():
        raise SystemExit(f"{REPORT} not found. Run missingness_report.py first.")
    x = pd.read_excel(REPORT, sheet_name=None)
    overview = x["Stream_Overview"]
    avail = x["Subject_Availability"].set_index("subid")
    pct = x["Subject_Pct_Complete"].set_index("subid").reindex(avail.index)
    detail = x["Subject_Stream_Detail"]
    overlap = x["Stream_Overlap"].set_index(x["Stream_Overlap"].columns[0])
    monthly = x["Monthly_Subjects"].set_index(x["Monthly_Subjects"].columns[0])
    clin = x["Clinical_Overview"]
    clin_visits = x["Clinical_Visit_Counts"]

    print("Loading raw streams for timelines...")
    rep = load_report_module()
    _, hs = rep.load_context()
    observed, _ = rep.observed_daily(hs)
    surveys = rep.load_surveys()

    figs = [
        ("01_stream_overview", fig_overview(overview, len(avail))),
        ("02_subject_stream_heatmap", fig_subject_heatmap(pct, avail)),
        ("03_completeness_spread", fig_spread(detail)),
        ("04_streams_per_subject", fig_streams_per_subject(avail)),
        ("05_stream_overlap", fig_overlap(overlap)),
        ("06_subjects_per_month", fig_monthly(monthly)),
    ]
    for s in DAILY:
        figs.append((f"07_timeline_{s}", fig_timeline(
            s, observed[s], "D", "Darker = more days of data that week.")))
    for s, freq, unit in [("SMART", "M", "month"), ("HUF", "W", "week")]:
        done = surveys[s][surveys[s]["finished"]]
        figs.append((f"07_timeline_{s}", fig_timeline(
            s, done[["subid", "date"]], freq, f"Dark = a finished survey that {unit}.")))
    figs.append(("08_clinical_by_visit", fig_clinical_by_visit(clin)))
    figs.append(("09_clinical_one_time", fig_clinical_one_time(clin, clin_visits, len(avail))))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR.parent / f"DETECT_Missingness_Figures_{TAG}.pdf"
    with PdfPages(pdf_path) as pdf:
        for name, fig in figs:
            save_png(fig, FIG_DIR / f"{name}.png")
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  {name}.png")
    print(f"Wrote {len(figs)} figures to {FIG_DIR}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
