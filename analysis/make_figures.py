#!/usr/bin/env python3
"""
analysis/make_figures.py

Builds Tables 1 and 2, the training-loss table, and Figures 1 to 10, from the derived
CSVs. Nothing here recomputes a result: every number is read from analysis/out_*/ or
documents/ so the figures and the text cannot drift apart. Run the analysis pipeline
first (see analysis/README.md).

    Table 1     registered grid, both seeds, with the two matched comparisons beneath
    Table 2     rollout motion: departure failures, latency, speed, completion time
    Table loss  final training loss against grid score

    Fig 1   Cube starting positions
    Fig 2   Aim invariance
    Fig 3   Aiming error
    Fig 4   Probe decoding
    Fig 5   Trajectory envelope
    Fig 6   Release point
    Fig 7   When the outcome becomes readable
    Fig 8   Training loss
    Fig 9   Four metrics by condition
    Fig 10  Bearing selection at the held-out positions

Requires scipy (ConvexHull, in fig_positions) alongside pandas and matplotlib. Must be
run from the repository root: it imports tools/calibrate_pose.py for the position table.

Outputs PDF for LaTeX and PNG for the web, into figures/.

RUN: python analysis/make_figures.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, "tools")
from calibrate_pose import BASE_X, T          # the ten training positions
from matplotlib.lines import Line2D

OUT = "figures"

# Validated categorical slots (dataviz reference palette, light mode). At most two hues
# are ever on screen together; NEUTRAL is context, not a category.
BLUE, ORANGE = "#2a78d6", "#eb6834"
NEUTRAL = "#8a8985"
INK, MUTED = "#0b0b0b", "#52514e"

T6_AZ = np.degrees(np.arctan2(15.5 - BASE_X, 10.0))                              # +24.23
CUP_AZ = np.degrees(np.arctan2(5.0 - BASE_X, 12.5))                              # -25.60
MID_AZ = np.degrees(np.arctan2((15.5 + 5.0) / 2 - BASE_X, (10.0 + 12.5) / 2))    #  -3.81

GRASP_S = 215 / 30
WINDOW_S = 3 * 50 / 30

SAME_TARGET = ["in_distribution", "reduced_lighting", "different_object", "distractors"]
CELLS = SAME_TARGET + ["new_positions"]
CONDITIONS = ["clean", "color", "recovery", "randomized"]
# Invariant policies first, so the eye lands on the outliers last.
POLICIES = ["clean", "clean-seed2000", "color", "color-seed2000",
            "recovery", "recovery-seed2000", "randomized", "randomized-seed2000"]

SHOW_TITLES = False

plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 6.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "figure.constrained_layout.use": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": False,
    "grid.color": "#e3e2df",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.4,
    "figure.dpi": 400,
})

def title(ax, text):
    if SHOW_TITLES:
        ax.set_title(text, loc="center", pad=8)

def need(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}. Run the analysis pipeline first (SETUP.md).")
    return path

def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.pdf and .png")


def pretty(name):
    """Display label for a policy name. Figures only; the tables use raw names."""
    s = name.replace("-seed2000", " (s2000)").replace("_", " ")
    return s[:1].upper() + s[1:]

def pick(df, *candidates):
    """First matching column name, or a clear error listing what is actually there."""
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"none of the {candidates} found. Columns are: {list(df.columns)}")

# ----------------------------------------------------------------------------
# Table 1
# ----------------------------------------------------------------------------

def table1():
    print("Table 1")
    grids, matched = {}, {}
    for seed in (1000, 2000):
        d = pd.read_csv(need(f"analysis/out_seed{seed}/success_by_cell.csv"))
        grids[seed] = d.pivot(index="condition", columns="eval_cell", values="successes")
        matched[seed] = pd.read_csv(need(f"analysis/out_seed{seed}/matched_comparisons.csv"))

    head = ["condition"] + [c.replace("_", " ") for c in CELLS] + ["total"]
    md, tex = [], []

    for seed in (1000, 2000):
        g = grids[seed].reindex(index=CONDITIONS, columns=CELLS)
        md.append(f"\n**Seed {seed}** (successes out of 15)\n")
        md.append("| " + " | ".join(head) + " |")
        md.append("|" + "---|" * len(head))
        tex.append(rf"\multicolumn{{{len(head)}}}{{l}}{{\textit{{Seed {seed}}}}} \\")
        for cond in CONDITIONS:
            vals = [int(g.loc[cond, c]) for c in CELLS]
            md.append(f"| {cond} | " + " | ".join(str(v) for v in vals)
                      + f" | {sum(vals)}/75 |")
            tex.append(f"{cond} & " + " & ".join(str(v) for v in vals)
                       + rf" & {sum(vals)}/75 \\")
        tex.append(r"\midrule")

    md.append("\n**Matched-axis comparisons** (pre-registered)\n")
    md.append("| seed | cell | matched | clean | difference | 95% CI | Fisher p |")
    md.append("|---|---|---|---|---|---|---|")
    for seed in (1000, 2000):
        for _, r in matched[seed].iterrows():
            ci = f"{r.diff_ci_low:+.3f} to {r.diff_ci_high:+.3f}"
            md.append(f"| {seed} | {r.eval_cell.replace('_', ' ')} "
                      f"({r.matched_condition}) | **{r.matched}** | {r.baseline_counts} "
                      f"| **{r.difference:+.3f}** | {ci} | {r.fisher_p:.3f} |")
            tex.append(rf"\textbf{{{r.matched_condition} vs clean, {r.eval_cell}}} (s{seed}) "
                       rf"& \multicolumn{{5}}{{l}}{{{r.matched} vs {r.baseline_counts}, "
                       rf"$\Delta$ {r.difference:+.3f} "
                       rf"[{r.diff_ci_low:+.3f}, {r.diff_ci_high:+.3f}], "
                       rf"$p$ = {r.fisher_p:.3f}}} \\")

    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "table1.md"), "w").write("\n".join(md) + "\n")
    open(os.path.join(OUT, "table1.tex"), "w").write("\n".join(tex) + "\n")
    print(f"  wrote {OUT}/table1.md and table1.tex")

# ----------------------------------------------------------------------------
# Table 2
# ----------------------------------------------------------------------------

def table2():
    print("Table 2")
    m = pd.read_csv(need("analysis/out_motion/rollout_motion_by_condition.csv"))

    c_cond = pick(m, "condition", "policy")
    c_seed = pick(m, "seed", "train_seed")
    c_n = pick(m, "n", "n_episodes", "episodes")
    c_nd = pick(m, "no_departure", "never_departed", "n_no_departure", "no_departure_n")
    c_lat = pick(m, "median_latency_s", "latency_median_s", "median_latency")
    c_dps = pick(m, "deg_step_success", "deg_per_step_success", "deg_per_step")
    c_dur = pick(m, "duration_success_s", "mean_duration_success_s", "mean_duration_s")

    # Registered conditions in the usual order, then anything exploratory at the end.
    key = {c: i for i, c in enumerate(CONDITIONS)}
    m = m.sort_values([c_cond, c_seed],
                      key=lambda s: s.map(key).fillna(99) if s.name == c_cond else s)

    head = ["condition", "seed", "never departed", "median latency (s)",
            "deg/step on successes", "mean duration of successes (s)"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = []

    for _, r in m.iterrows():
        cells = [str(r[c_cond]), str(int(r[c_seed])),
                 f"{int(r[c_nd])}/{int(r[c_n])}",
                 f"{r[c_lat]:.2f}", f"{r[c_dps]:.3f}", f"{r[c_dur]:.1f}"]
        md.append("| " + " | ".join(cells) + " |")
        tex.append(" & ".join(cells) + r" \\")

    md.append("")
    md.append("Clean is out of 91 rather than 75 because the 16 displacement-probe episodes "
              "per seed were run on the Clean policy and are pooled here.")

    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "table2.md"), "w").write("\n".join(md) + "\n")
    open(os.path.join(OUT, "table2.tex"), "w").write("\n".join(tex) + "\n")
    print(f"  wrote {OUT}/table2.md and table2.tex")


# ----------------------------------------------------------------------------
# Training-Loss Table
# ----------------------------------------------------------------------------

def table_loss():
    print("Table: training loss")
    d = pd.read_csv(need("documents/training_loss.csv"))
    c_step = pick(d, "_step", "step")
    c_loss = pick(d, "train/losses_after_rm_padding", "train_loss", "loss")
    final = d.sort_values(c_step).groupby("run")[c_loss].last()

    totals = {}
    for seed in (1000, 2000):
        g = pd.read_csv(need(f"analysis/out_seed{seed}/success_by_cell.csv"))
        s = g.pivot(index="condition", columns="eval_cell", values="successes")
        s = s.reindex(index=CONDITIONS, columns=CELLS)
        for cond in CONDITIONS:
            totals[(cond, seed)] = int(s.loc[cond].sum())

    def split(run):
        return (run[:-9], 2000) if run.endswith("-seed2000") else (run, 1000)

    rows = []
    for run, loss in final.items():
        cond, seed = split(str(run))
        t = totals.get((cond, seed))
        rows.append((cond, seed, float(loss), f"{t}/75" if t is not None else "n/a"))
    rows.sort(key=lambda r: r[2])

    head = ["condition", "seed", "final loss", "grid total"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = []
    for cond, seed, loss, tot in rows:
        cells = [cond, str(seed), f"{loss:.4f}", tot]
        md.append("| " + " | ".join(cells) + " |")
        tex.append(" & ".join(cells) + r" \\")

    md.append("")
    md.append(f"Final loss is the last logged value (step {int(d[c_step].max())}). "
              "Color-slowpace and density are exploratory and outside the registered grid.")

    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "table_loss.md"), "w").write("\n".join(md) + "\n")
    open(os.path.join(OUT, "table_loss.tex"), "w").write("\n".join(tex) + "\n")
    print(f"  wrote {OUT}/table_loss.md and table_loss.tex")


# ----------------------------------------------------------------------------
# Fig 1  Cube starting positions
# ----------------------------------------------------------------------------

FRAME_W, FRAME_H = 22.0, 17.0
CUP_XY = (5.0, 12.5)
TRAIN_XY = {"T1": (2.0, 2.5), "T2": (6.5, 7.5), "T3": (8.5, 15.0), "T4": (12.0, 14.0),
            "T5": (15.5, 2.5), "T7": (15.5, 14.25), "T8": (20.5, 2.5),
            "T9": (20.5, 6.5), "T10": (20.5, 10.0)}
T6_XY = (15.5, 10.0)
INTERP_XY = {"E2": (6.5, 2.5), "E3": (12.0, 10.0), "E4": (15.5, 6.5)}
EXTRAP_XY = {"E1": (2.0, 7.5), "E5": (19.5, 13.5)}

def fig_positions():
    print("Positions")
    from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, Polygon
    from scipy.spatial import ConvexHull

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.add_patch(Rectangle((0, 0), FRAME_W, FRAME_H, facecolor="#f1f0ed",
                           edgecolor=NEUTRAL, lw=0.8, zorder=0))
    ax.add_patch(Circle(CUP_XY, 1.45, facecolor="white", edgecolor=NEUTRAL,
                        lw=1.2, zorder=2))
    ax.text(*CUP_XY, "CUP", ha="center", va="center", fontsize=6,
            color=MUTED, zorder=3)
    ax.add_patch(FancyBboxPatch((BASE_X - 1.3, -0.2), 2.6, 3.0,
                                boxstyle="round,pad=0,rounding_size=0.35",
                                facecolor="#d9d8d4", edgecolor=NEUTRAL,
                                lw=0.8, zorder=2))
    ax.text(BASE_X, 1.3, "ARM", ha="center", va="center", fontsize=6,
            color=MUTED, zorder=3)

    # Labels default above the marker; these few would otherwise land on a hull edge.
    nudge = {"T8": (-7, -2.5), "T9": (-7, -2.5), "T10": (-7, -2.5), "T1": (0, -11)}

    def lab(name, x, y, colour, weight="normal"):
        dx, dy = nudge.get(name, (0, 6))
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="right" if dx < 0 else "center",
                    va="center" if dx < 0 else "baseline",
                    fontsize=6, color=colour, fontweight=weight)

    pts = np.array(list(TRAIN_XY.values()) + [T6_XY])
    ax.add_patch(Polygon(pts[ConvexHull(pts).vertices], closed=True,
                         facecolor="none", edgecolor=MUTED, lw=0.9,
                         ls=(0, (4, 3)), zorder=1))

    for name, (x, y) in TRAIN_XY.items():
        ax.scatter(x, y, s=42, color=INK, zorder=4)
        lab(name, x, y, MUTED)
    ax.scatter(*T6_XY, s=52, facecolors="white", edgecolors=INK, linewidths=1.1,
               linestyle="--", zorder=4)
    lab("T6", T6_XY[0], T6_XY[1], INK, weight="bold")
    for name, (x, y) in INTERP_XY.items():
        ax.scatter(x, y, s=48, facecolors="white", edgecolors=BLUE,
                   linewidths=1.4, zorder=4)
        lab(name, x, y, BLUE)
    for name, (x, y) in EXTRAP_XY.items():
        ax.scatter(x, y, s=48, facecolors="white", edgecolors=ORANGE,
                   linewidths=1.4, zorder=4)
        lab(name, x, y, ORANGE)

    ax.set_xlim(-0.6, 22.6)
    ax.set_ylim(-0.9, 17.9)
    ax.set_aspect("equal")
    ax.set_xlabel("x (inches)")
    ax.set_ylabel("y (inches)")
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, 5, 10, 15])
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=INK, markersize=5,
               label="Training (Randomized)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white",
               markeredgecolor=INK, markersize=5.5, label="T6 (all conditions)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white",
               markeredgecolor=BLUE, markersize=5.5, label="Held out, interpolation"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white",
               markeredgecolor=ORANGE, markersize=5.5, label="Held out, extrapolation"),
        Line2D([], [], color=MUTED, lw=0.9, ls=(0, (4, 3)), label="Training convex hull"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False,
       handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0, labelspacing=0.3)
    title(ax, "Cube starting positions")
    save(fig, "fig_positions")

# ----------------------------------------------------------------------------
# Fig 2  Aim invariance
# ----------------------------------------------------------------------------

def fig_aim_invariance():
    print("Aim invariance")
    aim = pd.read_csv(need("analysis/out_azimuth/aim_by_cell.csv"), index_col=0)
    order = [p for p in POLICIES if p in aim.index]
    y = np.arange(len(order))[::-1]

    fig, ax = plt.subplots(figsize=(3.1, 3.0))
    ax.axvline(T6_AZ, color=MUTED, lw=1.0, ls="--", zorder=1)

    for cell in [c for c in SAME_TARGET if c in aim.columns]:
        ax.scatter(aim[cell].reindex(order), y, s=30, facecolors="none",
                   edgecolors=NEUTRAL, linewidths=1.0, zorder=2)
    if "new_positions" in aim.columns:
        ax.scatter(aim["new_positions"].reindex(order), y, s=34, marker="D",
                   color=ORANGE, zorder=3)
        for p, yy in zip(order, y):
            v = float(aim.loc[p, "new_positions"])
            ref = float(aim.loc[p, "in_distribution"])
            if abs(v - ref) > 5:                      # label only what actually moved
                ax.annotate(f"{v - ref:+.1f}°", (v, yy), xytext=(0, 7),
                            textcoords="offset points", ha="center",
                            fontsize=7, color=ORANGE)

    ax.set_yticks(y, [pretty(p) for p in order])
    ax.set_xlabel("Median commanded bearing (deg)")
    ax.set_xlim(-6, 34)
    ax.set_ylim(-0.8, len(order) - 0.2)
    # y pinned to the axes, so the label cannot drift if the policy list changes.
    ax.annotate("Cube at T6", xy=(T6_AZ, 1.0), xycoords=("data", "axes fraction"),
                xytext=(3, -3), textcoords="offset points",
                ha="left", va="top", fontsize=7, color=MUTED)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    # Legend below the axes: the long cell label used to overprint "Cube at T6".
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor=NEUTRAL, markersize=5.5,
               label="Eval cells (in_distribution, etc.)"),
        Line2D([], [], marker="D", ls="", color=ORANGE, markersize=5,
               label="New positions"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False,
       handletextpad=0.4, borderaxespad=0.0, labelspacing=0.3)
    title(ax, "Only Randomized's aim moves when the cube does")
    save(fig, "fig_aim_invariance")


# ----------------------------------------------------------------------------
# Fig 3  Aiming error, commanded against required
# ----------------------------------------------------------------------------

def fig_aiming_error():
    print("Aiming error")
    aiming = pd.read_csv(need("analysis/out_azimuth/randomized_aiming.csv"))
    centroid = float(np.mean([np.degrees(np.arctan2(x - BASE_X, yy))
                              for x, yy in T.values()]))
    lim = (-82, 62)

    fig, ax = plt.subplots(figsize=(2.6, 2.7))
    # Both reference lines are labelled through the legend rather than with in-axes
    # text: a square panel this small has no free corner for a rotated label.
    ax.plot(lim, lim, ls="--", lw=1.0, color=MUTED, zorder=1, label="No error")
    ax.axhline(centroid, ls=":", lw=1.0, color=NEUTRAL, zorder=1,
               label=f"Training mean ({centroid:.0f}°)")

    for split, colour in (("interpolation", BLUE), ("extrapolation", ORANGE)):
        s = aiming[aiming.split == split]
        ax.scatter(s.true_az, s.az, s=30, color=colour, edgecolors="white",
                   linewidths=0.5, zorder=3,
                   label=f"{split[:1].upper()}{split[1:]} (n={len(s)})")

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Bearing required (deg)")
    ax.set_ylabel("Bearing commanded (deg)")
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    # Columns fill top to bottom, so the two reference lines sit left and the two
    # data splits sit right.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False,
              handletextpad=0.4, handlelength=1.6, columnspacing=1.2,
              borderaxespad=0.0, labelspacing=0.3)
    title(ax, "Randomized regresses to the centre\noutside its training hull")
    save(fig, "fig_aiming_error")


# ----------------------------------------------------------------------------
# Fig 4  Probe decoding
# ----------------------------------------------------------------------------

def fig_probe():
    print("Probe")
    probe = pd.read_csv(need("analysis/out_probe/position_probe.csv"))
    pr = probe.set_index("policy").reindex([p for p in POLICIES if p in set(probe.policy)])
    y = np.arange(len(pr))[::-1]
    h = 0.36

    fig, ax = plt.subplots(figsize=(4.0, 2.9))

    def whisk(mid, lo, hi):
        return np.vstack([pr[mid]-pr[lo], pr[hi]-pr[mid]])

    ax.barh(y + h / 2, pr.probe_by_episode_deg, h, color=BLUE,
            label="Held out by episode",
            xerr=whisk("probe_by_episode_deg", "probe_by_episode_lo", "probe_by_episode_hi"),
            error_kw=dict(lw=1.0, ecolor=INK, capsize=2.0))
    ax.barh(y - h / 2, pr.probe_by_position_deg, h, color=ORANGE,
            label="Held out by position",
            xerr=whisk("probe_by_position_deg", "probe_by_position_lo", "probe_by_position_hi"),
            error_kw=dict(lw=1.0, ecolor=INK, capsize=2.0))

    chance = float(pr.chance_med_deg.mean())
    ax.axvline(chance, ls="--", lw=1.0, color=MUTED, zorder=4)
    ax.text(chance + 0.8, y.max() + 0.62, f"Chance ({chance:.0f}°)",
            fontsize=7, color=MUTED, va="center")

    ax.set_yticks(y, [pretty(p) for p in pr.index])
    ax.set_xlabel("Median bearing error (deg)")
    ax.set_xlim(0, 68)
    ax.set_ylim(-0.75, len(pr) - 0.15)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              frameon=False, handletextpad=0.5, columnspacing=2.0)
    title(ax, "Every policy encodes position")

    save(fig, "fig_probe")


# ----------------------------------------------------------------------------
# Fig 5  Trajectory envelope, plotted as the margin over the bearing the cube demands
# ----------------------------------------------------------------------------

def fig_envelope():
    print("Envelope")
    env = pd.read_csv(need("analysis/out_azimuth/clean_envelope.csv"))
    cells = ["in_distribution", "near_1in", "near_2in"]
    xs = np.array([0.0, 1.0, 2.0])
    req = np.array([float(env[env.cell == c].required_az.iloc[0]) for c in cells])

    ylo, yhi = -4.6, 5.6

    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    # Everything above zero is a sweep that still passes over the cube. Shading the
    # band saves the reader from having to work out the sign of the y axis.
    ax.axhspan(0, yhi, color="#f1f0ed", zorder=0)
    ax.axhline(0, color=INK, lw=1.0, ls="--", zorder=2)
    ax.text(0.02, 0.975, "Cube inside the sweep", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.5, color=MUTED, zorder=6)

    for pol, colour, dx, up in (("clean", BLUE, -0.06, True),
                                ("clean-seed2000", ORANGE, 0.06, False)):
        e = env[env.policy == pol].set_index("cell").reindex(cells)
        m = e.median_max_az.to_numpy() - req
        lo = e.median_max_az.to_numpy() - e.q1.to_numpy()
        hi = e.q3.to_numpy() - e.median_max_az.to_numpy()
        ax.errorbar(xs + dx, m, yerr=[lo, hi], marker="o", ms=5, capsize=2.5,
                    color=colour, label=pretty(pol), zorder=3)

    ax.set_xticks(xs, ["0", "1", "2"])
    ax.set_xlim(-0.34, 2.48)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Cube displacement from T6 (inches)")
    ax.set_ylabel("Reached minus required (deg)")
    ax.grid(axis="y", zorder=1)
    ax.set_axisbelow(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False,
              handletextpad=0.4, columnspacing=1.5, borderaxespad=0.0)
    title(ax, "The sweep stops covering the cube past one inch")
    save(fig, "fig_envelope")


# ----------------------------------------------------------------------------
# Fig 6  Release point
# ----------------------------------------------------------------------------

def fig_release():
    print("Release point")
    demo = pd.read_csv(need("analysis/out_drops/demo_drops.csv"))
    loc = pd.read_csv(need("analysis/out_drops/drop_locations.csv"))
    roll = loc[loc.condition == "recovery"]

    fig, ax = plt.subplots(figsize=(3.0, 1.7))
    rng = np.random.default_rng(0)          # jitter only; no inference depends on it
    labels = []
    series = [(roll.az.to_numpy(), f"Recovery policy\n(n={len(roll)})", ORANGE),
              (demo.az.to_numpy(), f"Demonstrations\n(n={len(demo)})", BLUE)]

    for i, (vals, label, colour) in enumerate(series):
        yy = i + rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(vals, yy, s=14, color=colour, alpha=0.75,
                   edgecolors="white", linewidths=0.3, zorder=3)
        m = float(np.median(vals))
        ax.plot([m, m], [i - 0.27, i + 0.27], color=colour, lw=2.2, zorder=4)
        ax.annotate(f"{m:.1f}°", (m, i + 0.33), ha="center", fontsize=7.5, color=colour)
        labels.append(label)

    # The cube marker is omitted: it sits at +24.2 deg while every release is below
    # zero, so it stretched the axis by 30 deg of empty panel. Its bearing is given
    # in the caption. The midpoint still marks 50 percent of the cube to cup interval.
    for v, lab in ((CUP_AZ, "Cup"), (MID_AZ, "Carry midpoint")):
        ax.axvline(v, color=NEUTRAL, lw=1.0, ls=":", zorder=1)
        ax.text(v, 1.74, lab, ha="center", va="center", fontsize=7, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.6), zorder=5)

    ax.set_yticks([0, 1], labels)
    ax.set_ylim(-0.6, 1.95)
    ax.set_xlim(-40, 8)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Release bearing (deg)")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    title(ax, "The policy releases where the demonstrator released")
    save(fig, "fig_release")

# ----------------------------------------------------------------------------
# Fig 7  Success probe sweep
# ----------------------------------------------------------------------------

def fig_success_sweep():
    print("Success sweep")
    s = pd.read_csv(need("analysis/out_probe/success_sweep.csv")).sort_values(["policy", "offset"])

    c_n = pick(s, "episodes", "episode")
    if "t_end_s" not in s.columns:
        s["t_end_s"] = s.t_start_s + WINDOW_S
    if "complete" not in s.columns:
        s["complete"] = s.groupby("policy")[c_n].transform(lambda v: (v == v.iloc[0]).astype(int))

    s = s[s.complete == 1]
    xs = sorted(s.t_end_s.unique())
    ceiling = s.groupby("t_end_s").null_p95.max().reindex(xs).to_numpy()

    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    ax.fill_between(xs, 0.5, ceiling, color="#e3e2df", zorder=1)
    ax.axhline(0.5, color=MUTED, lw=1.0, ls="--", zorder=2)
    ax.axvline(GRASP_S, color=NEUTRAL, lw=1.0, ls=":", zorder=2)
    ax.text(GRASP_S + 0.15, 0.98, "Gripper closes", fontsize=6.5, color=NEUTRAL,
            va="top", transform=ax.get_xaxis_transform())
    # x in axes fractions, y in data units: both labels stay inside the panel and
    # off the curves however the x range comes out.
    ax.annotate("Permutation null", xy=(0.99, (0.5 + ceiling[-1]) / 2),
                xycoords=ax.get_yaxis_transform(), fontsize=6.5, color=MUTED,
                ha="right", va="center")
    ax.annotate("Chance", xy=(0.01, 0.5), xycoords=ax.get_yaxis_transform(),
                xytext=(0, -3), textcoords="offset points",
                fontsize=6.5, color=MUTED, ha="left", va="top")

    for pol, d in s.groupby("policy"):
        d = d.sort_values("t_end_s")
        hi = pol == "randomized"
        ax.plot(d.t_end_s, d.auc, lw=1.8 if hi else 1.0,
                color=ORANGE if hi else NEUTRAL, zorder=4 if hi else 3,
                alpha=1.0 if hi else 0.75)

    ax.set_xlabel("Information cutoff: end of the window (s)")
    ax.set_ylabel("Outcome AUC")
    ax.set_ylim(0.38, 1.06)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Line2D([], [], color=ORANGE, lw=1.8, label="Randomized"),
        Line2D([], [], color=NEUTRAL, lw=1.0, label="Other policies"),
    ], loc="lower right", frameon=False, handletextpad=0.5, borderaxespad=0.2)
    title(ax, "The outcome is unreadable until the gripper closes")
    save(fig, "fig_success_sweep")


# ----------------------------------------------------------------------------
# Fig 8  Training loss (appendix)
# ----------------------------------------------------------------------------

def fig_loss():
    print("Training loss")
    from matplotlib.ticker import ScalarFormatter, NullFormatter

    d = pd.read_csv(need("documents/training_loss.csv"))
    c_step = pick(d, "_step", "step")
    c_loss = pick(d, "train/losses_after_rm_padding", "train_loss", "loss")

    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    last = {}
    for run, g in d.groupby("run"):
        g = g.sort_values(c_step)
        # 49 logged points; W&B already subsampled, so smooth lightly or not at all.
        y = g[c_loss].rolling(3, min_periods=1).mean()
        hi = "randomized" in str(run).lower()
        ax.plot(g[c_step], y, lw=1.6 if hi else 0.9,
                color=ORANGE if hi else NEUTRAL,
                alpha=1.0 if hi else 0.75, zorder=4 if hi else 3)
        last[run] = float(g[c_loss].iloc[-1])
        
    # Log y: the runs separate by a factor the linear scale hides.
    ax.set_yscale("log")
    ax.set_ylim(0.030, 0.40)
    ax.set_yticks([0.04, 0.06, 0.1, 0.2, 0.3])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)

    lo = min(v for k, v in last.items() if "randomized" in k.lower())
    hi_ = max(last.values())
    xr = float(d[c_step].max())
    ax.annotate(f"{lo:.3f}", (xr, lo), xytext=(4, -1), textcoords="offset points",
                fontsize=6.5, color=ORANGE, va="center")
    ax.annotate(f"{hi_:.3f}", (xr, hi_), xytext=(4, 1), textcoords="offset points",
                fontsize=6.5, color=MUTED, va="center")
    ax.set_xlim(0, xr * 1.14)

    ax.legend(handles=[
        Line2D([], [], color=ORANGE, lw=1.6, label="Randomized (both seeds)"),
        Line2D([], [], color=NEUTRAL, lw=0.9, label="Other runs"),
    ], loc="upper right", frameon=False, handletextpad=0.5)
    title(ax, "Randomized fits best and performs worst")
    save(fig, "fig_loss")

# ----------------------------------------------------------------------------
# Fig 9  By condition
# ----------------------------------------------------------------------------

def fig_by_condition():
    """Four metrics per condition on one row each: success, execution, aim, training loss."""
    print("By condition")
    CONDS = ["clean", "color", "recovery", "randomized"]
    LABEL = {"clean": "Clean", "color": "Color",
             "recovery": "Recovery", "randomized": "Randomized"}
    SAME = ["in_distribution", "reduced_lighting", "different_object", "distractors"]
    REG = SAME + ["new_positions"]

    def policy(cond, seed):
        return cond if seed == 1000 else f"{cond}-seed{seed}"

    # 1. success rate per condition per seed
    sv = pd.read_csv(need("analysis/out_seed_variance/seed_variance.csv"))
    sv = sv[sv.condition != "ALL"].set_index("condition")

    # 2. never-departed fraction, registered cells only so every denominator is 75.
    #    rollout_motion_by_condition pools in the displacement probe, which gives Clean 91.
    me = pd.read_csv(need("analysis/out_motion/rollout_motion_episodes.csv"))
    me = me[me.cell.isin(REG) & me.condition.isin(CONDS)]
    nd = me.groupby(["condition", "seed"]).apply(
        lambda g: float((~g.departed.astype(bool)).mean()), include_groups=False)

    # 3. spread of cell medians across the same-target cells
    aim = pd.read_csv(need("analysis/out_azimuth/aim_by_cell.csv"), index_col=0)
    same = [c for c in SAME if c in aim.columns]
    spread = aim[same].max(axis=1) - aim[same].min(axis=1)

    # 4. final training loss, 3-point smoothed to match fig_loss
    tl = pd.read_csv(need("documents/training_loss.csv"))
    c_step = pick(tl, "_step", "step")
    c_loss = pick(tl, "train/losses_after_rm_padding", "train_loss", "loss")
    last = (tl.sort_values(c_step).groupby("run")[c_loss]
              .apply(lambda s: s.rolling(3, min_periods=1).mean().iloc[-1]))

    rows = []
    for cond in CONDS:
        for seed in (1000, 2000):
            pol = policy(cond, seed)
            rows.append({
                "condition": cond,
                "seed": seed,
                "success": float(sv.loc[cond, f"rate{seed}"]),
                "no_depart": float(nd.get((cond, seed), np.nan)),
                "aim": float(spread.get(pol, np.nan)),
                "loss": float(last.get(pol, np.nan)),
            })
    d = pd.DataFrame(rows)

    PANELS = [
        ("success",   "Success rate",                  (0, 1.0)),
        ("no_depart", "Never departed",                (-0.02, 0.30)),
        ("aim",       "Aim spread across cells (deg)", (-0.4, 9.0)),
        ("loss",      "Final training loss",           (0.030, 0.060)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(5.5, 2.05), sharey=True)
    ypos = {c: len(CONDS) - 1 - i for i, c in enumerate(CONDS)}   # Clean at the top

    for ax, (col, xlabel, xlim) in zip(axes, PANELS):
        # thin connector so the two seeds read as one condition
        for cond in CONDS:
            g = d[d.condition == cond]
            hi = cond == "randomized"
            ax.plot(g[col], [ypos[cond]] * len(g), lw=1.0,
                    color=ORANGE if hi else NEUTRAL,
                    alpha=1.0 if hi else 0.45, zorder=2)
        for _, r in d.iterrows():
            hi = r.condition == "randomized"
            ax.plot(r[col], ypos[r.condition],
                    marker="o" if r.seed == 1000 else "D",
                    ms=4.5 if hi else 3.6,
                    mfc=ORANGE if hi else "white",
                    mec=ORANGE if hi else NEUTRAL,
                    mew=1.1, ls="none",
                    zorder=4 if hi else 3)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel, fontsize=6.5, labelpad=2)
        ax.tick_params(axis="x", labelsize=6, pad=1)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", lw=0.4, alpha=0.25)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylim(-0.6, len(CONDS) + 0.15)
    axes[0].set_yticks([ypos[c] for c in CONDS])
    axes[0].set_yticklabels([LABEL[c] for c in CONDS], fontsize=7)
    axes[0].tick_params(axis="y", length=0, pad=2)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=3.6, mfc="white",
               mec=NEUTRAL, mew=1.1, label="Seed 1000"),
        Line2D([], [], marker="D", ls="none", ms=3.6, mfc="white",
               mec=NEUTRAL, mew=1.1, label="Seed 2000"),
    ]
    axes[-1].legend(handles=handles, loc="upper right", fontsize=6,
                    frameon=False, handletextpad=0.4, labelspacing=0.3,
                    borderpad=0.1, borderaxespad=0.2)

    if SHOW_TITLES:
        fig.suptitle("Randomized: worst success, most no-motion, most aim drift, lowest loss",
                     fontsize=8)

    fig.tight_layout(pad=0.4, w_pad=0.9)
    save(fig, "fig_by_condition")

# ----------------------------------------------------------------------------
# Fig 10  Bearing selection
# ----------------------------------------------------------------------------

def fig_bearing_selection():
    """Commanded bearing against required bearing at the five held-out positions."""
    print("Bearing selection")
    sn = pd.read_csv(need("analysis/out_azimuth/settled_new_positions.csv"))

    BASE_X = 11.0
    bear = lambda x, y: np.degrees(np.arctan2(x - BASE_X, y))
    T6, T2 = bear(15.5, 10.0), bear(6.5, 7.5)

    PANELS = [
        ("Randomized, 10 trained positions", ["randomized", "randomized-seed2000"], ORANGE),
        ("Density, 2 trained positions",     ["density"],                           BLUE),
    ]
    LO, HI = -78, 55

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6), sharex=True, sharey=True)

    for ax, (title, pols, color) in zip(axes, PANELS):
        g = sn[sn.policy.isin(pols)]
        ax.plot([LO, HI], [LO, HI], lw=0.8, color=MUTED, alpha=0.55, zorder=1)
        if pols == ["density"]:
            for y, lab in ((T6, "T6"), (T2, "T2")):
                ax.axhline(y, lw=0.8, ls=(0, (3, 2)), color=color, alpha=0.7, zorder=2)
                ax.text(HI - 1, y + 1.8, f"trained {lab}", fontsize=6,
                        color=color, ha="right", va="bottom")
        ax.plot(g.true_az, g.settled_az, marker="o", ls="none", ms=4,
                mfc="white", mec=color, mew=1.1, zorder=3)
        ax.set_title(title, fontsize=7.5, pad=4)
        ax.set_xlabel("Required bearing (deg)", fontsize=6.5, labelpad=2)
        ax.set_xlim(LO, HI)
        ax.set_ylim(LO, HI)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=6, pad=1)
        ax.grid(lw=0.4, alpha=0.2)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("Commanded bearing (deg)", fontsize=6.5, labelpad=2)
    axes[0].text(-24, -23, "identity", fontsize=6, color=MUTED, rotation=45,
                 rotation_mode="anchor")

    if SHOW_TITLES:
        fig.suptitle("Randomized tracks the target; Density selects between two trained bearings",
                     fontsize=8)

    fig.tight_layout(pad=0.4, w_pad=0.8)
    save(fig, "fig_bearing_selection")


if __name__ == "__main__":
    table1()
    table2()
    table_loss()
    fig_positions()
    fig_aim_invariance()
    fig_aiming_error()
    fig_probe()
    fig_envelope()
    fig_release()
    fig_success_sweep()
    fig_loss()
    fig_by_condition()
    fig_bearing_selection()
    print(f"\nall outputs in {OUT}/")