#!/usr/bin/env python3
"""tools/azimuth_analysis.py — aiming error in calibrated azimuth."""

import glob
import os
import sys

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibrate_pose import azimuth_fit, azimuth, BASE_X
from rollout_paths import resolve, parse_policy

# ------------------------ Set up ---------------------------------------

OUTDIR = "analysis/out_azimuth"

POS = {"E1": (2.0, 7.5), "E2": (6.5, 2.5), "E3": (12.0, 10.0), "E4": (15.5, 6.5), "E5": (19.5, 13.5)}
CELL_POS = {"in_distribution": (15.5, 10.0), "near_1in": (15.5, 9.0), "near_2in": (15.5, 8.0)}
INTERP = {"E2", "E3", "E4"}
CLEAN = ("clean", "clean-seed2000")

def rollout_actions(policy, cell):
    # Episode indices and the action array for the newest rollout of this pair
    root = resolve(policy, cell)
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df.episode_index != 0].sort_values(["episode_index", "frame_index"], kind="stable")
    return df.episode_index.to_numpy(), np.stack(df["action"].to_numpy()).astype(float)

os.makedirs(OUTDIR, exist_ok=True)

# ---------------------- Calibrate -------------------------------------

fit, X, az_true = azimuth_fit()
pan = X[:, [0]]
to_az = lambda p: fit.predict(np.asarray(p, float).reshape(-1, 1))

loo = cross_val_predict(LinearRegression(), pan, az_true, cv=LeaveOneOut())
resid = np.abs(loo - az_true)
r2 = 1 - ((loo - az_true) ** 2).sum() / ((az_true - az_true.mean()) ** 2).sum()

pd.DataFrame([{
    "slope_deg_per_unit": fit.coef_[0],
    "intercept_deg": fit.intercept_,
    "n": len(X),
    "loo_median_abs_err_deg": float(np.median(resid)),
    "loo_p90_abs_err_deg": float(np.percentile(resid, 90)),
    "loo_r2": float(r2),
}]).round(4).to_csv(os.path.join(OUTDIR, "calibration.csv"), index=False)

print("--- Calibration ---")
print(f"Azimuth = {fit.coef_[0]:.4f} * pan + {fit.intercept_:.3f} (n={len(X)})")
print(f"leave-one-out | error |: median {np.median(resid):.2f} deg,",
      f"90th percentile {np.percentile(resid, 90):.2f} deg, R2 {r2:.4f}")

ends = []
for path in sorted(glob.glob("analysis/out_endpoints/endpoints_*.csv")):
    df = pd.read_csv(path)
    df["policy"] = os.path.basename(path)[len("endpoints_"):-len(".csv")]
    df["az"] = to_az(df["pan"].to_numpy())
    ends.append(df)
ends = pd.concat(ends, ignore_index=True)

v = ends[ends.policy.isin(CLEAN) & (ends.cell == "in_distribution")]
print(f"\nvalidation: clean grasp azimuth at T6, true value {azimuth(15.5, 10.0):.2f} deg")
print(v.groupby("policy")["az"].agg(n="size", median="median").round(2).to_string())

# --------------------- Clean Trajectory Envelope -----------------------
# The grasp event is not comparable across these cells: the arm closes at frame ~215
# in distribution and at ~470 when the cube is displaced, because it reaches the location,
# doesn't find the cube, hovers, then closes late, so we're comparing trajectories and not
# grasp events

print("\n--- Clean trajectory envelope: furthest azimuth reached ---")
rows, store = [], {}
for pol in CLEAN:
    for cell, xy in CELL_POS.items():
        req = azimuth(*xy)
        ep, A = rollout_actions(pol, cell)
        mx = pd.Series(to_az(A[:, 0]), index=ep).groupby(level=0).max()
        store[(pol, cell)] = mx
        rows.append({
            "policy": pol,
            "cell": cell,
            "n": len(mx),
            "median_max_az": mx.median(),
            "q1": mx.quantile(0.25),
            "q3": mx.quantile(0.75),
            "required_az": req,
            "episodes_reaching_target": int((mx >= req).sum())
        })

env = pd.DataFrame(rows).round(2)
print(env.to_string(index=False))
env.to_csv(os.path.join(OUTDIR, "clean_envelope.csv"), index=False)

print()
for pol in CLEAN:
    a, b = store[(pol, "in_distribution")], store[(pol, "near_2in")]
    u = mannwhitneyu(b, a, alternative="greater")
    print(f"{pol}: near_2in vs in_distribution max azimuth, "
          f"median difference {b.median() - a.median():+.2f} deg"
          f" (required {azimuth(*CELL_POS['near_2in']) - azimuth(*CELL_POS['in_distribution']):+.2f}), "
          f"U={u.statistic:.0f}, p={u.pvalue:.3f}")

# ----------------------- Randomized Aiming ----------------------------

r = ends[ends.policy.str.startswith("randomized") & (ends.cell == "new_positions")].copy()
n_all = len(r)
r = r[r.instance.isin(POS)]
if len(r) < n_all:
    print(f"dropped {n_all - len(r)} new_positions rows with no recognized instance: "
          f"{sorted(set(ends.loc[ends.cell == 'new_positions', 'instance']) - set(POS))}")
    
r["true_az"] = [azimuth(*POS[i]) for i in r.instance]
r["err"] = r["az"] - r["true_az"]
r["split"] = np.where(r.instance.isin(INTERP), "interpolation", "extrapolation")
r["dist_in"] = [np.hypot(POS[i][0] - BASE_X, POS[i][1]) for i in r.instance]
r["cube_widths"] = r.err.abs() / np.degrees(2 * np.arctan(0.5 / r.dist_in))


print("\n--- Randomized Aiming Error at Held-out Positions ---")
print(r[["policy", "episode", "instance", "split", "az", "true_az", "err"]].round(1).to_string(index=False))
print("\n" + r.groupby("split")["err"].agg(
    n="size", mean_abs=lambda s: s.abs().mean(), median_abs=lambda s: s.abs().median()).round(2).to_string())

i = r.loc[r.split == "interpolation", "err"].abs()
e = r.loc[r.split == "extrapolation", "err"].abs()

print("\nMann-Whitney (extrapolation > interpolation):", mannwhitneyu(e, i, alternative="greater"))
print("extrapolation signed errors:", r.loc[r.split == "extrapolation", "err"].round(1).to_list())

print(r.groupby("split")["cube_widths"].agg(n="size", mean="mean", median="median").round(2))

r.round(2).to_csv(os.path.join(OUTDIR, "randomized_aiming.csv"), index=False)

# ------------------------- Aim Invariance Across Cells ----------------------

# Absolute pose is not comparable across policies as clean grasps with lift ~9 and elbow ~25, but recovery with lift ~25 and elbow ~8,
# This is a difference in teleoperated grasp style rather than aiming, as such, everything here is within a policy across cells

# New positions is excluded from IQR comparisons because the cube is at five difference places, and as such the policy will
# have a large spread
SAME_TARGET = ["in_distribution", "reduced_lighting", "different_object", "distractors"]
REGISTERED = SAME_TARGET + ["new_positions"]

med = ends.pivot_table(index="policy", columns="cell", values="az", aggfunc="median")
iqr = ends.pivot_table(index="policy", columns="cell", values="az", aggfunc=lambda s: s.quantile(0.75) - s.quantile(0.25))
same = [c for c in SAME_TARGET if c in med.columns]
allc = [c for c in REGISTERED if c in med.columns]

print(f"\n--- Commanded Azimuth by Cell (cube at T6 = {azimuth(15.5, 10.0):.2f} except new_positions) ---")
print(med[allc].round(1).to_string())

print("\nspread of cell medians within each policy, same-target cells only (deg)")
print((med[same].max(axis=1) - med[same].min(axis=1)).round(1).to_string())

print("\nsame, including new_positions (deg)")
print((med[allc].max(axis=1) - med[allc].min(axis=1)).round(1).to_string())

print("\nIQR width of commanded azimuth, same-target cells only (deg)")
print(iqr[same].round(1).to_string())

med[allc].round(3).to_csv(os.path.join(OUTDIR, "aim_by_cell.csv"))
iqr[allc].round(3).to_csv(os.path.join(OUTDIR, "aim_iqr_by_cell.csv"))

# ------------------------- Density Probe (PROTOCOL.md §8.30) ----------------------
# Amendment 30 chose T2 because it sits on the opposite side of the arm from T6, so one
# fixed sweep cannot succeed at both. Two questions: does it aim correctly at each
# trained position, and what does it do at a position it never saw.

T6_AZ_D, T2_AZ_D = azimuth(15.5, 10.0), azimuth(6.5, 7.5)
DENS_POS = {"in_distribution": (15.5, 10.0), "trained_t2": (6.5, 7.5)}

d_ends = ends[ends.policy == "density"]
if len(d_ends):
    rows = []
    for cell, xy in DENS_POS.items():
        g = d_ends[d_ends.cell == cell]
        if not len(g):
            continue
        true_az = azimuth(*xy)
        rows.append({"cell": cell, "n": len(g), "true_az": true_az,
                     "median_az": g.az.median(), "err": g.az.median() - true_az})
    if rows:
        dens = pd.DataFrame(rows).round(2)
        print("\n--- Density: grasp azimuth at the two trained positions ---")
        print(dens.to_string(index=False))
        dens.to_csv(os.path.join(OUTDIR, "density_aim.csv"), index=False)

# ---- Settled bearing at the held-out positions, same measure for every policy ----
# Almost nothing grasps at a held-out position, so the endpoint table is too thin to
# compare policies. Use the bearing the arm settles on instead: the median commanded
# azimuth over the last quarter of the episode, after it has committed to a direction.

LAB = pd.concat([pd.read_csv("documents/results_full.csv"),
                 pd.read_csv("documents/exploratory.csv")], ignore_index=True)

def settled_new_positions(policy):
    ep_idx, A = rollout_actions(policy, "new_positions")
    s = pd.Series(to_az(A[:, 0]), index=ep_idx)
    s.index.name = "episode"
    settled = s.groupby(level=0).apply(lambda g: g.iloc[int(len(g) * 0.75):].median())
    cond, seed = parse_policy(policy)
    inst = (LAB[(LAB.condition == cond) & (LAB.seed == seed)
                & (LAB.eval_cell == "new_positions")]
            .set_index("episode")["instance"])
    d = pd.DataFrame({"settled_az": settled}).join(inst)
    d["policy"] = policy
    d["true_az"] = [azimuth(*POS[i]) if i in POS else np.nan for i in d.instance]
    return d.reset_index()

frames = []
for pol in ["randomized", "randomized-seed2000", "density"]:
    try:
        frames.append(settled_new_positions(pol))
    except FileNotFoundError:
        print(f" no new_positions rollout for {pol}, skipped")

if frames:
    sn = pd.concat(frames, ignore_index=True)
    sn["err"] = sn.settled_az - sn.true_az
    sn["to_T6"] = (sn.settled_az - T6_AZ_D).abs()
    sn["to_T2"] = (sn.settled_az - T2_AZ_D).abs()
    sn["nearest_trained"] = np.where(sn.to_T6 <= sn.to_T2, "T6", "T2")
    sn.round(2).to_csv(os.path.join(OUTDIR, "settled_new_positions.csv"), index=False)

    print(f"\n--- Settled bearing at held-out positions "
          f"(density trained at T6 {T6_AZ_D:.1f} deg and T2 {T2_AZ_D:.1f} deg) ---")
    print(sn.groupby(["policy", "instance"]).agg(
        n=("settled_az", "size"),
        settled=("settled_az", "median"),
        true=("true_az", "first"),
        miss_vs_true=("err", lambda s: s.abs().median()),
        picked=("nearest_trained", lambda s: s.mode().iat[0]),
    ).round(1).to_string())

print(f"\nSaved to {OUTDIR}/")