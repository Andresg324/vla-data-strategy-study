#!/usr/bin/env python3
"""
tools/export_results.py

Regenerate every derived results file from the master workbook. The workbook is
the only file that should ever be edited by hand; everything under documents/
that ends in .csv is produced here.

    documents/results_raw_two_seeds.xlsx
    ├── sheet "results"              -> results_full.csv     600 rows, all columns
    │                                -> results.csv          600 rows, 5 columns
    │                                -> results_seed1000.csv
    │                                -> results_seed2000.csv
    └── sheet "exploratory results"  -> exploratory.csv      107 rows
                                     -> results_all.csv      707 rows, both concatenated

Validation runs before anything is written. If a check fails nothing is
overwritten, so a bad hand edit cannot propagate.

RUN: python tools/export_results.py
"""

import os
import sys

import pandas as pd

XL = "documents/results_raw_two_seeds.xlsx"
OUT = "documents"
SHEET_REG = "results"
SHEET_EXP = "exploratory results"

FIVE = ["condition", "eval_cell", "seed", "episode", "success"]
CONDITIONS = {"clean", "randomized", "recovery", "color"}
CELLS = {"in_distribution", "new_positions", "reduced_lighting",
         "different_object", "distractors"}
SEEDS = {1000, 2000}

VOCAB = {"success", "success_after_missed_grasp", "success_after_drop",
         "no_departure", "contact_no_grasp", "grasp_drop", "deliberate_drop",
         "cube_out_of_bounds", "cup_knocked", "timeout_other"}
SUCCESS_LABELS = {"success", "success_after_missed_grasp", "success_after_drop"}

# PROTOCOL.md §8.21 renames
LABEL_RENAMES = {
    "success_after_recovery": "success_after_missed_grasp",
    "success_after_regrasp": "success_after_drop",
}

def load(sheet):
    df = pd.read_excel(XL, sheet_name=sheet)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    if "failure_mode" in df:
        n = int(df.failure_mode.isin(LABEL_RENAMES.keys()).sum())
        if n:
            print(f" renamed {n} legacy failure_mode labels in sheet '{sheet}' (per §8.21)")
        df["failure_mode"] = df["failure_mode"].replace(LABEL_RENAMES)
    return df


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def validate_registered(df):
    print("registered grid:")
    good = True

    good &= check("600 rows", len(df) == 600, f"got {len(df)}")
    good &= check("columns present", set(FIVE).issubset(df.columns),
                  f"missing {set(FIVE) - set(df.columns)}")

    counts = df.groupby(["condition", "eval_cell", "seed"]).size()
    good &= check("20 cells x 2 seeds at 15 episodes", (counts == 15).all() and len(counts) == 40,
                  f"{len(counts)} groups, sizes {sorted(counts.unique())}")

    good &= check("conditions", set(df.condition) == CONDITIONS, str(set(df.condition)))
    good &= check("cells", set(df.eval_cell) == CELLS, str(set(df.eval_cell)))
    good &= check("seeds", set(df.seed) == SEEDS, str(set(df.seed)))
    good &= check("success is 0/1", set(df.success.unique()) <= {0, 1},
                  str(set(df.success.unique())))
    good &= check("episodes 1-15", set(df.episode.unique()) == set(range(1, 16)),
                  str(sorted(set(df.episode.unique()))))
    dup = df[df.duplicated(["condition", "eval_cell", "seed", "episode"], keep=False)]

    good &= check("no duplicate (condition, cell, seed, episode)", len(dup) == 0, f"{len(dup)} rows")

    if len(dup):
        print(dup[["condition", "eval_cell", "seed", "episode"]].to_string(index=False))

    if "failure_mode" in df:
        bad = set(df.failure_mode.dropna()) - VOCAB
        good &= check("failure_mode vocabulary", not bad, f"unknown: {bad}")
        mism = df[(df.success == 1) != df.failure_mode.isin(SUCCESS_LABELS)]
        good &= check("success agrees with failure_mode", len(mism) == 0,
                      f"{len(mism)} disagreements")
        if len(mism):
            print(mism[["condition", "eval_cell", "seed", "episode",
                        "success", "failure_mode"]].to_string(index=False))

    if "instance" in df:
        np_rows = df[df.eval_cell == "new_positions"]
        good &= check("new_positions instances are E1-E5",
                      set(np_rows.instance) == {"E1", "E2", "E3", "E4", "E5"},
                      str(set(np_rows.instance)))
    return good


def validate_exploratory(df):
    print("exploratory:")
    good = True
    for k, g in df.groupby(["condition", "eval_cell", "seed"]):
        good &= check(f"episodes contiguous {k}", sorted(g.episode) == list(range(1, len(g) + 1)), str(sorted(g.episode)))
    good &= check("107 rows", len(df) == 107, f"got {len(df)}")
    counts = df.groupby(["condition", "eval_cell", "seed"]).size().to_dict()

    # NOTE: "color-slowpace".startswith("color") is True. Any downstream filter that
    # selects the Color condition by prefix will silently include the exploratory policy.
    # Match conditions exactly, not by prefix.
    expected = {("color-slowpace", "in_distribution", 1000): 15,
                ("color-slowpace", "different_object", 1000): 15,
                ("clean", "near_1in", 1000): 8, ("clean", "near_1in", 2000): 8,
                ("clean", "near_2in", 1000): 8, ("clean", "near_2in", 2000): 8,
                ("density", "in_distribution", 1000): 15,
                ("density", "trained_t2", 1000): 15,
                ("density", "new_positions", 1000): 15}
    good &= check("cell sizes", counts == expected, f"got {counts}")
    good &= check("success is 0/1", set(df.success.unique()) <= {0, 1},
                  str(set(df.success.unique())))
    if "failure_mode" in df:
        bad = set(df.failure_mode.dropna()) - VOCAB
        good &= check("failure_mode vocabulary", not bad, f"unknown: {bad}")
        mism = df[(df.success == 1) != df.failure_mode.isin(SUCCESS_LABELS)]
        good &= check("success agrees with failure_mode", len(mism) == 0,
                      f"{len(mism)} disagreements")
        if len(mism):
            print(mism[["condition", "eval_cell", "seed", "episode",
                        "success", "failure_mode"]].to_string(index=False))
            
    return good


def main():
    if not os.path.exists(XL):
        sys.exit(f"missing {XL}")

    reg, exp = load(SHEET_REG), load(SHEET_EXP)
    ok = validate_registered(reg)
    print()
    ok &= validate_exploratory(exp)

    if not ok:
        sys.exit("\nvalidation failed, nothing written")

    reg.to_csv(f"{OUT}/results_full.csv", index=False)
    reg[FIVE].to_csv(f"{OUT}/results.csv", index=False)
    for s in sorted(SEEDS):
        reg[reg.seed == s][FIVE].to_csv(f"{OUT}/results_seed{s}.csv", index=False)
    exp.to_csv(f"{OUT}/exploratory.csv", index=False)
    pd.concat([reg.assign(source="registered"), exp.assign(source="exploratory")], ignore_index=True).to_csv(f"{OUT}/results_all.csv", index=False)

    print("\nwrote results_full.csv, results.csv, results_seed1000.csv, "
          "results_seed2000.csv, exploratory.csv, results_all.csv")
    print(f"registered {len(reg)} rows, exploratory {len(exp)}, "
          f"combined {len(reg) + len(exp)}")


if __name__ == "__main__":
    main()