"""
analysis/make_synthetic_results.py

Fake results.csv used to build and test analyze_results.py before any real data existed.
The numbers encode a hypothesis, not a result, and nothing here appears in the paper.

Kept for provenance: this file predates collection, which is why its cell names and seeds
are the pre-final ones (new_lighting rather than reduced_lighting, seed 0 rather than
1000/2000). Do not update.

Hypothesised rates were drafted with an LLM in July 2026.

RUN:
    python analysis/make_synthetic_results.py
    python analysis/analyze_results.py analysis/out/results_synthetic.csv --outdir analysis/out
"""

import os
import numpy as np
import pandas as pd

os.makedirs("analysis/out", exist_ok=True)

rng = np.random.default_rng(0) # seeded generator, makes it so every run gives the same synthetic data

conditions = ["clean", "randomized", "recovery", "color"]
cells = ["in_distribution", "new_positions", "new_lighting", "different_object", "distractors"]
seeds = [0]
episodes_per_cell = 15

# Hypothesized success probability per condition and cell; matched conditions expected to do better on their cell
# Numbers below are generated through Claude Opus 5, July 2026
p = {
    "clean":      {"in_distribution": .90, "new_positions": .20, "new_lighting": .40, "different_object": .30, "distractors": .45},
    "randomized": {"in_distribution": .85, "new_positions": .70, "new_lighting": .40, "different_object": .30, "distractors": .45},
    "recovery":   {"in_distribution": .88, "new_positions": .40, "new_lighting": .45, "different_object": .35, "distractors": .55},
    "color":      {"in_distribution": .85, "new_positions": .25, "new_lighting": .45, "different_object": .70, "distractors": .45},
}

rows = []
for c in conditions:
    for cell in cells:
        for s in seeds:
            for e in range(episodes_per_cell):
                success = int(rng.random() < p[c][cell]) # Success has probability p, filling in column with 1, else 0
                rows.append({"condition": c, "eval_cell": cell, "seed": s, "episode": e, "success": success})

pd.DataFrame(rows).to_csv("analysis/out/results_synthetic.csv", index=False)
print(f"wrote analysis/out/results_synthetic.csv ({len(rows)} rows)")