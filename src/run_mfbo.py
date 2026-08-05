"""
Step 4 + 6: the headline experiment.

Compare, on the SAME cost axis:
   - Single-fidelity BO : every evaluation is accurate (cost 10 each).
   - Multi-fidelity BO  : cheap scouting (cost 1) + rare accurate runs (cost 10).

X-axis is ACCUMULATED COST (not number of runs) -- because the whole claim is
"reach a good setup for less total cost". Averaged over seeds with spread bands.
Saves results/mfbo_vs_bo_cost.png.
"""

import numpy as np
import matplotlib.pyplot as plt

from simulator import lap_time
from optimizers import single_fidelity_bo
from mfbo import multi_fidelity_bo, COST_ACCURATE

N_SEEDS = 10


def bo_on_cost_axis(rng, n_init=5, n_iter=35):
    """Run single-fidelity BO and express its progress vs accumulated cost."""
    best = single_fidelity_bo(lap_time, rng, n_init=n_init, n_iter=n_iter)
    n_evals = n_init + np.arange(len(best))          # accurate evals used so far
    cost = n_evals * COST_ACCURATE
    return cost, best


# --- run both methods over several seeds ---
bo_best, mf_best = [], []
bo_cost = mf_cost = None
for seed in range(N_SEEDS):
    c_bo, b_bo = bo_on_cost_axis(np.random.default_rng(seed))
    c_mf, b_mf = multi_fidelity_bo(np.random.default_rng(seed))
    bo_cost, mf_cost = c_bo, c_mf
    bo_best.append(b_bo)
    mf_best.append(b_mf)

bo_best = np.array(bo_best)
mf_best = np.array(mf_best)

# --- report the headline number: cost to reach a target lap time ---
TARGET = 100.0
def cost_to_reach(cost_axis, best_curves, target):
    costs = []
    for curve in best_curves:
        hit = np.where(curve <= target)[0]
        costs.append(cost_axis[hit[0]] if len(hit) else np.nan)
    return np.nanmean(costs)

bo_c = cost_to_reach(bo_cost, bo_best, TARGET)
mf_c = cost_to_reach(mf_cost, mf_best, TARGET)
print(f"Mean cost to reach {TARGET:.0f} s lap:")
print(f"  single-fidelity BO : {bo_c:.0f}")
print(f"  multi-fidelity BO  : {mf_c:.0f}")
if not np.isnan(bo_c) and not np.isnan(mf_c):
    print(f"  => multi-fidelity used {100*(1-mf_c/bo_c):.0f}% less cost.")

# --- plot ---
plt.figure(figsize=(8, 5))
plt.plot(bo_cost, bo_best.mean(0), "g-o", ms=3, label="Single-fidelity BO")
plt.plot(mf_cost, mf_best.mean(0), "b-s", ms=3, label="Multi-fidelity BO (ours)")
plt.fill_between(bo_cost, bo_best.mean(0) - bo_best.std(0),
                 bo_best.mean(0) + bo_best.std(0), color="g", alpha=0.15)
plt.fill_between(mf_cost, mf_best.mean(0) - mf_best.std(0),
                 mf_best.mean(0) + mf_best.std(0), color="b", alpha=0.15)
plt.axhline(TARGET, color="k", ls="--", alpha=0.4, label=f"{TARGET:.0f}s target")
plt.xlabel("Accumulated evaluation cost  (cheap = 1, accurate = 10)")
plt.ylabel("Best accurate lap time so far (s)")
plt.title(f"Multi-fidelity reaches a good setup for less cost\n(mean ± std over {N_SEEDS} seeds, real Monza, 8 knobs)")
plt.legend()
plt.tight_layout()
plt.savefig("results/mfbo_vs_bo_cost.png", dpi=130)
print("saved results/mfbo_vs_bo_cost.png")
