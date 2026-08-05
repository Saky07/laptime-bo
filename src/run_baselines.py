"""
Step 3: run the baselines on the real 8-knob Monza problem.

Compares standard (single-fidelity) Bayesian optimization against random search,
averaged over several seeds. This is the yardstick the multi-fidelity method
(Step 4) must beat. Saves results/baselines.png.
"""

import numpy as np
import matplotlib.pyplot as plt

from simulator import lap_time            # the accurate (expensive) evaluator
from optimizers import single_fidelity_bo, random_search

N_SEEDS = 15
N_EVALS = 40                              # total evaluations each method may use

bo_curves, rs_curves = [], []
for seed in range(N_SEEDS):
    rng = np.random.default_rng(seed)
    bo_curves.append(single_fidelity_bo(lap_time, rng, n_init=5, n_iter=N_EVALS - 5))
    rng = np.random.default_rng(1000 + seed)
    rs_curves.append(random_search(lap_time, rng, n_total=N_EVALS))

bo = np.array(bo_curves)                  # shape (seeds, evals)
rs = np.array(rs_curves)

x_bo = np.arange(5, N_EVALS + 1)          # BO curve starts after 5 initial runs
x_rs = np.arange(1, N_EVALS + 1)

print(f"After {N_EVALS} evaluations (mean over {N_SEEDS} seeds):")
print(f"  single-fidelity BO : {bo[:, -1].mean():.2f} s")
print(f"  random search      : {rs[:, -1].mean():.2f} s")

plt.figure(figsize=(8, 5))
plt.plot(x_bo, bo.mean(0), "g-o", ms=3, label="Single-fidelity BO")
plt.plot(x_rs, rs.mean(0), "r-", label="Random search")
plt.fill_between(x_bo, bo.mean(0) - bo.std(0), bo.mean(0) + bo.std(0), color="g", alpha=0.15)
plt.fill_between(x_rs, rs.mean(0) - rs.std(0), rs.mean(0) + rs.std(0), color="r", alpha=0.10)
plt.xlabel("Number of (expensive) evaluations")
plt.ylabel("Best lap time found so far (s)")
plt.title(f"Baselines on real Monza, 8 knobs (mean ± std over {N_SEEDS} seeds)")
plt.legend()
plt.tight_layout()
plt.savefig("results/baselines.png", dpi=130)
print("saved results/baselines.png")
