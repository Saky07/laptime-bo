"""
Step 2: Is the CHEAP simulator actually related to the ACCURATE one?

Multi-fidelity Bayesian optimization only helps if the cheap, biased evaluation
still POINTS THE SAME DIRECTION as the expensive, accurate one. If a setting that
looks fast in the cheap sim tends to be fast in the accurate sim, we can trust the
cheap sim to scout. This script measures that agreement two ways:

  - Pearson correlation : do the actual lap-time VALUES track each other?
  - Spearman correlation: do the RANKINGS agree? (this is what matters most --
    we care that cheap ORDERS settings the same way, even if it's biased.)

It also saves a scatter plot so you can see the relationship.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from simulator import lap_time, lap_time_cheap, random_settings

rng = np.random.default_rng(42)

# sample a bunch of random settings and evaluate BOTH fidelities
N = 200
X = random_settings(rng, N)
accurate = np.array([lap_time(x) for x in X])
cheap = np.array([lap_time_cheap(x) for x in X])

pearson = pearsonr(cheap, accurate)[0]
spearman = spearmanr(cheap, accurate)[0]

print(f"Sampled {N} random settings.")
print(f"Pearson correlation  (values agree):   {pearson:.3f}")
print(f"Spearman correlation (rankings agree): {spearman:.3f}")
print(f"Mean bias (accurate - cheap):          {np.mean(accurate - cheap):.2f} s")
print()
if spearman > 0.9:
    print("=> Rankings agree strongly. The cheap sim is a trustworthy scout.")
    print("   Multi-fidelity BO is well justified.")
elif spearman > 0.7:
    print("=> Decent agreement. Multi-fidelity should help, with some caution.")
else:
    print("=> Weak agreement. Multi-fidelity may NOT help -- rethink the cheap sim.")

# scatter plot: each dot is one setting (cheap time vs accurate time)
plt.figure(figsize=(6.5, 6))
plt.scatter(cheap, accurate, s=18, alpha=0.6)
lo = min(cheap.min(), accurate.min())
hi = max(cheap.max(), accurate.max())
plt.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="perfect agreement")
plt.xlabel("CHEAP simulator lap time (s)")
plt.ylabel("ACCURATE simulator lap time (s)")
plt.title(f"Cheap vs accurate fidelity\nPearson={pearson:.2f}  Spearman={spearman:.2f}")
plt.legend()
plt.tight_layout()
plt.savefig("results/fidelity_correlation.png", dpi=130)
print("\nsaved results/fidelity_correlation.png")
