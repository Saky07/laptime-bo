"""
Step 4: Multi-Fidelity Bayesian Optimization (THE contribution).

Idea, in plain terms:
  We have two ways to score a car setting:
     - CHEAP  simulator: fast, slightly wrong, costs 1 unit  (lap_time_cheap)
     - ACCURATE simulator: slow/costly, correct, costs 10 units (lap_time)
  Standard BO only ever uses the accurate one, so it burns expensive runs.
  Multi-fidelity BO uses the cheap one to SCOUT broadly and only spends an
  accurate run to CONFIRM the most promising candidates -- reaching a good
  setup for far less total COST.

How it works here (a simple, honest multi-fidelity scheme):
  1. Keep a GP fit on the CHEAP data -- it's plentiful, so it models the
     landscape's SHAPE well (we proved cheap correlates with accurate, ~0.92).
  2. Each round, use the cheap GP to propose the most promising setting.
  3. Cheaply evaluate several such candidates; only the single best-looking one
     is "confirmed" with an expensive accurate evaluation.
  4. Track total COST (cheap=1, accurate=10) and the best ACCURATE lap time so far.

Returns two arrays: cumulative cost, and best accurate lap time at each accurate run.
"""

import numpy as np

from simulator import lap_time, lap_time_cheap, N_PARAMS
from optimizers import fit_gp

COST_CHEAP = 1.0
COST_ACCURATE = 10.0


def multi_fidelity_bo(rng, n_init=25, n_confirm=14,
                      scouts_per_round=15, n_candidates=1000):
    """Cost-aware multi-fidelity BO.

    The key to a COST win: spend only a SMALL number of accurate (expensive)
    confirmations, and let plentiful CHEAP scouting find where to spend them.

    n_init          : initial cheap samples to seed the GP (cheap is plentiful).
    n_confirm       : number of accurate (expensive) confirmations -- keep small.
    scouts_per_round: cheap evaluations per round to refine the cheap GP.
    """
    # --- seed generously with cheap samples ---
    Xc = rng.random((n_init, N_PARAMS))
    yc = np.array([lap_time_cheap(x) for x in Xc])
    cost = n_init * COST_CHEAP

    confirmed = set()                      # avoid confirming the same point twice
    cost_curve, best_curve = [], []
    best_accurate = np.inf

    for _ in range(n_confirm):
        # 1) fit GP on cheap data -> models the landscape shape cheaply
        gp = fit_gp(Xc, yc)

        # 2) scout: cheaply evaluate promising candidates to refine the GP
        cand = rng.random((n_candidates, N_PARAMS))
        mu, sd = gp.predict(cand, return_std=True)
        order = np.argsort(mu - 2.0 * sd)
        scouts = cand[order[:scouts_per_round]]
        for x in scouts:
            Xc = np.vstack([Xc, x])
            yc = np.append(yc, lap_time_cheap(x))
        cost += scouts_per_round * COST_CHEAP

        # 3) confirm ONE accurate run on the cheap GP's current best prediction
        gp = fit_gp(Xc, yc)
        mu = gp.predict(cand)
        for idx in np.argsort(mu):         # best predicted first, skip duplicates
            key = tuple(np.round(cand[idx], 4))
            if key not in confirmed:
                best_pred = cand[idx]
                confirmed.add(key)
                break
        t_acc = lap_time(best_pred)
        cost += COST_ACCURATE
        best_accurate = min(best_accurate, t_acc)

        cost_curve.append(cost)
        best_curve.append(best_accurate)

    return np.array(cost_curve), np.array(best_curve)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cost, best = multi_fidelity_bo(rng)
    print(f"Multi-fidelity BO: reached {best[-1]:.2f} s "
          f"using total cost {cost[-1]:.0f} "
          f"({len(best)} accurate runs + cheap scouting).")
