"""
Optimizers used in this project.

All of them search the SAME thing: the vector of car settings (N_PARAMS knobs,
each in [0,1]) that minimizes lap time. They differ in HOW they choose what to try.

  random_search       -- the dumb floor: try random settings, keep the best.
  single_fidelity_bo  -- standard Bayesian optimization (the strong baseline).
  (multi-fidelity BO lives in mfbo.py -- that's the contribution.)

The optimizers read the number of knobs from the simulator automatically, so they
never hard-code the dimension.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from simulator import N_PARAMS


def fit_gp(X, y):
    """Fit a Gaussian Process surrogate to the runs done so far."""
    kernel = ConstantKernel(1.0) * RBF(length_scale=0.5,
                                       length_scale_bounds=(0.02, 10.0))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-4,
                                  normalize_y=True, n_restarts_optimizer=3)
    gp.fit(X, y)
    return gp


def random_search(evaluate, rng, n_total=40):
    """Try n_total random settings; return best-lap-time-so-far after each."""
    X = rng.random((n_total, N_PARAMS))
    y = np.array([evaluate(x) for x in X])
    return np.minimum.accumulate(y)


def single_fidelity_bo(evaluate, rng, n_init=5, n_iter=35, n_candidates=800):
    """Standard Bayesian optimization on ONE (expensive) evaluator.

    evaluate: a function setting -> lap time (here, the accurate simulator).
    Returns best-lap-time-so-far after each evaluation.
    """
    X = rng.random((n_init, N_PARAMS))
    y = np.array([evaluate(x) for x in X])
    best = [y.min()]

    for _ in range(n_iter):
        gp = fit_gp(X, y)
        # propose candidates, score with Lower Confidence Bound (explore + exploit)
        cand = rng.random((n_candidates, N_PARAMS))
        mu, sd = gp.predict(cand, return_std=True)
        next_x = cand[np.argmin(mu - 2.0 * sd)]

        X = np.vstack([X, next_x])
        y = np.append(y, evaluate(next_x))
        best.append(y.min())

    return np.array(best)
