"""
Lap-time simulator on REAL track geometry, with a richer, physically meaningful
set of tunable knobs (the search space the optimizer explores).

A "setting" is a vector of N_PARAMS numbers, each in [0, 1]. See PARAM_NAMES.
Each knob maps to a real tuning decision and genuinely affects the lap time.

lap_time(settings)        -> accurate, expensive evaluation
lap_time_cheap(settings)  -> fast, slightly-wrong evaluation

Run directly to sanity-check:  python src/simulator.py
"""

import numpy as np
from track import load_track, DEFAULT_TRACK

# ---- The tunable knobs (search space) ----------------------------------
PARAM_NAMES = [
    "line_aggression",      # 0: straighter racing line through corners
    "grip_slow",            # 1: tyre grip used in slow (tight) corners
    "grip_fast",            # 2: tyre grip used in fast (open) corners
    "corner_threshold",     # 3: where "slow" vs "fast" corner is split
    "top_speed_cap",        # 4: top speed allowed on straights
    "brake_aggression",     # 5: how hard/late it brakes into corners
    "accel_aggression",     # 6: how hard it accelerates out of corners
    "entry_caution",        # 7: safety margin slowing corner entry (anti-spin)
]
N_PARAMS = len(PARAM_NAMES)

# Load the real track ONCE when this module is imported.
_X, _Y, _S, _CURV, _TOTAL = load_track(DEFAULT_TRACK)


def _simulate(settings, curvature, total, model_braking, mu):
    """Core lap-time calculation along a curvature profile."""
    p = np.clip(np.asarray(settings, dtype=float).ravel(), 0.0, 1.0)
    (line_aggression, grip_slow, grip_fast, corner_threshold,
     top_speed_cap, brake_aggression, accel_aggression, entry_caution) = p

    n = len(curvature)
    ds = total / n
    g = 9.81
    v_top = 90 * (0.5 + 0.5 * top_speed_cap)          # top speed cap (m/s)

    # a straighter racing line effectively opens up corners
    eff_curv = curvature * (1 - 0.45 * line_aggression)

    # split the track into "slow" (tight) and "fast" corners by curvature.
    # corner_threshold slides that boundary from 0.005 to 0.035 (1/m).
    threshold = 0.005 + 0.030 * corner_threshold
    grip_frac = np.where(eff_curv > threshold,
                         0.6 + 0.4 * grip_slow,       # tight corner
                         0.6 + 0.4 * grip_fast)       # fast corner / straight
    grip = mu * grip_frac

    # friction circle: max safe speed at each point
    v_corner = np.where(eff_curv > 1e-6,
                        np.sqrt(grip * g / np.maximum(eff_curv, 1e-6)),
                        v_top)
    # entry caution: shave a little speed off corners for safety (0.85..1.0)
    v_corner *= (0.85 + 0.15 * entry_caution)
    v = np.minimum(v_corner, v_top)

    if model_braking:
        # HIGH FIDELITY: finite braking + acceleration around the closed lap
        a_brake = 8.0 + 10.0 * brake_aggression
        a_accel = 5.0 + 7.0 * accel_aggression
        for _ in range(2):
            for i in range(n - 1, -1, -1):
                nxt = (i + 1) % n
                v[i] = min(v[i], np.sqrt(v[nxt] ** 2 + 2 * a_brake * ds))
            for i in range(n):
                prv = (i - 1) % n
                v[i] = min(v[i], np.sqrt(v[prv] ** 2 + 2 * a_accel * ds))

    v = np.maximum(v, 1.0)
    return float(np.sum(ds / v))


def lap_time(settings):
    """ACCURATE, EXPENSIVE evaluation (full resolution + braking model)."""
    return _simulate(settings, _CURV, _TOTAL, model_braking=True, mu=1.05)


def lap_time_cheap(settings):
    """CHEAP, slightly-WRONG evaluation: coarser track, no braking model,
    optimistic grip (the sim-to-real gap)."""
    coarse = _CURV[::6]
    return _simulate(settings, coarse, _TOTAL, model_braking=False, mu=1.12)


def random_settings(rng, n=1):
    """Helper: n random settings in the unit cube."""
    return rng.random((n, N_PARAMS))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"Track: {_TOTAL:.0f} m, {len(_CURV)} points, {N_PARAMS} tunable knobs\n")
    timid      = [0.0, 0.6, 0.6, 0.5, 0.5, 0.3, 0.3, 1.0]
    aggressive = [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.0]
    print("timid setting      -> accurate:", round(lap_time(timid), 2), "s")
    print("aggressive setting -> accurate:", round(lap_time(aggressive), 2), "s")
    print("aggressive setting -> cheap   :", round(lap_time_cheap(aggressive), 2), "s")
    print("\n5 random settings (accurate lap times):")
    for x in random_settings(rng, 5):
        print("  ", np.round(x, 2), "->", round(lap_time(x), 2), "s")
