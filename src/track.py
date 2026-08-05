"""
Load a REAL racetrack from a TUM-format CSV and turn it into the one thing the
lap-time simulator needs: curvature along the track.

CSV format (TUM racetrack-database):
    # x_m, y_m, w_tr_right_m, w_tr_left_m
The first two columns are the centerline (x, y) in metres.

Curvature at a point measures how sharply the track bends there:
    straight  -> curvature ~ 0
    tight corner -> large curvature (= 1 / radius)
"""

import os
import numpy as np


def load_track(csv_path):
    """Return (x, y, s, curvature, total_length) for a track CSV."""
    data = np.loadtxt(csv_path, delimiter=",", comments="#")
    x, y = data[:, 0], data[:, 1]

    # arc length: distance travelled along the centerline (closed loop)
    dx_seg = np.diff(x, append=x[0])
    dy_seg = np.diff(y, append=y[0])
    seg_len = np.sqrt(dx_seg ** 2 + dy_seg ** 2)
    total = float(seg_len.sum())
    s = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])

    # curvature from the centerline, differentiated w.r.t. arc length s
    dx = np.gradient(x, s)
    dy = np.gradient(y, s)
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5

    # light smoothing to remove numerical spikes from the raw GPS-derived data
    k = 5
    kernel = np.ones(k) / k
    curvature = np.convolve(curvature, kernel, mode="same")

    return x, y, s, curvature, total


# Convenience: the default track shipped in this repo.
DEFAULT_TRACK = os.path.join(os.path.dirname(__file__), "..", "data", "Monza.csv")


if __name__ == "__main__":
    # Visualise the track, coloured by curvature, to confirm the data is real.
    import matplotlib.pyplot as plt

    x, y, s, curv, total = load_track(DEFAULT_TRACK)
    print(f"Loaded track: {len(x)} points, length {total:.0f} m, "
          f"max curvature {curv.max():.3f} (1/m)")

    plt.figure(figsize=(7, 8))
    sc = plt.scatter(x, y, c=curv, cmap="plasma", s=10)
    plt.colorbar(sc, label="curvature (1/m)  —  brighter = tighter corner")
    plt.axis("equal")
    plt.title("Monza centerline (real data), coloured by corner tightness")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "results", "track_monza.png")
    plt.savefig(out, dpi=130)
    print("saved", out)
