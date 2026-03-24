"""Plot thin-loop self-inductance versus piecewise-linear loop discretization."""

import os
from pathlib import Path

import numpy as np

if os.getenv("CFSEM_TESTING"):
    import matplotlib

    matplotlib.use("Agg")

from matplotlib import pyplot as plt

import cfsem

LOOP_RADIUS = 1.0  # [m]
WIRE_RADIUS = 0.01  # [m]
NSEG_SWEEP = np.unique(
    np.rint(np.geomspace(8.0, 1e4, 9 if os.getenv("CFSEM_TESTING") else 25)).astype(int)
)


def circle_polyline(radius: float, nseg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed polyline on a circle with `nseg` linear segments."""
    phi = np.linspace(0.0, 2.0 * np.pi, nseg + 1)
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    z = np.zeros_like(phi)
    return x, y, z


def polyline_to_segments(
    xyzp: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Convert a closed polyline point series into segment starts, deltas, and midpoints."""
    x, y, z = xyzp
    dlx = x[1:] - x[:-1]
    dly = y[1:] - y[:-1]
    dlz = z[1:] - z[:-1]
    xyzfil = (x[:-1], y[:-1], z[:-1])
    dlxyzfil = (dlx, dly, dlz)
    xyzmid = (
        x[:-1] + 0.5 * dlx,
        y[:-1] + 0.5 * dly,
        z[:-1] + 0.5 * dlz,
    )
    return xyzfil, dlxyzfil, xyzmid


def loop_inductance_from_vector_potential(
    xyzp: tuple[np.ndarray, np.ndarray, np.ndarray],
    wire_radius: float,
) -> float:
    """Integrate A·dl around the loop using one evaluation point per target segment."""
    xyzfil, dlxyzfil, xyzmid = polyline_to_segments(xyzp)
    ifil = np.ones_like(xyzfil[0], dtype=np.float64)
    ax, ay, az = cfsem.vector_potential_linear_filament(
        xyzmid,
        xyzfil,
        dlxyzfil,
        ifil,
        wire_radius=wire_radius,
    )
    return float(np.sum(ax * dlxyzfil[0] + ay * dlxyzfil[1] + az * dlxyzfil[2]))


inductance_direct = np.empty(NSEG_SWEEP.size, dtype=np.float64)
inductance_from_a = np.empty(NSEG_SWEEP.size, dtype=np.float64)
wien_inductance = float(cfsem.self_inductance_circular_ring_wien(LOOP_RADIUS, WIRE_RADIUS))

for i, n in enumerate(NSEG_SWEEP):
    xyz = circle_polyline(LOOP_RADIUS, int(n))
    inductance_direct[i] = cfsem.self_inductance_piecewise_linear_filaments(
        xyz,
        wire_radius=WIRE_RADIUS,
    )
    inductance_from_a[i] = loop_inductance_from_vector_potential(xyz, WIRE_RADIUS)
    print(
        f"N={int(n):5d}: self_inductance_piecewise_linear_filaments={inductance_direct[i]:.6e} H, "
        f"A·dl with {WIRE_RADIUS * 1e2:.1f} cm wire radius={inductance_from_a[i]:.6e} H"
    )

fig, ax = plt.subplots(figsize=(7.0, 4.5))

ax.semilogx(
    NSEG_SWEEP,
    inductance_direct * 1e6,
    color="black",
    marker=".",
    markersize=7,
    linewidth=1.2,
    label="Direct cfsem self_inductance_piecewise_linear_filaments, wire radius = 1 cm",
)
ax.semilogx(
    NSEG_SWEEP,
    inductance_from_a * 1e6,
    color="tab:blue",
    marker="o",
    markersize=4,
    linewidth=1.2,
    label="A·dl from vector_potential_linear_filament, wire radius = 1 cm",
)
ax.axhline(
    wien_inductance * 1e6,
    color="tab:red",
    linestyle="--",
    linewidth=1.2,
    label="Wien formula, wire radius = 1 cm",
)
ax.set_xlabel("Loop discretization count [-]")
ax.set_ylabel("Self-inductance [$\\mu$H]")
ax.set_title("Thin Loop Self-Inductance vs. Loop Discretization")
ax.set_ylim(0.0, 2.0 * wien_inductance * 1e6)
ax.grid(True, which="both", linestyle=":", linewidth=0.7)
ax.legend(loc="best")
fig.tight_layout()

fpath = Path(__file__).with_suffix(".png")
fig.savefig(fpath, dpi=300)
print(f"Saved plot to {fpath}")

if not os.getenv("CFSEM_TESTING"):
    plt.show()
