"""Calculate the B-field from a Helmholtz coil pair."""

import os
import numpy as np
from matplotlib import pyplot as plt
from cfsem import flux_density_circular_filament, MU_0

coil_radius = 0.2  # [m]

# The helmholtz coil pair comprised two circular coils,
# separated by a distance equal to their radius.
# In our coordinate system, z = 0 is the midpoint between the coils.
rfil = [coil_radius, coil_radius]  # [m]
zfil = [-0.5 * coil_radius, 0.5 * coil_radius]  # [m]

# [A] The current * turns of each coil.
ifil = [100.0, 100.0]

# Define a mesh on which to evaluate the B-field.
# Note that we cannot evaluate the B-field at exactly r = 0.
r = np.linspace(1e-9, 2.0 * coil_radius, 100)  # [m]
z = np.linspace(-1.5 * coil_radius, 1.5 * coil_radius, 100)  # [m]
rmesh, zmesh = np.meshgrid(r, z)
rmesh_flat = rmesh.flatten()
zmesh_flat = zmesh.flatten()

# Calculate the B-field at every mesh point using cfsem.
Br_flat, Bz_flat = flux_density_circular_filament(
    ifil, rfil, zfil, rmesh_flat, zmesh_flat
)


Br = Br_flat.reshape(rmesh.shape)
Bz = Bz_flat.reshape(rmesh.shape)

Bmag = np.sqrt(Br**2 + Bz**2)  # [T] Magnitude of the B-field.

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax_map, ax_center = axes
ax_map.set_aspect("equal")
# Plot magnetic field lines.
ax_map.streamplot(rmesh, zmesh, Br, Bz, color="black", linewidth=0.5)
# Make a color plot of the magnetic field magnitude.
dr = r[1] - r[0]
dz = z[1] - z[0]
extent = (r[0] - dr / 2, r[-1] + dr / 2, z[0] - dz / 2, z[-1] + dz / 2)
im = ax_map.imshow(
    Bmag, extent=extent, origin="lower", interpolation="bicubic", norm="log"
)
fig.colorbar(im, label="$|\\vec{B}|$ [T]", ax=ax_map)
# Outline the region where the B-field magnitude is within 1% of the center value.
B_center = (4 / 5) ** (3 / 2) * MU_0 * ifil[0] / coil_radius  # [T]
ax_map.contour(
    rmesh,
    zmesh,
    np.abs(Bmag - B_center),
    levels=[0.01 * B_center],
    colors=["red"],
    linestyles="dashed",
)
# Draw the location of the coils.
ax_map.plot(rfil, zfil, color="black", marker="o", markersize=8, linestyle="none")
ax_map.set_title("$(r, z)$ map of $\\vec{B}$")

ax_map.set_xlabel("$r$ [m]")
ax_map.set_ylabel("$z$ [m]")
ax_map.set_xlim(0.0, r[-1])
ax_map.set_ylim(z[0], z[-1])


# Plot the centerline B-field from cfsem versus the analytic solution.
# Analytic formula from https://en.wikipedia.org/wiki/Helmholtz_coil#Derivation
def xi(x):
    """Helper function for calculating the centerline B-field analytically."""
    return (1 + (x / coil_radius) ** 2) ** (-3 / 2)


Bmag_analytic = (
    MU_0 / (2 * coil_radius) * (ifil[0] * xi(z - zfil[0]) + ifil[1] * xi(z - zfil[1]))
)
ax_center.plot(
    z, Bz[:, 0], label="cfsem", marker="+", color="tab:blue", linestyle="none"
)
ax_center.plot(z, Bmag_analytic, label="analytic", color="black")
# Annotate the coil locations.
for i in (0, 1):
    ax_center.axvline(zfil[i], color="gray", linestyle="--")
    ax_center.text(
        s=f"coil {i + 1}",
        x=zfil[i],
        y=0.5 * np.max(Bmag_analytic),
        rotation=90,
        ha="right",
        va="center",
        color="grey",
    )
ax_center.axvline(zfil[1], color="gray", linestyle="--")
ax_center.set_xlabel("$z$ [m]")
ax_center.set_ylabel("$|\\vec{B}|$ [T]")
ax_center.legend(loc="lower right")
ax_center.set_ylim(0.0, 1.1 * np.max(Bmag_analytic))
ax_center.set_title("Centerline ($r=0$) $B$-field")

fig.tight_layout()

if not os.environ.get("CFSEM_TESTING", False):
    fig.savefig("helmholtz.png", dpi=300)
