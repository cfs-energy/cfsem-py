"""
Comparison of self- and mutual- inductance of coils modeled as either an axisymmetric filament collection
or as a piecewise-linear helix.
"""

import numpy as np
import cfsem

# Center radius and height, winding pack width and height,
# and number of windings in r and z directions
# for two solenoids.
r1, z1, w1, h1, nr1, nz1 = (0.3, 0.0, 0.01, 1.0, 1, 10)
r2, z2, w2, h2, nr2, nz2 = (0.5, 0.2, 0.01, 0.5, 1, 5)

nt1 = nr1 * nz1  # Total number of turns for each coil
nt2 = nr2 * nz2

# Build axisymmetric filament representation
filaments_1 = cfsem.filament_coil(r1, z1, w1, h1, nt1, nr1, nz1)
filaments_2 = cfsem.filament_coil(r2, z2, w2, h2, nt2, nr2, nz2)

# Build helix representations with 100 points per turn.
# The first and last point in the helices span the full height of the winding pack
# such that each filament in the axisymmetric representation captures a half turn
# of the helical representation above and below its z-location.
angles1 = np.linspace(0.0, 2.0 * np.pi * nt1, 100 * nt1)  # [rad]
angles2 = np.linspace(0.0, 2.0 * np.pi * nt2, 100 * nt2)  # [rad]

xhelix1 = r1 * np.cos(angles1)  # [m]
yhelix1 = r1 * np.sin(angles1)  # [m]
zhelix1 = np.linspace(z1 - h1 / 2, z1 + h1 / 2, angles1.size)  # [m]

xhelix2 = r2 * np.cos(angles2)  # [m]
yhelix2 = r2 * np.sin(angles2)  # [m]
zhelix2 = np.linspace(z2 - h2 / 2, z2 + h2 / 2, angles2.size)  # [m]

# Estimate the self-inductance by 2 different methods,
# hand-calc and helical filaments.
self_inductance_handcalc_1 = cfsem.self_inductance_lyle6(r1, w1, h1, nt1)
self_inductance_handcalc_2 = cfsem.self_inductance_lyle6(r2, w2, h2, nt2)

self_inductance_helical_1 = cfsem.self_inductance_piecewise_linear_filaments(
    (xhelix1, yhelix1, zhelix1)
)
self_inductance_helical_2 = cfsem.self_inductance_piecewise_linear_filaments(
    (xhelix2, yhelix2, zhelix2)
)

print("First coil self-inductance")
print(f"    Handcalc: {self_inductance_handcalc_1:.2e} [H]")
print(f"    Helical: {self_inductance_helical_1:.2e} [H]")

print("Second coil self-inductance")
print(f"    Handcalc: {self_inductance_handcalc_2:.2e} [H]")
print(f"    Helical: {self_inductance_helical_2:.2e} [H]")

# Estimate mutual inductance by 2 different methods,
# axisymmetric filaments and helical filaments.
mutual_inductance_axisymmetric = cfsem.mutual_inductance_of_cylindrical_coils(
    filaments_1.T, filaments_2.T
)
mutual_inductance_helical = cfsem.mutual_inductance_piecewise_linear_filaments(
    (xhelix1, yhelix1, zhelix1), (xhelix2, yhelix2, zhelix2)
)
#    Mutual inductance is reflexive, so we get the same answer if we reverse the inputs
mutual_inductance_axisymmetric_reflexive = cfsem.mutual_inductance_of_cylindrical_coils(
    filaments_2.T, filaments_1.T
)
mutual_inductance_helical_reflexive = (
    cfsem.mutual_inductance_piecewise_linear_filaments(
        (xhelix2, yhelix2, zhelix2), (xhelix1, yhelix1, zhelix1)
    )
)

print("Mutual-inductance")
print(f"    Axisymmetric: {mutual_inductance_axisymmetric:.2e} [H]")
print(f"    Helical: {mutual_inductance_helical:.2e} [H]")
print(f"    Axisymmetric reflexive: {mutual_inductance_axisymmetric_reflexive:.2e} [H]")
print(f"    Helical reflexive: {mutual_inductance_helical_reflexive:.2e} [H]")
