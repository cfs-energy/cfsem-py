"""
Comparison of self- and mutual- inductance of coils modeled as either an axisymmetric filament collection
or as a piecewise-linear helix.
"""

import numpy as np

import cfsem

# Define a set of rectangular coils
r = [0.1, 0.15]        # Radial positions of coil centers [m]
z = [0.0, 0.2]         # Axial positions of coil centers [m]
dr = [0.05, 0.05]      # Radial sizes of coils [m]
dz = [0.1, 0.1]        # Axial sizes of coils [m]
j = [1e5, 2e5]         # Current densities [A/m^2]
nr = [10, 10]        # Radial discretizations for mutual inductance
nz = [10, 10]        # Axial discretizations for mutual inductance

L = cfsem.inductance_matrix_axisymmetric_coaxial_rectangular_coils(
        r=r,
        z=z,
        dr=dr,
        dz=dz,
        j=j,
        nr=nr,
        nz=nz,
    )

print("Inductance matrix [H]:")
print(L)
