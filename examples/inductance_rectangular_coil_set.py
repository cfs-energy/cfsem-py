"""
Example script for calculating the inductance matrix of a set of
rectangular coaxial coils with prescribed turn density.
"""

import numpy as np

import cfsem

# Define a set of rectangular coils
r = [0.1, 0.15]        # Radial positions of coil centers [m]
z = [0.0, 0.2]         # Axial positions of coil centers [m]
dr = [0.05, 0.05]      # Radial sizes of coils [m]
dz = [0.1, 0.1]        # Axial sizes of coils [m]
td = [2500.0, 2500.0]  # Turn density at 20mm x 20mm cross-section [turns/m^2]
nr = [10, 10]          # Radial discretizations for mutual inductance
nz = [10, 10]          # Axial discretizations for mutual inductance

L = cfsem.inductance_matrix_axisymmetric_coaxial_rectangular_coils(
        r=r,
        z=z,
        dr=dr,
        dz=dz,
        td=td,
        nr=nr,
        nz=nz,
    )

print("Inductance matrix [H]:")
print(L)