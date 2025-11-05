"""
Finite-difference solution to structural PDE for pancake or
infinite-length solenoid coil, following Iwasa 2e section 3.6.

Assumes
* Uniform current density within each r-section
* Zero R-Z shear (deck-of-cards structure)
* Zero Z-force,stress,strain (no axial compression accounted here; can be evaluated separately;
  usually relatively small)
* Isotropic material
    * This method could be extended to handle orthotropic material, with some effort
* Single material region
    * This method could be extended to handle multiple regions of isotropic materials

Supports
* Non-uniform r-grid
* Arbitrary current density and B-field defined on r-grid
* Nonzero pressure on inner/outer wall (fluid pressure, bucking load, etc)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import Literal
from collections.abc import Callable

import findiff
import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict
from pydantic_numpy.model import NumpyModel
from pydantic_numpy.typing import NpNDArray  # Array of any type or dimensionality
from scipy import io, sparse
from scipy.sparse import csc_matrix as CSC
from scipy.sparse import csr_matrix as CSR
from scipy.sparse.linalg import factorized


def solenoid_1d_structural_factor(elasticity_modulus: float, poisson_ratio: float) -> float:
    """Structural factor applied to RHS of solenoid stress solve"""
    c = (1.0 - poisson_ratio**2) / elasticity_modulus  # [m/N]
    return c


def solenoid_1d_structural_rhs(
    c: float,
    j: NDArray | list[float],
    bz: NDArray | list[float],
    pi: float = 0.0,
    po: float = 0.0,
) -> NDArray:
    """
    Right-hand-side for solenoid stress solve,
    including zero values at the BCs.

    From Iwasa 2e eqn. 3.64a

    Recommend padding the grid with a dummy value at either end
    to make room for the BCs without losing accounting of nonzero
    current density at the inner/outer radius.

    Padding for BCs can be done like:
    `rgrid = np.array([r0 - 1e-6] + rgrid.tolist() + [r1 + 1e-6])`

    Padding region is ultimately treated as structural material,
    so the padded region should be small to avoid introducing error,
    but not so small that it causes numerical error in the finite difference scheme.

    Args:
        c: [m/N] scalar structural factor; see `solenoid_1d_structural_factor()`
        j: [A/m^2] with shape (n x 1), current density at each point in the r-grid
        bz: [T] with shape (n x 1), Z-axis B-field at each point in the r-grid
        pi: [Pa] scalar pressure on inner wall, defined in +r direction
        po: [Pa] scalar pressure on outer wall, defined in -r direction

    Returns:
        -c * j * bz, [1/m^2] with shape (n x 1), the right-hand side of the solenoid stress PDE
    """
    # Guarantee arrays
    j = np.array(j)
    bz = np.array(bz)

    # RHS without BCs
    rhs = -c * j * bz

    # BC for r-stress at inner and outer radius
    # is the fluid or mechanical pressure, which is usually going to be set to zero
    # to represent an unsupported system, but could be set to a nonzero value
    # to represent a surface load.
    rhs[0] = -pi  # Sign convention: compression is negative stress
    rhs[-1] = -po

    return rhs


class SolenoidStress1D(NumpyModel):
    model_config = ConfigDict(validate_assignment=True, frozen=True, extra="forbid")

    rgrid: NpNDArray
    """[m] 1D grid of r-coordinates"""
    elasticity_modulus: float
    """[Pa] diagonal terms in material property matrix"""
    poisson_ratio: float
    """[dimensionless] factor determining off-diagonal terms in material property matrix"""
    order: Literal[2, 4] = 4
    """Finite-difference stencil polynomial order.
       Higher order operators produce excessive numerical error under typical use."""
    direct_inverse: bool = False
    """Whether to generate fully-dense direct inverse of the system, which
    can be useful as a linear operator. Alternatively, the system can be solved
    using an LU solver with reduced memory usage and better numerical conditioning."""

    @cached_property
    def operators(self) -> SolenoidStress1DOperators:
        """
        Linear operators for solving stress and strain in a pancake coil
        following Iwasa 2e section 3.6.
        """
        return solenoid_1d_structural_operators(
            np.array(self.rgrid), self.elasticity_modulus, self.poisson_ratio, self.order, self.direct_inverse
        )

    @cached_property
    def displacement_solver(self) -> Callable[[NDArray], NDArray]:
        """LU solver for load-displacement relation (A_ub)
        as an alternative to taking a direct inverse of A_bu"""
        return factorized(self.operators.a_bu)


@dataclass(frozen=True)
class SolenoidStress1DOperators:
    """
    Linear operators for solving stress and strain in a pancake coil
    following Iwasa 2e section 3.6.

    A_bu, (n x n) sparse operator mapping displacement to the RHS like A @ u_r = -c * j * bz
    A_ub, (n x n) fully-dense direct inverse of A_bu mapping RHS to displacement
    A_eu (2n x n), A_eu_radial (n x n), A_eu_hoop (n x n), sparse operators mapping displacement to strain
        * First entry is combined operator producing both strain components
        * Second and third entries are split operators, which are equivalent because they are fully decoupled
    A_se (2n x 2n), sparse operator mapping strain to stress
    """

    a_bu: CSC
    """(n x n) sparse operator mapping displacement to the RHS like A @ u_r = -c * j * bz"""
    a_ub: NDArray | None
    """(n x n) fully-dense direct inverse of A_bu mapping RHS to displacement.
        Only generated if `direct_inverse` flag is set."""
    a_eu: CSR
    """(2n x n), sparse operator mapping displacement to strain; contains both radial and hoop components"""
    a_eu_radial: CSR
    """(n x n), sparse operators mapping displacement to strain; radial component only"""
    a_eu_hoop: CSR
    """(n x n), sparse operators mapping displacement to strain; hoop component only"""
    a_se: CSR
    """(2n x 2n), sparse operator mapping strain to stress"""

    def write_mat(self, dst: str | Path) -> str:
        """Write the collection of operators in .mat format.

        Args:
            dst: Target directory to place the file named "stress_operators.mat"

        Raises:
            IOError: If the directory does not exist
        """
        # Check directory
        dst = Path(dst).absolute()
        fpath = dst / "stress_operators.mat"
        getLogger("cfsem").info(f"Saving stress operator data to {fpath}")

        to_save = {
            "A_bu": self.a_bu,
            "A_ub": self.a_ub,
            "A_eu": self.a_eu,
            "A_eu_radial": self.a_eu_radial,
            "A_eu_hoop": self.a_eu_hoop,
            "A_se": self.a_se,
        }

        if self.a_ub is None:  # savemat fails on None value
            to_save.pop("A_ub")

        #    Note this will implicitly convert all CSR matrices to CSC, which is .mat's preferred I/O
        io.savemat(fpath, to_save)

        return f"{fpath}"


def solenoid_1d_structural_operators(
    rgrid: NDArray | list,
    elasticity_modulus: float,
    poisson_ratio: float,
    order: Literal[2, 4],
    direct_inverse: bool = False,
) -> SolenoidStress1DOperators:
    """
    Linear operators for solving stress and strain in a pancake coil
    following Iwasa 2e section 3.6.

    Assumes
    * Uniform current density within each r-section
    * Zero R-Z shear (deck-of-cards structure)
    * Zero Z-force,stress,strain (no axial compression accounted here; can be evaluated separately;
      usually relatively small)
    * Isotropic material
      * This method could be extended to handle orthotropic material, with some effort
    * Single material region
      * This method could be extended to handle multiple regions of isotropic materials

    Supports
    * Non-uniform r-grid
    * Arbitrary current density and B-field defined on r-grid

    Args:
        rgrid: [m] with shape (n x 1), Grid of r-coordinates. Must be sorted ascending.
        elasticity_modulus: [N/m] Material property; Young's modulus
        poisson_ratio: [dimensionless] Material property; off-axis stress coupling term
        order: Finite-difference stencil polynomial order
        direct_inverse: Whether to generate fully-dense direct inverse of the system, which
                        can be useful as a linear operator.
                        Alternatively, the system can be solved using an LU solver with
                        reduced memory usage and better numerical conditioning.

    Returns:
        object containing linear operators
    """
    # Guarantee arrays
    rgrid = np.array(rgrid)
    nr = len(rgrid)

    #
    # Stencils / differential operators
    #

    # Note these will include the 1/dr and 1/dr^2 scalings,
    # not just the normalized stencil
    ddr = findiff.Diff(0, rgrid, acc=order)
    d2dr2 = ddr**2

    ddr = ddr.matrix((nr,))
    d2dr2 = d2dr2.matrix((nr,))

    rinv = CSR(np.diag(1.0 / rgrid))
    rinv2 = CSR(np.diag(1.0 / rgrid**2))

    #
    # Displacement-load relation
    #

    # The first and last row of the u-b relation will be modified later
    # to include the boundary conditions
    #
    # Iwasa 2e eqn. 3.64a left hand side
    #
    # This one is converted to CSC because, while it's easiest to build it as CSR,
    # it is more useful for solving a system as CSC
    a_bu = CSC(d2dr2 + (rinv @ ddr) - rinv2)  # Operator maps displacement TO rhs (-c*j*B)

    # Stress-strain relation components
    # that are needed for BCs
    # Iwasa 2e eqns. 3.63g
    a_stress_strain = np.array(
        [
            [1.0 / elasticity_modulus, -poisson_ratio / elasticity_modulus],
            [-poisson_ratio / elasticity_modulus, 1.0 / elasticity_modulus],
        ]
    )
    a_strain_stress = np.linalg.inv(a_stress_strain)
    diag_term = a_strain_stress[0, 0]
    off_diag_term = a_strain_stress[1, 0]

    # We only need the first and last row of this matrix.
    # Actualizing the whole thing is easy but unnecessary
    # This is extracted by hand-calc starting from
    # Iwasa 2e eqns. 3.63a-d
    bcmat = diag_term * ddr + off_diag_term * rinv

    # Apply BC to displacement operator
    a_bu[0, :] = bcmat[0, :]
    a_bu[-1, :] = bcmat[-1, :]

    # Invert A_bu directly to get A_ub
    # This is fully dense, which may be prohibitive in some situations
    a_ub = (
        None if not direct_inverse else np.linalg.inv(a_bu.todense())
    )  # Operator maps RHS=-c*j*bz to displacement

    #
    # Strain-displacement operator(s)
    #
    # Iwasa 2e eqns. 3.63a-d
    a_eu_hoop = rinv  # Just the hoop part
    a_eu_radial = ddr  # Just the radial part
    # fmt: off
    a_eu = CSR(sparse.block_array(
        [[a_eu_radial],
         [a_eu_hoop]]
    ))  # Operator for getting strain from displacement
    # fmt: on

    #
    # Stress-strain operators
    #
    # In this case, we're manually assembling the inverse of the A_es matrix
    # because we know all the components from handcalc arithmetic
    eye = sparse.eye(nr, nr)
    # fmt: off
    a_se = CSR(sparse.block_array(
        [[   diag_term * eye, off_diag_term * eye],
        [off_diag_term * eye, diag_term * eye]]
    )) # Operator maps strain to stress
    # fmt: on

    return SolenoidStress1DOperators(a_bu, a_ub, a_eu, a_eu_radial, a_eu_hoop, a_se)
