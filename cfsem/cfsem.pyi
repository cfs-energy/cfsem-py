from typing import Any, Literal

from numpy import float64, int64
from numpy.typing import NDArray

ArrayTriple = tuple[NDArray[float64], NDArray[float64], NDArray[float64]]

class HierarchicalDipoles:
    """Reusable single-source-tree hierarchical dipole field solver."""

    def __init__(self, theta: float = 0.01, construction_method: str = "recursive") -> None:
        """Create a reusable hierarchical dipole solver.

        Args:
            theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
            construction_method: Source-tree construction method.
        """
        ...

    def build(
        self,
        loc: ArrayTriple,
        obs: ArrayTriple,
        outer_radius: NDArray[float64] | None = None,
        par: bool = False,
    ) -> None:
        """Build source and target geometry for repeated dipole evaluations.

        Args:
            loc: Dipole source coordinates as component arrays.
            obs: Observation point coordinates as component arrays.
            outer_radius: Radius for the magnetized-sphere near-field treatment.
            par: Whether to parallelize CPU construction.
        """
        ...

    def build_sources(
        self,
        loc: ArrayTriple,
        outer_radius: NDArray[float64] | None = None,
        par: bool = False,
    ) -> None:
        """Build or replace the source tree for dipole geometry.

        Args:
            loc: Dipole source coordinates as component arrays.
            outer_radius: Radius for the magnetized-sphere near-field treatment.
            par: Whether to parallelize CPU construction.
        """
        ...

    def update_sources(
        self,
        loc: ArrayTriple,
        outer_radius: NDArray[float64] | None = None,
        par: bool = False,
    ) -> None:
        """Replace dipole source geometry and rebuild the source tree.

        Args:
            loc: Dipole source coordinates as component arrays.
            outer_radius: Radius for the magnetized-sphere near-field treatment.
            par: Whether to parallelize CPU construction.
        """
        ...

    def build_targets(self, obs: ArrayTriple, par: bool = False) -> None:
        """Build or replace observation point geometry.

        Args:
            obs: Observation point coordinates as component arrays.
            par: Reserved for API consistency.
        """
        ...

    def update_targets(self, obs: ArrayTriple, par: bool = False) -> None:
        """Replace observation point geometry.

        Args:
            obs: Observation point coordinates as component arrays.
            par: Reserved for API consistency.
        """
        ...

    def flux_density(
        self,
        moment: ArrayTriple,
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple:
        """Evaluate magnetic flux density for the current source and target geometry.

        Args:
            moment: Dipole magnetic moment components.
            par: Whether to evaluate target batches in parallel.
            out: Optional output component arrays to fill.

        Returns:
            Output component arrays. If `out` is provided, returns `out`.
        """
        ...

    def vector_potential(
        self,
        moment: ArrayTriple,
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple:
        """Evaluate magnetic vector potential for the current source and target geometry.

        Args:
            moment: Dipole magnetic moment components.
            par: Whether to evaluate target batches in parallel.
            out: Optional output component arrays to fill.

        Returns:
            Output component arrays. If `out` is provided, returns `out`.
        """
        ...

    def accepted_source_levels(
        self,
        moment: ArrayTriple,
        field: Literal["b", "a"] = "b",
    ) -> NDArray[float64]:
        """Return the accepted source-tree level diagnostic for each target.

        Args:
            moment: Dipole magnetic moment components.
            field: Field kernel to use for the acceptance diagnostic.

        Returns:
            Accepted source-tree level for each observation point.
        """
        ...

    def source_tree_aabbs(self) -> tuple[NDArray[float64], ...]:
        """Return source-tree AABB bounds and levels.

        Returns:
            Component arrays for minimum bounds, maximum bounds, and tree levels.
        """
        ...

class HierarchicalLinearFilaments:
    """Reusable single-source-tree hierarchical linear-filament field solver."""

    def __init__(self, theta: float = 0.05, construction_method: str = "recursive") -> None:
        """Create a reusable hierarchical linear-filament solver.

        Args:
            theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
            construction_method: Source-tree construction method.
        """
        ...

    def build(
        self,
        xyzfil: ArrayTriple,
        dlxyzfil: ArrayTriple,
        wire_radius: NDArray[float64],
        obs: ArrayTriple,
        par: bool = False,
    ) -> None:
        """Build source and target geometry for repeated filament evaluations.

        Args:
            xyzfil: Filament segment start coordinates as component arrays.
            dlxyzfil: Filament segment displacement vectors as component arrays.
            wire_radius: Wire radius for each segment.
            obs: Observation point coordinates as component arrays.
            par: Whether to parallelize CPU construction.
        """
        ...

    def build_sources(
        self,
        xyzfil: ArrayTriple,
        dlxyzfil: ArrayTriple,
        wire_radius: NDArray[float64],
        par: bool = False,
    ) -> None:
        """Build or replace the source tree for filament geometry.

        Args:
            xyzfil: Filament segment start coordinates as component arrays.
            dlxyzfil: Filament segment displacement vectors as component arrays.
            wire_radius: Wire radius for each segment.
            par: Whether to parallelize CPU construction.
        """
        ...

    def update_sources(
        self,
        xyzfil: ArrayTriple,
        dlxyzfil: ArrayTriple,
        wire_radius: NDArray[float64],
        par: bool = False,
    ) -> None:
        """Replace filament source geometry and rebuild the source tree.

        Args:
            xyzfil: Filament segment start coordinates as component arrays.
            dlxyzfil: Filament segment displacement vectors as component arrays.
            wire_radius: Wire radius for each segment.
            par: Whether to parallelize CPU construction.
        """
        ...

    def build_targets(self, obs: ArrayTriple, par: bool = False) -> None:
        """Build or replace observation point geometry.

        Args:
            obs: Observation point coordinates as component arrays.
            par: Reserved for API consistency.
        """
        ...

    def update_targets(self, obs: ArrayTriple, par: bool = False) -> None:
        """Replace observation point geometry.

        Args:
            obs: Observation point coordinates as component arrays.
            par: Reserved for API consistency.
        """
        ...

    def flux_density(
        self,
        current: NDArray[float64],
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple:
        """Evaluate magnetic flux density for the current source and target geometry.

        Args:
            current: Current in each filament segment.
            par: Whether to evaluate target batches in parallel.
            out: Optional output component arrays to fill.

        Returns:
            Output component arrays. If `out` is provided, returns `out`.
        """
        ...

    def vector_potential(
        self,
        current: NDArray[float64],
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple:
        """Evaluate magnetic vector potential for the current source and target geometry.

        Args:
            current: Current in each filament segment.
            par: Whether to evaluate target batches in parallel.
            out: Optional output component arrays to fill.

        Returns:
            Output component arrays. If `out` is provided, returns `out`.
        """
        ...

    def accepted_source_levels(
        self,
        current: NDArray[float64],
        field: Literal["b", "a"] = "b",
    ) -> NDArray[float64]:
        """Return the accepted source-tree level diagnostic for each target.

        Args:
            current: Current in each filament segment.
            field: Field kernel to use for the acceptance diagnostic.

        Returns:
            Accepted source-tree level for each observation point.
        """
        ...

    def source_tree_aabbs(self) -> tuple[NDArray[float64], ...]:
        """Return source-tree AABB bounds and levels.

        Returns:
            Component arrays for minimum bounds, maximum bounds, and tree levels.
        """
        ...

class HierarchicalBoundaryElements:
    """Reusable single-source-tree hierarchical triangular boundary-element field solver."""

    def __init__(
        self,
        theta: float = 0.05,
        quad: str = "dunavant3",
        construction_method: str = "recursive",
    ) -> None:
        """Create a reusable hierarchical boundary-element solver.

        Args:
            theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
            quad: Triangle quadrature rule.
            construction_method: Source-tree construction method.
        """
        ...

    def build(
        self,
        nodes: NDArray[float64],
        triangles: NDArray[int64],
        obs: ArrayTriple,
        par: bool = False,
    ) -> None:
        """Build source and target geometry for repeated triangle-mesh evaluations.

        Args:
            nodes: Mesh node coordinates with one node per row.
            triangles: Triangle node indices with one triangle per row.
            obs: Observation point coordinates as component arrays.
            par: Whether to parallelize CPU construction.
        """
        ...

    def build_sources(
        self,
        nodes: NDArray[float64],
        triangles: NDArray[int64],
        par: bool = False,
    ) -> None:
        """Build or replace the source tree for triangle geometry.

        Args:
            nodes: Mesh node coordinates with one node per row.
            triangles: Triangle node indices with one triangle per row.
            par: Whether to parallelize CPU construction.
        """
        ...

    def update_sources(
        self,
        nodes: NDArray[float64],
        triangles: NDArray[int64],
        par: bool = False,
    ) -> None:
        """Replace triangle source geometry and rebuild the source tree.

        Args:
            nodes: Mesh node coordinates with one node per row.
            triangles: Triangle node indices with one triangle per row.
            par: Whether to parallelize CPU construction.
        """
        ...

    def build_targets(self, obs: ArrayTriple, par: bool = False) -> None:
        """Build or replace observation point geometry.

        Args:
            obs: Observation point coordinates as component arrays.
            par: Reserved for API consistency.
        """
        ...

    def update_targets(self, obs: ArrayTriple, par: bool = False) -> None:
        """Replace observation point geometry.

        Args:
            obs: Observation point coordinates as component arrays.
            par: Reserved for API consistency.
        """
        ...

    def flux_density(
        self,
        current_density: ArrayTriple,
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple:
        """Evaluate magnetic flux density for the current source and target geometry.

        Args:
            current_density: Triangle-local current-density components.
            par: Whether to evaluate target batches in parallel.
            out: Optional output component arrays to fill.

        Returns:
            Output component arrays. If `out` is provided, returns `out`.
        """
        ...

    def vector_potential(
        self,
        current_density: ArrayTriple,
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple:
        """Evaluate magnetic vector potential for the current source and target geometry.

        Args:
            current_density: Triangle-local current-density components.
            par: Whether to evaluate target batches in parallel.
            out: Optional output component arrays to fill.

        Returns:
            Output component arrays. If `out` is provided, returns `out`.
        """
        ...

    def accepted_source_levels(
        self,
        current_density: ArrayTriple,
        field: Literal["b", "a"] = "b",
    ) -> NDArray[float64]:
        """Return the accepted source-tree level diagnostic for each target.

        Args:
            current_density: Triangle-local current-density components.
            field: Field kernel to use for the acceptance diagnostic.

        Returns:
            Accepted source-tree level for each observation point.
        """
        ...

    def source_tree_aabbs(self) -> tuple[NDArray[float64], ...]:
        """Return source-tree AABB bounds and levels.

        Returns:
            Component arrays for minimum bounds, maximum bounds, and tree levels.
        """
        ...

def __getattr__(name: str) -> Any: ...
