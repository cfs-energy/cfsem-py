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

    def set_sources(
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

    def flux_density(
        self,
        target: ArrayTriple,
        moment: ArrayTriple,
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple:
        """Evaluate magnetic flux density for the current source and target geometry.

        This is an approximate method, and no particular accuracy level is guaranteed.
        Truncated methods like this one may average entire local loop structures out of
        existence; as a result, maximum relative error is unbounded. This method must be
        tuned to a given use-case in order to be useful, and should not be used to
        calculate safety-related field limits.

        Args:
            target: Target point coordinates as component arrays.
            moment: Dipole magnetic moment components.
            par: Whether to evaluate target batches in parallel.
            out: Optional output component arrays to fill.

        Returns:
            Output component arrays. If `out` is provided, returns `out`.
        """
        ...

    def vector_potential(
        self,
        target: ArrayTriple,
        moment: ArrayTriple,
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple:
        """Evaluate magnetic vector potential for the current source and target geometry.

        This is an approximate method, and no particular accuracy level is guaranteed.
        Truncated methods like this one may average entire local loop structures out of
        existence; as a result, maximum relative error is unbounded. This method must be
        tuned to a given use-case in order to be useful, and should not be used to
        calculate safety-related field limits.

        Args:
            target: Target point coordinates as component arrays.
            moment: Dipole magnetic moment components.
            par: Whether to evaluate target batches in parallel.
            out: Optional output component arrays to fill.

        Returns:
            Output component arrays. If `out` is provided, returns `out`.
        """
        ...

    def accepted_source_levels(
        self,
        target: ArrayTriple,
        moment: ArrayTriple,
        field: Literal["b", "a"] = "b",
    ) -> NDArray[float64]:
        """Return the accepted source-tree level diagnostic for each target.

        Args:
            target: Target point coordinates as component arrays.
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

    def set_sources(
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

    def flux_density(
        self,
        target: ArrayTriple,
        current: NDArray[float64],
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple:
        """Evaluate magnetic flux density for the current source and target geometry.

        This is an approximate method, and no particular accuracy level is guaranteed.
        Truncated methods like this one may average entire local loop structures out of
        existence; as a result, maximum relative error is unbounded. This method must be
        tuned to a given use-case in order to be useful, and should not be used to
        calculate safety-related field limits.

        Args:
            target: Target point coordinates as component arrays.
            current: Current in each filament segment.
            par: Whether to evaluate target batches in parallel.
            out: Optional output component arrays to fill.

        Returns:
            Output component arrays. If `out` is provided, returns `out`.
        """
        ...

    def vector_potential(
        self,
        target: ArrayTriple,
        current: NDArray[float64],
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple:
        """Evaluate magnetic vector potential for the current source and target geometry.

        This is an approximate method, and no particular accuracy level is guaranteed.
        Truncated methods like this one may average entire local loop structures out of
        existence; as a result, maximum relative error is unbounded. This method must be
        tuned to a given use-case in order to be useful, and should not be used to
        calculate safety-related field limits.

        Args:
            target: Target point coordinates as component arrays.
            current: Current in each filament segment.
            par: Whether to evaluate target batches in parallel.
            out: Optional output component arrays to fill.

        Returns:
            Output component arrays. If `out` is provided, returns `out`.
        """
        ...

    def accepted_source_levels(
        self,
        target: ArrayTriple,
        current: NDArray[float64],
        field: Literal["b", "a"] = "b",
    ) -> NDArray[float64]:
        """Return the accepted source-tree level diagnostic for each target.

        Args:
            target: Target point coordinates as component arrays.
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

    def set_sources(
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

    def flux_density(
        self,
        target: ArrayTriple,
        current_density: ArrayTriple,
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple:
        """Evaluate magnetic flux density for the current source and target geometry.

        This is an approximate method, and no particular accuracy level is guaranteed.
        Truncated methods like this one may average entire local loop structures out of
        existence; as a result, maximum relative error is unbounded. This method must be
        tuned to a given use-case in order to be useful, and should not be used to
        calculate safety-related field limits.

        Args:
            target: Target point coordinates as component arrays.
            current_density: Triangle-local current-density components.
            par: Whether to evaluate target batches in parallel.
            out: Optional output component arrays to fill.

        Returns:
            Output component arrays. If `out` is provided, returns `out`.
        """
        ...

    def vector_potential(
        self,
        target: ArrayTriple,
        current_density: ArrayTriple,
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple:
        """Evaluate magnetic vector potential for the current source and target geometry.

        This is an approximate method, and no particular accuracy level is guaranteed.
        Truncated methods like this one may average entire local loop structures out of
        existence; as a result, maximum relative error is unbounded. This method must be
        tuned to a given use-case in order to be useful, and should not be used to
        calculate safety-related field limits.

        Args:
            target: Target point coordinates as component arrays.
            current_density: Triangle-local current-density components.
            par: Whether to evaluate target batches in parallel.
            out: Optional output component arrays to fill.

        Returns:
            Output component arrays. If `out` is provided, returns `out`.
        """
        ...

    def accepted_source_levels(
        self,
        target: ArrayTriple,
        current_density: ArrayTriple,
        field: Literal["b", "a"] = "b",
    ) -> NDArray[float64]:
        """Return the accepted source-tree level diagnostic for each target.

        Args:
            target: Target point coordinates as component arrays.
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
