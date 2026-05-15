from typing import Any, Literal

from numpy import float64, int64
from numpy.typing import NDArray

ArrayTriple = tuple[NDArray[float64], NDArray[float64], NDArray[float64]]

class HierarchicalDipoles:
    """Reusable single-source-tree hierarchical dipole field solver."""

    def __init__(self, theta: float = 0.01, construction_method: str = "recursive") -> None: ...
    def build(
        self,
        loc: ArrayTriple,
        obs: ArrayTriple,
        outer_radius: NDArray[float64] | None = None,
        par: bool = False,
    ) -> None: ...
    def build_sources(
        self,
        loc: ArrayTriple,
        outer_radius: NDArray[float64] | None = None,
        par: bool = False,
    ) -> None: ...
    def update_sources(
        self,
        loc: ArrayTriple,
        outer_radius: NDArray[float64] | None = None,
        par: bool = False,
    ) -> None: ...
    def build_targets(self, obs: ArrayTriple, par: bool = False) -> None: ...
    def update_targets(self, obs: ArrayTriple, par: bool = False) -> None: ...
    def flux_density(
        self,
        moment: ArrayTriple,
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple: ...
    def vector_potential(
        self,
        moment: ArrayTriple,
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple: ...
    def accepted_source_levels(
        self,
        moment: ArrayTriple,
        field: Literal["b", "a"] = "b",
    ) -> NDArray[float64]: ...
    def source_tree_aabbs(self) -> tuple[NDArray[float64], ...]: ...

class HierarchicalLinearFilaments:
    """Reusable single-source-tree hierarchical linear-filament field solver."""

    def __init__(self, theta: float = 0.05, construction_method: str = "recursive") -> None: ...
    def build(
        self,
        xyzfil: ArrayTriple,
        dlxyzfil: ArrayTriple,
        wire_radius: NDArray[float64],
        obs: ArrayTriple,
        par: bool = False,
    ) -> None: ...
    def build_sources(
        self,
        xyzfil: ArrayTriple,
        dlxyzfil: ArrayTriple,
        wire_radius: NDArray[float64],
        par: bool = False,
    ) -> None: ...
    def update_sources(
        self,
        xyzfil: ArrayTriple,
        dlxyzfil: ArrayTriple,
        wire_radius: NDArray[float64],
        par: bool = False,
    ) -> None: ...
    def build_targets(self, obs: ArrayTriple, par: bool = False) -> None: ...
    def update_targets(self, obs: ArrayTriple, par: bool = False) -> None: ...
    def flux_density(
        self,
        current: NDArray[float64],
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple: ...
    def vector_potential(
        self,
        current: NDArray[float64],
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple: ...
    def accepted_source_levels(
        self,
        current: NDArray[float64],
        field: Literal["b", "a"] = "b",
    ) -> NDArray[float64]: ...
    def source_tree_aabbs(self) -> tuple[NDArray[float64], ...]: ...

class HierarchicalBoundaryElements:
    """Reusable single-source-tree hierarchical triangular boundary-element field solver."""

    def __init__(
        self,
        theta: float = 0.05,
        quad: str = "dunavant3",
        construction_method: str = "recursive",
    ) -> None: ...
    def build(
        self,
        nodes: NDArray[float64],
        triangles: NDArray[int64],
        obs: ArrayTriple,
        par: bool = False,
    ) -> None: ...
    def build_sources(
        self,
        nodes: NDArray[float64],
        triangles: NDArray[int64],
        par: bool = False,
    ) -> None: ...
    def update_sources(
        self,
        nodes: NDArray[float64],
        triangles: NDArray[int64],
        par: bool = False,
    ) -> None: ...
    def build_targets(self, obs: ArrayTriple, par: bool = False) -> None: ...
    def update_targets(self, obs: ArrayTriple, par: bool = False) -> None: ...
    def flux_density(
        self,
        current_density: ArrayTriple,
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple: ...
    def vector_potential(
        self,
        current_density: ArrayTriple,
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple: ...
    def accepted_source_levels(
        self,
        current_density: ArrayTriple,
        field: Literal["b", "a"] = "b",
    ) -> NDArray[float64]: ...
    def source_tree_aabbs(self) -> tuple[NDArray[float64], ...]: ...

def __getattr__(name: str) -> Any: ...
