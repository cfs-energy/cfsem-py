from typing import TypeAlias, TypedDict

from numpy import complex128, float32, float64, int64, uint64
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[float64]
ComplexArray: TypeAlias = NDArray[complex128]
Float32Array: TypeAlias = NDArray[float32]
IntArray: TypeAlias = NDArray[int64]
UIntArray: TypeAlias = NDArray[uint64]
FloatMatrix: TypeAlias = NDArray[float64]
Float32Matrix: TypeAlias = NDArray[float32]
IntMatrix: TypeAlias = NDArray[int64]
UIntMatrix: TypeAlias = NDArray[uint64]
FloatTensor3: TypeAlias = NDArray[float64]
Float32Tensor3: TypeAlias = NDArray[float32]
ArrayTriple: TypeAlias = tuple[FloatArray, FloatArray, FloatArray]
ArrayPair: TypeAlias = tuple[FloatArray, FloatArray]
SparseF64: TypeAlias = tuple[FloatArray, UIntArray, UIntArray, int, int]
SparseF32: TypeAlias = tuple[Float32Array, UIntArray, UIntArray, int, int]
SourceTreeDiagnostics: TypeAlias = tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    UIntArray,
    UIntArray,
]
# Source tree diagnostic tuple order:
# min_x, min_y, min_z, max_x, max_y, max_z, level, left_child, right_child.

class HierarchicalDiagnostics:
    """Timing, size, and optional tree diagnostics from a hierarchical solve."""

    @property
    def construction_time(self) -> float: ...
    @property
    def evaluation_time(self) -> float: ...
    @property
    def source_count(self) -> int: ...
    @property
    def target_count(self) -> int: ...
    @property
    def source_tree(self) -> SourceTreeDiagnostics | None: ...
    @property
    def accepted_levels(self) -> FloatArray | None: ...

class SolveResult:
    """Field arrays and diagnostics returned by a hierarchical solve."""

    @property
    def field(self) -> ArrayTriple: ...
    @property
    def diagnostics(self) -> HierarchicalDiagnostics: ...

class QuadMeshQueryF64(TypedDict):
    nearest_node_indices: UIntArray
    nearest_node_points: FloatArray
    nearest_node_distances: FloatArray
    nearest_element_indices: UIntArray
    nearest_element_reference_points: FloatArray
    nearest_element_points: FloatArray
    nearest_element_distances: FloatArray
    nearest_face_element_indices: UIntArray
    nearest_face_local_faces: UIntArray
    nearest_face_reference_coordinates: FloatArray
    nearest_face_points: FloatArray
    nearest_face_distances: FloatArray

class QuadMeshQueryF32(TypedDict):
    nearest_node_indices: UIntArray
    nearest_node_points: Float32Array
    nearest_node_distances: Float32Array
    nearest_element_indices: UIntArray
    nearest_element_reference_points: Float32Array
    nearest_element_points: Float32Array
    nearest_element_distances: Float32Array
    nearest_face_element_indices: UIntArray
    nearest_face_local_faces: UIntArray
    nearest_face_reference_coordinates: Float32Array
    nearest_face_points: Float32Array
    nearest_face_distances: Float32Array

class DimensionalityError(Exception): ...

class SolenoidStress2dModelF64:
    @property
    def ndof_full(self) -> int: ...
    @property
    def ndof_reduced(self) -> int: ...
    @property
    def nelem(self) -> int: ...
    @property
    def n_temperature_nodes(self) -> int: ...
    @property
    def nq_per_element(self) -> int: ...
    @property
    def nodes_per_element(self) -> int: ...
    @property
    def element_type(self) -> str: ...
    @property
    def formulation(self) -> str: ...
    def analysis_nodes_flat(self) -> FloatArray: ...
    def analysis_elements_flat(self) -> UIntArray: ...
    def pressure_faces_flat(self) -> UIntArray: ...
    def traction_faces_flat(self) -> UIntArray: ...
    def free_dofs(self) -> UIntArray: ...
    def fixed_dofs(self) -> UIntArray: ...
    def fixed_values(self) -> FloatArray: ...
    def constant_rhs(self) -> FloatArray: ...
    def quadrature_points_flat(self) -> FloatArray: ...
    def strain_constant(self) -> FloatArray: ...
    def stress_constant(self) -> FloatArray: ...
    def thermal_strain_constant(self) -> FloatArray: ...
    def thermal_stress_constant(self) -> FloatArray: ...
    def stiffness_csc(self) -> SparseF64: ...
    def body_force_to_rhs_csr(self) -> SparseF64: ...
    def pressure_to_rhs_csr(self) -> SparseF64: ...
    def traction_to_rhs_csr(self) -> SparseF64: ...
    def temperature_to_rhs_csr(self) -> SparseF64: ...
    def strain_operator_csr(self) -> SparseF64: ...
    def stress_operator_csr(self) -> SparseF64: ...
    def thermal_strain_operator_csr(self) -> SparseF64: ...
    def thermal_stress_operator_csr(self) -> SparseF64: ...
    def build_rhs(
        self,
        body_force: FloatArray | None = None,
        pressure_values: FloatArray | None = None,
        traction_values: FloatArray | None = None,
        nodal_temperature: FloatArray | None = None,
    ) -> FloatArray: ...
    def solve(self, rhs: FloatArray) -> FloatArray: ...
    def element_quadrature(self) -> tuple[FloatArray, FloatArray, FloatArray, int]: ...
    def element_measures(self) -> tuple[FloatArray, FloatArray]: ...
    def evaluate_quadrature(
        self, displacements_full: FloatArray, nodal_temperature: FloatArray | None = None
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, int]: ...

class SolenoidStress2dModelF32:
    @property
    def ndof_full(self) -> int: ...
    @property
    def ndof_reduced(self) -> int: ...
    @property
    def nelem(self) -> int: ...
    @property
    def n_temperature_nodes(self) -> int: ...
    @property
    def nq_per_element(self) -> int: ...
    @property
    def nodes_per_element(self) -> int: ...
    @property
    def element_type(self) -> str: ...
    @property
    def formulation(self) -> str: ...
    def analysis_nodes_flat(self) -> Float32Array: ...
    def analysis_elements_flat(self) -> UIntArray: ...
    def pressure_faces_flat(self) -> UIntArray: ...
    def traction_faces_flat(self) -> UIntArray: ...
    def free_dofs(self) -> UIntArray: ...
    def fixed_dofs(self) -> UIntArray: ...
    def fixed_values(self) -> Float32Array: ...
    def constant_rhs(self) -> Float32Array: ...
    def quadrature_points_flat(self) -> Float32Array: ...
    def strain_constant(self) -> Float32Array: ...
    def stress_constant(self) -> Float32Array: ...
    def thermal_strain_constant(self) -> Float32Array: ...
    def thermal_stress_constant(self) -> Float32Array: ...
    def stiffness_csc(self) -> SparseF32: ...
    def body_force_to_rhs_csr(self) -> SparseF32: ...
    def pressure_to_rhs_csr(self) -> SparseF32: ...
    def traction_to_rhs_csr(self) -> SparseF32: ...
    def temperature_to_rhs_csr(self) -> SparseF32: ...
    def strain_operator_csr(self) -> SparseF32: ...
    def stress_operator_csr(self) -> SparseF32: ...
    def thermal_strain_operator_csr(self) -> SparseF32: ...
    def thermal_stress_operator_csr(self) -> SparseF32: ...
    def build_rhs(
        self,
        body_force: Float32Array | None = None,
        pressure_values: Float32Array | None = None,
        traction_values: Float32Array | None = None,
        nodal_temperature: Float32Array | None = None,
    ) -> Float32Array: ...
    def solve(self, rhs: Float32Array) -> Float32Array: ...
    def element_quadrature(self) -> tuple[Float32Array, Float32Array, Float32Array, int]: ...
    def element_measures(self) -> tuple[Float32Array, Float32Array]: ...
    def evaluate_quadrature(
        self, displacements_full: Float32Array, nodal_temperature: Float32Array | None = None
    ) -> tuple[Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, int]: ...

def body_force_density_circular_filament_cartesian(
    current: FloatArray,
    rfil: FloatArray,
    zfil: FloatArray,
    obs: ArrayTriple,
    j: ArrayTriple,
    par: bool,
) -> ArrayTriple: ...
def body_force_density_linear_filament(
    xyzfil: ArrayTriple,
    dlxyzfil: ArrayTriple,
    ifil: FloatArray,
    obs: ArrayTriple,
    j: ArrayTriple,
    wire_radius: FloatArray,
    par: bool = True,
) -> ArrayTriple: ...
def ellipe(x: float) -> float: ...
def ellipk(x: float) -> float: ...
def hyp2f1(
    a: ComplexArray,
    b: ComplexArray,
    c: ComplexArray,
    z: ComplexArray,
    par: bool = True,
) -> ComplexArray: ...
def filament_helix_path(
    path: ArrayTriple,
    helix_start_offset: tuple[float, float, float],
    twist_pitch: float,
    angle_offset: float,
    out: ArrayTriple,
) -> None: ...
def rotate_filaments_about_path(path: ArrayTriple, angle_offset: float, out: ArrayTriple) -> None: ...
def flux_circular_filament(
    current: FloatArray,
    rfil: FloatArray,
    zfil: FloatArray,
    rprime: FloatArray,
    zprime: FloatArray,
    par: bool,
) -> FloatArray: ...
def vector_potential_circular_filament(
    current: FloatArray,
    rfil: FloatArray,
    zfil: FloatArray,
    rprime: FloatArray,
    zprime: FloatArray,
    par: bool,
) -> FloatArray: ...
def flux_density_circular_filament(
    current: FloatArray,
    rfil: FloatArray,
    zfil: FloatArray,
    rprime: FloatArray,
    zprime: FloatArray,
    par: bool,
) -> ArrayPair: ...
def flux_density_circular_filament_cartesian(
    current: FloatArray,
    rfil: FloatArray,
    zfil: FloatArray,
    xyzobs: ArrayTriple,
    par: bool,
) -> ArrayTriple: ...
def mutual_inductance_circular_to_linear(
    rfil: FloatArray,
    zfil: FloatArray,
    nfil: FloatArray,
    xyzfil: ArrayTriple,
    dlxyzfil: ArrayTriple,
    par: bool,
) -> float: ...
def flux_density_dipole(
    loc: ArrayTriple,
    moment: ArrayTriple,
    obs: ArrayTriple,
    outer_radius: FloatArray,
    par: bool,
) -> ArrayTriple: ...
def vector_potential_dipole(
    loc: ArrayTriple,
    moment: ArrayTriple,
    obs: ArrayTriple,
    outer_radius: FloatArray,
    par: bool,
) -> ArrayTriple: ...
def flux_density_dipole_hierarchical(
    loc: ArrayTriple,
    moment: ArrayTriple,
    obs: ArrayTriple,
    outer_radius: FloatArray,
    theta: float = 0.01,
    construction_method: str = "longest_axis",
    par: bool = True,
    out: ArrayTriple | None = None,
    extra_diagnostics: bool = False,
) -> SolveResult:
    """Hierarchical magnetic flux density of dipoles in Cartesian coordinates.

    This one-shot method builds the source tree internally. This is an approximate method,
    and no particular accuracy level is guaranteed. Truncated methods like this one may
    average entire local loop structures out of existence; as a result, maximum relative
    error is unbounded. This method must be tuned to a given use-case in order to be
    useful, and should not be used to calculate safety-related field limits.

    Args:
        loc: Dipole source coordinates as component arrays.
        moment: Dipole magnetic moment components.
        obs: Target point coordinates as component arrays.
        outer_radius: Magnetized-sphere radius for each source. Use zeros for point dipoles.
        theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
        construction_method: Source-tree construction method, either `"longest_axis"` or
            `"morton_lbvh"`.
        par: Whether to evaluate target batches in parallel.
        out: Optional contiguous and aligned output component arrays to fill.
        extra_diagnostics: Whether to populate source-tree diagnostics that require extra data.

    Returns:
        Field component arrays and diagnostics. If `out` is provided, returns `out` in
        `SolveResult.field`.
    """
    ...

def vector_potential_dipole_hierarchical(
    loc: ArrayTriple,
    moment: ArrayTriple,
    obs: ArrayTriple,
    outer_radius: FloatArray,
    theta: float = 0.01,
    construction_method: str = "longest_axis",
    par: bool = True,
    out: ArrayTriple | None = None,
    extra_diagnostics: bool = False,
) -> SolveResult:
    """Hierarchical magnetic vector potential of dipoles in Cartesian coordinates.

    This one-shot method builds the source tree internally. This is an approximate method,
    and no particular accuracy level is guaranteed. Truncated methods like this one may
    average entire local loop structures out of existence; as a result, maximum relative
    error is unbounded. This method must be tuned to a given use-case in order to be
    useful, and should not be used to calculate safety-related field limits.

    Args:
        loc: Dipole source coordinates as component arrays.
        moment: Dipole magnetic moment components.
        obs: Target point coordinates as component arrays.
        outer_radius: Magnetized-sphere radius for each source. Use zeros for point dipoles.
        theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
        construction_method: Source-tree construction method, either `"longest_axis"` or
            `"morton_lbvh"`.
        par: Whether to evaluate target batches in parallel.
        out: Optional contiguous and aligned output component arrays to fill.
        extra_diagnostics: Whether to populate source-tree diagnostics that require extra data.

    Returns:
        Field component arrays and diagnostics. If `out` is provided, returns `out` in
        `SolveResult.field`.
    """
    ...

def flux_density_linear_filament(
    xyzp: ArrayTriple,
    xyzfil: ArrayTriple,
    dlxyzfil: ArrayTriple,
    ifil: FloatArray,
    wire_radius: FloatArray,
    par: bool = True,
) -> ArrayTriple: ...
def flux_density_linear_filament_hierarchical(
    xyzp: ArrayTriple,
    xyzfil: ArrayTriple,
    dlxyzfil: ArrayTriple,
    ifil: FloatArray,
    wire_radius: FloatArray,
    theta: float = 0.05,
    construction_method: str = "longest_axis",
    par: bool = True,
    out: ArrayTriple | None = None,
    extra_diagnostics: bool = False,
) -> SolveResult:
    """Hierarchical B-field calculation for many linear filament segments.

    This one-shot method builds the source tree internally. This is an approximate method,
    and no particular accuracy level is guaranteed. Truncated methods like this one may
    average entire local loop structures out of existence; as a result, maximum relative
    error is unbounded. This method must be tuned to a given use-case in order to be
    useful, and should not be used to calculate safety-related field limits.

    Args:
        xyzp: Target point coordinates as component arrays.
        xyzfil: Filament segment start coordinates as component arrays.
        dlxyzfil: Filament segment start-to-end displacement components.
        ifil: Current in each filament segment.
        wire_radius: Wire radius for each filament segment. Use zeros for thin wires.
        theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
        construction_method: Source-tree construction method, either `"longest_axis"` or
            `"morton_lbvh"`.
        par: Whether to evaluate target batches in parallel.
        out: Optional contiguous and aligned output component arrays to fill.
        extra_diagnostics: Whether to populate source-tree diagnostics that require extra data.

    Returns:
        Field component arrays and diagnostics. If `out` is provided, returns `out` in
        `SolveResult.field`.
    """
    ...

def flux_density_linear_filament_matrix(
    xyzp: ArrayTriple,
    xyzfil: ArrayTriple,
    dlxyzfil: ArrayTriple,
    ifil: FloatArray,
    wire_radius: FloatArray,
    par: bool = True,
) -> ArrayTriple: ...
def flux_density_point_segment(
    xyzp: ArrayTriple,
    xyzfil: ArrayTriple,
    dlxyzfil: ArrayTriple,
    ifil: FloatArray,
    par: bool,
) -> ArrayTriple: ...
def vector_potential_linear_filament(
    xyzp: ArrayTriple,
    xyzfil: ArrayTriple,
    dlxyzfil: ArrayTriple,
    ifil: FloatArray,
    wire_radius: FloatArray,
    par: bool = True,
) -> ArrayTriple: ...
def vector_potential_linear_filament_hierarchical(
    xyzp: ArrayTriple,
    xyzfil: ArrayTriple,
    dlxyzfil: ArrayTriple,
    ifil: FloatArray,
    wire_radius: FloatArray,
    theta: float = 0.05,
    construction_method: str = "longest_axis",
    par: bool = True,
    out: ArrayTriple | None = None,
    extra_diagnostics: bool = False,
) -> SolveResult:
    """Hierarchical A-field calculation for many linear filament segments.

    This one-shot method builds the source tree internally. This is an approximate method,
    and no particular accuracy level is guaranteed. Truncated methods like this one may
    average entire local loop structures out of existence; as a result, maximum relative
    error is unbounded. This method must be tuned to a given use-case in order to be
    useful, and should not be used to calculate safety-related field limits.

    Args:
        xyzp: Target point coordinates as component arrays.
        xyzfil: Filament segment start coordinates as component arrays.
        dlxyzfil: Filament segment start-to-end displacement components.
        ifil: Current in each filament segment.
        wire_radius: Wire radius for each filament segment. Use zeros for thin wires.
        theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
        construction_method: Source-tree construction method, either `"longest_axis"` or
            `"morton_lbvh"`.
        par: Whether to evaluate target batches in parallel.
        out: Optional contiguous and aligned output component arrays to fill.
        extra_diagnostics: Whether to populate source-tree diagnostics that require extra data.

    Returns:
        Field component arrays and diagnostics. If `out` is provided, returns `out` in
        `SolveResult.field`.
    """
    ...

def vector_potential_linear_filament_matrix(
    xyzp: ArrayTriple,
    xyzfil: ArrayTriple,
    dlxyzfil: ArrayTriple,
    ifil: FloatArray,
    wire_radius: FloatArray,
    par: bool = True,
) -> ArrayTriple: ...
def vector_potential_point_segment(
    xyzp: ArrayTriple,
    xyzfil: ArrayTriple,
    dlxyzfil: ArrayTriple,
    ifil: FloatArray,
    par: bool,
) -> ArrayTriple: ...
def inductance_piecewise_linear_filaments(
    xyzfil0: ArrayTriple,
    dlxyzfil0: ArrayTriple,
    xyzfil1: ArrayTriple,
    dlxyzfil1: ArrayTriple,
    wire_radius: FloatArray,
) -> float: ...
def inductance_linear_filaments(
    xyzfil_tgt: ArrayTriple,
    dlxyzfil_tgt: ArrayTriple,
    xyzfil_src: ArrayTriple,
    dlxyzfil_src: ArrayTriple,
    wire_radius_src: FloatArray,
) -> FloatArray: ...
def inductance_linear_filaments_matrix(
    xyzfil_tgt: ArrayTriple,
    dlxyzfil_tgt: ArrayTriple,
    xyzfil_src: ArrayTriple,
    dlxyzfil_src: ArrayTriple,
    wire_radius_src: FloatArray,
    par: bool = True,
) -> FloatArray: ...
def gs_operator_order2(rs: FloatArray, zs: FloatArray) -> tuple[FloatArray, UIntArray, UIntArray]: ...
def gs_operator_order4(rs: FloatArray, zs: FloatArray) -> tuple[FloatArray, UIntArray, UIntArray]: ...
def flux_density_triangle_mesh(
    obs: FloatMatrix,
    nodes: FloatMatrix,
    triangles: IntMatrix,
    s: FloatArray,
    par: bool = True,
) -> ArrayTriple: ...
def vector_potential_triangle_mesh(
    obs: FloatMatrix,
    nodes: FloatMatrix,
    triangles: IntMatrix,
    s: FloatArray,
    par: bool = True,
) -> ArrayTriple: ...
def flux_density_triangle_mesh_hierarchical(
    obs: FloatMatrix,
    nodes: FloatMatrix,
    triangles: IntMatrix,
    s: FloatArray,
    theta: float = 0.05,
    construction_method: str = "longest_axis",
    par: bool = True,
    out: ArrayTriple | None = None,
    extra_diagnostics: bool = False,
) -> SolveResult:
    """Hierarchical B-field calculation for a triangle mesh with nodal stream-function values.

    This one-shot method builds the source tree internally. This is an approximate method,
    and no particular accuracy level is guaranteed. Truncated methods like this one may
    average entire local loop structures out of existence; as a result, maximum relative
    error is unbounded. This method must be tuned to a given use-case in order to be
    useful, and should not be used to calculate safety-related field limits.

    Direct triangle interactions use the analytic uniform-triangle field. A source
    triangle contributes zero at target points geometrically on that triangle.

    References:
        D. R. Wilton, J. Rivero, W. A. Johnson, and F. Vipiana,
        “Evaluation of Static Potential Integrals on Triangular Domains,”
        IEEE Access, vol. 8, pp. 99806–99819, 2020.
        <https://doi.org/10.1109/ACCESS.2020.2997287>

    Args:
        obs: Target point coordinates with one point per row.
        nodes: Mesh node coordinates with one node per row.
        triangles: Triangle node indices with one triangle per row.
        s: Nodal stream-function values.
        theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
        construction_method: Source-tree construction method, either `"longest_axis"` or
            `"morton_lbvh"`.
        par: Whether to evaluate target batches in parallel.
        out: Optional contiguous and aligned output component arrays to fill.
        extra_diagnostics: Whether to populate source-tree diagnostics that require extra data.

    Returns:
        Field component arrays and diagnostics. If `out` is provided, returns `out` in
        `SolveResult.field`.
    """
    ...

def vector_potential_triangle_mesh_hierarchical(
    obs: FloatMatrix,
    nodes: FloatMatrix,
    triangles: IntMatrix,
    s: FloatArray,
    theta: float = 0.05,
    construction_method: str = "longest_axis",
    par: bool = True,
    out: ArrayTriple | None = None,
    extra_diagnostics: bool = False,
) -> SolveResult:
    """Hierarchical A-field calculation for a triangle mesh with nodal stream-function values.

    This one-shot method builds the source tree internally. This is an approximate method,
    and no particular accuracy level is guaranteed. Truncated methods like this one may
    average entire local loop structures out of existence; as a result, maximum relative
    error is unbounded. This method must be tuned to a given use-case in order to be
    useful, and should not be used to calculate safety-related field limits.

    Direct triangle interactions use the exact uniform-triangle potential, which
    is finite and continuous on triangle interiors, edges, and vertices.

    References:
        D. R. Wilton, J. Rivero, W. A. Johnson, and F. Vipiana,
        “Evaluation of Static Potential Integrals on Triangular Domains,”
        IEEE Access, vol. 8, pp. 99806–99819, 2020.
        <https://doi.org/10.1109/ACCESS.2020.2997287>

    Args:
        obs: Target point coordinates with one point per row.
        nodes: Mesh node coordinates with one node per row.
        triangles: Triangle node indices with one triangle per row.
        s: Nodal stream-function values.
        theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
        construction_method: Source-tree construction method, either `"longest_axis"` or
            `"morton_lbvh"`.
        par: Whether to evaluate target batches in parallel.
        out: Optional contiguous and aligned output component arrays to fill.
        extra_diagnostics: Whether to populate source-tree diagnostics that require extra data.

    Returns:
        Field component arrays and diagnostics. If `out` is provided, returns `out` in
        `SolveResult.field`.
    """
    ...

def flux_density_triangle_mesh_mapping(
    obs: FloatMatrix,
    nodes: FloatMatrix,
    triangles: IntMatrix,
    par: bool = True,
) -> ArrayTriple: ...
def vector_potential_triangle_mesh_mapping(
    obs: FloatMatrix,
    nodes: FloatMatrix,
    triangles: IntMatrix,
    par: bool = True,
) -> ArrayTriple: ...
def triangle_mesh_current_density(nodes: FloatMatrix, triangles: IntMatrix, s: FloatArray) -> ArrayTriple: ...
def triangle_mesh_quadrature_points(
    nodes: FloatMatrix, triangles: IntMatrix, quad: str = "dunavant3"
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, int]: ...
def triangle_mesh_inductance_matrix(
    nodes: FloatMatrix, triangles: IntMatrix, par: bool = True, quad: str = "dunavant3"
) -> FloatArray: ...
def triangle_mesh_inductance_mapping_from_linear_filaments(
    xyzfil: ArrayTriple,
    dlxyzfil: ArrayTriple,
    wire_radius: FloatArray,
    nodes_tgt: FloatMatrix,
    triangles_tgt: IntMatrix,
    par: bool = True,
    quad: str = "dunavant3",
) -> FloatArray: ...
def triangle_mesh_inductance_mapping_from_circular_filaments(
    rfil: FloatArray,
    zfil: FloatArray,
    nodes_tgt: FloatMatrix,
    triangles_tgt: IntMatrix,
    par: bool = True,
    quad: str = "dunavant3",
) -> FloatArray: ...
def triangle_mesh_flux_linkage_mapping_from_dipoles(
    loc: ArrayTriple,
    moment_dir: ArrayTriple,
    outer_radius: FloatArray,
    nodes_tgt: FloatMatrix,
    triangles_tgt: IntMatrix,
    par: bool = True,
    quad: str = "dunavant3",
) -> FloatArray: ...
def triangle_mesh_force_mapping(
    nodes_src: FloatMatrix,
    triangles_src: IntMatrix,
    nodes_tgt: FloatMatrix,
    triangles_tgt: IntMatrix,
    s_tgt: FloatArray,
    par: bool = True,
    quad: str = "dunavant3",
) -> ArrayTriple: ...
def triangle_mesh_self_force_mapping(
    nodes: FloatMatrix,
    triangles: IntMatrix,
    s: FloatArray,
    par: bool = True,
    quad: str = "dunavant3",
) -> ArrayTriple: ...
def triangle_mesh_force_mapping_from_linear_filaments(
    xyzfil: ArrayTriple,
    dlxyzfil: ArrayTriple,
    wire_radius: FloatArray,
    nodes_tgt: FloatMatrix,
    triangles_tgt: IntMatrix,
    s_tgt: FloatArray,
    par: bool = True,
    quad: str = "dunavant3",
) -> ArrayTriple: ...
def triangle_mesh_force_mapping_from_circular_filaments(
    rfil: FloatArray,
    zfil: FloatArray,
    nodes_tgt: FloatMatrix,
    triangles_tgt: IntMatrix,
    s_tgt: FloatArray,
    par: bool = True,
    quad: str = "dunavant3",
) -> ArrayTriple: ...
def triangle_mesh_force_mapping_from_dipoles(
    loc: ArrayTriple,
    moment_dir: ArrayTriple,
    outer_radius: FloatArray,
    nodes_tgt: FloatMatrix,
    triangles_tgt: IntMatrix,
    s_tgt: FloatArray,
    par: bool = True,
    quad: str = "dunavant3",
) -> ArrayTriple: ...
def solenoid_stress_fem_assemble_model_2d_f64(
    nodes: FloatMatrix,
    elements: UIntMatrix,
    material_ids: UIntArray,
    material_table: FloatTensor3,
    pressure_faces: UIntMatrix,
    traction_faces: UIntMatrix,
    thermal_material_table: FloatMatrix,
    material_orientation_angles: FloatArray,
    prescribed_dofs: UIntArray,
    prescribed_values: FloatArray,
    element_type: int,
    formulation: int,
    thickness: float,
    quadrature: int,
    par: bool,
) -> SolenoidStress2dModelF64: ...
def solenoid_stress_fem_assemble_model_2d_f32(
    nodes: Float32Matrix,
    elements: UIntMatrix,
    material_ids: UIntArray,
    material_table: Float32Tensor3,
    pressure_faces: UIntMatrix,
    traction_faces: UIntMatrix,
    thermal_material_table: Float32Matrix,
    material_orientation_angles: Float32Array,
    prescribed_dofs: UIntArray,
    prescribed_values: Float32Array,
    element_type: int,
    formulation: int,
    thickness: float,
    quadrature: int,
    par: bool,
) -> SolenoidStress2dModelF32: ...
def solenoid_stress_fem_cfsem_radial_material_f64(
    youngs_modulus: float, poisson_ratio: float
) -> FloatArray: ...
def solenoid_stress_fem_cfsem_radial_material_f32(
    youngs_modulus: float, poisson_ratio: float
) -> Float32Array: ...
def solenoid_stress_fem_isotropic_axisymmetric_material_f64(
    youngs_modulus: float, poisson_ratio: float
) -> FloatArray: ...
def solenoid_stress_fem_isotropic_axisymmetric_material_f32(
    youngs_modulus: float, poisson_ratio: float
) -> Float32Array: ...
def solenoid_stress_fem_isotropic_plane_strain_material_f64(
    youngs_modulus: float, poisson_ratio: float
) -> FloatArray: ...
def solenoid_stress_fem_isotropic_plane_strain_material_f32(
    youngs_modulus: float, poisson_ratio: float
) -> Float32Array: ...
def solenoid_stress_fem_isotropic_axisymmetric_thermal_material_f64(
    alpha: float, reference_temperature: float
) -> FloatArray: ...
def solenoid_stress_fem_isotropic_axisymmetric_thermal_material_f32(
    alpha: float, reference_temperature: float
) -> Float32Array: ...
def solenoid_stress_fem_isotropic_plane_strain_thermal_material_f64(
    alpha: float, reference_temperature: float
) -> FloatArray: ...
def solenoid_stress_fem_isotropic_plane_strain_thermal_material_f32(
    alpha: float, reference_temperature: float
) -> Float32Array: ...
def solenoid_stress_fem_orthotropic_axisymmetric_thermal_material_f64(
    alpha_r: float, alpha_z: float, alpha_t: float, reference_temperature: float
) -> FloatArray: ...
def solenoid_stress_fem_orthotropic_axisymmetric_thermal_material_f32(
    alpha_r: float, alpha_z: float, alpha_t: float, reference_temperature: float
) -> Float32Array: ...
def solenoid_stress_fem_infer_quad9_mesh_f64(
    nodes: FloatMatrix, elements: UIntMatrix
) -> tuple[FloatArray, UIntArray, UIntArray, UIntArray, UIntArray]: ...
def solenoid_stress_fem_infer_quad9_mesh_f32(
    nodes: Float32Matrix, elements: UIntMatrix
) -> tuple[Float32Array, UIntArray, UIntArray, UIntArray, UIntArray]: ...
def solenoid_stress_fem_quad_mesh_query_f64(
    nodes: FloatMatrix,
    elements: UIntMatrix,
    points: FloatMatrix,
    element_type: str,
    max_iterations: int,
) -> QuadMeshQueryF64: ...
def solenoid_stress_fem_quad_mesh_query_f32(
    nodes: Float32Matrix,
    elements: UIntMatrix,
    points: Float32Matrix,
    element_type: str,
    max_iterations: int,
) -> QuadMeshQueryF32: ...
def solenoid_stress_fem_quad_mesh_interpolation_operator_f64(
    nodes: FloatMatrix,
    elements: UIntMatrix,
    element_indices: UIntArray,
    reference_points: FloatMatrix,
    element_type: str,
) -> SparseF64: ...
def solenoid_stress_fem_quad_mesh_interpolation_operator_f32(
    nodes: Float32Matrix,
    elements: UIntMatrix,
    element_indices: UIntArray,
    reference_points: Float32Matrix,
    element_type: str,
) -> SparseF32: ...
def solenoid_stress_fem_quad_mesh_strain_operator_f64(
    nodes: FloatMatrix,
    elements: UIntMatrix,
    element_indices: UIntArray,
    reference_points: FloatMatrix,
    element_type: str,
    formulation: int,
    thickness: float,
) -> SparseF64: ...
def solenoid_stress_fem_quad_mesh_strain_operator_f32(
    nodes: Float32Matrix,
    elements: UIntMatrix,
    element_indices: UIntArray,
    reference_points: Float32Matrix,
    element_type: str,
    formulation: int,
    thickness: float,
) -> SparseF32: ...
def solenoid_stress_fem_quad_mesh_stress_operator_f64(
    nodes: FloatMatrix,
    elements: UIntMatrix,
    element_indices: UIntArray,
    reference_points: FloatMatrix,
    material_ids: UIntArray,
    material_table: FloatTensor3,
    material_orientation_angles: FloatArray,
    element_type: str,
    formulation: int,
    thickness: float,
) -> SparseF64: ...
def solenoid_stress_fem_quad_mesh_stress_operator_f32(
    nodes: Float32Matrix,
    elements: UIntMatrix,
    element_indices: UIntArray,
    reference_points: Float32Matrix,
    material_ids: UIntArray,
    material_table: Float32Tensor3,
    material_orientation_angles: Float32Array,
    element_type: str,
    formulation: int,
    thickness: float,
) -> SparseF32: ...
