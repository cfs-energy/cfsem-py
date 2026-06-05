//! Shared numeric traits and constants for the solenoid-stress backend.

use deimos_numerics::sparse::CompensatedField;
use faer_traits::RealField;
use num_traits::{Float, FromPrimitive};

/// Floating-point trait bound used throughout the solenoid-stress backend.
///
/// Keeping the bound in one place makes it easier to support both `f32` and `f64` entry points
/// without duplicating generic constraints everywhere else.
pub trait Real:
    Float
    + FromPrimitive
    + RealField
    + CompensatedField
    + Copy
    + std::fmt::Debug
    + Send
    + Sync
    + 'static
{
}

impl<T> Real for T where
    T: Float
        + FromPrimitive
        + RealField
        + CompensatedField
        + Copy
        + std::fmt::Debug
        + Send
        + Sync
        + 'static
{
}

/// Cast a literal `f64` constant into the active floating-point type.
pub fn cast<F: Real>(value: f64) -> F {
    F::from_f64(value).expect("finite f64 literal should cast to target float")
}

/// Return the constant `2*pi` in the active floating-point type.
pub fn two_pi<F: Real>() -> F {
    cast(2.0 * core::f64::consts::PI)
}

/// Structural 2D reduction used by the quadrilateral FEM backend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Structural2dFormulation<F: Real> {
    /// Axisymmetric reduction in `(r, z)` with hoop strain `e_tt = u_r / r`.
    Axisymmetric,
    /// Plane-strain reduction in `(x, y)` with `e_zz = 0` and finite model thickness.
    PlaneStrain {
        /// Out-of-plane thickness used to convert analysis-plane integrals into 3D volume.
        thickness: F,
    },
}

impl<F: Real> Structural2dFormulation<F> {
    /// Parse the compact formulation code used by the low-level Python binding.
    pub fn from_code(code: u8, thickness: F) -> Result<Self, String> {
        match code {
            0 => Ok(Self::Axisymmetric),
            1 => {
                if thickness <= F::zero() {
                    return Err(format!(
                        "plane-strain thickness must be positive; got {thickness:?}"
                    ));
                }
                Ok(Self::PlaneStrain { thickness })
            }
            _ => Err(format!(
                "unsupported structural 2D formulation code {code}; use 0 for axisymmetric or 1 for plane_strain"
            )),
        }
    }

    /// Return the canonical public string spelling used by the Python wrapper and docs.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Axisymmetric => "axisymmetric",
            Self::PlaneStrain { .. } => "plane_strain",
        }
    }

    /// Return the axisymmetric swept-volume or planar-thickness measure for one volume sample.
    pub fn volume_scale(self, point: [F; 2], det_j: F, weight: F) -> Result<F, String> {
        match self {
            Self::Axisymmetric => {
                if point[0] < F::zero() {
                    return Err(format!(
                        "quadrature point has negative radius {:?}; axisymmetric radius must be nonnegative",
                        point[0]
                    ));
                }
                Ok(two_pi::<F>() * point[0] * det_j * weight)
            }
            Self::PlaneStrain { thickness } => Ok(thickness * det_j * weight),
        }
    }

    /// Return the axisymmetric swept-surface or planar-thickness measure for one face sample.
    pub fn face_scale(self, point: [F; 2], line_jacobian: F, weight: F) -> Result<F, String> {
        match self {
            Self::Axisymmetric => {
                if point[0] < F::zero() {
                    return Err(format!(
                        "face quadrature point has negative radius {:?}; axisymmetric radius must be nonnegative",
                        point[0]
                    ));
                }
                Ok(two_pi::<F>() * point[0] * line_jacobian * weight)
            }
            Self::PlaneStrain { thickness } => Ok(thickness * line_jacobian * weight),
        }
    }
}

/// Number of displacement unknowns carried by each node in the axisymmetric structural solver.
pub const DOF_PER_NODE: usize = 2;

/// Return the number of structural displacement unknowns carried by one element with the given
/// node count.
pub const fn dof_per_element(nodes_per_element: usize) -> usize {
    DOF_PER_NODE * nodes_per_element
}

/// Map one element's node indices to the global displacement DOF indices
/// `[u_r1, u_z1, u_r2, u_z2, ...]`.
pub fn local_dofs<const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize>(
    nodes: &[usize; NODES_PER_ELEMENT],
) -> [usize; DOF_PER_ELEMENT] {
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    let mut local_dofs = [0usize; DOF_PER_ELEMENT];
    for (local_node, global_node) in nodes.iter().copied().enumerate() {
        local_dofs[2 * local_node] = 2 * global_node;
        local_dofs[2 * local_node + 1] = 2 * global_node + 1;
    }
    local_dofs
}

/// Validate per-element material-index arrays shared by assembly and recovery code.
pub(crate) fn validate_element_material_inputs<F: Real>(
    nelem: usize,
    material_ids: &[usize],
    material_orientation_angles: Option<&[F]>,
) -> Result<(), String> {
    if material_ids.len() != nelem {
        return Err(format!(
            "material_ids has length {}, but mesh has {} elements",
            material_ids.len(),
            nelem
        ));
    }
    if let Some(angles) = material_orientation_angles
        && angles.len() != nelem
    {
        return Err(format!(
            "material_orientation_angles has length {}, but mesh has {} elements",
            angles.len(),
            nelem
        ));
    }
    Ok(())
}

/// Scatter one local vector into sparse triplet storage.
pub(crate) fn scatter_local_vector<F: Real, const NROW: usize>(
    rows: &mut Vec<usize>,
    cols: &mut Vec<usize>,
    vals: &mut Vec<F>,
    global_rows: &[usize; NROW],
    global_col: usize,
    local: &[F; NROW],
) {
    for row in 0..NROW {
        let value = local[row];
        if value != F::zero() {
            rows.push(global_rows[row]);
            cols.push(global_col);
            vals.push(value);
        }
    }
}

/// Scatter one local dense block into sparse triplet storage.
pub(crate) fn scatter_local_matrix<F: Real, const NROW: usize, const NCOL: usize>(
    rows: &mut Vec<usize>,
    cols: &mut Vec<usize>,
    vals: &mut Vec<F>,
    global_rows: &[usize; NROW],
    global_cols: &[usize; NCOL],
    local: &[[F; NCOL]; NROW],
) {
    for row in 0..NROW {
        for col in 0..NCOL {
            let value = local[row][col];
            if value != F::zero() {
                rows.push(global_rows[row]);
                cols.push(global_cols[col]);
                vals.push(value);
            }
        }
    }
}

/// One scalar normal-pressure load topology entry for one element face.
///
/// The pressure amplitude is supplied later at `build_rhs(...)` time, not stored here.
#[derive(Clone, Copy, Debug)]
pub struct PressureLoad {
    /// Element index receiving the load.
    pub element: usize,
    /// Local face index in the element-family numbering used by `face_reference`:
    /// `0 = bottom`, `1 = right`, `2 = top`, `3 = left`.
    pub local_face: u8,
}

/// One traction-vector load topology entry for one element face.
///
/// The traction amplitudes are supplied later at `build_rhs(...)` time, not stored here.
#[derive(Clone, Copy, Debug)]
pub struct TractionLoad {
    /// Element index receiving the load.
    pub element: usize,
    /// Local face index in the element-family numbering used by `face_reference`:
    /// `0 = bottom`, `1 = right`, `2 = top`, `3 = left`.
    pub local_face: u8,
}

/// Per-material thermal-expansion data for the 2D thermoelastic model.
#[derive(Clone, Copy, Debug)]
pub struct ThermalMaterial<F: Real> {
    /// Thermal strain coefficients in the active four-component strain order.
    ///
    /// Axisymmetric models use `[rr, zz, tt, rz]`; plane-strain models use `[xx, yy, zz, xy]`.
    ///
    /// Units: `[strain / temperature]`.
    pub alpha: [F; 4],
    /// Stress-free reference temperature for this material.
    ///
    /// Units: `[temperature]`.
    pub reference_temperature: F,
}

/// Sparse triplets for the assembled structural stiffness matrix before CSC compression.
///
/// Each entry has units
/// `[generalized nodal force / displacement] = [energy / distance^2]`,
/// which is the axisymmetric analogue of stiffness.
#[derive(Debug, Clone)]
pub struct StiffnessTriplets<F: Real> {
    /// Sparse row indices for the assembled stiffness-operator triplets.
    pub rows: Vec<usize>,
    /// Sparse column indices for the assembled stiffness-operator triplets.
    pub cols: Vec<usize>,
    /// Sparse values for the assembled stiffness-operator triplets.
    pub vals: Vec<F>,
}
