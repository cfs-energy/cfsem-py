//! Rust-native convenience helpers layered on top of the axisymmetric FEM backend.
//!
//! These helpers package common construction and postprocessing tasks so both Rust and Python can
//! use the same domain-level operations without duplicating elastic stress-strain matrix or
//! mesh-elevation logic.

use std::collections::HashMap;

use crate::mesh::elements::quad2d::quad4;
use crate::physics::solenoid_stress::types::{Real, ThermalMaterial, cast};

/// Per-element quadrature data in element-major flattened form.
///
/// `points_rz`, `weights_area`, and `weights_volume` are stored in the same element-major order.
/// `points_rz` has flattened shape `(nelem * nq_per_element, 2)`,
/// `weights_area` and `weights_volume` have flattened shape `(nelem * nq_per_element,)`, and
/// `nq_per_element` gives the number of consecutive quadrature entries belonging to each element.
#[derive(Debug, Clone)]
pub struct AxisymmetricElementQuadrature<F: Real> {
    /// Physical quadrature-point coordinates `(r, z)` in element-major order.
    ///
    /// Flattened shape: `(nelem * nq_per_element, 2)`.
    /// Units: `[length]`.
    pub points_rz: Vec<[F; 2]>,
    /// Mapped meridian-area weights `det(J) w` in element-major order.
    ///
    /// Flattened shape: `(nelem * nq_per_element,)`.
    /// Units: `[area]`.
    pub weights_area: Vec<F>,
    /// Mapped swept-volume weights `2*pi*r det(J) w` in element-major order.
    ///
    /// Flattened shape: `(nelem * nq_per_element,)`.
    /// Units: `[volume]`.
    pub weights_volume: Vec<F>,
    /// Number of quadrature points contributed by each element.
    pub nq_per_element: usize,
}

/// Per-element meridian area and swept volume.
///
/// Both vectors have shape `(nelem,)`.
#[derive(Debug, Clone)]
pub struct AxisymmetricElementMeasures<F: Real> {
    /// Meridian-plane area of each element.
    ///
    /// Shape: `(nelem,)`.
    /// Units: `[area]`.
    pub areas: Vec<F>,
    /// Swept 3D volume represented by each axisymmetric element.
    ///
    /// Shape: `(nelem,)`.
    /// Units: `[volume]`.
    pub swept_volumes: Vec<F>,
}

/// Recovered quadrature-point fields in element-major flattened form.
///
/// Each field vector stores one `[rr, zz, tt, rz]` sample per quadrature point in element-major
/// order. `points_rz` has flattened shape `(nelem * nq_per_element, 2)`. Each tensor field has
/// flattened shape `(nelem * nq_per_element, 4)`. `nq_per_element` records how many consecutive
/// samples belong to each element.
#[derive(Debug, Clone)]
pub struct QuadratureFieldSamples<F: Real> {
    /// Physical quadrature-point coordinates `(r, z)` in element-major order.
    ///
    /// Flattened shape: `(nelem * nq_per_element, 2)`.
    /// Units: `[length]`.
    pub points_rz: Vec<[F; 2]>,
    /// Total strain samples `[e_rr, e_zz, e_tt, g_rz]`.
    ///
    /// Flattened shape: `(nelem * nq_per_element, 4)`.
    /// Units: `[strain]`.
    pub strain: Vec<[F; 4]>,
    /// Thermal strain samples in the same ordering as `strain`.
    ///
    /// Flattened shape: `(nelem * nq_per_element, 4)`.
    /// Units: `[strain]`.
    pub thermal_strain: Vec<[F; 4]>,
    /// Elastic strain samples `strain - thermal_strain`.
    ///
    /// Flattened shape: `(nelem * nq_per_element, 4)`.
    /// Units: `[strain]`.
    pub elastic_strain: Vec<[F; 4]>,
    /// Stress samples `[sigma_rr, sigma_zz, sigma_tt, tau_rz]`.
    ///
    /// Flattened shape: `(nelem * nq_per_element, 4)`.
    /// Units: `[stress]`.
    pub stress: Vec<[F; 4]>,
    /// Number of quadrature points contributed by each element.
    pub nq_per_element: usize,
}

/// Explicit 9-node analysis mesh inferred from a corner-only 4-node quadrilateral mesh.
///
/// `input_nodes` has shape `(nnode, 2)`, `input_elements` has shape `(nelem, 4)`,
/// `analysis_nodes` has shape `(n_analysis_nodes, 2)`, and `analysis_elements` has shape
/// `(nelem, 9)`.
#[derive(Debug, Clone)]
pub struct ElevatedQuad9Mesh<F: Real> {
    /// Input corner-node coordinates.
    ///
    /// Shape: `(nnode, 2)`.
    /// Units: `[length]`.
    pub input_nodes: Vec<[F; 2]>,
    /// Input quad4 connectivity.
    ///
    /// Shape: `(nelem, 4)`.
    pub input_elements: Vec<[usize; 4]>,
    /// Elevated quad9 node coordinates.
    ///
    /// Shape: `(n_analysis_nodes, 2)`.
    /// Units: `[length]`.
    pub analysis_nodes: Vec<[F; 2]>,
    /// Elevated quad9 connectivity.
    ///
    /// Shape: `(nelem, 9)`.
    /// Each element stores nodes in the local quad9 order:
    /// - corners `0..=3` in counter-clockwise order `[bottom-left, bottom-right, top-right, top-left]`
    /// - midsides `4..=7` on faces `[bottom, right, top, left]`
    /// - center node `8`
    pub analysis_elements: Vec<[usize; 9]>,
    /// Indices of the corner nodes in `analysis_nodes`.
    ///
    /// Shape: `(nnode,)`.
    pub corner_node_indices: Vec<usize>,
    /// Indices of the unique midside nodes in `analysis_nodes`.
    ///
    /// Shape: `(n_midside_nodes,)`.
    pub midside_node_indices: Vec<usize>,
    /// Indices of the center nodes in `analysis_nodes`, one per element.
    ///
    /// Shape: `(nelem,)`.
    pub center_node_indices: Vec<usize>,
}

/// Construct the isotropic axisymmetric elastic stress-strain matrix.
///
/// Args:
///     youngs_modulus: Young's modulus with units `[pressure]`.
///     poisson_ratio: Poisson ratio with units `[dimensionless]`.
///
/// Returns:
///     Elastic stress-strain matrix with shape `(4, 4)` in component order
///     `[rr, zz, tt, rz]`. Units are `[stress / strain] = [pressure]`.
pub fn isotropic_axisymmetric_material<F: Real>(
    youngs_modulus: F,
    poisson_ratio: F,
) -> [[F; 4]; 4] {
    let two = cast::<F>(2.0);
    let lam = youngs_modulus * poisson_ratio
        / ((F::one() + poisson_ratio) * (F::one() - two * poisson_ratio));
    let mu = youngs_modulus / (two * (F::one() + poisson_ratio));
    [
        [lam + two * mu, lam, lam, F::zero()],
        [lam, lam + two * mu, lam, F::zero()],
        [lam, lam, lam + two * mu, F::zero()],
        [F::zero(), F::zero(), F::zero(), mu],
    ]
}

/// Construct isotropic thermal-expansion data.
///
/// Args:
///     alpha: Isotropic thermal expansion coefficient with units `[strain / temperature]`.
///     reference_temperature: Stress-free reference temperature with units `[temperature]`.
///
/// Returns:
///     Thermal material data with row shape `(5,)`, stored as
///     `[alpha_r, alpha_z, alpha_t, alpha_rz, T_ref]`. The first four entries have units
///     `[strain / temperature]`; `T_ref` has units `[temperature]`.
pub fn isotropic_axisymmetric_thermal_material<F: Real>(
    alpha: F,
    reference_temperature: F,
) -> ThermalMaterial<F> {
    ThermalMaterial {
        alpha: [alpha, alpha, alpha, F::zero()],
        reference_temperature,
    }
}

/// Construct orthotropic thermal-expansion data.
///
/// Args:
///     alpha_r: Radial thermal expansion coefficient with units `[strain / temperature]`.
///     alpha_z: Axial thermal expansion coefficient with units `[strain / temperature]`.
///     alpha_t: Hoop thermal expansion coefficient with units `[strain / temperature]`.
///     reference_temperature: Stress-free reference temperature with units `[temperature]`.
///
/// Returns:
///     Thermal material data with row shape `(5,)`, stored as
///     `[alpha_r, alpha_z, alpha_t, alpha_rz, T_ref]`. The first four entries have units
///     `[strain / temperature]`; `T_ref` has units `[temperature]`.
pub fn orthotropic_axisymmetric_thermal_material<F: Real>(
    alpha_r: F,
    alpha_z: F,
    alpha_t: F,
    reference_temperature: F,
) -> ThermalMaterial<F> {
    ThermalMaterial {
        alpha: [alpha_r, alpha_z, alpha_t, F::zero()],
        reference_temperature,
    }
}

/// Construct the reduced elastic matrix used by the 1D radial solver.
///
/// Args:
///     youngs_modulus: Young's modulus with units `[pressure]`.
///     poisson_ratio: Poisson ratio with units `[dimensionless]`.
///
/// Returns:
///     Elastic stress-strain matrix with shape `(4, 4)` in component order
///     `[rr, zz, tt, rz]`. Units are `[stress / strain] = [pressure]`.
pub fn cfsem_radial_material<F: Real>(youngs_modulus: F, poisson_ratio: F) -> [[F; 4]; 4] {
    let factor = youngs_modulus / (F::one() - poisson_ratio * poisson_ratio);
    let shear = youngs_modulus / (cast::<F>(2.0) * (F::one() + poisson_ratio));
    [
        [factor, F::zero(), factor * poisson_ratio, F::zero()],
        [F::zero(), youngs_modulus, F::zero(), F::zero()],
        [factor * poisson_ratio, F::zero(), factor, F::zero()],
        [F::zero(), F::zero(), F::zero(), shear],
    ]
}

/// Elevate a corner-only quad4 mesh to an explicit quad9 analysis mesh.
///
/// The output `analysis_elements` use the standard local quad9 ordering documented on
/// [`ElevatedQuad9Mesh::analysis_elements`]. In particular, midside nodes `4..=7` follow the
/// quad local face numbering `[bottom, right, top, left]`.
///
/// Args:
///     nodes_rz: Corner-node coordinates with shape `(nnode, 2)` in `(r, z)` order. Units are
///         `[length]`.
///     elements: Quad4 connectivity with shape `(nelem, 4)`. Corner nodes must be ordered
///         counter-clockwise in the `(r, z)` plane.
///
/// Returns:
///     Elevated quad9 mesh with:
///     - `input_nodes` shape `(nnode, 2)` and units `[length]`
///     - `input_elements` shape `(nelem, 4)`
///     - `analysis_nodes` shape `(n_analysis_nodes, 2)` and units `[length]`
///     - `analysis_elements` shape `(nelem, 9)`
///     - `corner_node_indices`, `midside_node_indices`, and `center_node_indices` as
///       one-dimensional index vectors.
pub fn infer_quad9_mesh<F: Real>(
    nodes_rz: &[[F; 2]],
    elements: &[[usize; 4]],
) -> Result<ElevatedQuad9Mesh<F>, String> {
    let node_count = nodes_rz.len();
    for (element_index, conn) in elements.iter().enumerate() {
        for &node in conn {
            if node >= node_count {
                return Err(format!(
                    "element {element_index} references node {node}, but mesh has only {node_count} nodes"
                ));
            }
        }
    }

    let mut analysis_nodes = nodes_rz.to_vec();
    let mut analysis_elements = vec![[0usize; 9]; elements.len()];
    let mut edge_to_midpoint = HashMap::<(usize, usize), usize>::new();
    let mut midside_node_indices = Vec::new();
    let mut center_node_indices = Vec::with_capacity(elements.len());

    for (element_index, conn) in elements.iter().copied().enumerate() {
        analysis_elements[element_index][..4].copy_from_slice(&conn);
        let coords = conn.map(|node| nodes_rz[node]);
        for (local_edge, (local_a, local_b)) in quad4::FACE_NODE_PAIRS.into_iter().enumerate() {
            let node_a = conn[local_a];
            let node_b = conn[local_b];
            let edge_key = if node_a < node_b {
                (node_a, node_b)
            } else {
                (node_b, node_a)
            };
            let midpoint_index = if let Some(&index) = edge_to_midpoint.get(&edge_key) {
                index
            } else {
                let midpoint = [
                    cast::<F>(0.5) * (coords[local_a][0] + coords[local_b][0]),
                    cast::<F>(0.5) * (coords[local_a][1] + coords[local_b][1]),
                ];
                let index = analysis_nodes.len();
                analysis_nodes.push(midpoint);
                edge_to_midpoint.insert(edge_key, index);
                midside_node_indices.push(index);
                index
            };
            analysis_elements[element_index][4 + local_edge] = midpoint_index;
        }

        let quarter = cast::<F>(0.25);
        let center = [
            quarter * (coords[0][0] + coords[1][0] + coords[2][0] + coords[3][0]),
            quarter * (coords[0][1] + coords[1][1] + coords[2][1] + coords[3][1]),
        ];
        let center_index = analysis_nodes.len();
        analysis_nodes.push(center);
        center_node_indices.push(center_index);
        analysis_elements[element_index][8] = center_index;
    }

    Ok(ElevatedQuad9Mesh {
        input_nodes: nodes_rz.to_vec(),
        input_elements: elements.to_vec(),
        analysis_nodes,
        analysis_elements,
        corner_node_indices: (0..nodes_rz.len()).collect(),
        midside_node_indices,
        center_node_indices,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        cfsem_radial_material, infer_quad9_mesh, isotropic_axisymmetric_material,
        orthotropic_axisymmetric_thermal_material,
    };

    fn two_element_strip_mesh() -> (Vec<[f64; 2]>, Vec<[usize; 4]>) {
        let nodes = vec![
            [0.5_f64, 0.0],
            [0.75, 0.0],
            [1.0, 0.0],
            [0.5, 0.2],
            [0.75, 0.2],
            [1.0, 0.2],
        ];
        let elements = vec![[0usize, 1, 4, 3], [1usize, 2, 5, 4]];
        (nodes, elements)
    }

    #[test]
    fn material_convenience_helpers_preserve_expected_invariants() {
        let isotropic = isotropic_axisymmetric_material(200.0e9_f64, 0.3_f64);
        for row in 0..4 {
            for col in 0..4 {
                assert!((isotropic[row][col] - isotropic[col][row]).abs() < 1.0e-12);
            }
        }

        let radial = cfsem_radial_material(200.0e9_f64, 0.3_f64);
        assert_eq!(radial[1][1], 200.0e9_f64);

        let thermal = orthotropic_axisymmetric_thermal_material(1.0_f64, 2.0_f64, 3.0_f64, 4.0);
        assert_eq!(thermal.alpha, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(thermal.reference_temperature, 4.0);
    }

    #[test]
    fn infer_quad9_mesh_reuses_shared_edge_midpoints_and_preserves_local_face_order() {
        let (nodes, elements) = two_element_strip_mesh();
        let elevated = infer_quad9_mesh(&nodes, &elements).expect("quad9 elevation");
        assert_eq!(elevated.analysis_elements.len(), 2);
        assert_eq!(elevated.corner_node_indices, vec![0, 1, 2, 3, 4, 5]);

        let first = elevated.analysis_elements[0];
        assert_eq!(elevated.analysis_nodes[first[4]], [0.625, 0.0]);
        assert_eq!(elevated.analysis_nodes[first[5]], [0.75, 0.1]);
        assert_eq!(elevated.analysis_nodes[first[6]], [0.625, 0.2]);
        assert_eq!(elevated.analysis_nodes[first[7]], [0.5, 0.1]);
        assert_eq!(elevated.analysis_nodes[first[8]], [0.625, 0.1]);
        assert_eq!(
            elevated.analysis_elements[0][5],
            elevated.analysis_elements[1][7]
        );
        assert_eq!(
            elevated.center_node_indices.len(),
            elevated.analysis_elements.len()
        );
    }
}
