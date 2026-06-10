//! Borrowed view of 2D quadrilateral mesh geometry and point-query helpers.

use std::collections::HashSet;

use crate::mesh::elements::quad2d::{mapping, quad4, quad9};
use crate::mesh::{Scalar, cast};
use crate::physics::solenoid_stress::{
    DOF_PER_NODE, Structural2dFormulation, build_b_matrix, rotate_material_in_plane,
    validate_element_material_inputs,
};

/// Borrowed view of a 2D quadrilateral mesh with fixed nodes per element.
#[derive(Clone, Copy)]
pub struct QuadMeshView2d<'a, F: Copy, const NODES_PER_ELEMENT: usize> {
    /// Node coordinates stored as a caller-defined 2D pair.
    pub nodes_rz: &'a [[F; 2]],
    /// Element connectivity in family-specific local-node order.
    pub elements: &'a [[usize; NODES_PER_ELEMENT]],
}

impl<'a, F: Copy, const NODES_PER_ELEMENT: usize> QuadMeshView2d<'a, F, NODES_PER_ELEMENT> {
    /// Number of mesh nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes_rz.len()
    }

    /// Number of elements.
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }

    /// Validate that every connectivity entry references an existing node.
    pub fn validate_connectivity(&self) -> Result<(), String> {
        let node_count = self.num_nodes();
        let mut duplicate_elements = Vec::new();
        for (element_index, element) in self.elements.iter().enumerate() {
            let mut unique_nodes = HashSet::with_capacity(NODES_PER_ELEMENT);
            for &node in element {
                if node >= node_count {
                    return Err(format!(
                        "element {element_index} references node {node}, but mesh has only {node_count} nodes"
                    ));
                }
                unique_nodes.insert(node);
            }
            if unique_nodes.len() != NODES_PER_ELEMENT {
                duplicate_elements.push((element_index, *element));
            }
        }
        if !duplicate_elements.is_empty() {
            let entries = duplicate_elements
                .iter()
                .map(|(element_index, element)| format!("{element_index} {element:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!(
                "warning: {} elements contain duplicate node indices: {entries}",
                duplicate_elements.len()
            );
        }
        Ok(())
    }

    /// Return the node indices of one element.
    pub fn element_nodes(
        &self,
        element_index: usize,
    ) -> Result<[usize; NODES_PER_ELEMENT], String> {
        self.elements
            .get(element_index)
            .copied()
            .ok_or_else(|| format!("element index {element_index} out of bounds"))
    }

    /// Gather the physical coordinates of one element's nodes.
    pub fn element_coords(
        &self,
        element_index: usize,
    ) -> Result<[[F; 2]; NODES_PER_ELEMENT], String> {
        let nodes = self.element_nodes(element_index)?;
        Ok(std::array::from_fn(|local_index| {
            self.nodes_rz[nodes[local_index]]
        }))
    }
}

/// Reference-element operations needed by generic quadrilateral mesh queries.
pub trait QuadReferenceElement<const NODES_PER_ELEMENT: usize> {
    /// Evaluate reference shape functions at `(\xi, \eta)`.
    fn shape<F: Scalar>(xi: F, eta: F) -> [F; NODES_PER_ELEMENT];

    /// Evaluate reference shape-function gradients at `(\xi, \eta)`.
    fn grad_ref<F: Scalar>(xi: F, eta: F) -> [[F; 2]; NODES_PER_ELEMENT];

    /// Map a face-local coordinate `s` onto the reference element.
    fn face_reference<F: Scalar>(local_face: u8, s: F) -> Result<(F, F, [F; 2]), String>;
}

/// Bilinear quad4 reference element marker for generic mesh queries.
pub struct Quad4ReferenceElement;

impl QuadReferenceElement<4> for Quad4ReferenceElement {
    fn shape<F: Scalar>(xi: F, eta: F) -> [F; 4] {
        quad4::shape(xi, eta)
    }

    fn grad_ref<F: Scalar>(xi: F, eta: F) -> [[F; 2]; 4] {
        quad4::grad_ref(xi, eta)
    }

    fn face_reference<F: Scalar>(local_face: u8, s: F) -> Result<(F, F, [F; 2]), String> {
        quad4::face_reference(local_face, s)
    }
}

/// Biquadratic quad9 reference element marker for generic mesh queries.
pub struct Quad9ReferenceElement;

impl QuadReferenceElement<9> for Quad9ReferenceElement {
    fn shape<F: Scalar>(xi: F, eta: F) -> [F; 9] {
        quad9::shape(xi, eta)
    }

    fn grad_ref<F: Scalar>(xi: F, eta: F) -> [[F; 2]; 9] {
        quad9::grad_ref(xi, eta)
    }

    fn face_reference<F: Scalar>(local_face: u8, s: F) -> Result<(F, F, [F; 2]), String> {
        quad9::face_reference(local_face, s)
    }
}

/// One-pass mesh query results for a batch of physical points.
///
/// The query records the nearest node, nearest element, and nearest face for each point.
/// Downstream interpolation and recovery operators can reuse the element/reference arrays without
/// repeating the `O(nquery * nelem)` geometric search. For points inside the mesh, the nearest
/// element is the containing element and `nearest_element_distances` is zero to numerical
/// tolerance.
#[derive(Debug, Clone)]
pub struct QuadMeshQueryResult<F: Scalar> {
    /// Nearest-node index for each query point.
    pub nearest_node_indices: Vec<usize>,
    /// Coordinates of each nearest node.
    pub nearest_node_points: Vec<[F; 2]>,
    /// Distance from each query point to its nearest node.
    pub nearest_node_distances: Vec<F>,
    /// Nearest element index for each query point.
    pub nearest_element_indices: Vec<usize>,
    /// Reference coordinates of the closest point on each nearest element.
    pub nearest_element_reference_points: Vec<[F; 2]>,
    /// Closest physical point on each nearest element.
    pub nearest_element_points: Vec<[F; 2]>,
    /// Distance from each query point to its nearest element.
    pub nearest_element_distances: Vec<F>,
    /// Element index of the nearest face for each query point.
    pub nearest_face_element_indices: Vec<usize>,
    /// Local face index of the nearest face for each query point.
    pub nearest_face_local_faces: Vec<u8>,
    /// Face-local coordinate `s in [-1, 1]` of each nearest face point.
    pub nearest_face_reference_coordinates: Vec<F>,
    /// Closest physical point on each nearest face.
    pub nearest_face_points: Vec<[F; 2]>,
    /// Distance from each query point to its nearest face.
    pub nearest_face_distances: Vec<F>,
}

/// Sparse triplets for mesh interpolation or recovery operators.
#[derive(Debug, Clone)]
pub struct QuadMeshSparseOperator<F: Scalar> {
    /// Output row index for each nonzero.
    pub rows: Vec<usize>,
    /// Input column index for each nonzero.
    pub cols: Vec<usize>,
    /// Nonzero values in triplet order.
    pub vals: Vec<F>,
    /// Number of output rows.
    pub nrow: usize,
    /// Number of input columns.
    pub ncol: usize,
}

#[derive(Clone, Copy, Debug)]
struct ElementProjection<F: Scalar> {
    element_index: usize,
    reference: [F; 2],
    point: [F; 2],
    distance: F,
}

#[derive(Clone, Copy, Debug)]
struct FaceProjection<F: Scalar> {
    element_index: usize,
    local_face: u8,
    s: F,
    point: [F; 2],
    distance: F,
}

fn squared_distance<F: Scalar>(a: [F; 2], b: [F; 2]) -> F {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

fn solve_2x2<F: Scalar>(matrix: [[F; 2]; 2], rhs: [F; 2]) -> Option<[F; 2]> {
    let det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    let eps = F::epsilon() * cast::<F>(64.0);
    if det.abs() <= eps {
        return None;
    }
    Some([
        (rhs[0] * matrix[1][1] - matrix[0][1] * rhs[1]) / det,
        (matrix[0][0] * rhs[1] - rhs[0] * matrix[1][0]) / det,
    ])
}

fn clamp_reference<F: Scalar>(reference: [F; 2]) -> [F; 2] {
    [
        reference[0].max(-F::one()).min(F::one()),
        reference[1].max(-F::one()).min(F::one()),
    ]
}

fn reference_map<E, F, const NODES_PER_ELEMENT: usize>(
    coords: &[[F; 2]; NODES_PER_ELEMENT],
    reference: [F; 2],
) -> [F; 2]
where
    E: QuadReferenceElement<NODES_PER_ELEMENT>,
    F: Scalar,
{
    let shape = E::shape(reference[0], reference[1]);
    mapping::map_point(coords, &shape)
}

fn closest_reference_point<E, F, const NODES_PER_ELEMENT: usize>(
    coords: &[[F; 2]; NODES_PER_ELEMENT],
    target: [F; 2],
    max_iterations: usize,
) -> ([F; 2], [F; 2], F)
where
    E: QuadReferenceElement<NODES_PER_ELEMENT>,
    F: Scalar,
{
    // Project one physical point onto one element by minimizing physical distance over the
    // bounded reference square. This is still local element work, with a fixed number of seeds, so
    // complexity is O(max_iterations * NODES_PER_ELEMENT).
    let seeds = [
        [F::zero(), F::zero()],
        [-F::one(), -F::one()],
        [F::one(), -F::one()],
        [F::one(), F::one()],
        [-F::one(), F::one()],
        [F::zero(), -F::one()],
        [F::one(), F::zero()],
        [F::zero(), F::one()],
        [-F::one(), F::zero()],
    ];
    let mut best_reference = seeds[0];
    let mut best_point = reference_map::<E, F, NODES_PER_ELEMENT>(coords, best_reference);
    let mut best_distance_sq = squared_distance(best_point, target);

    for seed in seeds {
        let mut reference = seed;
        for _ in 0..max_iterations {
            let point = reference_map::<E, F, NODES_PER_ELEMENT>(coords, reference);
            let residual = [point[0] - target[0], point[1] - target[1]];
            let grad = E::grad_ref(reference[0], reference[1]);
            let jac = mapping::jacobian(coords, &grad);
            let gradient = [
                jac[0][0] * residual[0] + jac[1][0] * residual[1],
                jac[0][1] * residual[0] + jac[1][1] * residual[1],
            ];
            let hessian = [
                [
                    jac[0][0] * jac[0][0] + jac[1][0] * jac[1][0],
                    jac[0][0] * jac[0][1] + jac[1][0] * jac[1][1],
                ],
                [
                    jac[0][1] * jac[0][0] + jac[1][1] * jac[1][0],
                    jac[0][1] * jac[0][1] + jac[1][1] * jac[1][1],
                ],
            ];
            let Some(update) = solve_2x2(hessian, gradient) else {
                break;
            };
            let next = clamp_reference([reference[0] - update[0], reference[1] - update[1]]);
            let step_sq = squared_distance(reference, next);
            reference = next;
            if step_sq <= F::epsilon() * F::epsilon() {
                break;
            }
        }
        let point = reference_map::<E, F, NODES_PER_ELEMENT>(coords, reference);
        let distance_sq = squared_distance(point, target);
        if distance_sq < best_distance_sq {
            best_reference = reference;
            best_point = point;
            best_distance_sq = distance_sq;
        }
    }

    (best_reference, best_point, best_distance_sq.sqrt())
}

fn closest_face_point<E, F, const NODES_PER_ELEMENT: usize>(
    coords: &[[F; 2]; NODES_PER_ELEMENT],
    local_face: u8,
    target: [F; 2],
    max_iterations: usize,
) -> Result<([F; 2], F, F), String>
where
    E: QuadReferenceElement<NODES_PER_ELEMENT>,
    F: Scalar,
{
    // Project one physical point onto one element face. This is independent of global mesh size
    // and costs O(max_iterations * NODES_PER_ELEMENT).
    let seeds = [-F::one(), F::zero(), F::one()];
    let mut best_s = F::zero();
    let mut best_point = reference_map::<E, F, NODES_PER_ELEMENT>(coords, {
        let (xi, eta, _) = E::face_reference(local_face, best_s)?;
        [xi, eta]
    });
    let mut best_distance_sq = squared_distance(best_point, target);

    for seed in seeds {
        let mut s = seed;
        for _ in 0..max_iterations {
            let (xi, eta, ds_reference) = E::face_reference(local_face, s)?;
            let reference = [xi, eta];
            let shape = E::shape(reference[0], reference[1]);
            let point = mapping::map_point(coords, &shape);
            let residual = [point[0] - target[0], point[1] - target[1]];
            let grad = E::grad_ref(reference[0], reference[1]);
            let jac = mapping::jacobian(coords, &grad);
            let tangent = mapping::face_tangent(&jac, ds_reference);
            let tangent_sq = tangent[0] * tangent[0] + tangent[1] * tangent[1];
            if tangent_sq <= F::epsilon() {
                break;
            }
            let gradient = tangent[0] * residual[0] + tangent[1] * residual[1];
            let next = (s - gradient / tangent_sq).max(-F::one()).min(F::one());
            let step = (next - s).abs();
            s = next;
            if step <= F::epsilon() {
                break;
            }
        }
        let (xi, eta, _) = E::face_reference(local_face, s)?;
        let point = reference_map::<E, F, NODES_PER_ELEMENT>(coords, [xi, eta]);
        let distance_sq = squared_distance(point, target);
        if distance_sq < best_distance_sq {
            best_s = s;
            best_point = point;
            best_distance_sq = distance_sq;
        }
    }

    Ok((best_point, best_s, best_distance_sq.sqrt()))
}

fn nearest_node_projection<F: Scalar>(nodes: &[[F; 2]], query: [F; 2]) -> (usize, [F; 2], F) {
    let mut best_index = 0usize;
    let mut best_distance_sq = squared_distance(nodes[0], query);
    for (node_index, &node) in nodes.iter().enumerate().skip(1) {
        let distance_sq = squared_distance(node, query);
        if distance_sq < best_distance_sq {
            best_index = node_index;
            best_distance_sq = distance_sq;
        }
    }
    (best_index, nodes[best_index], best_distance_sq.sqrt())
}

/// Query nearest node, nearest element, and nearest face in one mesh pass.
///
/// For each query point, this scans all nodes once and all elements once. During the element scan
/// it computes nearest-element and nearest-face data together. Complexity is
/// `O(nquery * (nnode + nelem * max_iterations))`, with a fixed factor from four faces per
/// element. No spatial acceleration structure is built or cached. For contained points, the
/// nearest element is the containing element and its distance is zero to numerical tolerance.
pub fn query_quad_mesh<E, F, const NODES_PER_ELEMENT: usize>(
    mesh: QuadMeshView2d<'_, F, NODES_PER_ELEMENT>,
    query_points: &[[F; 2]],
    max_iterations: usize,
) -> Result<QuadMeshQueryResult<F>, String>
where
    E: QuadReferenceElement<NODES_PER_ELEMENT>,
    F: Scalar,
{
    mesh.validate_connectivity()?;
    if mesh.num_nodes() == 0 {
        return Err("cannot query an empty mesh with no nodes".to_string());
    }
    if mesh.num_elements() == 0 {
        return Err("cannot query an empty mesh with no elements".to_string());
    }

    let mut nearest_node_indices = Vec::with_capacity(query_points.len());
    let mut nearest_node_points = Vec::with_capacity(query_points.len());
    let mut nearest_node_distances = Vec::with_capacity(query_points.len());
    let mut nearest_element_indices = Vec::with_capacity(query_points.len());
    let mut nearest_element_reference_points = Vec::with_capacity(query_points.len());
    let mut nearest_element_points = Vec::with_capacity(query_points.len());
    let mut nearest_element_distances = Vec::with_capacity(query_points.len());
    let mut nearest_face_element_indices = Vec::with_capacity(query_points.len());
    let mut nearest_face_local_faces = Vec::with_capacity(query_points.len());
    let mut nearest_face_reference_coordinates = Vec::with_capacity(query_points.len());
    let mut nearest_face_points = Vec::with_capacity(query_points.len());
    let mut nearest_face_distances = Vec::with_capacity(query_points.len());

    for &query in query_points {
        let (node_index, node_point, node_distance) = nearest_node_projection(mesh.nodes_rz, query);
        nearest_node_indices.push(node_index);
        nearest_node_points.push(node_point);
        nearest_node_distances.push(node_distance);

        let mut nearest: Option<ElementProjection<F>> = None;
        let mut nearest_face: Option<FaceProjection<F>> = None;

        // One O(nelem) pass supplies all element- and face-level query data for this point.
        for element_index in 0..mesh.num_elements() {
            let coords = mesh.element_coords(element_index)?;

            let (reference, point, distance) =
                closest_reference_point::<E, F, NODES_PER_ELEMENT>(&coords, query, max_iterations);
            let projection = ElementProjection {
                element_index,
                reference,
                point,
                distance,
            };
            if nearest.is_none_or(|current| projection.distance < current.distance) {
                nearest = Some(projection);
            }

            for local_face in 0..4 {
                let (point, s, distance) = closest_face_point::<E, F, NODES_PER_ELEMENT>(
                    &coords,
                    local_face,
                    query,
                    max_iterations,
                )?;
                let projection = FaceProjection {
                    element_index,
                    local_face,
                    s,
                    point,
                    distance,
                };
                if nearest_face.is_none_or(|current| projection.distance < current.distance) {
                    nearest_face = Some(projection);
                }
            }
        }

        let nearest = nearest.expect("non-empty mesh should have a nearest element");
        nearest_element_indices.push(nearest.element_index);
        nearest_element_reference_points.push(nearest.reference);
        nearest_element_points.push(nearest.point);
        nearest_element_distances.push(nearest.distance);

        let nearest_face = nearest_face.expect("non-empty mesh should have a nearest face");
        nearest_face_element_indices.push(nearest_face.element_index);
        nearest_face_local_faces.push(nearest_face.local_face);
        nearest_face_reference_coordinates.push(nearest_face.s);
        nearest_face_points.push(nearest_face.point);
        nearest_face_distances.push(nearest_face.distance);
    }

    Ok(QuadMeshQueryResult {
        nearest_node_indices,
        nearest_node_points,
        nearest_node_distances,
        nearest_element_indices,
        nearest_element_reference_points,
        nearest_element_points,
        nearest_element_distances,
        nearest_face_element_indices,
        nearest_face_local_faces,
        nearest_face_reference_coordinates,
        nearest_face_points,
        nearest_face_distances,
    })
}

/// Build a sparse interpolation operator from query element/reference coordinates.
///
/// The operator has shape `(nquery, nnode)` and maps scalar nodal values to scalar values at the
/// query points. It can also be applied to a dense `(nnode, ncomponent)` array on the Python side
/// to interpolate multiple nodal fields with the same geometry query.
pub fn quad_mesh_interpolation_operator<E, F, const NODES_PER_ELEMENT: usize>(
    mesh: QuadMeshView2d<'_, F, NODES_PER_ELEMENT>,
    element_indices: &[usize],
    reference_points: &[[F; 2]],
) -> Result<QuadMeshSparseOperator<F>, String>
where
    E: QuadReferenceElement<NODES_PER_ELEMENT>,
    F: Scalar,
{
    mesh.validate_connectivity()?;
    if element_indices.len() != reference_points.len() {
        return Err(format!(
            "element_indices has length {}, but reference_points has length {}",
            element_indices.len(),
            reference_points.len()
        ));
    }

    let mut rows = Vec::with_capacity(element_indices.len() * NODES_PER_ELEMENT);
    let mut cols = Vec::with_capacity(element_indices.len() * NODES_PER_ELEMENT);
    let mut vals = Vec::with_capacity(element_indices.len() * NODES_PER_ELEMENT);

    for (query_index, (&element_index, &reference)) in
        element_indices.iter().zip(reference_points).enumerate()
    {
        let element_nodes = mesh.element_nodes(element_index)?;
        let shape = E::shape(reference[0], reference[1]);
        for local_node in 0..NODES_PER_ELEMENT {
            let value = shape[local_node];
            if value != F::zero() {
                rows.push(query_index);
                cols.push(element_nodes[local_node]);
                vals.push(value);
            }
        }
    }

    Ok(QuadMeshSparseOperator {
        rows,
        cols,
        vals,
        nrow: element_indices.len(),
        ncol: mesh.num_nodes(),
    })
}

/// Build a sparse strain-recovery operator from query element/reference coordinates.
///
/// The operator has shape `(4 * nquery, 2 * nnode)` and maps full nodal displacement values to the
/// four-component strain vector at each query point. Rows are grouped per query point in the same
/// component ordering as the structural formulation.
pub fn quad_mesh_strain_operator<E, const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize>(
    mesh: QuadMeshView2d<'_, f64, NODES_PER_ELEMENT>,
    element_indices: &[usize],
    reference_points: &[[f64; 2]],
    formulation: Structural2dFormulation,
) -> Result<QuadMeshSparseOperator<f64>, String>
where
    E: QuadReferenceElement<NODES_PER_ELEMENT>,
{
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    mesh.validate_connectivity()?;
    if element_indices.len() != reference_points.len() {
        return Err(format!(
            "element_indices has length {}, but reference_points has length {}",
            element_indices.len(),
            reference_points.len()
        ));
    }

    let max_nonzeros = element_indices.len() * 4 * DOF_PER_ELEMENT;
    let mut rows = Vec::with_capacity(max_nonzeros);
    let mut cols = Vec::with_capacity(max_nonzeros);
    let mut vals = Vec::with_capacity(max_nonzeros);

    for (query_index, (&element_index, &reference)) in
        element_indices.iter().zip(reference_points).enumerate()
    {
        let coords = mesh.element_coords(element_index)?;
        let nodes = mesh.element_nodes(element_index)?;
        let shape = E::shape(reference[0], reference[1]);
        let grad_ref = E::grad_ref(reference[0], reference[1]);
        let jac = mapping::jacobian(&coords, &grad_ref);
        let inv_jac = mapping::inv_j(&jac)?;
        let grad_phys = mapping::grad_phys(&grad_ref, &inv_jac);
        let point = mapping::map_point(&coords, &shape);
        let b = build_b_matrix::<NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            formulation,
            &shape,
            &grad_phys,
            point,
        )?;
        for component in 0..4 {
            let row = 4 * query_index + component;
            for local_node in 0..NODES_PER_ELEMENT {
                for dof_component in 0..DOF_PER_NODE {
                    let local_dof = DOF_PER_NODE * local_node + dof_component;
                    let value = b[component][local_dof];
                    if value != 0.0 {
                        rows.push(row);
                        cols.push(DOF_PER_NODE * nodes[local_node] + dof_component);
                        vals.push(value);
                    }
                }
            }
        }
    }

    Ok(QuadMeshSparseOperator {
        rows,
        cols,
        vals,
        nrow: 4 * element_indices.len(),
        ncol: DOF_PER_NODE * mesh.num_nodes(),
    })
}

/// Build a sparse stress-recovery operator from query element/reference coordinates.
///
/// The operator has shape `(4 * nquery, 2 * nnode)` and maps full nodal displacement values to the
/// four-component stress vector at each query point. Rows are grouped per query point in the same
/// component ordering as the structural formulation. Each query point uses the material assigned to
/// its query element; for contained points this is the containing element.
///
/// `mesh.nodes_rz` and `reference_points` define geometry with physical node coordinates in
/// `[length]` and unitless reference coordinates. `element_indices` and `material_ids` are unitless
/// indices with lengths `nquery` and `nelem`, respectively. `material_table` has shape
/// `(nmat, 4, 4)` and units `[stress / strain] = [pressure]`. Optional
/// `material_orientation_angles` has shape `(nelem,)` and unitless radians. Operator entries have
/// units `[pressure / length]`, so multiplying by nodal displacements with units `[length]`
/// returns stresses with units `[pressure]`.
pub fn quad_mesh_stress_operator<E, const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize>(
    mesh: QuadMeshView2d<'_, f64, NODES_PER_ELEMENT>,
    element_indices: &[usize],
    reference_points: &[[f64; 2]],
    material_ids: &[usize],
    material_table: &[[[f64; 4]; 4]],
    material_orientation_angles: Option<&[f64]>,
    formulation: Structural2dFormulation,
) -> Result<QuadMeshSparseOperator<f64>, String>
where
    E: QuadReferenceElement<NODES_PER_ELEMENT>,
{
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    mesh.validate_connectivity()?;
    if element_indices.len() != reference_points.len() {
        return Err(format!(
            "element_indices has length {}, but reference_points has length {}",
            element_indices.len(),
            reference_points.len()
        ));
    }
    validate_element_material_inputs(
        mesh.num_elements(),
        material_ids,
        material_orientation_angles,
    )?;

    let max_nonzeros = element_indices.len() * 4 * DOF_PER_ELEMENT;
    let mut rows = Vec::with_capacity(max_nonzeros);
    let mut cols = Vec::with_capacity(max_nonzeros);
    let mut vals = Vec::with_capacity(max_nonzeros);

    for (query_index, (&element_index, &reference)) in
        element_indices.iter().zip(reference_points).enumerate()
    {
        let coords = mesh.element_coords(element_index)?;
        let nodes = mesh.element_nodes(element_index)?;
        let material_id = material_ids[element_index];
        let material = material_table.get(material_id).ok_or_else(|| {
            format!("material_id {material_id} on element {element_index} is out of range")
        })?;
        let material_storage;
        let material = if let Some(angles) = material_orientation_angles {
            material_storage = rotate_material_in_plane(material, angles[element_index]);
            &material_storage
        } else {
            material
        };

        let shape = E::shape(reference[0], reference[1]);
        let grad_ref = E::grad_ref(reference[0], reference[1]);
        let jac = mapping::jacobian(&coords, &grad_ref);
        let inv_jac = mapping::inv_j(&jac)?;
        let grad_phys = mapping::grad_phys(&grad_ref, &inv_jac);
        let point = mapping::map_point(&coords, &shape);
        let b = build_b_matrix::<NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            formulation,
            &shape,
            &grad_phys,
            point,
        )?;

        for component in 0..4 {
            let row = 4 * query_index + component;
            for local_node in 0..NODES_PER_ELEMENT {
                for dof_component in 0..DOF_PER_NODE {
                    let local_dof = DOF_PER_NODE * local_node + dof_component;
                    let mut value = 0.0;
                    for strain_component in 0..4 {
                        value = value
                            + material[component][strain_component]
                                * b[strain_component][local_dof];
                    }
                    if value != 0.0 {
                        rows.push(row);
                        cols.push(DOF_PER_NODE * nodes[local_node] + dof_component);
                        vals.push(value);
                    }
                }
            }
        }
    }

    Ok(QuadMeshSparseOperator {
        rows,
        cols,
        vals,
        nrow: 4 * element_indices.len(),
        ncol: DOF_PER_NODE * mesh.num_nodes(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        Quad4ReferenceElement, Quad9ReferenceElement, QuadMeshView2d,
        quad_mesh_interpolation_operator, quad_mesh_stress_operator, query_quad_mesh,
    };
    use crate::mesh::elements::quad2d::{mapping, quad9};
    use crate::physics::solenoid_stress::Structural2dFormulation;

    #[test]
    fn quad_mesh_query_and_interpolation_operator_handle_quad4() {
        let nodes = [[0.0_f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let elements = [[0usize, 1, 2, 3]];
        let mesh = QuadMeshView2d {
            nodes_rz: &nodes,
            elements: &elements,
        };

        let query = query_quad_mesh::<Quad4ReferenceElement, _, 4>(
            mesh,
            &[[0.25, 0.75], [1.4, 0.5], [0.5, -0.2]],
            20,
        )
        .expect("mesh query");
        assert_eq!(query.nearest_node_indices, vec![3, 1, 0]);
        assert_eq!(query.nearest_element_indices, vec![0, 0, 0]);
        assert!(query.nearest_element_distances[0] <= 1.0e-12);
        assert!(query.nearest_element_distances[1] > 0.0);
        assert!((query.nearest_element_reference_points[0][0] + 0.5).abs() < 1.0e-12);
        assert!((query.nearest_element_reference_points[0][1] - 0.5).abs() < 1.0e-12);
        assert!((query.nearest_element_points[1][0] - 1.0).abs() < 1.0e-12);
        assert!((query.nearest_element_points[1][1] - 0.5).abs() < 1.0e-12);
        assert_eq!(query.nearest_face_local_faces[2], 0);
        assert!(query.nearest_face_reference_coordinates[2].abs() < 1.0e-12);

        let nodal_values = [0.0, 1.0, 3.0, 2.0];
        let interpolation = quad_mesh_interpolation_operator::<Quad4ReferenceElement, _, 4>(
            mesh,
            &query.nearest_element_indices,
            &query.nearest_element_reference_points,
        )
        .expect("interpolation operator");
        let first_value = interpolation
            .rows
            .iter()
            .zip(&interpolation.cols)
            .zip(&interpolation.vals)
            .filter(|((row, _), _)| **row == 0)
            .map(|((_, col), value)| *value * nodal_values[*col])
            .sum::<f64>();
        assert!((first_value - 1.75).abs() < 1.0e-12);
    }

    #[test]
    fn quad_mesh_interpolation_uses_quad9_curved_geometry() {
        let nodes = [
            [0.0_f64, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, -0.2],
            [1.2, 0.5],
            [0.5, 1.1],
            [-0.1, 0.5],
            [0.45, 0.55],
        ];
        let elements = [[0usize, 1, 2, 3, 4, 5, 6, 7, 8]];
        let mesh = QuadMeshView2d {
            nodes_rz: &nodes,
            elements: &elements,
        };
        let reference = [0.0, -0.5];
        let shape = quad9::shape(reference[0], reference[1]);
        let point = mapping::map_point(&nodes, &shape);
        let nodal_values = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let query =
            query_quad_mesh::<Quad9ReferenceElement, _, 9>(mesh, &[point], 20).expect("mesh query");
        let interpolation = quad_mesh_interpolation_operator::<Quad9ReferenceElement, _, 9>(
            mesh,
            &query.nearest_element_indices,
            &query.nearest_element_reference_points,
        )
        .expect("quad9 interpolation operator");
        let interpolated = interpolation
            .rows
            .iter()
            .zip(&interpolation.cols)
            .zip(&interpolation.vals)
            .filter(|((row, _), _)| **row == 0)
            .map(|((_, col), value)| *value * nodal_values[*col])
            .sum::<f64>();
        let expected = shape
            .iter()
            .zip(nodal_values)
            .map(|(n, value)| n * value)
            .sum::<f64>();
        assert!((query.nearest_element_reference_points[0][0] - reference[0]).abs() < 1.0e-10);
        assert!((query.nearest_element_reference_points[0][1] - reference[1]).abs() < 1.0e-10);
        assert!((interpolated - expected).abs() < 1.0e-10);
    }

    #[test]
    fn quad_mesh_stress_operator_uses_query_element_material() {
        let nodes = [
            [0.0_f64, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ];
        let elements = [[0usize, 1, 4, 3], [1, 2, 5, 4]];
        let mesh = QuadMeshView2d {
            nodes_rz: &nodes,
            elements: &elements,
        };
        let material_ids = [0usize, 1usize];
        let material_table = [
            [
                [2.0, 0.25, 0.0, 0.0],
                [0.5, 3.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 7.0],
            ],
            [
                [11.0, 1.5, 0.0, 0.0],
                [2.0, 13.0, 0.0, 0.0],
                [0.0, 0.0, 17.0, 0.0],
                [0.0, 0.0, 0.0, 19.0],
            ],
        ];
        let query =
            query_quad_mesh::<Quad4ReferenceElement, _, 4>(mesh, &[[0.25, 0.5], [1.75, 0.5]], 20)
                .expect("mesh query");
        let operator = quad_mesh_stress_operator::<Quad4ReferenceElement, 4, 8>(
            mesh,
            &query.nearest_element_indices,
            &query.nearest_element_reference_points,
            &material_ids,
            &material_table,
            None,
            Structural2dFormulation::PlaneStrain { thickness: 1.0 },
        )
        .expect("stress operator");
        let displacement = [
            0.0_f64, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 2.0, 2.0, 2.0,
        ];
        let mut stress = [0.0_f64; 8];
        for ((&row, &col), &value) in operator.rows.iter().zip(&operator.cols).zip(&operator.vals) {
            stress[row] += value * displacement[col];
        }

        let expected = [2.5, 6.5, 0.0, 0.0, 14.0, 28.0, 0.0, 0.0];
        for (actual, expected) in stress.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    }
}
