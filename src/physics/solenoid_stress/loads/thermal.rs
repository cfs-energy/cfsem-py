use rayon::prelude::*;

use crate::chunksize;
use crate::mesh::{QuadMeshView2d, QuadratureRule};
use crate::physics::solenoid_stress::axisym::{
    accumulate_b_transpose_vector, build_b_matrix, constitutive_times_strain,
};
use crate::physics::solenoid_stress::convenience::{
    rotate_material_in_plane, rotate_thermal_material_in_plane,
};
use crate::physics::solenoid_stress::family::QuadElementFamily;
use crate::physics::solenoid_stress::geometry::{VolumeSample, validate_structural_2d_mesh};
use crate::physics::solenoid_stress::types::{
    DOF_PER_NODE, Real, Structural2dFormulation, ThermalMaterial, local_dofs, scatter_local_matrix,
    validate_element_material_inputs,
};

use super::{SparseOperator, ThermalLoadOperator, concat_thermal_load_operators, ranges_for_len};

/// Local thermal operator data for one element.
///
/// `temperature_to_rhs` maps element nodal temperatures `[temperature]` to the element's
/// consistent nodal thermal load vector `[energy / distance]`, so its entries have units
/// `[force / temperature]`.
///
/// `reference_rhs` is the constant offset contributed by the material's stress-free reference
/// temperature and has units `[energy / distance]`.
struct LocalThermalKernel<F: Real, const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize> {
    temperature_to_rhs: [[F; NODES_PER_ELEMENT]; DOF_PER_ELEMENT],
    reference_rhs: [F; DOF_PER_ELEMENT],
}

/// Build the local dense thermal load operator for one element.
///
/// The thermal strain model is
/// `epsilon_th = alpha * (T - T_ref)`,
/// where `alpha` has units `[strain / temperature]`.  The returned block therefore maps nodal
/// temperatures directly to generalized nodal loads.
fn thermal_element_kernel<F: Real, const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize>(
    samples: &[VolumeSample<F, NODES_PER_ELEMENT>],
    material: &[[F; 4]; 4],
    thermal: &ThermalMaterial<F>,
    formulation: Structural2dFormulation<F>,
) -> Result<LocalThermalKernel<F, NODES_PER_ELEMENT, DOF_PER_ELEMENT>, String> {
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    let mut local = LocalThermalKernel {
        temperature_to_rhs: [[F::zero(); NODES_PER_ELEMENT]; DOF_PER_ELEMENT],
        reference_rhs: [F::zero(); DOF_PER_ELEMENT],
    };
    let thermal_stress_unit = constitutive_times_strain(material, &thermal.alpha);

    for sample in samples {
        // `B` has units `[1 / length]`, `D * alpha` has units `[stress / temperature]`, and the
        // quadrature scale contributes a physical volume. The resulting local block has units
        // `[force / temperature]`.
        let b = build_b_matrix::<F, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            formulation,
            &sample.n,
            &sample.grad_phys,
            sample.point,
        )?;
        let scale = formulation.volume_scale(sample.point, sample.det_j, sample.weight)?;
        let mut local_unit_rhs = [F::zero(); DOF_PER_ELEMENT];
        accumulate_b_transpose_vector(&mut local_unit_rhs, &b, &thermal_stress_unit, scale);

        for local_temp_node in 0..NODES_PER_ELEMENT {
            // Interpolating the nodal temperature field with `N_i` converts the unit thermal load
            // for a uniform `DeltaT` into one column per nodal temperature DOF.
            let scale_node = sample.n[local_temp_node];
            for dof in 0..DOF_PER_ELEMENT {
                local.temperature_to_rhs[dof][local_temp_node] = local.temperature_to_rhs[dof]
                    [local_temp_node]
                    + local_unit_rhs[dof] * scale_node;
            }
        }
        // The reference-temperature term is a constant RHS offset because `T_ref` is prescribed
        // by the material model, not by a nodal unknown.
        for dof in 0..DOF_PER_ELEMENT {
            local.reference_rhs[dof] =
                local.reference_rhs[dof] - local_unit_rhs[dof] * thermal.reference_temperature;
        }
    }

    Ok(local)
}

/// Assemble the global temperature-to-RHS operator and reference-temperature offset.
///
/// Output shapes:
/// - `temperature_to_rhs`: `(2 * mesh.num_nodes(), mesh.num_nodes())`
/// - `reference_rhs`: `(2 * mesh.num_nodes(),)`
///
/// Row meaning:
/// - row `2*a` is the radial generalized-force equation for node `a`,
/// - row `2*a + 1` is the axial generalized-force equation for node `a`.
///
/// Column meaning:
/// - column `j` is nodal temperature at node `j`.
///
/// Entry units:
/// - `temperature_to_rhs`: `[generalized force / temperature] = [energy / (distance * temperature)]`
/// - `reference_rhs`: `[generalized force] = [energy / distance]`
pub(crate) fn temperature_operator_for_family<
    F: Real,
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, F, NODES_PER_ELEMENT>,
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    thermal_material_table: &[ThermalMaterial<F>],
    material_orientation_angles: Option<&[F]>,
    formulation: Structural2dFormulation<F>,
    quadrature: QuadratureRule,
) -> Result<ThermalLoadOperator<F>, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_structural_2d_mesh(mesh, formulation)?;
    validate_element_material_inputs(
        mesh.num_elements(),
        material_ids,
        material_orientation_angles,
    )?;
    temperature_operator_range_for_family::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
        mesh,
        material_ids,
        material_table,
        thermal_material_table,
        material_orientation_angles,
        formulation,
        quadrature,
        0,
        mesh.num_elements(),
    )
}

/// Assemble thermal load operators using element ranges split across Rayon workers.
pub(crate) fn temperature_operator_for_family_par<
    F: Real,
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, F, NODES_PER_ELEMENT>,
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    thermal_material_table: &[ThermalMaterial<F>],
    material_orientation_angles: Option<&[F]>,
    formulation: Structural2dFormulation<F>,
    quadrature: QuadratureRule,
) -> Result<ThermalLoadOperator<F>, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_structural_2d_mesh(mesh, formulation)?;
    validate_element_material_inputs(
        mesh.num_elements(),
        material_ids,
        material_orientation_angles,
    )?;
    let nelem = mesh.num_elements();
    let ranges = ranges_for_len(nelem, chunksize(nelem));
    let chunks = ranges
        .into_par_iter()
        .map(|(start, end)| {
            temperature_operator_range_for_family::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                mesh,
                material_ids,
                material_table,
                thermal_material_table,
                material_orientation_angles,
                formulation,
                quadrature,
                start,
                end,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(concat_thermal_load_operators(
        chunks,
        mesh.num_nodes() * 2,
        mesh.num_nodes(),
        nelem * DOF_PER_ELEMENT * NODES_PER_ELEMENT,
    ))
}

#[allow(clippy::too_many_arguments)]
fn temperature_operator_range_for_family<
    F: Real,
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, F, NODES_PER_ELEMENT>,
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    thermal_material_table: &[ThermalMaterial<F>],
    material_orientation_angles: Option<&[F]>,
    formulation: Structural2dFormulation<F>,
    quadrature: QuadratureRule,
    element_start: usize,
    element_end: usize,
) -> Result<ThermalLoadOperator<F>, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    let ndof = mesh.num_nodes() * 2;
    let ncol = mesh.num_nodes();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let mut reference_rhs = vec![F::zero(); ndof];

    for element_index in element_start..element_end {
        let coords = mesh.element_coords(element_index)?;
        let nodes = mesh.element_nodes(element_index)?;
        let material_id = material_ids[element_index];
        let material = material_table.get(material_id).ok_or_else(|| {
            format!("material_id {material_id} on element {element_index} is out of range")
        })?;
        let thermal = thermal_material_table.get(material_id).ok_or_else(|| {
            format!("thermal material_id {material_id} on element {element_index} is out of range")
        })?;
        let material_storage;
        let thermal_storage;
        let (material, thermal) = if let Some(angles) = material_orientation_angles {
            material_storage = rotate_material_in_plane(material, angles[element_index]);
            thermal_storage = rotate_thermal_material_in_plane(thermal, angles[element_index]);
            (&material_storage, &thermal_storage)
        } else {
            (material, thermal)
        };
        let samples = Family::volume_samples::<F>(&coords, quadrature)?;
        let local = thermal_element_kernel::<F, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            &samples,
            material,
            thermal,
            formulation,
        )?;
        let global_rows = local_dofs::<NODES_PER_ELEMENT, DOF_PER_ELEMENT>(&nodes);
        scatter_local_matrix(
            &mut rows,
            &mut cols,
            &mut vals,
            &global_rows,
            &nodes,
            &local.temperature_to_rhs,
        );
        for dof in 0..DOF_PER_ELEMENT {
            reference_rhs[global_rows[dof]] =
                reference_rhs[global_rows[dof]] + local.reference_rhs[dof];
        }
    }

    Ok(ThermalLoadOperator {
        temperature_to_rhs: SparseOperator {
            rows,
            cols,
            vals,
            nrow: ndof,
            ncol,
        },
        reference_rhs,
    })
}
