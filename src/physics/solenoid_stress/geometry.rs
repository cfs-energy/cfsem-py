use crate::physics::solenoid_stress::mesh::MeshView;
use crate::physics::solenoid_stress::quad4::{
    NODES_PER_ELEMENT, face_reference, grad_phys, grad_ref, inv_j, jacobian, map_point,
};
use crate::physics::solenoid_stress::quadrature::{QuadratureRule, gauss_face, gauss_volume};
use crate::physics::solenoid_stress::types::{Real, two_pi};

#[derive(Debug, Clone)]
pub struct ElementMeasures<F: Real> {
    pub areas: Vec<F>,
    pub swept_volumes: Vec<F>,
}

#[derive(Debug, Clone)]
pub struct ElementQuadrature<F: Real> {
    pub points_rz: Vec<[F; 2]>,
    pub weights_area: Vec<F>,
    pub weights_volume: Vec<F>,
    pub nq_per_element: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct VolumeSample<F: Real> {
    pub n: [F; NODES_PER_ELEMENT],
    pub grad_phys: [[F; 2]; NODES_PER_ELEMENT],
    pub det_j: F,
    pub point: [F; 2],
    pub weight: F,
}

#[derive(Debug, Clone, Copy)]
pub struct FaceSample<F: Real> {
    pub n: [F; NODES_PER_ELEMENT],
    pub tangent: [F; 2],
    pub point: [F; 2],
    pub weight: F,
}

pub fn volume_samples<F: Real>(
    coords: &[[F; 2]; NODES_PER_ELEMENT],
    quadrature: QuadratureRule,
) -> Result<Vec<VolumeSample<F>>, String> {
    let samples = gauss_volume::<F>(quadrature);
    let mut out = Vec::with_capacity(samples.len());
    for ([xi, eta], weight) in samples {
        let n = crate::physics::solenoid_stress::quad4::shape(xi, eta);
        let grad_reference = grad_ref(xi, eta);
        let jac = jacobian(coords, &grad_reference);
        let inv = inv_j(&jac)?;
        let det = crate::physics::solenoid_stress::quad4::det_j(&jac);
        let point = map_point(coords, &n);
        if point[0] < F::zero() {
            return Err(format!(
                "quadrature point has negative radius {:?}; axisymmetric radius must be nonnegative",
                point[0]
            ));
        }
        out.push(VolumeSample {
            n,
            grad_phys: grad_phys(&grad_reference, &inv),
            det_j: det,
            point,
            weight,
        });
    }
    Ok(out)
}

pub fn face_samples<F: Real>(
    coords: &[[F; 2]; NODES_PER_ELEMENT],
    local_face: u8,
    quadrature: QuadratureRule,
) -> Result<Vec<FaceSample<F>>, String> {
    let samples = gauss_face::<F>(quadrature);
    let mut out = Vec::with_capacity(samples.len());
    for (s, weight) in samples {
        let (xi, eta, ds_reference) = face_reference(local_face, s)?;
        let n = crate::physics::solenoid_stress::quad4::shape(xi, eta);
        let grad_reference = grad_ref(xi, eta);
        let jac = jacobian(coords, &grad_reference);
        let point = map_point(coords, &n);
        if point[0] < F::zero() {
            return Err(format!(
                "face quadrature point has negative radius {:?}; axisymmetric radius must be nonnegative",
                point[0]
            ));
        }
        let tangent = [
            jac[0][0] * ds_reference[0] + jac[0][1] * ds_reference[1],
            jac[1][0] * ds_reference[0] + jac[1][1] * ds_reference[1],
        ];
        let tangent_norm_sq = tangent[0] * tangent[0] + tangent[1] * tangent[1];
        if tangent_norm_sq <= F::zero() {
            return Err(format!(
                "degenerate face tangent on local face {local_face}; tangent squared norm is {tangent_norm_sq:?}"
            ));
        }
        out.push(FaceSample {
            n,
            tangent,
            point,
            weight,
        });
    }
    Ok(out)
}

pub fn element_measures<F: Real>(
    mesh: MeshView<'_, F>,
    quadrature: QuadratureRule,
) -> Result<ElementMeasures<F>, String> {
    mesh.validate_nodes()?;
    mesh.validate_connectivity()?;
    let mut areas = Vec::with_capacity(mesh.num_elements());
    let mut swept_volumes = Vec::with_capacity(mesh.num_elements());
    let two_pi = two_pi::<F>();
    for element_index in 0..mesh.num_elements() {
        let coords = mesh.element_coords(element_index)?;
        let mut area = F::zero();
        let mut swept = F::zero();
        for sample in volume_samples(&coords, quadrature)? {
            area = area + sample.det_j * sample.weight;
            swept = swept + two_pi * sample.point[0] * sample.det_j * sample.weight;
        }
        areas.push(area);
        swept_volumes.push(swept);
    }
    Ok(ElementMeasures {
        areas,
        swept_volumes,
    })
}

pub fn element_quadrature<F: Real>(
    mesh: MeshView<'_, F>,
    quadrature: QuadratureRule,
) -> Result<ElementQuadrature<F>, String> {
    mesh.validate_nodes()?;
    mesh.validate_connectivity()?;
    let nq = quadrature.points_per_element();
    let mut points_rz = Vec::with_capacity(mesh.num_elements() * nq);
    let mut weights_area = Vec::with_capacity(mesh.num_elements() * nq);
    let mut weights_volume = Vec::with_capacity(mesh.num_elements() * nq);
    let two_pi = two_pi::<F>();
    for element_index in 0..mesh.num_elements() {
        let coords = mesh.element_coords(element_index)?;
        for sample in volume_samples(&coords, quadrature)? {
            points_rz.push(sample.point);
            weights_area.push(sample.det_j * sample.weight);
            weights_volume.push(two_pi * sample.point[0] * sample.det_j * sample.weight);
        }
    }
    Ok(ElementQuadrature {
        points_rz,
        weights_area,
        weights_volume,
        nq_per_element: nq,
    })
}
