use crate::MU0_OVER_4PI;
use crate::math::{cross3, rss3};

/// Second-order quadrature integration weights on a triangular surface
/// Format is [Weights, U, V]
///
/// References:
/// * F. Hussain, M. S. Karim, and R. Ahamad, “Appropriate Gaussian quadrature formulae for triangles”.
const TABLE_GAUSS_LEGENDRE_2: [[f64; 3]; 4] = [
    [0.5283121635e-01, 0.1666666667e+00, 0.7886751346e+00],
    [0.1971687836e+00, 0.6220084679e+00, 0.2113248654e+00],
    [0.5283121635e-01, 0.4465819874e-01, 0.7886751346e+00],
    [0.1971687836e+00, 0.1666666667e+00, 0.2113248654e+00],
];

/// Third-order quadrature integration weights on a triangular surface
/// Format is [Weights, U, V]
///
/// References:
/// * F. Hussain, M. S. Karim, and R. Ahamad, “Appropriate Gaussian quadrature formulae for triangles”.
const TABLE_GAUSS_LEGENDRE_3: [[f64; 3]; 9] = [
    [0.9876542474e-01, 0.2500000000e+00, 0.5000000000e+00],
    [0.1391378575e-01, 0.5635083269e-01, 0.8872983346e+00],
    [0.1095430035e+00, 0.4436491673e+00, 0.1127016654e+00],
    [0.6172839460e-01, 0.4436491673e+00, 0.5000000000e+00],
    [0.8696116674e-02, 0.1000000000e+00, 0.8872983346e+00],
    [0.6846438175e-01, 0.7872983346e+00, 0.1127016654e+00],
    [0.6172839460e-01, 0.5635083269e-01, 0.5000000000e+00],
    [0.8696116674e-02, 0.1270166538e-01, 0.8872983346e+00],
    [0.6846438175e-01, 0.1000000000e+00, 0.1127016654e+00],
];

#[derive(Clone, Copy)]
pub enum QuadratureKind {
    GaussLegendre,
    Dunavant,
}

/// Isoparametric mapping of a point on a 3D triangle
/// from U-V coordinates on the triangle's surface.
#[inline]
pub fn map_tri_uv(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3], pin_uv: [f64; 2]) -> [f64; 3] {
    // Barycentric interpolation
    let mut pout = [0.0; 3];

    // Precalulate third uv component
    let w = 1.0 - pin_uv[0] - pin_uv[1];
    pout[0] = n0[0] * w + n1[0] * pin_uv[0] + n2[0] * pin_uv[1];
    pout[1] = n0[1] * w + n1[1] * pin_uv[0] + n2[1] * pin_uv[1];
    pout[2] = n0[2] * w + n1[2] * pin_uv[0] + n2[2] * pin_uv[1];

    return pout;
}

/// Area of a 3D triangle.
#[inline]
pub fn calc_tri_area(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3]) -> f64 {
    // Get two directional vectors from node 0 to nodes 1 and 2
    let v01 = [n1[0] - n0[0], n1[1] - n0[1], n1[2] - n0[2]];
    let v02 = [n2[0] - n0[0], n2[1] - n0[1], n2[2] - n0[2]];

    // Area of parallelgram is norm of crossproduct of the vectors.
    let cross = cross3(v01[0], v01[1], v01[2], v02[0], v02[1], v02[2]);
    let area = 0.5 * rss3(cross.0, cross.1, cross.2);
    return area;
}

/// Normal vector of a triangle.
///
/// Direction is non-unique; the order of the points determines whether
/// the returned normal points "up" or "down" relative to the triangle.
#[inline]
pub fn calc_tri_normal(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3]) -> [f64; 3] {
    // Get two directional vectors from node 0 to nodes 1 and 2
    let v01 = [n1[0] - n0[0], n1[1] - n0[1], n1[2] - n0[2]];
    let v02 = [n2[0] - n0[0], n2[1] - n0[1], n2[2] - n0[2]];

    // Get cross product for the two directional vectors
    let cross = cross3(v01[0], v01[1], v01[2], v02[0], v02[1], v02[2]);

    // Normalize the normal vector
    let norm = rss3(cross.0, cross.1, cross.2);
    let out = [cross.0 / norm, cross.1 / norm, cross.2 / norm];

    return out;
}

/// Magnetic flux density (B-field) contribution of a given triangle's basis function
/// with unit weighting to a given obseration point.
///
/// Assumes a basis function living on the triangle's first node.
#[inline]
pub fn triangle_flux_density_basis(
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
    obs: [f64; 3],
    quad_kind: QuadratureKind,
    quad_order: usize,
) -> [f64; 3] {
    // Calculate directional vectors between nodes
    let v01 = [n1[0] - n0[0], n1[1] - n0[1], n1[2] - n0[2]];
    let v02 = [n2[0] - n0[0], n2[1] - n0[1], n2[2] - n0[2]];

    // Triangle area and normal vector
    let tri_area = calc_tri_area(n0, n1, n2);

    // Reference current density vector for this triangle
    let jref = [
        (v02[0] - v01[0]) / (2.0 * tri_area),
        (v02[1] - v01[1]) / (2.0 * tri_area),
        (v02[2] - v01[2]) / (2.0 * tri_area),
    ];

    // Match quadrature kind and order to list of quadrature points
    let quad_points: &[[f64; 3]] = match (quad_kind, quad_order) {
        (QuadratureKind::GaussLegendre, 2) => &TABLE_GAUSS_LEGENDRE_2,
        (QuadratureKind::GaussLegendre, 3) => &TABLE_GAUSS_LEGENDRE_3,
        _ => panic!(),
    };

    let mut b = [0.0; 3];

    for i in 0..quad_points.len() {
        // Unpack current quad point
        let (c, u, v) = (quad_points[i][0], quad_points[i][1], quad_points[i][2]);

        // Transform current quad points for the given triangle
        let qp = map_tri_uv(n0, n1, n2, [u, v]);

        // Biot-Savart kernel for a constant surface current density over the triangle
        let r = [obs[0] - qp[0], obs[1] - qp[1], obs[2] - qp[2]];
        let d = rss3(r[0], r[1], r[2]);
        let inv_d3 = 1.0 / (d * d * d);
        let j_cross_r = cross3(jref[0], jref[1], jref[2], r[0], r[1], r[2]);

        // Add B-field contribution for current quadrature point
        b[0] += c * j_cross_r.0 * inv_d3 * tri_area;
        b[1] += c * j_cross_r.1 * inv_d3 * tri_area;
        b[2] += c * j_cross_r.2 * inv_d3 * tri_area;
    }

    return b;
}

/// Flux density (B-field) of triangular surface current density distribution
/// at a target point due to scalar current density potential `s`
/// at each node.
#[inline]
pub fn flux_density_triangle(
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
    s: [f64; 3],
    obs: [f64; 3],
    quad_kind: QuadratureKind,
    quad_order: usize,
) -> [f64; 3] {
    let mut out = [0.0; 3];

    // Collect B-field contributions for the three basis functions living on n0, n1, and n2
    let b_n0 = triangle_flux_density_basis(n0, n1, n2, obs, quad_kind, quad_order);
    let b_n1 = triangle_flux_density_basis(n1, n2, n0, obs, quad_kind, quad_order);
    let b_n2 = triangle_flux_density_basis(n2, n0, n1, obs, quad_kind, quad_order);

    // Sum contributions by each basis function weighted by the basis function value
    out[0] = (s[0] * b_n0[0] + s[1] * b_n1[0] + s[2] * b_n2[0]) * MU0_OVER_4PI;
    out[1] = (s[0] * b_n0[1] + s[1] * b_n1[1] + s[2] * b_n2[1]) * MU0_OVER_4PI;
    out[2] = (s[0] * b_n0[2] + s[1] * b_n1[2] + s[2] * b_n2[2]) * MU0_OVER_4PI;

    return out;
}

#[cfg(test)]
mod tests {
    use core::f64::consts::PI;

    use super::{QuadratureKind, calc_tri_normal, flux_density_triangle};
    use crate::physics::circular_filament::flux_density_circular_filament_cartesian_scalar;
    use crate::testing::approx;

    #[derive(Clone, Copy)]
    struct TrianglePatch {
        nodes: [[f64; 3]; 3],
        s: [f64; 3],
    }

    fn circular_strip_triangles(
        radius: f64,
        height: f64,
        s0: f64,
        nphi: usize,
    ) -> Vec<TrianglePatch> {
        assert!(nphi >= 3);

        let dphi = 2.0 * PI / nphi as f64;
        let mut tris = Vec::with_capacity(2 * nphi);

        for i in 0..nphi {
            let phi0 = i as f64 * dphi;
            let phi1 = (i + 1) as f64 * dphi;

            let lower0 = [radius * phi0.cos(), radius * phi0.sin(), -height / 2.0];
            let lower1 = [radius * phi1.cos(), radius * phi1.sin(), -height / 2.0];
            let upper0 = [radius * phi0.cos(), radius * phi0.sin(), height / 2.0];
            let upper1 = [radius * phi1.cos(), radius * phi1.sin(), height / 2.0];

            let tri0 = TrianglePatch {
                nodes: [lower0, lower1, upper1],
                s: [-s0, -s0, s0],
            };
            let tri1 = TrianglePatch {
                nodes: [lower0, upper1, upper0],
                s: [-s0, s0, s0],
            };

            let radial = [(phi0 + 0.5 * dphi).cos(), (phi0 + 0.5 * dphi).sin(), 0.0];
            for tri in [tri0, tri1] {
                let normal = calc_tri_normal(tri.nodes[0], tri.nodes[1], tri.nodes[2]);
                let alignment = normal[0] * radial[0] + normal[1] * radial[1];
                assert!(
                    alignment > 0.0,
                    "triangle winding is not radially consistent"
                );
                tris.push(tri);
            }
        }

        tris
    }

    fn strip_flux_density(tris: &[TrianglePatch], obs: [f64; 3]) -> [f64; 3] {
        let mut out = [0.0; 3];
        for tri in tris {
            let contrib = flux_density_triangle(
                tri.nodes[0],
                tri.nodes[1],
                tri.nodes[2],
                tri.s,
                obs,
                QuadratureKind::GaussLegendre,
                3,
            );
            out[0] += contrib[0];
            out[1] += contrib[1];
            out[2] += contrib[2];
        }
        out
    }

    fn max_abs_component(vectors: &[[f64; 3]]) -> f64 {
        vectors
            .iter()
            .flat_map(|v| v.iter())
            .map(|v| v.abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn test_flux_density_triangle_circular_strip_matches_circular_filament_far_field() {
        let radius = 0.7312345987;
        let height = radius * 1e-3;
        let nphi = 256;

        let loop_current = 1.7; // Some not-special number to check current scaling
        let s0 = loop_current; // Potential jump for this strip construction matches the loop current

        let strip = circular_strip_triangles(radius, height, s0, nphi);
        let obs = [
            [2.70, 0.95, 0.85],
            [3.10, -1.15, 1.05],
            [3.45, 0.75, -1.25],
            [3.80, 1.30, 1.60],
            [4.20, -0.90, -1.55],
            [4.55, 1.10, -2.05],
        ];

        let mut b_strip = Vec::with_capacity(obs.len());
        let mut b_loop = Vec::with_capacity(obs.len());
        for point in obs {
            b_strip.push(strip_flux_density(&strip, point));
            let b_ref = flux_density_circular_filament_cartesian_scalar(
                (radius, 0.0, loop_current),
                (point[0], point[1], point[2]),
            );
            b_loop.push([b_ref.0, b_ref.1, b_ref.2]);
        }

        let axis_names = ["Bx", "By", "Bz"];
        let bfield_rtol = 1e-3;
        let bfield_atol = max_abs_component(&b_loop) * 1e-12;

        for i in 0..obs.len() {
            for axis in 0..3 {
                assert!(
                    approx(b_loop[i][axis], b_strip[i][axis], bfield_rtol, bfield_atol),
                    "{} mismatch at point {}: strip={:.6e}, reference={:.6e}, obs={:?}",
                    axis_names[axis],
                    i,
                    b_strip[i][axis],
                    b_loop[i][axis],
                    obs[i],
                );
            }
        }
    }

    #[test]
    fn test_flux_density_triangle_circular_strip_matches_circular_filament_near_axis() {
        let radius = 0.75;
        let height = radius * 1e-3;
        let nphi = 256;

        let loop_current = 1.7; // Some not-special number to check current scaling
        let s0 = loop_current; // Potential jump for this strip construction matches the loop current

        let strip = circular_strip_triangles(radius, height, s0, nphi);

        let mut obs = Vec::with_capacity(202);
        for i in 0..=200 {
            let z = -1.0 + i as f64 * 0.01;
            obs.push([1e-8, 0.0, z]);
        }
        obs.push([0.0, 1e-8, 0.0]);

        let axis_names = ["Bx", "By", "Bz"];
        let bz_rtol = 1e-3;
        let mut b_loop = Vec::with_capacity(obs.len());
        for point in &obs {
            let b_ref = flux_density_circular_filament_cartesian_scalar(
                (radius, 0.0, loop_current),
                (point[0], point[1], point[2]),
            );
            b_loop.push([b_ref.0, b_ref.1, b_ref.2]);
        }
        let transverse_atol = max_abs_component(&b_loop) * 1e-8 + 1e-14;

        for (i, point) in obs.iter().enumerate() {
            let b_strip = strip_flux_density(&strip, *point);

            assert!(
                approx(b_loop[i][2], b_strip[2], bz_rtol, 0.0),
                "Bz mismatch at point {}: strip={:.6e}, reference={:.6e}, obs={:?}",
                i,
                b_strip[2],
                b_loop[i][2],
                point,
            );

            for axis in 0..2 {
                assert!(
                    b_strip[axis].abs() < transverse_atol,
                    "{} should be near zero at point {}: strip={:.6e}, atol={:.6e}, obs={:?}",
                    axis_names[axis],
                    i,
                    b_strip[axis],
                    transverse_atol,
                    point,
                );
            }
        }
    }
}
