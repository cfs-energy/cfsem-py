//! Exact static potential of a uniformly loaded flat triangle.
//!
//! This boundary-line reduction is equivalent to the robust constant-source
//! formulation in Wilton et al., IEEE Access 2020, Eqs. (5)-(9), (15), and
//! (19)-(21).

use crate::math::{Scalar, cross3, dot3, norm3, scale3, sub3};
use crate::mesh::elements::tri::tri3::closest_point;

const SURFACE_EPSILON_FACTOR: f64 = 64.0;

/// Cached geometry for exact potential evaluation on a nondegenerate triangle.
///
/// The triangle carries a uniform unit source density. [`Self::scalar_potential`]
/// returns `∫ dS' / |r - r'|` over the triangle and
/// [`Self::scalar_potential_gradient`] its observation-point gradient, both in
/// closed form and finite at every observation point except the gradient on
/// the triangle's own edges. Multiplying by `μ0/4π` and a constant surface
/// current density gives the triangle's vector potential.
#[derive(Clone, Copy, Debug)]
pub struct UniformTriangle<T: Scalar> {
    nodes: [[T; 3]; 3],
    normal: [T; 3],
    edge_tangent: [[T; 3]; 3],
    edge_outward: [[T; 3]; 3],
    edge_length: [T; 3],
    max_edge: T,
}

impl<T: Scalar> UniformTriangle<T> {
    /// Construct cached geometry for a nondegenerate, oriented triangle.
    ///
    /// Args:
    ///     n0: Vertex 0 coordinates `[x, y, z]` (m).
    ///     n1: Vertex 1 coordinates `[x, y, z]` (m).
    ///     n2: Vertex 2 coordinates `[x, y, z]` (m).
    ///
    /// Returns:
    ///     Cached edge frames, unit normal `(n1 - n0) x (n2 - n0)` normalized,
    ///     and edge lengths. A degenerate triangle trips a debug assertion and
    ///     yields non-finite geometry in release builds.
    #[inline]
    pub fn new(n0: [T; 3], n1: [T; 3], n2: [T; 3]) -> Self {
        let nodes = [n0, n1, n2];
        let raw_normal = cross3(sub3(n1, n0), sub3(n2, n0));
        let twice_area = norm3(raw_normal);
        debug_assert!(twice_area > T::ZERO, "triangle must be nondegenerate");
        let normal = scale3(raw_normal, T::ONE / twice_area);

        let mut edge_tangent = [[T::ZERO; 3]; 3];
        let mut edge_outward = [[T::ZERO; 3]; 3];
        let mut edge_length = [T::ZERO; 3];
        let mut max_edge = T::ZERO;
        for edge in 0..3 {
            let delta = sub3(nodes[(edge + 1) % 3], nodes[edge]);
            let length = norm3(delta);
            let tangent = scale3(delta, T::ONE / length);
            edge_tangent[edge] = tangent;
            edge_outward[edge] = cross3(tangent, normal);
            edge_length[edge] = length;
            if length > max_edge {
                max_edge = length;
            }
        }

        Self {
            nodes,
            normal,
            edge_tangent,
            edge_outward,
            edge_length,
            max_edge,
        }
    }

    /// Triangle vertices `[n0, n1, n2]` (m).
    #[inline]
    pub fn nodes(&self) -> [[T; 3]; 3] {
        self.nodes
    }

    /// Unit normal from the vertex orientation (dimensionless).
    ///
    /// Right-handed with respect to `n0 -> n1 -> n2`.
    #[inline]
    pub fn normal(&self) -> [T; 3] {
        self.normal
    }

    /// Return the exact scalar potential `∫ dS' / |r - r'|` (m).
    ///
    /// Finite and continuous everywhere, including on the triangle, its edges
    /// and its vertices.
    ///
    /// Args:
    ///     obs: Observation point `[x, y, z]` (m).
    ///
    /// Returns:
    ///     The potential of a unit uniform source density on the triangle (m).
    ///
    /// References:
    /// - \[8\], Eqs. (5)-(9), (15), and (19)-(21), including limiting cases.
    #[inline]
    pub fn scalar_potential(&self, obs: [T; 3]) -> T {
        let d = dot3(sub3(obs, self.nodes[0]), self.normal);
        let mut value = T::ZERO;
        let d2 = d * d;
        for edge in 0..3 {
            let rel = sub3(self.nodes[edge], obs);
            let h = dot3(rel, self.edge_outward[edge]);
            // Apply the analytic h log(h) -> 0 limit on an edge or vertex.
            if h == T::ZERO && d == T::ZERO {
                continue;
            }
            let along_start = dot3(rel, self.edge_tangent[edge]);
            let along_end = along_start + self.edge_length[edge];
            let transverse = h.mul_add(h, d2).sqrt();
            let integral = if transverse > T::ZERO {
                asinh_difference(along_end / transverse, along_start / transverse)
            } else {
                self.edge_integral(edge, obs, d)
            };
            value = value + h * integral;
        }
        value - d * self.signed_solid_angle(obs)
    }

    /// Return the exact observation-point gradient (dimensionless).
    ///
    /// The caller chooses a convention on the source triangle, where the
    /// normal component is discontinuous.
    ///
    /// Args:
    ///     obs: Observation point `[x, y, z]` (m).
    ///
    /// Returns:
    ///     `∇_obs ∫ dS' / |r - r'|` (dimensionless); non-finite on the
    ///     triangle's edges.
    ///
    /// References:
    /// - \[8\], Eqs. (5)-(9), (15), and (19)-(21), including limiting cases.
    #[inline]
    pub fn scalar_potential_gradient(&self, obs: [T; 3]) -> [T; 3] {
        let d = dot3(sub3(obs, self.nodes[0]), self.normal);
        let mut gradient = [T::ZERO; 3];
        for edge in 0..3 {
            let integral = self.edge_integral(edge, obs, d);
            for axis in 0..3 {
                gradient[axis] = gradient[axis] - self.edge_outward[edge][axis] * integral;
            }
        }
        let omega = self.signed_solid_angle(obs);
        for axis in 0..3 {
            gradient[axis] = gradient[axis] - self.normal[axis] * omega;
        }
        gradient
    }

    /// Return true only when the point is represented on the finite triangle.
    ///
    /// The tolerance is a fixed multiple of machine epsilon times the larger
    /// of the point's coordinate magnitude and the longest edge, so recomputed
    /// mesh points (centroids, quadrature points) are classified on-surface.
    ///
    /// Args:
    ///     obs: Observation point `[x, y, z]` (m).
    ///
    /// Returns:
    ///     Whether `obs` lies on the closed triangle to that tolerance.
    #[inline]
    pub fn contains_on_surface(&self, obs: [T; 3]) -> bool {
        let closest = closest_point(obs, self.nodes[0], self.nodes[1], self.nodes[2]);
        let delta = sub3(obs, closest);
        let factor = crate::math::cast::<T>(SURFACE_EPSILON_FACTOR);
        let obs_norm = norm3(obs);
        // Recomputing a mesh point in global coordinates introduces a plane
        // residual proportional to the coordinate magnitude, not just the edge length.
        let scale = obs_norm.max(self.max_edge);
        let tolerance = factor * T::epsilon() * scale;
        dot3(delta, delta) <= tolerance * tolerance
    }

    /// Integral of inverse distance along one edge (dimensionless).
    ///
    /// References:
    /// - \[8\], Eqs. (5)-(9) and limiting cases in Eqs. (19)-(21).
    #[inline]
    fn edge_integral(&self, edge: usize, obs: [T; 3], d: T) -> T {
        let start = self.nodes[edge];
        let tangent = self.edge_tangent[edge];
        let along_start = dot3(sub3(start, obs), tangent);
        let along_end = along_start + self.edge_length[edge];
        let h = dot3(sub3(start, obs), self.edge_outward[edge]);
        let transverse = h.mul_add(h, d * d).sqrt();

        if transverse > T::ZERO {
            return (along_end / transverse).asinh() - (along_start / transverse).asinh();
        }

        // Coplanar and on the edge line, but outside the finite segment.
        if along_start != T::ZERO
            && along_end != T::ZERO
            && (along_start > T::ZERO) == (along_end > T::ZERO)
        {
            return (along_end.abs() / along_start.abs()).ln().abs();
        }
        T::infinity()
    }

    /// Signed solid angle used by the exact potential and gradient.
    ///
    /// References:
    /// - \[8\], Eq. (15) and the source-plane limits in Eqs. (19)-(21).
    #[inline]
    fn signed_solid_angle(&self, obs: [T; 3]) -> T {
        let r0 = sub3(obs, self.nodes[0]);
        let r1 = sub3(obs, self.nodes[1]);
        let r2 = sub3(obs, self.nodes[2]);
        let n0 = norm3(r0);
        let n1 = norm3(r1);
        let n2 = norm3(r2);
        let numerator = dot3(r0, cross3(r1, r2));
        let denominator = n0 * n1 * n2 + dot3(r0, r1) * n2 + dot3(r1, r2) * n0 + dot3(r2, r0) * n1;
        crate::math::cast::<T>(2.0) * numerator.atan2(denominator)
    }
}

/// `asinh(a) - asinh(b)` with one logarithm instead of two (dimensionless).
///
/// With `p(x) = |x| + sqrt(x² + 1)` and `asinh(x) = sign(x)·ln p(x)`, the
/// difference is `ln(p(a)^{±1} / p(b)^{±1})`. It is evaluated as `ln_1p` of
/// an argument built from `q(x) = p(x) - 1 = |x| + x² / (sqrt(x² + 1) + 1)`,
/// which keeps full relative precision when both arguments are small (the
/// far-field regime, where the two `asinh` values nearly cancel).
#[inline]
fn asinh_difference<T: Scalar>(a: T, b: T) -> T {
    #[inline]
    fn q<T: Scalar>(x: T) -> T {
        let root = x.mul_add(x, T::ONE).sqrt();
        x.abs() + x * x / (root + T::ONE)
    }
    let (qa, qb) = (q(a), q(b));
    match (a >= T::ZERO, b >= T::ZERO) {
        // ln(pa / pb)
        (true, true) => ((qa - qb) / (T::ONE + qb)).ln_1p(),
        // ln(pa · pb)
        (true, false) => qa.mul_add(qb, qa + qb).ln_1p(),
        // -ln(pa · pb)
        (false, true) => T::ZERO - qa.mul_add(qb, qa + qb).ln_1p(),
        // ln(pb / pa)
        (false, false) => ((qb - qa) / (T::ONE + qa)).ln_1p(),
    }
}

#[cfg(test)]
mod tests {
    use super::{UniformTriangle, asinh_difference};

    #[test]
    fn asinh_difference_matches_libm() {
        for (a, b) in [
            (3.0_f64, 1.0_f64),
            (-3.0, -1.0),
            (3.0, -1.0),
            (-0.2, 0.7),
            (1e6, 1e6 - 3.0),
            (-1e6, -1e6 + 2.0),
            (1e-9, -1e-9),
            (0.0, 2.0),
        ] {
            let exact = a.asinh() - b.asinh();
            let fast = asinh_difference(a, b);
            // libm's difference itself carries the rounding of two large asinh values.
            let tolerance = 8.0 * f64::EPSILON * (a.asinh().abs() + b.asinh().abs() + exact.abs());
            assert!(
                (fast - exact).abs() <= tolerance,
                "a={a} b={b}: {fast} vs {exact}"
            );
        }
    }

    #[test]
    fn potential_matches_vector_potential_triangle() {
        // The public vector potential is MU0/4π · J · Φ with J the unit
        // basis current density.
        use crate::physics::boundary_element::{
            triangle_basis_current_densities, vector_potential_triangle,
        };
        let n = [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.006, 0.03, 0.0]];
        let tri = UniformTriangle::new(n[0], n[1], n[2]);
        for obs in [[0.01, 0.01, 0.0], [0.01, 0.01, 0.003], [0.2, -0.1, 0.05]] {
            let a = vector_potential_triangle(n[0], n[1], n[2], [1.0, 0.0, 0.0], obs);
            let j = triangle_basis_current_densities(n[0], n[1], n[2])[0];
            let expected = crate::MU0_OVER_4PI * tri.scalar_potential(obs);
            for axis in 0..3 {
                assert!(
                    approx(a[axis], expected * j[axis], 1e-13, 1e-20),
                    "obs={obs:?} axis={axis} vector_potential={} potential={}",
                    a[axis],
                    expected * j[axis]
                );
            }
        }
    }

    #[test]
    fn normal_is_right_handed_unit() {
        let tri = UniformTriangle::new([0.0_f64, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]);
        assert_eq!(tri.normal(), [0.0, 0.0, 1.0]);
        let flipped = UniformTriangle::new([0.0_f64, 0.0, 0.0], [0.0, 3.0, 0.0], [2.0, 0.0, 0.0]);
        assert_eq!(flipped.normal(), [0.0, 0.0, -1.0]);
    }

    fn approx(a: f64, b: f64, rel: f64, abs: f64) -> bool {
        (a - b).abs() <= abs.max(rel * a.abs().max(b.abs()))
    }

    #[test]
    fn gradient_matches_potential_finite_difference() {
        let tri = UniformTriangle::new([0.0_f64, 0.0, 0.0], [0.9, 0.1, 0.0], [0.2, 0.7, 0.0]);
        for obs in [[0.2, 0.2, 0.4], [0.2, 0.2, -0.4], [1.1, -0.2, 0.03]] {
            let gradient = tri.scalar_potential_gradient(obs);
            let step = 1e-6;
            for axis in 0..3 {
                let mut plus = obs;
                let mut minus = obs;
                plus[axis] += step;
                minus[axis] -= step;
                let fd = (tri.scalar_potential(plus) - tri.scalar_potential(minus)) / (2.0 * step);
                assert!(
                    approx(gradient[axis], fd, 2e-8, 2e-9),
                    "obs={obs:?} axis={axis} analytic={} fd={fd}",
                    gradient[axis]
                );
            }
        }
    }

    #[test]
    fn surface_predicate_is_finite_triangle_only() {
        let tri = UniformTriangle::new([0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        for obs in [[0.2, 0.2, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]] {
            assert!(tri.contains_on_surface(obs));
        }
        assert!(!tri.contains_on_surface([1.0, 1.0, 0.0]));
        assert!(!tri.contains_on_surface([0.2, 0.2, 1e-12]));
        assert!(!tri.contains_on_surface([0.2, 0.2, -1e-12]));
    }

    #[test]
    fn surface_predicate_accepts_recomputed_centroids_at_machine_coordinates() {
        let major_radius = 1.85_f64;
        let minor_radius = 0.5_f64;
        let edge = 5e-3_f64;

        let dphi = edge / (major_radius + minor_radius);
        let dtheta = edge / minor_radius;
        let torus_point = |phi: f64, theta: f64| {
            let (sin_phi, cos_phi) = phi.sin_cos();
            let (sin_theta, cos_theta) = theta.sin_cos();
            let radius = major_radius + minor_radius * cos_theta;
            [radius * cos_phi, radius * sin_phi, minor_radius * sin_theta]
        };

        for iphi in 0..20 {
            let phi0 = -0.02 + dphi * iphi as f64;
            let phi1 = phi0 + dphi;
            for itheta in 0..20 {
                let theta0 = -0.1 + dtheta * itheta as f64;
                let theta1 = theta0 + dtheta;
                let p00 = torus_point(phi0, theta0);
                let p10 = torus_point(phi1, theta0);
                let p01 = torus_point(phi0, theta1);
                let p11 = torus_point(phi1, theta1);

                for nodes in [[p00, p10, p11], [p00, p11, p01]] {
                    let centroid = std::array::from_fn(|axis| {
                        (nodes[0][axis] + nodes[1][axis] + nodes[2][axis]) / 3.0
                    });
                    let triangle = UniformTriangle::new(nodes[0], nodes[1], nodes[2]);
                    assert!(
                        triangle.contains_on_surface(centroid),
                        "recomputed centroid was classified off-surface: nodes={nodes:?}, centroid={centroid:?}",
                    );

                    let normal_offset =
                        std::array::from_fn(|axis| centroid[axis] + 1e-12 * triangle.normal[axis]);
                    assert!(
                        !triangle.contains_on_surface(normal_offset),
                        "1 pm normal offset was classified on-surface",
                    );
                }
            }
        }
    }

    #[test]
    fn potential_scales_and_gradient_is_scale_invariant() {
        let base = UniformTriangle::new([0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let scale = 1e5;
        let scaled =
            UniformTriangle::new([0.0_f64, 0.0, 0.0], [scale, 0.0, 0.0], [0.0, scale, 0.0]);
        let obs = [0.2, 0.3, 0.7];
        let obs_scaled = [obs[0] * scale, obs[1] * scale, obs[2] * scale];
        assert!(approx(
            scaled.scalar_potential(obs_scaled),
            scale * base.scalar_potential(obs),
            2e-14,
            1e-12
        ));
        let g0 = base.scalar_potential_gradient(obs);
        let g1 = scaled.scalar_potential_gradient(obs_scaled);
        for axis in 0..3 {
            assert!(approx(g0[axis], g1[axis], 2e-14, 1e-14));
        }
    }

    #[test]
    fn potential_is_finite_on_interior_edges_and_vertices() {
        let tri = UniformTriangle::new([0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        for obs in [[0.2, 0.2, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]] {
            let value = tri.scalar_potential(obs);
            assert!(value.is_finite() && value > 0.0);
        }
    }

    #[test]
    fn far_field_matches_monopole_limit() {
        let tri = UniformTriangle::new([0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let centroid = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        let area = 0.5;
        for distance in [10.0_f64, 1e2, 1e3, 1e4, 1e6, 1e8] {
            let obs = [centroid[0], centroid[1], distance];
            let expected_potential = area / distance;
            let expected_gradient_z = -area / distance.powi(2);
            let potential = tri.scalar_potential(obs);
            let gradient = tri.scalar_potential_gradient(obs);
            assert!(
                approx(potential, expected_potential, 2e-3, 1e-15),
                "distance={distance:e} potential={potential:e} expected={expected_potential:e}"
            );
            assert!(
                approx(gradient[2], expected_gradient_z, 4e-3, 1e-24),
                "distance={distance:e} gradient_z={} expected={expected_gradient_z:e}",
                gradient[2]
            );
        }
    }
}
