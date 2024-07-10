//! Methods for performing linear filament calcs on inputs
//! defined in mesh edge list format.
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use num_traits::{Float, NumCast};

use crate::{
    MU0_OVER_4PI,
    math::rss3,
    mesh::{MeshEdgeList, convert_point},
};
use crate::{
    math::{decompose_filament, dot3},
    physics::linear_filament::vector_potential_linear_filament_scalar,
};

/// Mutual inductance from each edge in mesh 1 to each edge in mesh 2.
/// If mesh 2 is not populated, the self-inductance of mesh 1 is taken,
/// using the thin-filament scalar self-inductance for self-terms.
///
/// Can take in geometry and return mutual inductances in an arbitrary choice of
/// float format, but in all cases, internal calculations are done with 64-bit floats
/// and parallelized over rows of the resulting matrix (each chunk accounts for
/// the contribution of one edge in mesh 1 to every edge in mesh 2).
///
/// # Panics
///
/// * On encountering a value in the input that is not representable as 64-bit float.
///   For inputs in lower widths of standard float types (f16, f32), this is unreachable.
///
/// # Arguments
///
/// * `m1`: Mesh node coordinates (in meters) and edge indices for first mesh
/// * `m2`: Optional mesh node coordinates (in meters) and edge indices for second mesh.
///       If not populated, the self-inductance matrix of mesh `m1` is calculated.
///
/// # Returns
///
/// * (m*n)-length mutual inductance matrix from edges of `m1` to `m2`, flattened, in canonical array order,
///   where `m = m1.edges().len()` and `n = m2.edges.len()`.
///   Reshape like (m, n) to restore square format without reallocating.
pub fn mesh_edge_inductance<T>(m1: &MeshEdgeList<T>, m2: Option<&MeshEdgeList<T>>) -> Vec<T>
where
    T: Float + Send + Sync,
{
    // If there is no second mesh, we're doing self inductance
    let self_inductance = m2.is_none();
    let m2 = m2.unwrap_or(m1);

    // Allocate for the output
    let n1 = m1.edges().len();
    let n2 = m2.edges().len();
    let n_out = n1 * n2;
    let mut out = vec![T::zero(); n_out];

    // Chunk output,
    // taking each chunk as all the contributions of an edge in mesh 1
    // to each edge in mesh 2
    let outc = out.par_chunks_mut(n1);

    // Loop over pairs of edges, taking M = dot((A/I), dL) .
    outc.enumerate().for_each(|(i, o)| {
        let (first, second) = m1.edges()[i];
        let xyzifil1 = (
            convert_point(m1.nodes()[first]),
            convert_point(m1.nodes()[second]),
            1.0,
        );
        for j in 0..m2.edges().len() {
            // Handle segment self-inductance case
            if self_inductance && i == j {
                let (start, end, _) = xyzifil1;
                let length = rss3(end.0 - start.0, end.1 - start.1, end.2 - start.2);
                o[j] = NumCast::from(0.5 * MU0_OVER_4PI * length).unwrap();
                continue;
            }

            // Handle mutual-inductance case
            let edge2 = m2.edges()[j];
            let (start2, end2) = (
                convert_point(m2.nodes()[edge2.0]),
                convert_point(m2.nodes()[edge2.1]),
            );

            //    First, get vector potential from edge 1 (with its midpoint as the source) to the midpoint of edge 2
            //    with unit current in edge 1 in order to extract A/I.
            let (midpoint2, dl2) = decompose_filament(start2, end2);
            let (ax_per_amp, ay_per_amp, az_per_amp) =
                vector_potential_linear_filament_scalar(xyzifil1, midpoint2);

            //    Take M = dot((A/I), dL)
            let m = dot3(ax_per_amp, ay_per_amp, az_per_amp, dl2.0, dl2.1, dl2.2);
            o[j] = NumCast::from(m).unwrap();
        }
    });

    out
}

#[cfg(test)]
mod test {
    use crate::testing::*;

    use super::mesh_edge_inductance;

    #[test]
    fn test_mesh_edge_inductance() {
        // Vector potential method should give the exact same result
        // as Neumann's formula
        let (rtol, atol) = (1e-10, 1e-10);

        // Build meshes from the same helix as the filaments
        let mesh64 = example_mesh::<f64>();
        let mesh32 = example_mesh::<f32>();

        // Total self-inductance for 64-bit mesh
        let mesh_self_inductance_f64: f64 = mesh_edge_inductance(&mesh64, None).iter().sum();

        // Total self-inductance for 32-bit mesh; do sum as f64 to avoid excessive roundoff
        let mesh_self_inductance_f32: f64 = mesh_edge_inductance::<f32>(&mesh32, None)
            .iter()
            .map(|v| *v as f64)
            .sum();

        // 64-bit filament
        let (x, y, z) = example_helix();
        let dl = (&diff(&x)[..], &diff(&y)[..], &diff(&z)[..]);
        let n = x.len() - 1;
        let xyzfil = (&x[..n], &y[..n], &z[..n]);
        let filament_self_inductance =
            crate::physics::linear_filament::inductance_piecewise_linear_filaments(
                xyzfil, dl, xyzfil, dl, true,
            )
            .unwrap();

        // Make sure mesh calcs match
        assert!(approx(
            filament_self_inductance,
            mesh_self_inductance_f64,
            rtol,
            atol
        ));
        assert!(approx(
            filament_self_inductance,
            mesh_self_inductance_f32,
            rtol,
            atol
        ));
    }
}
