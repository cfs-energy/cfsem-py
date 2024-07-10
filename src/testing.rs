//! Test utilities

use core::f64::consts::{E, PI};

use num_traits::{Float, NumCast};

use itertools::{self, Itertools};

use crate::mesh::MeshEdgeList;

/// Divide-by-zero-resistant approximate comparison
pub(crate) fn approx(truth: f64, val: f64, rtol: f64, atol: f64) -> bool {
    let abs_err = (val - truth).abs();
    let lim = rtol * truth.abs() + atol;
    abs_err < lim
}

/// Evenly spaced values from start to end
pub(crate) fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| start + (i as f64 / (n - 1) as f64) * (end - start))
        .collect::<Vec<f64>>()
}

/// Dense N-dimensional meshgrid; cartesian product of
/// input grids, in canonical array order
pub(crate) fn meshgrid(grids: &[&[f64]]) -> Vec<Vec<f64>> {
    let ngrids = grids.len();

    // Interleaved cartesian product
    let interleaved = grids
        .iter()
        .map(|&x| x.iter())
        .multi_cartesian_product()
        .flatten()
        .cloned()
        .collect_vec();

    // Deinterleave
    let mut meshes = Vec::with_capacity(ngrids);
    for i in 0..ngrids {
        meshes.push(
            interleaved[i..]
                .iter()
                .step_by(ngrids)
                .cloned()
                .collect_vec(),
        )
    }

    meshes
}

/// First-order forward difference; returns n-1 sized output
pub(crate) fn diff(v: &[f64]) -> Vec<f64> {
    v[1..]
        .iter()
        .zip(v[0..v.len() - 1].iter())
        .map(|(&b, &a)| b - a)
        .collect::<Vec<f64>>()
}

/// Convert circular filament to ndiscr-1 piecewise linear segments
/// (ndiscr points)
pub(crate) fn discretize_circular_filament(
    r: f64,
    z: f64,
    ndiscr: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = linspace(0.0, 2.0 * PI, ndiscr)
        .iter()
        .map(|v| r * libm::cos(*v))
        .collect();
    let y: Vec<f64> = linspace(0.0, 2.0 * PI, ndiscr)
        .iter()
        .map(|v| r * libm::sin(*v))
        .collect();
    let z: Vec<f64> = (0..ndiscr).map(|_| z).collect();
    (x, y, z)
}

/// Make an example helical path for testing
pub(crate) fn example_helix() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Make a slightly tilted helical piecewise-linear filament
    let n = 10_000;
    let xc = [0.1, -0.1]; // Start and end of centerline path
    let yc = [-0.05, 0.2];
    let zc = [-2.0, 2.0];

    let xc: Vec<f64> = linspace(xc[0], xc[1], n);
    let yc: Vec<f64> = linspace(yc[0], yc[1], n);
    let zc: Vec<f64> = linspace(zc[0], zc[1], n);

    let mut x = xc.clone();
    let mut y = xc.clone();
    let mut z = xc.clone();
    crate::mesh::filament_helix_path(
        (&xc, &yc, &zc),
        (2.0 * E / 3.0, 0.0, 0.0),
        0.5,
        0.0,
        (&mut x, &mut y, &mut z),
    )
    .unwrap();

    (x, y, z)
}

/// Example set of circular filaments with (r, z, n_turns) values
pub(crate) fn example_circular_filaments() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Make some circular filaments
    let r = 1.0 / PI; // [m] some number
    let z = 1.0 / E; // [m] some number

    let rfil = [r, r + E / 4.0, r + E / 2.0].to_vec();
    let zfil = [z, -z, 0.0].to_vec();
    let nfil = [PI, E, E / 2.0].to_vec();

    (rfil, zfil, nfil)
}

// Midpoints between each pair of points.
// Output is 1 index shorter than input.
pub(crate) fn midpoints(x: &[f64]) -> Vec<f64> {
    x[..]
        .iter()
        .zip(x[1..].iter())
        .map(|(this, next)| (this + next) / 2.0)
        .collect()
}

/// The example_helix filaments in mesh format
pub(crate) fn example_mesh<T>() -> MeshEdgeList<T>
where
    T: Float + Send + Sync,
{
    let (x, y, z) = example_helix();
    let mut nodes: Vec<(T, T, T)> = Vec::with_capacity(x.len());
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(x.len() - 1);

    for i in 0..x.len() {
        nodes.push((
            NumCast::from(x[i]).unwrap(),
            NumCast::from(y[i]).unwrap(),
            NumCast::from(z[i]).unwrap(),
        ));
    }

    for i in 0..x.len() - 1 {
        edges.push((i, i + 1));
    }

    MeshEdgeList::new(nodes, edges).unwrap()
}

#[cfg(test)]
mod test {
    use super::*;

    /// Check that meshgrid returns the correct shape and values for
    /// a 2-dimensional input.
    #[test]
    fn test_meshgrid_2d() {
        let x = [0.0, 1.0, 2.0];
        let y = [3.0, 4.0, 5.0, 6.0];
        let m = meshgrid(&[&x, &y]);

        // Check that the meshgrid has the correct shape
        assert_eq!(2, m.len()); // should be number of dimensions
        for mi in m.iter() {
            assert_eq!(x.len() * y.len(), mi.len());
        }

        // Check that the meshgrid contains the correct values
        for (i, &xi) in x.iter().enumerate() {
            for (j, &yj) in y.iter().enumerate() {
                let k = i * y.len() + j;
                assert_eq!(xi, m[0][k]);
                assert_eq!(yj, m[1][k]);
            }
        }
    }
}
