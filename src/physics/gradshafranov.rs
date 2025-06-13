//! Functions for use in [Grad-Shafranov](https://en.wikipedia.org/wiki/Grad%E2%80%93Shafranov_equation) solvers.
use std::collections::HashMap;

// Finite-difference coeffs
// https://en.wikipedia.org/wiki/Finite_difference_coefficient

/// 4th-order central difference for second derivative
const DDX2_CENTRAL: [(isize, f64); 5] = [
    (-2_isize, -1f64 / 12f64),
    (-1_isize, 4f64 / 3f64),
    (0_isize, -5f64 / 2f64),
    (1_isize, 4f64 / 3f64),
    (2_isize, -1f64 / 12f64),
];

/// 4th-order central difference for first derivative
const DDX_CENTRAL: [(isize, f64); 4] = [
    (-2_isize, 1f64 / 12f64),
    (-1_isize, -2f64 / 3f64),
    // (0_isize, 0.0f64),
    (1_isize, 2f64 / 3f64),
    (2_isize, -1f64 / 12f64),
];

/// 4th-order forward difference for second derivative
const DDX2_FWD: [(isize, f64); 6] = [
    (0_isize, 15f64 / 4f64),
    (1_isize, -77f64 / 6f64),
    (2_isize, 107f64 / 6f64),
    (3_isize, -13f64),
    (4_isize, 61f64 / 12f64),
    (5_isize, -5f64 / 6f64),
];

/// 4th-order forward difference for first derivative
const DDX_FWD: [(isize, f64); 5] = [
    (0_isize, -25f64 / 12f64),
    (1_isize, 4f64),
    (2_isize, -3f64),
    (3_isize, 4f64 / 3f64),
    (4_isize, -1f64 / 4f64),
];

/// Grad-Shafranov $\Delta^{\*}$ elliptic operator assembled to second order accuracy per Jardin eq. 4.54.
///
/// # Arguments
///
/// * `rs`: (m) 1D grid of r-coordinates, length `nr`, corresponding to a mesh of shape (nr, nz)
/// * `zs`: (m) 1D grid of z-coordinates, length `nz`
///
/// # Commentary
///
/// Values corresponding to boundary elements are unity, because this is how boundary conditions are implemented.
///
/// Returns a triplet-format (x, i, j) sparse matrix to be used to initialize a compressed sparse matrix,
/// with no particular sorting order.
///
/// # References
///
///   \[1\] S. Jardin, *Computational Methods in Plasma Physics*, 1st ed. USA: CRC Press, Inc., 2010.
pub fn gs_operator_order2(rs: &[f64], zs: &[f64]) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    let nr: usize = rs.len();
    let nz: usize = zs.len();

    let dr: f64 = rs[1] - rs[0];
    let dz: f64 = zs[1] - zs[0];

    let invdr2: f64 = 1.0 / (dr * dr);
    let invdz2: f64 = 1.0 / (dz * dz);

    // Populate finite difference matrix
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    for i in 0..nr {
        let r: f64 = rs[i];
        let invrp: f64 = 1.0 / (r + dr / 2.0); // R_{i + 1/2}
        let invrm: f64 = 1.0 / (r - dr / 2.0); // R_{i - 1/2}
        let x: f64 = r * invdr2;
        // Populate this row of the matrix
        for j in 0..nz {
            let row = i * nz + j;

            // Add eye() values along boundary
            // and finite difference scheme everywhere else
            if i == 0 || i == nr - 1 || j == 0 || j == nz - 1 {
                // This is a boundary point
                rows.push(row);
                cols.push(row);
                vals.push(1.0);
            } else {
                // This is not a boundary point
                let new_rows: [usize; 5] = [row; 5];
                rows.extend(new_rows.iter());
                let new_cols: [usize; 5] = [(row - 1), (row - nz), row, (row + nz), (row + 1)];
                cols.extend(new_cols.iter());
                let new_vals: [f64; 5] = [
                    invdz2,
                    x * invrm,
                    -2.0 * invdz2 - x * invrm - x * invrp,
                    x * invrp,
                    invdz2,
                ];
                vals.extend(new_vals.iter());
            }
        }
    }

    (vals, rows, cols)
}

/// Grad-Shafranov $\Delta^{\*}$ elliptic operator assembled to 4th order accuracy.
///
/// # Arguments
///
/// * `rs`: (m) 1D grid of r-coordinates, length `nr`, corresponding to a mesh of shape (nr, nz)
/// * `zs`: (m) 1D grid of z-coordinates, length `nz`
///
/// # Commentary
///
/// Values corresponding to boundary elements are unity, because this is how boundary conditions are implemented.
///
/// Returns a triplet-format (x, i, j) sparse matrix to be used to initialize a compressed sparse matrix,
/// with no particular sorting order.
///
/// This is a fairly direct translation of the $\Delta^{\*}$ operator from Jardin equation 4.5
///
/// $$
/// \Delta^{\*} \Psi = R \frac{\partial}{\partial R} \left( \frac{1}{R} \frac{\partial \Psi}{\partial R} \right)
///                    \+ \frac{\partial^2 \Psi}{\partial Z^2}
/// $$
///
/// to
///
/// $$
/// \Delta^{\*} \Psi = \frac{\partial^2 \Psi}{\partial Z^2} + \frac{\partial^2 \Psi}{\partial R^2}
///                    \- \frac{1}{R} \frac{\partial \Psi}{\partial R}
/// $$
///
/// from one step of product rule.
///
///
/// # References
///
///   \[1\] S. Jardin, *Computational Methods in Plasma Physics*, 1st ed. USA: CRC Press, Inc., 2010.
pub fn gs_operator_order4(rs: &[f64], zs: &[f64]) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    let nr = rs.len();
    let nz = zs.len();

    let dr = rs[1] - rs[0];
    let dz = zs[1] - zs[0];

    let dr2 = dr * dr;
    let dz2 = dz * dz;

    // Assemble finite difference coeffs
    let ddx2_central: Vec<(isize, f64)> = Vec::from(DDX2_CENTRAL);
    let ddx_central: Vec<(isize, f64)> = Vec::from(DDX_CENTRAL);

    let ddx2_fwd: Vec<(isize, f64)> = Vec::from(DDX2_FWD);
    let ddx2_bwd: Vec<(isize, f64)> = ddx2_fwd.iter().map(|x| (-x.0, x.1)).collect(); // Copy & do not flip signs

    let ddx_fwd: Vec<(isize, f64)> = Vec::from(DDX_FWD);
    let ddx_bwd: Vec<(isize, f64)> = ddx_fwd.iter().map(|x| (-x.0, -x.1)).collect(); // Copy & flip signs

    // Populate finite difference matrix
    let components: Vec<_> = (0..nr)
        .map(|i| {
            let r: f64 = rs[i];
            let mrdr = -(r * dr);
            let mut map: HashMap<(usize, usize), f64> = HashMap::new();
            // Populate this row of the matrix
            for j in 0..nz {
                let row = i * nz + j;

                // Add eye() values along boundary
                // and finite difference scheme everywhere else
                if i == 0 || i == nr - 1 || j == 0 || j == nz - 1 {
                    // This is a boundary point
                    map.insert((row, row), 1.0);
                } else {
                    // This is not a boundary point

                    /*
                    Choose finite difference stencils based on proximity to boundary
                    */
                    let (ddr2_stencil, ddr_stencil) = match i {
                        1 => (&ddx2_fwd, &ddx_fwd),
                        x if x == nr - 2 => (&ddx2_bwd, &ddx_bwd),
                        _ => (&ddx2_central, &ddx_central),
                    };
                    let ddz2_stencil = match j {
                        1 => &ddx2_fwd,
                        x if x == nz - 2 => &ddx2_bwd,
                        _ => &ddx2_central,
                    };

                    /*
                    Sum contributions
                    */
                    // d2/dr2 terms
                    for (ioffset, w) in ddr2_stencil {
                        let col = (row as isize + ioffset * nz as isize) as usize;
                        let k = (row, col);
                        let val = *w / dr2;
                        *map.entry(k).or_insert(0.0) += val;
                    }
                    // -1/R d/dr terms
                    for (ioffset, w) in ddr_stencil {
                        let col = (row as isize + ioffset * nz as isize) as usize;
                        let k = (row, col);
                        let val = *w / mrdr;
                        *map.entry(k).or_insert(0.0) += val;
                    }
                    // d2/dz2 terms
                    for (ioffset, w) in ddz2_stencil {
                        let col = (row as isize + ioffset) as usize;
                        let k = (row, col);
                        let val = *w / dz2;
                        *map.entry(k).or_insert(0.0) += val;
                    }
                }
            }
            map
        })
        .collect();

    // Reduce components
    let mut map: HashMap<(usize, usize), f64> = HashMap::new();
    for map_component in components {
        for (k, v) in map_component {
            *map.entry(k).or_insert(0.0) += v;
        }
    }

    // Convert to triplet format
    let vals: Vec<f64> = map.values().copied().collect();
    let rows: Vec<usize> = map.keys().map(|x| x.0).collect();
    let cols: Vec<usize> = map.keys().map(|x| x.1).collect();

    (vals, rows, cols)
}
