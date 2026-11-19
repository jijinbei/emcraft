//! Microstrip on a substrate of thickness h and permittivity εr, a strip of width W and no
//! thickness, open above (no cover): the closed-form fits of E. Hammerstad and Ø. Jensen,
//! *Accurate models for microstrip computer-aided design*, IEEE MTT-S International Microwave
//! Symposium Digest, 1980, pp. 407–409 — the quasi-static effective permittivity (within 0.2 %
//! for 0.01 ≤ W/h ≤ 100, εr ≤ 128) and characteristic impedance (within 0.01 % for the air line,
//! W/h ≤ 1000). A published fit, not an exact solution: a reference to the fit's accuracy.

use std::f64::consts::PI;

use emcraft_model::units::{C0, MU0};

/// The quasi-static effective permittivity, u = W/h.
pub fn eps_eff(u: f64, eps_r: f64) -> f64 {
    let a = 1.0 + ((u.powi(4) + (u / 52.0).powi(2)) / (u.powi(4) + 0.432)).ln() / 49.0 + (1.0 + (u / 18.1).powi(3)).ln() / 18.7;
    let b = 0.564 * ((eps_r - 0.9) / (eps_r + 3.0)).powf(0.053);
    (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 10.0 / u).powf(-a * b)
}

/// The quasi-static characteristic impedance (Ω), u = W/h.
///
/// ```
/// use emcraft_theory::microstrip;
///
/// // A strip as wide as its alumina substrate is thick (εr = 9.8) is close to 50 Ω.
/// let z = microstrip::z0(1.0, 9.8);
/// assert!((45.0..52.0).contains(&z));
/// // Wider strips have lower impedance; air lines are faster than loaded ones.
/// assert!(microstrip::z0(2.0, 9.8) < z);
/// assert!(microstrip::eps_eff(1.0, 9.8) > 1.0 && microstrip::eps_eff(1.0, 9.8) < 9.8);
/// ```
pub fn z0(u: f64, eps_r: f64) -> f64 {
    let eta0 = MU0 * C0;
    let f = 6.0 + (2.0 * PI - 6.0) * (-(30.666 / u).powf(0.7528)).exp();
    eta0 / (2.0 * PI * eps_eff(u, eps_r).sqrt()) * (f / u + (1.0 + (2.0 / u).powi(2)).sqrt()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fits_meet_their_limits() {
        // A wide strip: nearly all the field in the substrate; a narrow one: towards half of it
        // in each medium, slowly ((1 + 10/u)^{−ab}, ab ≈ 0.37).
        approx::assert_abs_diff_eq!(eps_eff(1000.0, 9.8), 9.8, epsilon = 0.3);
        let narrow = eps_eff(0.01, 9.8) - (9.8 + 1.0) / 2.0;
        assert!(narrow > 0.0 && narrow < 0.5, "{narrow}");
        // The 50 Ω line on alumina is close to W = h (W/h ≈ 0.95); on εr = 1 the air line,
        // whose impedance is η₀/(2π) ln(f/u + √(1 + 4/u²)).
        assert!((z0(0.95, 9.8) - 50.0).abs() < 1.5, "{}", z0(0.95, 9.8));
        approx::assert_abs_diff_eq!(eps_eff(3.0, 1.0), 1.0, epsilon = 1e-12);
    }

    /// The strip's capacitance per length over ε₀ by the method of moments, h = 1: the strip in
    /// `cells` of uniform charge, each at 1 V at its centre. A line charge on the substrate's
    /// surface gives the potential (1/(2πε₀(1 + εr))) Σₙ Kⁿ⁻¹ ln[(x² + (2n)²)/(x² + (2n − 2)²)]
    /// there, K = (1 − εr)/(1 + εr), by its images in the ground and the surface (P. Silvester,
    /// *TEM wave properties of microstrip transmission lines*, Proc. IEE 115(1):43–48, 1968).
    fn capacitance(u: f64, eps_r: f64, cells: usize) -> f64 {
        // ∫ ln(x² + c²) dx.
        let primitive = |x: f64, c: f64| {
            if c == 0.0 {
                if x == 0.0 { 0.0 } else { x * (x * x).ln() - 2.0 * x }
            } else {
                x * (x * x + c * c).ln() - 2.0 * x + 2.0 * c * (x / c).atan()
            }
        };
        let k = (1.0 - eps_r) / (1.0 + eps_r);
        let terms = if k == 0.0 { 1 } else { (1e-13f64.ln() / k.abs().ln()).ceil() as usize };
        let width = u / cells as f64;
        let potential = |from: f64| -> f64 {
            let over = |c: f64| primitive(from + width / 2.0, c) - primitive(from - width / 2.0, c);
            (1..=terms).map(|n| k.powi(n as i32 - 1) * (over(2.0 * n as f64) - over(2.0 * n as f64 - 2.0))).sum::<f64>() / (2.0 * PI * (1.0 + eps_r))
        };
        // The matrix depends on the cells' distance alone.
        let by_distance: Vec<f64> = (0..cells).map(|i| potential(i as f64 * width)).collect();
        let mut a: Vec<Vec<f64>> = (0..cells).map(|i| (0..cells).map(|j| by_distance[i.abs_diff(j)]).chain([1.0]).collect()).collect();
        for c in 0..cells {
            let pivot = (c..cells).max_by(|x, y| a[*x][c].abs().total_cmp(&a[*y][c].abs())).unwrap();
            a.swap(c, pivot);
            for r in 0..cells {
                if r != c {
                    let f = a[r][c] / a[c][c];
                    for q in c..=cells {
                        a[r][q] -= f * a[c][q];
                    }
                }
            }
        }
        (0..cells).map(|i| a[i][cells] / a[i][i]).sum::<f64>() * width
    }

    /// The fits against the moment method's line, Z = η₀/√(C C_air) and εe = C/C_air with the
    /// capacitances over ε₀, each from 50 and 100 cells (the error falls as one over their
    /// number: 2 C₁₀₀ − C₅₀): the impedance within 0.1 %, the permittivity within the fit's
    /// 0.2 %, for narrow and wide strips on soft and hard substrates.
    #[test]
    fn the_fits_are_the_moment_methods_microstrip() {
        for (u, eps_r) in [(0.2, 2.2), (1.0, 9.8), (3.0, 4.2), (10.0, 2.2), (0.5, 12.9)] {
            let c = |e: f64| 2.0 * capacitance(u, e, 100) - capacitance(u, e, 50);
            let (air, loaded) = (c(1.0), c(eps_r));
            let (z, ee) = (MU0 * C0 / (air * loaded).sqrt(), loaded / air);
            assert!((z0(u, eps_r) / z - 1.0).abs() < 1e-3, "u {u}, εr {eps_r}: {} against {z}", z0(u, eps_r));
            assert!((eps_eff(u, eps_r) / ee - 1.0).abs() < 2e-3, "u {u}, εr {eps_r}: {} against {ee}", eps_eff(u, eps_r));
        }
    }
}
