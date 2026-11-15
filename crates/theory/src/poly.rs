//! Polynomials: their products and their complex roots, all at once by the Durand–Kerner
//! (Weierstrass) iteration (E. Durand, *Solutions numériques des équations algébriques*, 1960;
//! I. O. Kerner, Numer. Math. 8:290, 1966): each root estimate moved by p(zᵢ)/∏ⱼ≠ᵢ(zᵢ − zⱼ).
//!
//! ```
//! use emcraft_theory::poly::{Seeds, product, roots};
//! use num_complex::Complex64 as Z;
//! // (x − 1)(x + 2) = x² + x − 2, low coefficients first.
//! assert_eq!(product(&[-1.0, 1.0], &[2.0, 1.0]), vec![-2.0, 1.0, 1.0]);
//! // z³ − 1: the cube roots of unity.
//! let r: [Z; 3] = roots(|z| z * z * z - 1.0, Seeds { scale: 1.0, spread: 2.1 }, 500);
//! assert!(r.iter().all(|z| (z * z * z - 1.0).norm() < 1e-12));
//! ```

use num_complex::Complex64 as Z;

/// The product of two polynomials given by their coefficients, the constant first.
pub fn product(p: &[f64], q: &[f64]) -> Vec<f64> {
    let mut r = vec![0.0; (p.len() + q.len()).saturating_sub(1)];
    for (i, x) in p.iter().enumerate() {
        for (j, y) in q.iter().enumerate() {
            r[i + j] += x * y;
        }
    }
    r
}

/// Where the iteration starts: the roots' estimates on the circle |z| = `scale` (about the
/// roots' size) at angles 0.4 + `spread`·k rad — off the real axis, where real coefficients
/// would keep them, and apart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seeds {
    pub scale: f64,
    pub spread: f64,
}

/// The iteration has settled once no root moves more than this share of the seeds' scale.
const SETTLED: f64 = 1e-15;

/// The `N` roots of the monic polynomial `p` of degree `N`, from `seeds`, in at most
/// `max_iterations` sweeps (they settle to [`SETTLED`] of the scale long before, unless roots
/// coincide).
pub fn roots<const N: usize>(p: impl Fn(Z) -> Z, seeds: Seeds, max_iterations: usize) -> [Z; N] {
    let Seeds { scale, spread } = seeds;
    let mut r: [Z; N] = std::array::from_fn(|i| Z::from_polar(scale, 0.4 + spread * i as f64));
    for _ in 0..max_iterations {
        let mut moved = 0.0f64;
        for i in 0..N {
            let denom = (0..N).filter(|j| *j != i).fold(Z::new(1.0, 0.0), |acc, j| acc * (r[i] - r[j]));
            if denom.norm() == 0.0 {
                continue;
            }
            let step = p(r[i]) / denom;
            r[i] -= step;
            moved = moved.max(step.norm() / scale);
        }
        if moved < SETTLED {
            break;
        }
    }
    r
}
