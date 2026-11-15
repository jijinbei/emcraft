//! A round guide with a sinusoidally rippled wall, r_w(z) = r₀ + h cos(2πz/d): its TM0 Bloch
//! modes, the slow-wave structure of relativistic backward-wave oscillators
//! (J. A. Swegle, J. W. Poukey, G. T. Leifeste, *Backward wave oscillators with rippled wall
//! resonators: analytic theory and numerical simulation*, Phys. Fluids 28(9):2882–2894, 1985;
//! the Floquet expansion as theirs).
//!
//! **The field.** A TM0 Bloch wave of phase ψ per period is a sum of space harmonics,
//! E_z = Σₙ Aₙ J₀(pₙr) e^{−jkₙz}, kₙ = (ψ + 2πn)/d, pₙ² = (ω/c)² − kₙ², each with
//! E_r = jkₙ r [J₁(pₙr)/(pₙr)] Aₙ e^{−jkₙz}; a harmonic slower than light has pₙ² < 0 and its
//! Bessel functions become I₀, I₁ (both written here as power series in pₙ²r², real either way).
//!
//! **The wall.** The tangential E vanishes on it, E_z + r_w′E_r = 0 at r = r_w(z). Taken against
//! e^{jkₘz} over a period, harmonics −N..N each way, it is a real matrix (r_w even in z, r_w′
//! odd):
//!
//! Mₘₙ = ⟨cos((n − m)k₀z) J₀(pₙr_w)⟩ + kₙ⟨sin((n − m)k₀z) r_w r_w′ J₁(pₙr_w)/(pₙr_w)⟩,
//!
//! k₀ = 2π/d, the averages by the trapezoid rule over the period (exact to rounding for these
//! smooth periodic integrands). The modes are where det M = 0, found by its sign changes along
//! the frequency and bisection. The expansion converges for ripples within Rayleigh's limit
//! (about k₀h < 0.45 for a sinusoid); beyond it the truncation's roots wander with N.

use std::f64::consts::PI;

use emcraft_model::units::C0;

use crate::bessel::{Radial, radial};

/// The trapezoid points over a period.
const POINTS: usize = 128;

/// How finely a mode's frequency is found: a part in 10¹², far below the expansion's own error.
const ROOT_TOLERANCE: f64 = 1e-12;

/// The most space harmonics each way.
pub const MAX_HARMONICS: usize = 12;

/// The rippled guide: its mean radius r₀, ripple amplitude h and period d (m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RippledGuide {
    pub radius: f64,
    pub ripple: f64,
    pub period: f64,
}

impl RippledGuide {
    fn valid(&self) -> bool {
        [self.radius, self.ripple, self.period].iter().all(|x| x.is_finite())
            && self.radius > 0.0
            && self.period > 0.0
            && self.ripple >= 0.0
            && self.ripple < self.radius
    }

    /// det M at angular frequency `w` (rad/s), phase `psi` per period (rad),
    /// harmonics −`n`..`n`; scaled so it neither overflows nor underflows (only its sign and
    /// zeros matter).
    fn mismatch(&self, w: f64, psi: f64, n: usize) -> f64 {
        let size = 2 * n + 1;
        let k0 = 2.0 * PI / self.period;
        let wall: Vec<(f64, f64, f64)> = (0..POINTS)
            .map(|q| {
                let z = self.period * q as f64 / POINTS as f64;
                let r = self.radius + self.ripple * (k0 * z).cos();
                (z, r, -self.ripple * k0 * (k0 * z).sin())
            })
            .collect();
        let mut m = vec![0.0; size * size];
        for col in 0..size {
            let harmonic = col as f64 - n as f64;
            let kn = (psi + 2.0 * PI * harmonic) / self.period;
            let p2 = (w / C0).powi(2) - kn * kn;
            let values: Vec<(f64, f64, f64)> = wall
                .iter()
                .map(|&(z, r, slope)| {
                    let Radial { j0: f0, j1_over_x: f1 } = radial(p2 * r * r);
                    (z, f0, r * slope * f1)
                })
                .collect();
            // Each harmonic's column scaled by its largest value on the wall: an evanescent one's
            // I₀ grows as e^{|p|r}, and only the determinant's sign is wanted.
            let scale = values.iter().map(|v| v.1.abs().max((kn * v.2).abs())).fold(0.0, f64::max).max(1e-300);
            for row in 0..size {
                let shift = (col as f64 - row as f64) * k0;
                let (mut a, mut b) = (0.0, 0.0);
                for &(z, f0, rf1) in &values {
                    let (s, c) = (shift * z).sin_cos();
                    a += c * f0;
                    b += s * rf1;
                }
                m[row * size + col] = (a + kn * b) / (POINTS as f64 * scale);
            }
        }
        crate::linalg::det(m, size)
    }

    /// The TM0 modes' frequencies (Hz) at phase `psi` per period (rad), from `low` to `high`
    /// (Hz), lowest first: det M's sign changes on a grid of `steps`, each bisected to 1e-12
    /// relative; `harmonics` each way (1 to [`MAX_HARMONICS`]). None for hostile input.
    ///
    /// ```
    /// use emcraft_theory::ripple::RippledGuide;
    /// // Without a ripple, the smooth guide's TM01: f = c √((ψ/d)² + (x₀₁/r₀)²)/2π.
    /// let (r0, d) = (0.015, 0.0167);
    /// let smooth = RippledGuide { radius: r0, ripple: 0.0, period: d };
    /// let psi = 0.5;
    /// let f = smooth.frequencies(psi, 5e9, 9.5e9, 400, 4).unwrap();
    /// let want = 299_792_458.0 * ((psi / d).powi(2) + (2.404_825_557_695_773 / r0).powi(2)).sqrt() / (2.0 * std::f64::consts::PI);
    /// assert!((f[0] / want - 1.0).abs() < 1e-9, "{f:?} against {want}");
    /// assert!(smooth.frequencies(psi, 5e9, 9e9, 400, 0).is_none());
    /// ```
    pub fn frequencies(&self, psi: f64, low: f64, high: f64, steps: usize, harmonics: usize) -> Option<Vec<f64>> {
        let ok = self.valid()
            && psi.is_finite()
            && low.is_finite()
            && high.is_finite()
            && low > 0.0
            && high > low
            && (2..=1_000_000).contains(&steps)
            && (1..=MAX_HARMONICS).contains(&harmonics);
        if !ok {
            return None;
        }
        let at = |f: f64| self.mismatch(2.0 * PI * f, psi, harmonics);
        let grid = (0..=steps).map(|i| low + (high - low) * i as f64 / steps as f64);
        Some(crate::roots::roots_on_grid(at, grid, ROOT_TOLERANCE))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_series_are_the_bessel_functions() {
        // J₀(2) = 0.2238907791, J₁(2) = 0.5767248078; I₀(3) = 4.8807925859, I₁(3) = 3.9533702174
        // (Abramowitz & Stegun, Tables 9.1 and 9.8).
        let Radial { j0, j1_over_x: j1 } = radial(4.0);
        approx::assert_abs_diff_eq!(j0, 0.223_890_779_1, epsilon = 1e-10);
        approx::assert_abs_diff_eq!(j1, 0.576_724_807_8 / 2.0, epsilon = 1e-10);
        let Radial { j0: i0, j1_over_x: i1 } = radial(-9.0);
        approx::assert_abs_diff_eq!(i0, 4.880_792_585_9, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(i1, 3.953_370_217_4 / 3.0, epsilon = 1e-9);
    }

    #[test]
    fn a_shallow_ripple_converges_with_its_harmonics_and_opens_a_gap_at_pi() {
        // Within Rayleigh's limit (k₀h = 0.38) the band's frequencies settle as harmonics are
        // added; at ψ = π the empty lattice's two folded TM01 lines split into a stop band.
        let g = RippledGuide { radius: 0.015, ripple: 0.001, period: 0.0167 };
        let at = |n: usize| g.frequencies(PI, 5e9, 14e9, 900, n).unwrap();
        let (f4, f8) = (at(4), at(8));
        assert!(f4.len() >= 2 && f8.len() >= 2, "{f4:?} {f8:?}");
        assert!((f4[0] / f8[0] - 1.0).abs() < 1e-6 && (f4[1] / f8[1] - 1.0).abs() < 1e-6, "{f4:?} {f8:?}");
        let empty = C0 * ((PI / 0.0167f64).powi(2) + (2.404_825_557_695_773 / 0.015f64).powi(2)).sqrt() / (2.0 * PI);
        assert!(f8[0] < empty && f8[1] > empty, "{f8:?} about {empty}");
    }
}
