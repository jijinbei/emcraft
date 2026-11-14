//! The backward-wave oscillator's start of oscillation (docs/adr/0012-particles.md, stage D).
//!
//! A beam of velocity v₀ in step with a backward space harmonic of a slow-wave structure
//! (H. R. Johnson, *Backward-wave oscillators*, Proc. IRE 43:684–697, 1955; J. R. Pierce,
//! *Traveling-Wave Tubes*, Van Nostrand 1950, chs. II and VII): with the circuit field ∝
//! e^{−jβₑz + βₑCδz}, βₑ = ω/v₀, and a lossless circuit, the three waves' δ solve
//!
//! (δ² + 4QC)(δ + jb) = j,
//!
//! b the velocity parameter, (v₀ − v_p)/(C v_p), QC Pierce's space-charge parameter. The beam
//! enters unmodulated: its current and velocity, ∝ Σ Aᵢ/(δᵢ² + 4QC) and Σ Aᵢδᵢ/(δᵢ² + 4QC),
//! that is Σ Aᵢ(δᵢ + jb) and Σ Aᵢδᵢ(δᵢ + jb), vanish at z = 0; nothing comes in at the collector
//! end (Σ Aᵢ e^{2πCNδᵢ} = 0 at z = L, N = L/λₑ beam wavelengths). The three conditions'
//! determinant vanishes at the start of oscillation: CN = 0.314, b = 1.522 without space charge
//! (Johnson), CN rising with QC. Solved here by Newton's method on (CN, b), the cubic's roots by
//! the Durand–Kerner iteration.
//!
//! **Pierce's gain parameter** C³ = K I₀/(4V₀), K = |E|²/(2β²P) the coupling impedance of the
//! harmonic. A relativistic beam bunches under its longitudinal mass γ³m: C³ = K I₀ q/(2 γ³ m
//! v₀²) (B. Levush, T. M. Antonsen, A. Bromborsky, W. R. Lou, Y. Carmel, *Theory of relativistic
//! backward-wave oscillators with end reflections*, IEEE Trans. Plasma Sci. 20(3):263–280,
//! 1992), which is Pierce's at v₀ ≪ c. So the start current is I_st = (CN/N)³ · 2γ³mv₀²/(qK).
//!
//! **Space charge** 4QC = (ω_q/(ωC))², ω_q = R ω_p/γ^{3/2} the reduced plasma frequency
//! (ω_p² = n q²/(ε₀m)). For a thin ring beam of radius b and thickness t inside a conducting
//! tube of radius a, the ring's own field of a modulation e^{−jβₑz} decays as
//! I₀(κr), K₀(κr), κ = βₑ/γ, and against that of an unbounded beam reduces it by
//! R² = βₑ² t b G/γ², G = I₀(κb)[K₀(κb) − K₀(κa) I₀(κb)/I₀(κa)] (G. M. Branch, T. G. Mihran,
//! *Plasma frequency reduction factors in electron beams*, IRE Trans. Electron Devices ED-2:3–11,
//! 1955, here for a thin ring and relativistic κ).

use emcraft_model::units::{C0, EPS0};
use num_complex::Complex64 as Z;

use crate::bessel::{i0, k0};

/// Johnson's start condition: CN and the velocity parameter b.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Start {
    pub cn: f64,
    pub b: f64,
}

/// The roots of (δ² + 4QC)(δ + jb) − j by the Durand–Kerner iteration, bounded.
fn roots(b: f64, qc: f64) -> [Z; 3] {
    let jb = Z::new(0.0, b);
    let p = |z: Z| (z * z + 4.0 * qc) * (z + jb) - Z::new(0.0, 1.0);
    let scale = 1.0 + b.abs() + qc.abs().sqrt();
    let mut r: [Z; 3] = std::array::from_fn(|i| Z::from_polar(scale, 0.4 + 2.1 * i as f64));
    for _ in 0..500 {
        let mut moved = 0.0f64;
        for i in 0..3 {
            let denom = (0..3).filter(|j| *j != i).fold(Z::new(1.0, 0.0), |acc, j| acc * (r[i] - r[j]));
            if denom.norm() == 0.0 {
                continue;
            }
            let step = p(r[i]) / denom;
            r[i] -= step;
            moved = moved.max(step.norm() / scale);
        }
        if moved < 1e-15 {
            break;
        }
    }
    r
}

/// The boundary conditions' determinant at (CN, b) over the roots' differences, `echo` the collector end's backward wave per
/// unit of the gun end's (ρe^{jΦ}; 0 for matched ends).
fn determinant(cn: f64, b: f64, qc: f64, echo: Z) -> Z {
    let d = roots(b, qc);
    let jb = Z::new(0.0, b);
    let rows: [[Z; 3]; 3] = [d.map(|x| x + jb), d.map(|x| x * (x + jb)), d.map(|x| (2.0 * std::f64::consts::PI * cn * x).exp() - echo)];
    let m = |i: usize, j: usize| rows[1][i] * rows[2][j] - rows[1][j] * rows[2][i];
    // Over the roots' differences: two equal roots make two columns equal, a zero that is no start.
    let apart = (d[0] - d[1]) * (d[1] - d[2]) * (d[2] - d[0]);
    (rows[0][0] * m(1, 2) - rows[0][1] * m(0, 2) + rows[0][2] * m(0, 1)) / apart
}

/// Johnson's start of oscillation of a lossless BWO at space-charge parameter `qc` (0 to 1):
/// CN = 0.314, b = 1.522 without space charge. None if Newton's method does not settle.
///
/// ```
/// use emcraft_theory::bwo::johnson_start;
/// let s = johnson_start(0.0).unwrap();
/// assert!((s.cn - 0.314).abs() < 5e-4 && (s.b - 1.522).abs() < 5e-4);
/// // Space charge raises the start.
/// assert!(johnson_start(0.25).unwrap().cn > 0.34);
/// assert!(johnson_start(f64::NAN).is_none());
/// ```
pub fn johnson_start(qc: f64) -> Option<Start> {
    if !(qc.is_finite() && (0.0..=1.0).contains(&qc)) {
        return None;
    }
    let (mut cn, mut b) = (0.3, 1.5);
    let h = 1e-7;
    // From no space charge to qc in steps, each Newton's start the last step's root.
    let stages = (qc / 0.05).ceil().max(1.0) as usize;
    for stage in 1..=stages {
        let q = qc * stage as f64 / stages as f64;
        let mut settled = false;
        for _ in 0..100 {
            let f = determinant(cn, b, q, Z::new(0.0, 0.0));
            if f.norm() < 1e-13 {
                settled = true;
                break;
            }
            let (fa, fb) = ((determinant(cn + h, b, q, Z::new(0.0, 0.0)) - f) / h, (determinant(cn, b + h, q, Z::new(0.0, 0.0)) - f) / h);
            let det = fa.re * fb.im - fb.re * fa.im;
            if det == 0.0 || !det.is_finite() {
                return None;
            }
            cn -= (f.re * fb.im - f.im * fb.re) / det;
            b -= (fa.re * f.im - fa.im * f.re) / det;
        }
        if !settled {
            return None;
        }
    }
    Some(Start { cn, b })
}

/// A thin ring beam: its radius and thickness, and the radius of the conducting tube around it
/// (m), for its space charge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ring {
    pub radius: f64,
    pub thickness: f64,
    pub tube: f64,
}

/// A backward-wave oscillator: the coupling impedance on the beam (Ω), the length (m) and the
/// frequency (Hz); the beam at `gamma0`, of particles of `charge` (C) and `mass` (kg), its space
/// charge counted if it is a `ring`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Oscillator {
    pub impedance: f64,
    pub length: f64,
    pub frequency: f64,
    pub gamma0: f64,
    pub charge: f64,
    pub mass: f64,
    pub ring: Option<Ring>,
}

impl Oscillator {
    fn valid(&self) -> bool {
        let ring = self
            .ring
            .is_none_or(|r| [r.radius, r.thickness, r.tube].iter().all(|x| x.is_finite() && *x > 0.0) && r.radius + 0.5 * r.thickness < r.tube);
        [self.impedance, self.length, self.frequency, self.gamma0, self.charge, self.mass].iter().all(|x| x.is_finite())
            && self.impedance > 0.0
            && self.length > 0.0
            && self.frequency > 0.0
            && self.gamma0 > 1.0
            && self.charge != 0.0
            && self.mass > 0.0
            && ring
    }

    fn speed(&self) -> f64 {
        C0 * (1.0 - 1.0 / (self.gamma0 * self.gamma0)).sqrt()
    }

    /// 2γ³mv₀²/(qK): the current (A) at C = 1.
    fn scale(&self) -> f64 {
        let v0 = self.speed();
        2.0 * self.gamma0.powi(3) * self.mass * v0 * v0 / (self.charge.abs() * self.impedance)
    }

    /// Pierce's QC at `current` (A); 0 without a ring.
    ///
    /// ```
    /// use emcraft_theory::bwo::{Oscillator, Ring};
    /// let ring = Ring { radius: 8e-3, thickness: 1e-3, tube: 10e-3 };
    /// let o = Oscillator { impedance: 0.6, length: 0.24, frequency: 11.8e9, gamma0: 1.0 + 1e5 / 510_998.95, charge: -1.602_176_634e-19, mass: 9.109_383_701_5e-31, ring: Some(ring) };
    /// let qc = o.space_charge(5.0).unwrap();
    /// assert!(qc > 0.05 && qc < 0.3);
    /// // QC grows as the cube root of the current.
    /// assert!((o.space_charge(40.0).unwrap() / qc - 2.0).abs() < 1e-9);
    /// ```
    pub fn space_charge(&self, current: f64) -> Option<f64> {
        if !(self.valid() && current.is_finite() && current > 0.0) {
            return None;
        }
        let Some(ring) = self.ring else { return Some(0.0) };
        let v0 = self.speed();
        let w = 2.0 * std::f64::consts::PI * self.frequency;
        let beta = w / v0;
        let kappa = beta / self.gamma0;
        let (ib, ia) = (i0(kappa * ring.radius)?, i0(kappa * ring.tube)?);
        let g = ib * (k0(kappa * ring.radius)? - k0(kappa * ring.tube)? * ib / ia);
        let r2 = beta * beta * ring.thickness * ring.radius * g / (self.gamma0 * self.gamma0);
        let density = current / (self.charge.abs() * v0 * 2.0 * std::f64::consts::PI * ring.radius * ring.thickness);
        let wq2 = r2 * density * self.charge * self.charge / (EPS0 * self.mass) / self.gamma0.powi(3);
        let c = (current / self.scale()).cbrt();
        Some(wq2 / (w * c).powi(2) / 4.0)
    }

    /// The start current (A): (CN/N)³ times 2γ³mv₀²/(qK), CN Johnson's at the space charge of
    /// that current (a fixed point, bounded). None for hostile input.
    ///
    /// ```
    /// use emcraft_theory::bwo::{Oscillator, Ring};
    /// // 100 kV electrons (γ₀ = 1.1957, β₀ = 0.5482), 1 Ω, 12 GHz, 120 mm: N = 8.762, 24.18 A.
    /// let mut o = Oscillator { impedance: 1.0, length: 0.12, frequency: 12e9, gamma0: 1.0 + 1e5 / 510_998.95, charge: -1.602_176_634e-19, mass: 9.109_383_701_5e-31, ring: None };
    /// let i = o.start_current().unwrap();
    /// assert!((i / 24.18 - 1.0).abs() < 2e-3);
    /// // Twice as long: an eighth of it (Johnson's L⁻³).
    /// o.length = 0.24;
    /// assert!((i / o.start_current().unwrap() - 8.0).abs() < 1e-9);
    /// // A ring's space charge raises it.
    /// o.ring = Some(Ring { radius: 8e-3, thickness: 1e-3, tube: 10e-3 });
    /// assert!(o.start_current().unwrap() > i / 8.0);
    /// o.impedance = 0.0;
    /// assert!(o.start_current().is_none());
    /// ```
    pub fn start_current(&self) -> Option<f64> {
        if !self.valid() {
            return None;
        }
        let n = self.length * self.frequency / self.speed();
        let at = |cn: f64| (cn / n).powi(3) * self.scale();
        let mut current = at(johnson_start(0.0)?.cn);
        for _ in 0..100 {
            let qc = self.space_charge(current)?;
            let next = at(johnson_start(qc.min(1.0))?.cn);
            let settled = (next - current).abs() < 1e-12 * current;
            current = next;
            if settled {
                return Some(current);
            }
        }
        None
    }
}

/// The circuit at one frequency (Hz): the Bloch phase ψ per period of its forward wave (rad), the
/// coupling impedance of the backward harmonic on the beam (Ω) and the ends' round-trip
/// reflection ρ = Γ₀Γ_L of the Bloch waves, referred to the structure's two end planes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub frequency: f64,
    pub phase: f64,
    pub impedance: f64,
    pub reflection: Z,
}

/// A backward-wave oscillator of `periods` cells of `period` (m) whose ends reflect (Levush et
/// al. 1992): the backward wave reaching the gun end comes back as the circuit's forward wave,
/// Γ₀ of it, which the beam leaves alone (out of step), and at the collector end Γ_L of that
/// returns as the backward wave. With B and F the Bloch amplitudes at the end planes, F(0) =
/// Γ₀B(0) and B(L) = Γ_L F(0) e^{−jψN_p}; the backward wave's harmonic e^{−jβₑz + βₑCδz} at the
/// end planes is its Bloch amplitude, so the collector condition becomes
///
/// Σ Aᵢ [e^{2πCNδᵢ} − ρ e^{jΦ}] = 0, Φ = 2πN − ψN_p,
///
/// which is Johnson's at ρ = 0. b is the circuit's: βₑ(1 + Cb) = (2π − ψ)/d. The start is where
/// the determinant vanishes in (I, f), the circuit interpolated linearly between its samples.
#[derive(Clone, Debug, PartialEq)]
pub struct Reflecting {
    /// By rising frequency.
    pub circuit: Vec<Sample>,
    pub period: f64,
    pub periods: usize,
    pub gamma0: f64,
    pub charge: f64,
    pub mass: f64,
    pub ring: Option<Ring>,
}

/// The most cells a reflecting oscillator may have.
const MAX_PERIODS: usize = 100_000;

impl Reflecting {
    fn valid(&self) -> bool {
        let finite = |s: &Sample| [s.frequency, s.phase, s.impedance, s.reflection.re, s.reflection.im].iter().all(|x| x.is_finite());
        self.circuit.len() >= 2
            && self.circuit.iter().all(|s| finite(s) && s.frequency > 0.0 && s.impedance > 0.0 && s.phase > 0.0 && s.phase < std::f64::consts::PI)
            && self.circuit.windows(2).all(|w| w[1].frequency > w[0].frequency)
            && self.period.is_finite()
            && self.period > 0.0
            && (1..=MAX_PERIODS).contains(&self.periods)
    }

    /// The circuit at `f` (Hz), between its samples.
    fn at(&self, f: f64) -> Option<Sample> {
        let pair = self.circuit.windows(2).find(|w| f >= w[0].frequency && f <= w[1].frequency)?;
        let (a, b) = (pair[0], pair[1]);
        let t = (f - a.frequency) / (b.frequency - a.frequency);
        Some(Sample {
            frequency: f,
            phase: a.phase + t * (b.phase - a.phase),
            impedance: (a.impedance.ln() + t * (b.impedance.ln() - a.impedance.ln())).exp(),
            reflection: a.reflection + (b.reflection - a.reflection) * t,
        })
    }

    /// The matched oscillator at the circuit's sample `s`.
    fn matched(&self, s: &Sample) -> Oscillator {
        Oscillator {
            impedance: s.impedance,
            length: self.periods as f64 * self.period,
            frequency: s.frequency,
            gamma0: self.gamma0,
            charge: self.charge,
            mass: self.mass,
            ring: self.ring,
        }
    }

    /// Pierce's CN, b and QC and the collector's echo ρe^{jΦ} at `current` (A) and `f` (Hz).
    fn parameters(&self, current: f64, f: f64) -> Option<(f64, f64, f64, Z)> {
        let s = self.at(f)?;
        let o = self.matched(&s);
        let c = (current / o.scale()).cbrt();
        let qc = o.space_charge(current)?;
        let v0 = o.speed();
        let be = 2.0 * std::f64::consts::PI * f / v0;
        let b = ((2.0 * std::f64::consts::PI - s.phase) / (be * self.period) - 1.0) / c;
        let n = o.length * f / v0;
        let phi = 2.0 * std::f64::consts::PI * n - s.phase * self.periods as f64;
        Some((c * n, b, qc, s.reflection * Z::from_polar(1.0, phi)))
    }

    /// The boundary conditions' determinant at `current` (A) and `f` (Hz).
    fn mismatch(&self, current: f64, f: f64) -> Option<Z> {
        let (cn, b, qc, echo) = self.parameters(current, f)?;
        Some(determinant(cn, b, qc.min(1.0), echo))
    }

    /// The start of oscillation: the lowest current (A) at which the determinant vanishes within
    /// the circuit's samples, and its frequency (Hz); Newton's method on (ln I, f) from Johnson's
    /// matched start at each sample, bounded. None for hostile input or if none settles.
    ///
    /// ```
    /// use emcraft_theory::bwo::{Oscillator, Reflecting, Sample};
    /// use num_complex::Complex64 as Z;
    /// let (q, m, gamma0) = (-1.602_176_634e-19, 9.109_383_701_5e-31, 1.0 + 1e5 / 510_998.95);
    /// // A circuit in step with a 100 kV beam near ψ = 100°, 11.9 GHz, ends matched.
    /// let sample = |f: f64, deg: f64, k: f64| Sample { frequency: f, phase: deg.to_radians(), impedance: k, reflection: Z::new(0.0, 0.0) };
    /// let circuit = vec![sample(11.25e9, 80.0, 0.35), sample(11.91e9, 100.0, 0.59), sample(12.58e9, 120.0, 1.13)];
    /// let o = Reflecting { circuit: circuit.clone(), period: 0.01, periods: 24, gamma0, charge: q, mass: m, ring: None };
    /// let (i, f) = o.start().unwrap();
    /// // Johnson's at that frequency's impedance.
    /// let s = circuit[1];
    /// let k = (s.impedance.ln() + (f - s.frequency) / (circuit[0].frequency - s.frequency) * (circuit[0].impedance.ln() - s.impedance.ln())).exp();
    /// let johnson = Oscillator { impedance: k, length: 0.24, frequency: f, gamma0, charge: q, mass: m, ring: None };
    /// assert!((i / johnson.start_current().unwrap() - 1.0).abs() < 1e-6);
    /// assert!(f > 11.5e9 && f < 11.9e9);
    /// // Reflecting ends move it.
    /// let echo = Reflecting { circuit: circuit.iter().map(|s| Sample { reflection: Z::new(0.3, 0.0), ..*s }).collect(), ..o.clone() };
    /// assert!((echo.start().unwrap().0 / i - 1.0).abs() > 0.05);
    /// assert!(Reflecting { periods: 0, ..o }.start().is_none());
    /// ```
    pub fn start(&self) -> Option<(f64, f64)> {
        if !self.valid()
            || !(self.gamma0.is_finite()
                && self.gamma0 > 1.0
                && self.charge.is_finite()
                && self.charge != 0.0
                && self.mass.is_finite()
                && self.mass > 0.0)
        {
            return None;
        }
        let ghz = 1e9;
        // Newton's method, each step halved until it lowers |det| (at most 30 times).
        let solve = |guess: (f64, f64)| -> Option<(f64, f64)> {
            let (mut x, mut y) = (guess.0.ln(), guess.1 / ghz);
            let mut d = self.mismatch(x.exp(), y * ghz)?;
            for _ in 0..100 {
                let h = 1e-6;
                let dx = (self.mismatch((x + h).exp(), y * ghz)? - d) / h;
                let dy = (self.mismatch(x.exp(), (y + h) * ghz)? - d) / h;
                let det = dx.re * dy.im - dy.re * dx.im;
                if det == 0.0 || !det.is_finite() {
                    return None;
                }
                let (mut sx, mut sy) =
                    (((d.re * dy.im - d.im * dy.re) / det).clamp(-0.5, 0.5), ((dx.re * d.im - dx.im * d.re) / det).clamp(-0.05, 0.05));
                let mut next = None;
                for _ in 0..30 {
                    if let Some(e) = self.mismatch((x - sx).exp(), (y - sy) * ghz).filter(|e| e.norm() < d.norm()) {
                        next = Some(e);
                        break;
                    }
                    (sx, sy) = (0.5 * sx, 0.5 * sy);
                }
                d = next?;
                // Settled to 1e-9 of the current and 1 Hz (the numerical derivative's noise), at a
                // zero rather than a stalled step.
                if sx.abs() < 1e-9 && sy.abs() < 1e-9 {
                    return (d.norm() < 1e-8).then(|| ((x - sx).exp(), (y - sy) * ghz));
                }
                (x, y) = (x - sx, y - sy);
            }
            None
        };
        // Guesses every 1 % of the circuit's span: each frequency with Johnson's matched start.
        let (lo, hi) = (self.circuit[0].frequency, self.circuit[self.circuit.len() - 1].frequency);
        (0..=100)
            .map(|k| lo + (hi - lo) * k as f64 / 100.0)
            .filter_map(|f| {
                let s = self.at(f)?;
                solve((self.matched(&s).start_current()?, f))
            })
            // A start, not the determinant's vanishing as C → 0 (b without bound).
            .filter(|(i, f)| i.is_finite() && self.parameters(*i, *f).is_some_and(|(cn, b, ..)| cn > 0.05 && b.abs() < 20.0))
            .min_by(|a, b| a.0.total_cmp(&b.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_start_condition_is_johnsons() {
        // Johnson (1955), Fig. 6 at QC = 0, d = 0: CN = 0.314, b = 1.522; the waves then cancel at
        // the gun to the iteration's tolerance.
        let s = johnson_start(0.0).unwrap();
        approx::assert_abs_diff_eq!(s.cn, 0.314, epsilon = 2e-4);
        approx::assert_abs_diff_eq!(s.b, 1.522, epsilon = 3e-4);
        assert!(determinant(s.cn, s.b, 0.0, Z::new(0.0, 0.0)).norm() < 1e-12);
        // Johnson's Fig. 6: CN rises with QC, slowly at first.
        let cns: Vec<f64> = [0.1, 0.2, 0.3, 0.4].iter().map(|q| johnson_start(*q).unwrap().cn).collect();
        assert!(cns.windows(2).all(|w| w[1] > w[0]) && cns[0] > 0.314 && cns[0] < 0.33);
    }

    #[test]
    fn a_slow_beams_start_current_is_pierces() {
        // At v₀ ≪ c the relativistic C³ is Pierce's K I/(4V₀) (Pierce 1950, ch. II), V₀ = m v₀²/2q.
        let (q, m) = (-1.602_176_634e-19f64, 9.109_383_701_5e-31);
        let v0 = 1e-3 * C0;
        let gamma0 = 1.0 / (1.0 - 1e-6f64).sqrt();
        let o = Oscillator { impedance: 5.0, length: 0.1, frequency: 3e9, gamma0, charge: q, mass: m, ring: None };
        let v = 0.5 * m * v0 * v0 / q.abs();
        let pierce = 4.0 * v * (0.314_055 / (0.1 * 3e9 / v0)).powi(3) / 5.0;
        approx::assert_abs_diff_eq!(o.start_current().unwrap() / pierce, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn a_thin_rings_space_charge_is_reduced_as_branch_and_mihran() {
        // A ring far from the wall and thin against 1/κ: G → I₀K₀(κb), and at κb ≫ 1 that tends to
        // 1/(2κb) (Abramowitz & Stegun 9.7.1–9.7.2), so R² → βₑ² t/(2κγ²) = βₑ t/(2γ).
        let ring = Ring { radius: 0.05, thickness: 1e-4, tube: 0.3 };
        let o = Oscillator {
            impedance: 1.0,
            length: 0.1,
            frequency: 30e9,
            gamma0: 1.1,
            charge: -1.602_176_634e-19,
            mass: 9.109_383_701_5e-31,
            ring: Some(ring),
        };
        let (v0, w) = (o.speed(), 2.0 * std::f64::consts::PI * o.frequency);
        let current = 3.0;
        let density = current / (o.charge.abs() * v0 * 2.0 * std::f64::consts::PI * ring.radius * ring.thickness);
        let wp2 = density * o.charge * o.charge / (EPS0 * o.mass);
        let r2 = (w / v0) * ring.thickness / (2.0 * o.gamma0);
        let c = (current / o.scale()).cbrt();
        let want = r2 * wp2 / o.gamma0.powi(3) / (w * c).powi(2) / 4.0;
        // κb = 69: the asymptote's next term is 1/(8(κb)²) relative, the wall's e^{−2κ(a−b)}.
        approx::assert_abs_diff_eq!(o.space_charge(current).unwrap() / want, 1.0, epsilon = 1e-3);
    }
}
