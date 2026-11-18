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

use emcraft_model::particles::Species;
use emcraft_model::units::{C0, EPS0};
use num_complex::Complex64 as Z;

use crate::bessel::{i0, k0};

/// Johnson's start condition: CN and the velocity parameter b.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Start {
    pub cn: f64,
    pub b: f64,
}

/// The cubic's roots: their seeds' spacing (rad, about 2π/3) and the most sweeps.
const CUBIC_SPREAD: f64 = 2.1;
const CUBIC_SWEEPS: usize = 500;

/// The roots of (δ² + 4QC)(δ + jb) − j ([`crate::poly::roots`]).
fn roots(b: f64, qc: f64) -> [Z; 3] {
    let jb = Z::new(0.0, b);
    let p = |z: Z| (z * z + 4.0 * qc) * (z + jb) - Z::new(0.0, 1.0);
    let scale = 1.0 + b.abs() + qc.abs().sqrt();
    crate::poly::roots(p, crate::poly::Seeds { scale, spread: CUBIC_SPREAD }, CUBIC_SWEEPS)
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

/// Newton's step for f(x, y) = 0, f complex (its real and imaginary parts two equations), from
/// `(x, y)` where f is `here`: the Jacobian by forward differences of `h`, the 2 × 2 system by
/// Cramer's rule. None where f has no value or the Jacobian is singular.
fn newton_step(f: impl Fn(f64, f64) -> Option<Z>, (x, y): (f64, f64), here: Z, h: f64) -> Option<[f64; 2]> {
    let dx = (f(x + h, y)? - here) / h;
    let dy = (f(x, y + h)? - here) / h;
    let det = dx.re * dy.im - dy.re * dx.im;
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    Some([(here.re * dy.im - here.im * dy.re) / det, (dx.re * here.im - dx.im * here.re) / det])
}

/// The most Newton steps a root takes.
const MAX_NEWTON: usize = 100;

/// Johnson's start: Newton's first guess (CN, b) near his matched values; the space charge
/// raised from none in steps of this QC, each step's root the next one's guess; the forward
/// differences' step (well above the determinant's rounding, below its curvature) and the
/// determinant's size at which a root is settled.
const JOHNSON_GUESS: (f64, f64) = (0.3, 1.5);
const SPACE_CHARGE_STEP: f64 = 0.05;
const JOHNSON_H: f64 = 1e-7;
const JOHNSON_SETTLED: f64 = 1e-13;

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
    let (mut cn, mut b) = JOHNSON_GUESS;
    // From no space charge to qc in steps, each Newton's start the last step's root.
    let stages = (qc / SPACE_CHARGE_STEP).ceil().max(1.0) as usize;
    for stage in 1..=stages {
        let q = qc * stage as f64 / stages as f64;
        let det = |cn: f64, b: f64| Some(determinant(cn, b, q, Z::new(0.0, 0.0)));
        let mut settled = false;
        for _ in 0..MAX_NEWTON {
            let f = determinant(cn, b, q, Z::new(0.0, 0.0));
            if f.norm() < JOHNSON_SETTLED {
                settled = true;
                break;
            }
            let [dcn, db] = newton_step(det, (cn, b), f, JOHNSON_H)?;
            (cn, b) = (cn - dcn, b - db);
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

/// The beam: injected at `gamma0`, of particles of `species`, its space charge counted if it is
/// a `ring`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Beam {
    pub gamma0: f64,
    pub species: Species,
    pub ring: Option<Ring>,
}

impl Beam {
    /// Its numbers finite, γ₀ > 1, a charge and a mass > 0, a ring inside its tube.
    fn valid(&self) -> bool {
        let ring = self
            .ring
            .is_none_or(|r| [r.radius, r.thickness, r.tube].iter().all(|x| x.is_finite() && *x > 0.0) && r.radius + 0.5 * r.thickness < r.tube);
        let Species { charge, mass } = self.species;
        [self.gamma0, charge, mass].iter().all(|x| x.is_finite()) && self.gamma0 > 1.0 && charge != 0.0 && mass > 0.0 && ring
    }

    /// Its speed v₀ (m/s).
    fn speed(&self) -> f64 {
        C0 * (1.0 - 1.0 / (self.gamma0 * self.gamma0)).sqrt()
    }
}

/// A backward-wave oscillator: the coupling impedance on the beam (Ω), the length (m) and the
/// frequency (Hz), and its beam.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Oscillator {
    pub impedance: f64,
    pub length: f64,
    pub frequency: f64,
    pub beam: Beam,
}

impl Oscillator {
    fn valid(&self) -> bool {
        [self.impedance, self.length, self.frequency].iter().all(|x| x.is_finite())
            && self.impedance > 0.0
            && self.length > 0.0
            && self.frequency > 0.0
            && self.beam.valid()
    }

    fn speed(&self) -> f64 {
        self.beam.speed()
    }

    /// 2γ³mv₀²/(qK): the current (A) at C = 1.
    fn scale(&self) -> f64 {
        let v0 = self.speed();
        let Beam { gamma0, species, .. } = self.beam;
        2.0 * gamma0.powi(3) * species.mass * v0 * v0 / (species.charge.abs() * self.impedance)
    }

    /// Pierce's QC at `current` (A); 0 without a ring.
    ///
    /// ```
    /// use emcraft_model::particles::ELECTRON;
    /// use emcraft_theory::bwo::{Beam, Oscillator, Ring};
    /// let ring = Ring { radius: 8e-3, thickness: 1e-3, tube: 10e-3 };
    /// let beam = Beam { gamma0: 1.0 + 1e5 / 510_998.95, species: ELECTRON, ring: Some(ring) };
    /// let o = Oscillator { impedance: 0.6, length: 0.24, frequency: 11.8e9, beam };
    /// let qc = o.space_charge(5.0).unwrap();
    /// assert!(qc > 0.05 && qc < 0.3);
    /// // QC grows as the cube root of the current.
    /// assert!((o.space_charge(40.0).unwrap() / qc - 2.0).abs() < 1e-9);
    /// ```
    pub fn space_charge(&self, current: f64) -> Option<f64> {
        if !(self.valid() && current.is_finite() && current > 0.0) {
            return None;
        }
        let Beam { gamma0, species: Species { charge, mass }, ring } = self.beam;
        let Some(ring) = ring else { return Some(0.0) };
        let v0 = self.speed();
        let w = 2.0 * std::f64::consts::PI * self.frequency;
        let beta = w / v0;
        let kappa = beta / gamma0;
        let (ib, ia) = (i0(kappa * ring.radius)?, i0(kappa * ring.tube)?);
        let g = ib * (k0(kappa * ring.radius)? - k0(kappa * ring.tube)? * ib / ia);
        let r2 = beta * beta * ring.thickness * ring.radius * g / (gamma0 * gamma0);
        let density = current / (charge.abs() * v0 * 2.0 * std::f64::consts::PI * ring.radius * ring.thickness);
        let wq2 = r2 * density * charge * charge / (EPS0 * mass) / gamma0.powi(3);
        let c = (current / self.scale()).cbrt();
        Some(wq2 / (w * c).powi(2) / 4.0)
    }

    /// The start current (A): (CN/N)³ times 2γ³mv₀²/(qK), CN Johnson's at the space charge of
    /// that current (a fixed point, bounded). None for hostile input.
    ///
    /// ```
    /// use emcraft_model::particles::ELECTRON;
    /// use emcraft_theory::bwo::{Beam, Oscillator, Ring};
    /// // 100 kV electrons (γ₀ = 1.1957, β₀ = 0.5482), 1 Ω, 12 GHz, 120 mm: N = 8.762, 24.18 A.
    /// let beam = Beam { gamma0: 1.0 + 1e5 / 510_998.95, species: ELECTRON, ring: None };
    /// let mut o = Oscillator { impedance: 1.0, length: 0.12, frequency: 12e9, beam };
    /// let i = o.start_current().unwrap();
    /// assert!((i / 24.18 - 1.0).abs() < 2e-3);
    /// // Twice as long: an eighth of it (Johnson's L⁻³).
    /// o.length = 0.24;
    /// assert!((i / o.start_current().unwrap() - 8.0).abs() < 1e-9);
    /// // A ring's space charge raises it.
    /// o.beam.ring = Some(Ring { radius: 8e-3, thickness: 1e-3, tube: 10e-3 });
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
    pub beam: Beam,
}

/// Pierce's CN, b and QC at a current and frequency, and the collector's echo ρe^{jΦ}.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Pierce {
    cn: f64,
    b: f64,
    qc: f64,
    echo: Z,
}

/// Where an oscillator starts: its current (A) and frequency (Hz).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StartPoint {
    pub current: f64,
    pub frequency: f64,
}

/// The reflecting oscillator's Newton: the forward differences' step in (ln I, f/GHz); each step
/// clamped to these (a factor e^½ of the current, 50 MHz) and halved at most this often until
/// it lowers |det|; settled once a step is below this and |det| below that (the numerical
/// derivative's noise).
const REFLECTING_H: f64 = 1e-6;
const MAX_LN_CURRENT_STEP: f64 = 0.5;
const MAX_GHZ_STEP: f64 = 0.05;
const HALVINGS: usize = 30;
const SETTLED_STEP: f64 = 1e-9;
const SETTLED_DET: f64 = 1e-8;

/// Newton's guesses along the circuit (this many steps across its span), and what a start is,
/// not the determinant vanishing as C → 0 (b without bound): CN and |b| within these.
const GUESSES: usize = 100;
const MIN_START_CN: f64 = 0.05;
const MAX_START_B: f64 = 20.0;

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
            && self.beam.valid()
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
        Oscillator { impedance: s.impedance, length: self.periods as f64 * self.period, frequency: s.frequency, beam: self.beam }
    }

    /// Pierce's parameters at `current` (A) and `f` (Hz).
    fn parameters(&self, current: f64, f: f64) -> Option<Pierce> {
        let s = self.at(f)?;
        let o = self.matched(&s);
        let c = (current / o.scale()).cbrt();
        let qc = o.space_charge(current)?;
        let v0 = o.speed();
        let be = 2.0 * std::f64::consts::PI * f / v0;
        let b = ((2.0 * std::f64::consts::PI - s.phase) / (be * self.period) - 1.0) / c;
        let n = o.length * f / v0;
        let phi = 2.0 * std::f64::consts::PI * n - s.phase * self.periods as f64;
        Some(Pierce { cn: c * n, b, qc, echo: s.reflection * Z::from_polar(1.0, phi) })
    }

    /// The boundary conditions' determinant at `current` (A) and `f` (Hz).
    fn mismatch(&self, current: f64, f: f64) -> Option<Z> {
        let Pierce { cn, b, qc, echo } = self.parameters(current, f)?;
        Some(determinant(cn, b, qc.min(1.0), echo))
    }

    /// The start of oscillation: the lowest current (A) at which the determinant vanishes within
    /// the circuit's samples, and its frequency (Hz); Newton's method on (ln I, f) from Johnson's
    /// matched start at each sample, bounded. None for hostile input or if none settles.
    ///
    /// ```
    /// use emcraft_model::particles::ELECTRON;
    /// use emcraft_theory::bwo::{Beam, Oscillator, Reflecting, Sample};
    /// use num_complex::Complex64 as Z;
    /// let beam = Beam { gamma0: 1.0 + 1e5 / 510_998.95, species: ELECTRON, ring: None };
    /// // A circuit in step with a 100 kV beam near ψ = 100°, 11.9 GHz, ends matched.
    /// let sample = |f: f64, deg: f64, k: f64| Sample { frequency: f, phase: deg.to_radians(), impedance: k, reflection: Z::new(0.0, 0.0) };
    /// let circuit = vec![sample(11.25e9, 80.0, 0.35), sample(11.91e9, 100.0, 0.59), sample(12.58e9, 120.0, 1.13)];
    /// let o = Reflecting { circuit: circuit.clone(), period: 0.01, periods: 24, beam };
    /// let start = o.start().unwrap();
    /// let (i, f) = (start.current, start.frequency);
    /// // Johnson's at that frequency's impedance.
    /// let s = circuit[1];
    /// let k = (s.impedance.ln() + (f - s.frequency) / (circuit[0].frequency - s.frequency) * (circuit[0].impedance.ln() - s.impedance.ln())).exp();
    /// let johnson = Oscillator { impedance: k, length: 0.24, frequency: f, beam };
    /// assert!((i / johnson.start_current().unwrap() - 1.0).abs() < 1e-6);
    /// assert!(f > 11.5e9 && f < 11.9e9);
    /// // Reflecting ends move it.
    /// let echo = Reflecting { circuit: circuit.iter().map(|s| Sample { reflection: Z::new(0.3, 0.0), ..*s }).collect(), ..o.clone() };
    /// assert!((echo.start().unwrap().current / i - 1.0).abs() > 0.05);
    /// assert!(Reflecting { periods: 0, ..o }.start().is_none());
    /// ```
    pub fn start(&self) -> Option<StartPoint> {
        if !self.valid() {
            return None;
        }
        let ghz = 1e9;
        let det = |x: f64, y: f64| self.mismatch(x.exp(), y * ghz);
        // Newton's method on (ln I, f/GHz), each step halved until it lowers |det|.
        let solve = |guess: (f64, f64)| -> Option<StartPoint> {
            let (mut x, mut y) = (guess.0.ln(), guess.1 / ghz);
            let mut d = det(x, y)?;
            for _ in 0..MAX_NEWTON {
                let [sx, sy] = newton_step(det, (x, y), d, REFLECTING_H)?;
                let (mut sx, mut sy) = (sx.clamp(-MAX_LN_CURRENT_STEP, MAX_LN_CURRENT_STEP), sy.clamp(-MAX_GHZ_STEP, MAX_GHZ_STEP));
                let mut next = None;
                for _ in 0..HALVINGS {
                    if let Some(e) = det(x - sx, y - sy).filter(|e| e.norm() < d.norm()) {
                        next = Some(e);
                        break;
                    }
                    (sx, sy) = (0.5 * sx, 0.5 * sy);
                }
                d = next?;
                // At a zero rather than a stalled step.
                if sx.abs() < SETTLED_STEP && sy.abs() < SETTLED_STEP {
                    return (d.norm() < SETTLED_DET).then(|| StartPoint { current: (x - sx).exp(), frequency: (y - sy) * ghz });
                }
                (x, y) = (x - sx, y - sy);
            }
            None
        };
        // Guesses across the circuit's span: each frequency with Johnson's matched start.
        let (lo, hi) = (self.circuit[0].frequency, self.circuit[self.circuit.len() - 1].frequency);
        (0..=GUESSES)
            .map(|k| lo + (hi - lo) * k as f64 / GUESSES as f64)
            .filter_map(|f| {
                let s = self.at(f)?;
                solve((self.matched(&s).start_current()?, f))
            })
            .filter(|p| {
                p.current.is_finite() && self.parameters(p.current, p.frequency).is_some_and(|q| q.cn > MIN_START_CN && q.b.abs() < MAX_START_B)
            })
            .min_by(|a, b| a.current.total_cmp(&b.current))
    }
}

#[cfg(test)]
mod tests {
    use emcraft_model::particles::ELECTRON;

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
        let Species { charge: q, mass: m } = ELECTRON;
        let v0 = 1e-3 * C0;
        let gamma0 = 1.0 / (1.0 - 1e-6f64).sqrt();
        let o = Oscillator { impedance: 5.0, length: 0.1, frequency: 3e9, beam: Beam { gamma0, species: ELECTRON, ring: None } };
        let v = 0.5 * m * v0 * v0 / q.abs();
        let pierce = 4.0 * v * (0.314_055 / (0.1 * 3e9 / v0)).powi(3) / 5.0;
        approx::assert_abs_diff_eq!(o.start_current().unwrap() / pierce, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn a_thin_rings_space_charge_is_reduced_as_branch_and_mihran() {
        // A ring far from the wall and thin against 1/κ: G → I₀K₀(κb), and at κb ≫ 1 that tends to
        // 1/(2κb) (Abramowitz & Stegun 9.7.1–9.7.2), so R² → βₑ² t/(2κγ²) = βₑ t/(2γ).
        let ring = Ring { radius: 0.05, thickness: 1e-4, tube: 0.3 };
        let gamma0 = 1.1;
        let o = Oscillator { impedance: 1.0, length: 0.1, frequency: 30e9, beam: Beam { gamma0, species: ELECTRON, ring: Some(ring) } };
        let Species { charge: q, mass: m } = ELECTRON;
        let (v0, w) = (o.speed(), 2.0 * std::f64::consts::PI * o.frequency);
        let current = 3.0;
        let density = current / (q.abs() * v0 * 2.0 * std::f64::consts::PI * ring.radius * ring.thickness);
        let wp2 = density * q * q / (EPS0 * m);
        let r2 = (w / v0) * ring.thickness / (2.0 * gamma0);
        let c = (current / o.scale()).cbrt();
        let want = r2 * wp2 / gamma0.powi(3) / (w * c).powi(2) / 4.0;
        // κb = 69: the asymptote's next term is 1/(8(κb)²) relative, the wall's e^{−2κ(a−b)}.
        approx::assert_abs_diff_eq!(o.space_charge(current).unwrap() / want, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn hostile_oscillators_have_no_start() {
        // One number wrong at a time, the rest a working oscillator's.
        let ring = Ring { radius: 8e-3, thickness: 1e-3, tube: 10e-3 };
        let beam = Beam { gamma0: 1.2, species: ELECTRON, ring: Some(ring) };
        let o = Oscillator { impedance: 0.6, length: 0.24, frequency: 11.8e9, beam };
        assert!(o.start_current().is_some() && o.space_charge(5.0).is_some());
        let with = |ring: Ring| Oscillator { beam: Beam { ring: Some(ring), ..beam }, ..o };
        let species = |charge: f64, mass: f64| Oscillator { beam: Beam { species: Species { charge, mass }, ..beam }, ..o };
        let bad = [
            Oscillator { impedance: 0.0, ..o },
            Oscillator { impedance: f64::INFINITY, ..o },
            Oscillator { length: 0.0, ..o },
            Oscillator { length: f64::NAN, ..o },
            Oscillator { frequency: 0.0, ..o },
            Oscillator { frequency: f64::INFINITY, ..o },
            Oscillator { beam: Beam { gamma0: 1.0, ..beam }, ..o },
            Oscillator { beam: Beam { gamma0: f64::INFINITY, ..beam }, ..o },
            species(0.0, ELECTRON.mass),
            species(f64::NAN, ELECTRON.mass),
            species(ELECTRON.charge, 0.0),
            species(ELECTRON.charge, f64::INFINITY),
            with(Ring { radius: 0.0, ..ring }),
            with(Ring { thickness: 0.0, ..ring }),
            with(Ring { tube: f64::INFINITY, ..ring }),
            // The ring's outer edge on the tube: 9.5 mm + 0.5 mm.
            with(Ring { radius: 9.5e-3, ..ring }),
            with(Ring { radius: 12e-3, ..ring }),
        ];
        for b in bad {
            assert!(b.start_current().is_none() && b.space_charge(5.0).is_none(), "{b:?}");
        }
        assert!(o.space_charge(0.0).is_none() && o.space_charge(f64::NAN).is_none());

        let sample = |f: f64, deg: f64, k: f64| Sample { frequency: f, phase: deg.to_radians(), impedance: k, reflection: Z::new(0.0, 0.0) };
        let circuit = vec![sample(11.25e9, 80.0, 0.35), sample(11.91e9, 100.0, 0.59), sample(12.58e9, 120.0, 1.13)];
        let beam = Beam { gamma0: 1.0 + 1e5 / 510_998.95, species: ELECTRON, ring: None };
        let r = Reflecting { circuit: circuit.clone(), period: 0.01, periods: 24, beam };
        assert!(r.start().is_some());
        let second = |s: Sample| Reflecting { circuit: vec![circuit[0], s, circuit[2]], ..r.clone() };
        let bad = [
            Reflecting { circuit: vec![circuit[1]], ..r.clone() },
            Reflecting { circuit: vec![circuit[1], circuit[0], circuit[2]], ..r.clone() },
            Reflecting { circuit: vec![circuit[0], circuit[0], circuit[2]], ..r.clone() },
            second(Sample { frequency: f64::NAN, ..circuit[1] }),
            second(Sample { impedance: 0.0, ..circuit[1] }),
            second(Sample { phase: 0.0, ..circuit[1] }),
            second(Sample { phase: std::f64::consts::PI, ..circuit[1] }),
            second(Sample { reflection: Z::new(0.0, f64::INFINITY), ..circuit[1] }),
            Reflecting { circuit: vec![Sample { frequency: 0.0, ..circuit[0] }, circuit[1], circuit[2]], ..r.clone() },
            Reflecting { period: 0.0, ..r.clone() },
            Reflecting { period: f64::INFINITY, ..r.clone() },
            Reflecting { periods: 0, ..r.clone() },
            Reflecting { periods: MAX_PERIODS + 1, ..r.clone() },
            Reflecting { beam: Beam { gamma0: 1.0, ..beam }, ..r.clone() },
        ];
        for b in bad {
            assert!(b.start().is_none(), "{b:?}");
        }
    }
}
