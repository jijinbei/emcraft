//! Backward-wave oscillators on the r–z grid against their theories (docs/adr/0012-particles.md,
//! stages D and E): a slow-wave structure's cold band, Pierce impedance and ends' reflections
//! measured on the grid itself; Johnson's start of oscillation with Pierce's space charge and
//! Levush's echo from them (`emcraft_theory::bwo`); a rippled tube's band against its Floquet
//! modes (`emcraft_theory::ripple`).

use std::f64::consts::PI;

use emcraft_model::units::{EPS0, MU0};
use emcraft_particles::{ELECTRON, Pusher};
use emcraft_theory::bessel::j;
use emcraft_theory::bwo::{Reflecting, Ring, Sample};
use num_complex::Complex64 as Z;

use super::*;
use crate::pic::{Pulse, Wall};

/// The first zero of J₀.
const X01: f64 = 2.404_825_557_695_773;

/// A slow-wave structure: its period (m) and its wall's radius at z (m from its start,
/// periodic), its narrowest radius, the drift tubes' radii before and after it, and the grid's.
#[derive(Clone, Copy)]
struct Structure {
    period: f64,
    wall: fn(f64) -> f64,
    narrowest: f64,
    drift: [f64; 2],
    outer: f64,
}

/// A tube 10 mm inside its teeth, grooves 5 mm wide to 13 mm, period 10 mm (grooves at the
/// period's first half): the non-relativistic BWO's.
const CORRUGATED: Structure = Structure { period: 0.010, wall: corrugated, narrowest: 0.010, drift: [0.013; 2], outer: 0.013 };

fn corrugated(z: f64) -> f64 {
    if ((z / 0.010).rem_euclid(1.0) - 0.25).abs() < 0.25 { 0.013 } else { 0.010 }
}

/// A tube rippled sinusoidally, 15 ± 1 mm with a period of 16.7 mm (Swegle et al.'s kind of
/// structure, within Rayleigh's limit), the drift tubes at its widest.
const RIPPLED: Structure = Structure { period: 0.0167, wall: rippled, narrowest: 0.014, drift: [0.016; 2], outer: 0.017 };

fn rippled(z: f64) -> f64 {
    0.015 + 0.001 * (2.0 * PI * z / 0.0167).cos()
}

/// The Ku-band relativistic BWO of D.-Y. Na, Y. A. Omelchenko, H. Moon, B.-H. V. Borges,
/// F. L. Teixeira (*Axisymmetric charge-conservative electromagnetic particle simulation
/// algorithm on unstructured grids: application to microwave vacuum electronic devices*,
/// J. Comput. Phys. 346:295–317, 2017; arXiv:1701.03784, §5.1): a ripple
/// R(z) = (A + B)/2 + (A − B)/2 cos(2πz/C), A = 17.1 mm, B = 12.9 mm, C = 7.5 mm, 14.5 periods,
/// begun here at a trough and ended at a crest, the drift tubes at those radii.
const KU: Structure = Structure { period: 0.0075, wall: ku, narrowest: 0.0129, drift: [0.0129, 0.0171], outer: 0.018 };

fn ku(z: f64) -> f64 {
    0.015 - 0.0021 * (2.0 * PI * z / 0.0075).cos()
}

/// Cells `h` across and `h / fine` along z (fine along z for relativistic beams: the numerical
/// Cherenkov instability, emcraft_fdtd::axi).
#[derive(Clone, Copy)]
struct Cells {
    h: f64,
    fine: usize,
}

impl Cells {
    fn grid(&self, outer: f64, length: f64, wall: Wall) -> AxiGrid {
        let dz = self.h / self.fine as f64;
        let nz = (length / dz).round() as usize;
        AxiGrid::new((outer / self.h).round() as usize, self.h, Axis { cells: nz, h: length / nz as f64, wall }).unwrap()
    }
}

/// `periods` of `s` with periodic ends on `cells`, E_z = J₀(x₀₁r/a) cos(kz + `shift`) in it (a
/// the narrowest radius, k = 2πm over its length).
fn ringing(s: Structure, cells: Cells, periods: usize, m: usize, shift: f64) -> AxiPic {
    let lz = periods as f64 * s.period;
    let g = cells.grid(s.outer, lz, Wall::Periodic);
    let mut pic = AxiPic::new(g, g.time_step(0.95), Pusher::Boris).unwrap();
    let k = 2.0 * PI * m as f64 / lz;
    let a = s.narrowest;
    pic.set(Component::Ez, |r, z| if r < a { j(0, X01 * r / a) * (k * z + shift).cos() } else { 0.0 });
    pic.set_metal(|r, z| r > (s.wall)(z));
    pic
}

/// The strongest line within `band` (Hz) of a `trace` of (t, value): to 10 MHz, then to 0.1 MHz.
fn strongest(trace: &[(f64, f64)], band: (f64, f64)) -> f64 {
    let line = |f: f64| trace.iter().map(|(t, e)| Z::from_polar(*e, -2.0 * PI * f * t)).sum::<Z>().norm();
    let best = |lo: f64, hi: f64, n: usize| {
        (0..=n).map(|i| lo + (hi - lo) * i as f64 / n as f64).map(|f| (line(f), f)).fold((0.0, 0.0), |x, y| if y.0 > x.0 { y } else { x }).1
    };
    let coarse = best(band.0, band.1, ((band.1 - band.0) / 10e6).round() as usize);
    best(coarse - 10e6, coarse + 10e6, 200)
}

/// The frequency (Hz) of a TM0 Bloch mode of `s` (as [`ringing`]) let ring 40 ns: the strongest
/// line within `band` at a point off the axis, the trace's mean taken out.
fn bloch_line(s: Structure, cells: Cells, periods: usize, m: usize, shift: f64, band: (f64, f64)) -> f64 {
    let mut pic = ringing(s, cells, periods, m, shift);
    let mut trace = Vec::new();
    for _ in 0..(40e-9 / pic.dt()) as usize {
        pic.step().unwrap();
        trace.push((pic.time(), pic.field_at(0.6 * s.narrowest, 0.3 * s.period).0[2]));
    }
    // The start's divergence leaves a static field behind (no charge holds it): its mean taken
    // out, lest it leak into the band's lowest lines.
    let mean = trace.iter().map(|x| x.1).sum::<f64>() / trace.len().max(1) as f64;
    let trace: Vec<(f64, f64)> = trace.into_iter().map(|(t, e)| (t, e - mean)).collect();
    strongest(&trace, band)
}

/// A TM01 mode of a structure: its Bloch phase per period (rad), frequency (Hz), Pierce
/// impedance of its backward harmonic k = (2π − ψ)/d at each asked radius (Ω), group velocity.
struct ColdMode {
    psi: f64,
    f: f64,
    impedance: Vec<f64>,
    vg: f64,
}

/// `s`'s cold TM01 band on a periodic stretch of `periods`, at the Bloch phases 2πm/periods for
/// `modes`: its line within `band` ([`bloch_line`]), then every component's phasor there
/// accumulated over 40 ns; the stored energy W = ¼∫(ε₀|Ê|² + |B̂|²/μ₀) dV, half of it each of
/// the two waves ±ψ; the harmonic's amplitude E the spatial transform of Ê_z along z; v_g = dω/dk
/// from the neighbouring phases; then K = E²/(2k²P), P = v_g (W/2)/L (Pierce 1950, ch. II).
fn cold_band(s: Structure, cells: Cells, periods: usize, modes: std::ops::RangeInclusive<usize>, radii: &[f64], band: (f64, f64)) -> Vec<ColdMode> {
    let lz = periods as f64 * s.period;
    let zero = Z::new(0.0, 0.0);
    // Each phase on a thread of its own.
    let mode = |m: usize| -> (f64, f64, Vec<f64>, f64) {
        let f = bloch_line(s, cells, periods, m, 0.0, band);
        let mut pic = ringing(s, cells, periods, m, 0.0);
        let mut ph: [Vec<Z>; 6] = std::array::from_fn(|w| vec![zero; pic.lats[w].len()]);
        let steps = (40e-9 / pic.dt()) as usize;
        for _ in 0..steps {
            pic.step().unwrap();
            let rot = Z::from_polar(2.0 / steps as f64, -2.0 * PI * f * pic.time());
            let fields = [&pic.f.er, &pic.f.ez, &pic.f.et, &pic.f.br, &pic.f.bz, &pic.f.bt];
            for (p, v) in ph.iter_mut().zip(fields) {
                for (a, b) in p.iter_mut().zip(v.iter()) {
                    *a += rot * b;
                }
            }
        }
        let g = pic.grid;
        let mut w = 0.0;
        for (which, p) in ph.iter().enumerate() {
            let lat = pic.lats[which];
            let coefficient = if which < 3 { 0.25 * EPS0 } else { 0.25 / MU0 };
            for kk in 0..lat.nz {
                for i in 0..lat.nr {
                    let r = (i as f64 + if lat.r_node { 0.0 } else { 0.5 }) * g.dr;
                    let ring = if lat.r_node && i == 0 { PI * 0.25 * g.dr * g.dr } else { 2.0 * PI * r * g.dr };
                    w += coefficient * p[lat.at(i, kk)].norm_sqr() * ring * g.z.h;
                }
            }
        }
        let k = 2.0 * PI * m as f64 / lz;
        let kb = 2.0 * PI / s.period - k;
        let lez = pic.lats[EZ];
        let amplitudes = radii
            .iter()
            .map(|rb| {
                let i = (rb / g.dr).round() as usize;
                (0..lez.nz).map(|kk| ph[EZ][lez.at(i, kk)] * Z::from_polar(g.z.h / lz, -kb * (kk as f64 + 0.5) * g.z.h)).sum::<Z>().norm()
            })
            .collect();
        (k * s.period, f, amplitudes, w)
    };
    let out: Vec<(f64, f64, Vec<f64>, f64)> = std::thread::scope(|sc| {
        let jobs: Vec<_> = modes.map(|m| sc.spawn(move || mode(m))).collect();
        jobs.into_iter().map(|j| j.join().unwrap()).collect()
    });
    let n = out.len();
    (0..n)
        .map(|m| {
            let (psi, f, ref a, w) = out[m];
            let (lo, hi) = (m.saturating_sub(1), (m + 1).min(n - 1));
            let vg = 2.0 * PI * (out[hi].1 - out[lo].1) / ((out[hi].0 - out[lo].0) / s.period);
            let kb = (2.0 * PI - psi) / s.period;
            let p = vg.abs() * 0.5 * w / lz;
            ColdMode { psi, f, impedance: a.iter().map(|a| a * a / (2.0 * kb * kb * p)).collect(), vg }
        })
        .collect()
}

/// `periods` of `s` between 20 mm of drift tube each end and a CPML of 10 × fine cells (at most
/// 64): the run, and where the structure begins (m).
fn tube(s: Structure, cells: Cells, periods: f64) -> (AxiPic, f64) {
    let layer = (10 * cells.fine).min(crate::pic::MAX_LAYER);
    let dz = cells.h / cells.fine as f64;
    let (buffer, length) = (0.02f64, periods * s.period);
    let g = cells.grid(s.outer, length + 2.0 * (layer as f64 * dz + buffer), Wall::Conductor);
    let mut pic = AxiPic::new(g, g.time_step(0.95), Pusher::Boris).unwrap();
    let z0 = layer as f64 * g.z.h + buffer;
    pic.set_metal(|r, z| {
        r > match z {
            z if z <= z0 => s.drift[0],
            z if z >= z0 + length => s.drift[1],
            z => (s.wall)(z - z0),
        }
    });
    pic.absorb(false, layer).unwrap();
    pic.absorb(true, layer).unwrap();
    (pic, z0)
}

/// A ring beam: its kinetic energy (eV), current (A), radii (m) and the axial field holding it
/// (T).
#[derive(Clone, Copy)]
struct Beam {
    energy: f64,
    current: f64,
    inner: f64,
    outer: f64,
    b: f64,
}

impl Beam {
    fn gamma0(&self) -> f64 {
        1.0 + self.energy * -ELECTRON.charge / ELECTRON.rest_energy()
    }
}

/// A BWO's run: each ns, the strongest line of E_z within the band watched (21 lines) near the
/// structure's cathode end (the window's mean, the beam's own static field, taken out) and its
/// frequency; and the line over the run's last third, to 0.1 MHz.
struct Oscillation {
    lines: Vec<(f64, f64)>,
    last: f64,
}

/// A BWO: `beam` through the cathode end of the [`tube`] of `periods`, watched within `band`
/// (Hz).
fn bwo(s: Structure, cells: Cells, periods: f64, beam: Beam, time: f64, band: (f64, f64)) -> Oscillation {
    let (mut pic, z0) = tube(s, cells, periods);
    let dt = pic.dt();
    pic.set(Component::Bz, |_, _| beam.b);
    let k = pic.add(Rings { species: ELECTRON, rings: Vec::new() }).unwrap();
    let pulse = Pulse::Steady { current: beam.current, rise: 1e-9 };
    let energy = beam.energy * -ELECTRON.charge;
    pic.inject(Injection { high: false, population: k, pulse, inner: beam.inner, radius: beam.outer, energy, per_step: 8 }).unwrap();
    let (pr, pz) = (0.9 * s.narrowest, z0 + 0.25 * s.period);
    let window = (1e-9 / dt) as usize;
    let freqs: Vec<f64> = (0..=20).map(|i| band.0 + (band.1 - band.0) * i as f64 / 20.0).collect();
    let zero = Z::new(0.0, 0.0);
    let (mut sums, mut basis, mut mean) = (vec![zero; freqs.len()], vec![zero; freqs.len()], 0.0);
    let mut out = Vec::new();
    let steps = (time / dt) as usize;
    let mut tail = Vec::new();
    for n in 0..steps {
        pic.step().unwrap();
        let (e, t) = (pic.field_at(pr, pz).0[2], pic.time());
        if n >= steps - steps / 3 {
            tail.push((t, e));
        }
        mean += e;
        for ((s, b), f) in sums.iter_mut().zip(basis.iter_mut()).zip(&freqs) {
            let w = Z::from_polar(1.0, -2.0 * PI * f * t);
            *s += w * e;
            *b += w;
        }
        if (n + 1) % window == 0 {
            let m = mean / window as f64;
            let lines = sums.iter().zip(&basis).zip(&freqs).map(|((s, b), f)| ((s - b * m).norm() / window as f64, *f));
            out.push(lines.fold((0.0, 0.0), |a, b| if b.0 > a.0 { b } else { a }));
            sums.iter_mut().chain(basis.iter_mut()).for_each(|s| *s = zero);
            mean = 0.0;
        }
    }
    let average = tail.iter().map(|x| x.1).sum::<f64>() / tail.len().max(1) as f64;
    let tail: Vec<(f64, f64)> = tail.into_iter().map(|(t, e)| (t, e - average)).collect();
    Oscillation { lines: out, last: strongest(&tail, band) }
}

/// The round-trip reflection ρ = Γ₀Γ_L of the [`tube`]'s ends for its TM01 Bloch waves at each
/// of `freqs` (Hz), about the band's phase `psi(f)` (rad per period): a pulse in the middle rung
/// down to nothing, E_z sampled at the same point s of 8 neighbouring cells from `inset` cells
/// inside each end, and the waves along ±z fitted to them by least squares,
/// E_m = a e^{−jψm} + b e^{jψm}, m counted from the end plane, ψ too. Left of the pulse a wave along +z
/// comes only from the cathode end's reflection, right of it one along −z only from the
/// collector's: a/b = Γ₀e^{−2jψs} there, b/a = Γ_L e^{2jψs} at the other end.
fn reflections(s: Structure, cells: Cells, periods: usize, inset: usize, freqs: &[f64], psi: impl Fn(f64) -> f64) -> Vec<Z> {
    let (mut pic, z0) = tube(s, cells, periods as f64);
    let mid = z0 + 0.5 * periods as f64 * s.period;
    let a = s.narrowest;
    pic.set(Component::Ez, |r, z| if r < a { j(0, X01 * r / a) * (-((z - mid) / (0.3 * s.period)).powi(2)).exp() } else { 0.0 });
    let count = 8;
    let ms: Vec<f64> = (inset..inset + count).map(|m| m as f64).collect();
    let z_left = ms.iter().map(|m| z0 + (m + 0.3) * s.period);
    let z_right = ms.iter().map(|m| z0 + (periods as f64 - m + 0.3) * s.period);
    let points: Vec<f64> = z_left.chain(z_right).collect();
    let zero = Z::new(0.0, 0.0);
    let mut sums = vec![vec![zero; points.len()]; freqs.len()];
    for _ in 0..(80e-9 / pic.dt()) as usize {
        pic.step().unwrap();
        let t = pic.time();
        let e: Vec<f64> = points.iter().map(|z| pic.field_at(0.9 * a, *z).0[2]).collect();
        for (s, f) in sums.iter_mut().zip(freqs) {
            let w = Z::from_polar(1.0, -2.0 * PI * f * t);
            for (a, v) in s.iter_mut().zip(&e) {
                *a += w * v;
            }
        }
    }
    // The least-squares (a, b) of E_m = a u_m + b v_m (the 2 × 2 normal equations) and its
    // residual; ψ itself refined about the band's, where the residual is least (golden section,
    // ±0.1 rad): a ψ a little off leaks the incident wave into the reflected one.
    let split = |e: &[Z], m: &[f64], psi: f64| {
        let u: Vec<Z> = m.iter().map(|m| Z::from_polar(1.0, -psi * m)).collect();
        let v: Vec<Z> = m.iter().map(|m| Z::from_polar(1.0, psi * m)).collect();
        let dot = |x: &[Z], y: &[Z]| x.iter().zip(y).map(|(x, y)| x.conj() * y).sum::<Z>();
        let (uu, uv, vv, ue, ve) = (dot(&u, &u), dot(&u, &v), dot(&v, &v), dot(&u, e), dot(&v, e));
        let det = uu * vv - uv * uv.conj();
        let (a, b) = ((vv * ue - uv * ve) / det, (uu * ve - uv.conj() * ue) / det);
        let residual: f64 = e.iter().zip(u.iter().zip(&v)).map(|(e, (u, v))| (e - a * u - b * v).norm_sqr()).sum();
        (a, b, residual)
    };
    let fit = |e: &[Z], m: &[f64], psi: f64| {
        let best = emcraft_theory::search::golden(psi - 0.1, psi + 0.1, 60, |p| split(e, m, p).2, |a, b| a < b);
        let (a, b, _) = split(e, m, best.at);
        (a, b)
    };
    let right: Vec<f64> = ms.iter().map(|m| -m).collect();
    freqs
        .iter()
        .zip(&sums)
        .map(|(f, s)| {
            let p = psi(*f);
            let (a0, b0) = fit(&s[..count], &ms, p);
            let (al, bl) = fit(&s[count..], &right, p);
            a0 / b0 * (bl / al)
        })
        .collect()
}

/// The circuit a BWO's start condition takes, measured on the grid: `band`'s phase and impedance
/// (the latter averaged over the ring's area from its inner, middle and outer radii, Simpson
/// r-weighted, interpolated in ln K) at each of `freqs` with the ends' reflection `rho` there
/// (or none, matched ends).
fn circuit(band: &[ColdMode], radii: [f64; 3], freqs: &[f64], rho: &[Z], echo: bool) -> Vec<Sample> {
    let between = |f: f64| {
        let w = band.windows(2).find(|w| f <= w[1].f).unwrap_or_else(|| &band[band.len() - 2..]);
        (&w[0], &w[1], (f - w[0].f) / (w[1].f - w[0].f))
    };
    let [r0, r1, r2] = radii;
    let mean = |m: &ColdMode| (r0 * m.impedance[0] + 4.0 * r1 * m.impedance[1] + r2 * m.impedance[2]) / (r0 + 4.0 * r1 + r2);
    freqs
        .iter()
        .zip(rho)
        .map(|(f, r)| {
            let (a, b, t) = between(*f);
            let impedance = (mean(a).ln() + t * (mean(b).ln() - mean(a).ln())).exp();
            Sample { frequency: *f, phase: a.psi + t * (b.psi - a.psi), impedance, reflection: if echo { *r } else { Z::new(0.0, 0.0) } }
        })
        .collect()
}

/// The band's phase at `f` (Hz), between its modes.
fn phase_of(band: &[ColdMode], f: f64) -> f64 {
    let w = band.windows(2).find(|w| f <= w[1].f).unwrap_or_else(|| &band[band.len() - 2..]);
    w[0].psi + (f - w[0].f) / (w[1].f - w[0].f) * (w[1].psi - w[0].psi)
}

/// The strongest line over a stretch of ns of a [`bwo`] run, and its frequency.
fn peak(r: &[(f64, f64)], from: usize, to: usize) -> (f64, f64) {
    r[from..to].iter().copied().fold((0.0, 0.0), |a, b| if b.0 > a.0 { b } else { a })
}

fn every(r: &[(f64, f64)]) -> String {
    r.iter().step_by(5).map(|(a, f)| format!("{a:.1e}@{:.2}", f * 1e-9)).collect::<Vec<_>>().join(" ")
}

#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn a_backward_wave_oscillator_starts_where_its_reflecting_ends_say() {
    // Johnson's start (Proc. IRE 43:684, 1955) with Pierce's space charge (Branch & Mihran's
    // reduction for a thin ring), the relativistic C³ and the ends' reflections (Levush et al.,
    // IEEE Trans. Plasma Sci. 20:263, 1992), the circuit measured on this grid: the cold TM01 band
    // and Pierce impedance on a periodic stretch, the ends' round-trip reflection ρ (|ρ| ≈ 0.17)
    // from a ring-down of the whole tube. With matched ends 12 periods would start at about 210 A;
    // the ends' echo brings it to about 87 A, and 24 periods to about 6.7 A. A 100 kV beam below
    // the reflecting ends' start (0.65 and 0.6 of it) grows nothing in 100 ns, above it (1.35 and
    // 1.8 times) oscillates near the predicted frequency. The starts measured on the grid (about
    // 90 A, 21.5 A for 18 periods against 26 A, 10 A for 24) agree with the theory if the echo's
    // round-trip phase is about 60° later than measured: the beam's modulation in the drift tube
    // by the wave leaving the structure, and the band's phase to a few degrees, both outside the
    // theory.
    let (s, cells) = (CORRUGATED, Cells { h: 0.5e-3, fine: 1 });
    let radii = [0.0075, 0.008, 0.0085];
    let band = cold_band(s, cells, 36, 7..=13, &radii, (8e9, 15e9));
    let freqs: Vec<f64> = (0..=56).map(|i| 11.0e9 + 0.025e9 * i as f64).collect();
    let rho = reflections(s, cells, 24, 2, &freqs, |f| phase_of(&band, f));
    let beam = |current: f64| Beam { energy: 1e5, current, inner: 0.0075, outer: 0.0085, b: 2.0 };
    let ring = Some(Ring { radius: 0.008, thickness: 0.001, tube: s.narrowest });
    let start = |periods: usize, echo: bool| {
        let circuit = circuit(&band, radii, &freqs, &rho, echo);
        Reflecting { circuit, period: s.period, periods, beam: emcraft_theory::bwo::Beam { gamma0: beam(0.0).gamma0(), species: ELECTRON, ring } }
            .start()
            .map(|p| (p.current, p.frequency))
            .unwrap()
    };
    let (short, long, matched) = (start(12, true), start(24, true), start(12, false));
    assert!(band.iter().all(|m| m.vg > 0.0) && rho.iter().all(|r| r.norm() > 0.1 && r.norm() < 0.3), "{rho:?}");
    assert!(matched.0 > 2.0 * short.0 && (60.0..120.0).contains(&short.0) && (5.0..9.0).contains(&long.0), "{short:?} {long:?} matched {matched:?}");
    // Each run watched within 2 % of its predicted line: the slow parasitic near 14.2 GHz and the
    // band's edges left out.
    let runs = [(12, 0.65, short), (12, 1.35, short), (24, 0.6, long), (24, 1.8, long)];
    let lines: Vec<Vec<(f64, f64)>> = std::thread::scope(|sc| {
        let handles: Vec<_> =
            runs.iter().map(|&(n, x, (i, f))| sc.spawn(move || bwo(s, cells, n as f64, beam(x * i), 100e-9, (0.98 * f, 1.02 * f)).lines)).collect();
        handles.into_iter().map(|t| t.join().unwrap()).collect()
    });
    for (pair, (_, f_start)) in lines.chunks(2).zip([short, long]) {
        let noise = peak(&pair[0], 10, 30).0;
        assert!(peak(&pair[0], 80, 100).0 < 3.0 * noise, "below the start: {}", every(&pair[0]));
        let (top, f) = peak(&pair[1], 0, 100);
        assert!(top > 100.0 * noise, "above the start: {}", every(&pair[1]));
        assert!((f / f_start - 1.0).abs() < 0.01, "{} GHz against {} GHz", f * 1e-9, f_start * 1e-9);
    }
}

#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn a_rippled_tubes_band_converges_to_its_floquet_modes() {
    // The sinusoidally rippled tube of relativistic BWOs, r_w = r₀ + h cos(2πz/d) (Swegle,
    // Poukey & Leifeste, Phys. Fluids 28:2882, 1985): r₀ = 15 mm, h = 1 mm, d = 16.7 mm, within
    // Rayleigh's limit (k₀h = 0.38), against emcraft_theory::ripple's Floquet modes. Its wall a
    // staircase, the grid's band sits below them by an error of the cell's first order (−0.35 %
    // at ψ = π on 0.25 mm cells, −0.77 % on 0.5 mm); from both, Richardson's extrapolation meets
    // them to 0.1 % across the band and at both edges of the stop band at π.
    use emcraft_theory::ripple::RippledGuide;
    let guide = RippledGuide { radius: 0.015, ripple: 0.001, period: RIPPLED.period };
    // (m of 8 periods, the initial standing wave's shift, which of the theory's modes)
    let modes = [(1usize, 0.0, 0usize), (2, 0.0, 0), (3, 0.0, 0), (4, 0.0, 0), (4, PI / 2.0, 1)];
    let lines: Vec<(f64, f64, f64)> = std::thread::scope(|sc| {
        let jobs: Vec<_> = modes
            .iter()
            .map(|&(m, shift, which)| {
                sc.spawn(move || {
                    let at = |h: f64| bloch_line(RIPPLED, Cells { h, fine: 1 }, 8, m, shift, (6e9, 14e9));
                    let theory = guide.frequencies(PI * m as f64 / 4.0, 6e9, 14e9, 800, 8).unwrap()[which];
                    (at(0.5e-3), at(0.25e-3), theory)
                })
            })
            .collect();
        jobs.into_iter().map(|j| j.join().unwrap()).collect()
    });
    for (coarse, fine, theory) in lines {
        assert!(fine < theory && (fine / theory - 1.0).abs() < 0.005, "{fine} against {theory}");
        assert!(((2.0 * fine - coarse) / theory - 1.0).abs() < 1e-3, "{coarse}, {fine} against {theory}");
    }
}

#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn a_relativistic_backward_wave_oscillator_starts_where_its_theory_says() {
    // A 255 keV beam (γ₀ = 1.5; its longitudinal mass γ³m = 3.4 m) on an 11.5–12.5 mm ring held
    // by 3 T, through 16 periods of the rippled tube (15 ± 1 mm, d = 16.7 mm) on cells 4 times
    // finer along z (the numerical Cherenkov instability, emcraft_fdtd::axi). Johnson's start
    // (Proc. IRE 43:684, 1955) with the relativistic C³ and the ends' echo (Levush et al., IEEE
    // Trans. Plasma Sci. 20:263, 1992) and Pierce's space charge (Branch & Mihran's thin ring),
    // from the circuit measured on the grid — the cold band and impedance on a periodic stretch
    // (0.23 Ω on the ring at ψ ≈ 105°), the ends' round-trip reflection from a ring-down of 24
    // periods (|ρ| ≈ 0.045: the ripple meets the drift tubes at its crest, smoothly) — puts the
    // start at about 330 A near 9.11 GHz. At 0.6 of it nothing grows in 60 ns; at 1.4 times it
    // oscillates within 1.5 % of that frequency (9.02–9.06 GHz: the current pulls it down); at
    // 1.0 the line holds about level, at the start.
    let (s, cells) = (RIPPLED, Cells { h: 0.5e-3, fine: 4 });
    let radii = [0.0115, 0.012, 0.0125];
    let band = cold_band(s, cells, 20, 5..=7, &radii, (7e9, 11.5e9));
    let freqs: Vec<f64> = (0..=40).map(|i| 8.6e9 + 0.025e9 * i as f64).collect();
    // The ends' reflection on square cells (the same staircase across; ψ refined by the fit).
    let rho = reflections(s, Cells { fine: 1, ..cells }, 24, 2, &freqs, |f| phase_of(&band, f));
    let beam = |current: f64| Beam { energy: 2.555e5, current, inner: 0.0115, outer: 0.0125, b: 3.0 };
    let ring = Some(Ring { radius: 0.012, thickness: 0.001, tube: s.narrowest });
    let start = |echo: bool| {
        let circuit = circuit(&band, radii, &freqs, &rho, echo);
        Reflecting { circuit, period: s.period, periods: 16, beam: emcraft_theory::bwo::Beam { gamma0: beam(0.0).gamma0(), species: ELECTRON, ring } }
            .start()
            .map(|p| (p.current, p.frequency))
            .unwrap()
    };
    let (matched, (current, f_start)) = (start(false), start(true));
    assert!(rho.iter().all(|r| r.norm() < 0.1), "{rho:?}");
    assert!((200.0..500.0).contains(&current) && (matched.0 / current - 1.0).abs() < 0.1, "{current} A at {f_start} Hz, matched {matched:?}");
    let lines: Vec<Vec<(f64, f64)>> = std::thread::scope(|sc| {
        let jobs: Vec<_> = [0.6, 1.4]
            .iter()
            .map(|&x| sc.spawn(move || bwo(s, cells, 16.0, beam(x * current), 60e-9, (0.95 * f_start, 1.05 * f_start)).lines))
            .collect();
        jobs.into_iter().map(|j| j.join().unwrap()).collect()
    });
    let noise = peak(&lines[0], 10, 30).0;
    assert!(peak(&lines[0], 40, 60).0 < 3.0 * noise, "below the start: {}", every(&lines[0]));
    let (top, f) = peak(&lines[1], 0, 60);
    assert!(top > 100.0 * noise, "above the start: {}", every(&lines[1]));
    assert!((f / f_start - 1.0).abs() < 0.015, "{} GHz against {} GHz", f * 1e-9, f_start * 1e-9);
}

#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn a_ku_band_relativistic_bwo_oscillates_at_its_published_frequency() {
    // Na et al. (J. Comput. Phys. 346:295, 2017; arXiv:1701.03784, §5.1): their Ku-band BWO
    // (the KU ripple, a 1.5 kA beam at 0.877c on a 9–11 mm ring, 5 T) oscillates at 15.59 GHz,
    // its walls drawn smoothly (15.575–15.61 GHz as their mesh is refined; on staircased walls
    // lower, 14.81 and 15.16 GHz). Here the walls are staircases of 0.5 and 0.25 mm cells (4 times
    // finer along z). The cold π-point the BWO runs just below converges at the cell's second
    // order (15.25, 15.58 GHz; extrapolated 15.68 GHz, emcraft_theory::ripple's to 0.05 %); the
    // oscillation, 15.05 and 15.35 GHz, below it, extrapolated to 15.46 GHz at second order and
    // 15.66 GHz at first: the published line between them, each within 1 % of it.
    use emcraft_theory::ripple::RippledGuide;
    let theory = RippledGuide { radius: 0.015, ripple: 0.0021, period: 0.0075 }.frequencies(PI, 10e9, 20e9, 1000, 8).unwrap()[0];
    let gamma0 = 1.0 / (1.0 - 0.877f64.powi(2)).sqrt();
    let energy = (gamma0 - 1.0) * ELECTRON.rest_energy() / -ELECTRON.charge;
    let runs: Vec<(f64, f64)> = std::thread::scope(|sc| {
        let jobs: Vec<_> = [0.5e-3, 0.25e-3]
            .iter()
            .map(|&h| {
                sc.spawn(move || {
                    let cells = Cells { h, fine: 4 };
                    // The π-point's standing wave odd about the ripple's trough (the even one, the stop
                    // band's other edge, lies above 20 GHz).
                    let pi = bloch_line(KU, cells, 2, 1, PI / 2.0, (14.5e9, 17e9));
                    let beam = Beam { energy, current: 1500.0, inner: 0.009, outer: 0.011, b: 5.0 };
                    (pi, bwo(KU, cells, 14.5, beam, 50e-9, (14.5e9, 16.5e9)).last)
                })
            })
            .collect();
        jobs.into_iter().map(|j| j.join().unwrap()).collect()
    });
    let [(pi_coarse, coarse), (pi_fine, fine)] = [runs[0], runs[1]];
    let published = 15.59e9;
    let (first, second) = (2.0 * fine - coarse, fine + (fine - coarse) / 3.0);
    assert!(coarse < fine && fine < published && second < published && published < first, "{coarse}, {fine} against {published}");
    assert!([first, second].iter().all(|f| (f / published - 1.0).abs() < 0.01), "{first}, {second} against {published}");
    let pi = pi_fine + (pi_fine - pi_coarse) / 3.0;
    assert!(pi_coarse < pi_fine && pi_fine < theory && (pi / theory - 1.0).abs() < 5e-4, "π-point {pi_coarse}, {pi_fine} against {theory}");
    assert!(coarse < pi_coarse && fine < pi_fine, "the oscillation below the π-point");
}
