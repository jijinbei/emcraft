//! The axisymmetric PIC against closed forms: a pillbox's TM010, Gauss's law kept by moving
//! rings, a bunch's loss factor through a pillbox, a ring beam's space-charge depression.

use std::f64::consts::PI;

use emcraft_model::units::C0;
use emcraft_particles::{ELECTRON, Pusher, Species};

use super::*;
use crate::pic::{Pulse, Wall};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn pillbox(h: f64) -> (AxiPic, f64, f64) {
    let (a, l) = (0.01f64, 0.01306f64);
    let g = AxiGrid::new((a / h).round() as usize, (l / h).round() as usize, h, l / (l / h).round(), Wall::Conductor).unwrap();
    let dt = g.time_step(0.95);
    (AxiPic::new(g, dt, Pusher::Boris).unwrap(), a, l)
}

#[test]
fn a_pillbox_rings_at_tm010() {
    // E_z = J₀(x₀₁r/a) in a closed cylinder rings at x₀₁c/(2πa) (Pozar §6.4): 11.4743 GHz for
    // 10 mm; 1.1e-4 low on 0.25 mm cells.
    use emcraft_theory::bessel::j;
    let (mut pic, a, l) = pillbox(0.25e-3);
    let dt = pic.dt();
    let x01 = 2.404_825_557_695_773;
    pic.set(2, |r, _| j(0, x01 * r / a));
    let mut ups = Vec::new();
    let mut last = pic.field_at(0.0, 0.5 * l).0[2];
    for n in 0..((5.0 / 11.47e9) / dt) as usize {
        pic.step().unwrap();
        let now = pic.field_at(0.0, 0.5 * l).0[2];
        if last < 0.0 && now >= 0.0 {
            ups.push((n as f64 + last / (last - now)) * dt);
        }
        last = now;
    }
    let f = (ups.len() - 1) as f64 / (ups[ups.len() - 1] - ups[0]);
    let want = x01 * C0 / (2.0 * PI * a);
    assert!((f / want - 1.0).abs() < 3e-4, "{f:e} Hz against {want:e}");
}

#[rstest::rstest]
#[case::closed(Wall::Conductor)]
#[case::periodic(Wall::Periodic)]
fn rings_keep_gauss_law_to_rounding(#[case] wall: Wall) {
    // Electron and positron rings up to γ ≈ 2 in every direction, a static B_z, a metal block
    // and the walls: the flux form ε₀∮E·dA − Q stays where it started at every node.
    let g = AxiGrid::new(12, 30, 1e-3, 0.8e-3, wall).unwrap();
    let mut pic = AxiPic::new(g, g.time_step(0.9), Pusher::Vay).unwrap();
    pic.set_metal(|r, z| r > 0.009 && (z - 0.012).abs() < 0.002);
    let mut rnd = Lcg(5);
    let positron = Species { charge: -ELECTRON.charge, mass: ELECTRON.mass };
    for species in [ELECTRON, positron] {
        let rings = (0..300)
            .map(|_| Ring { r: 0.011 * rnd.next(), z: 0.024 * rnd.next(), u: [(); 3].map(|()| 2.0 * C0 * (2.0 * rnd.next() - 1.0)), w: 1e6 })
            .collect();
        pic.add(Rings { species, rings }).unwrap();
    }
    pic.set(5, |_, _| 0.2);
    let before = pic.gauss();
    let scale = before.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    for _ in 0..150 {
        pic.step().unwrap();
    }
    let drift = before.iter().zip(pic.gauss()).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
    assert!(drift < 1e-12 * scale, "{drift:e} of {scale:e}");
    assert!(pic.rings() < 600, "some reach the walls");
}

#[test]
fn a_bunch_on_the_axis_leaves_the_pillboxs_loss_factor() {
    // As the Cartesian grid's test, with the cylinder exact: q² Σ k_n e^{−(ω_nσ_t)²} over the
    // TM0np modes (Wilson, SLAC-PUB-4547; emcraft_theory::wake), 5.805 V/pC; 0.05 % on 0.5 mm.
    let (mut pic, a, l) = pillbox(0.5e-3);
    let dt = pic.dt();
    let k = pic.add(Rings { species: ELECTRON, rings: Vec::new() }).unwrap();
    let sigma = 2e-3 / C0;
    let pulse = Pulse::Gaussian { charge: 1e-9, t0: 5.0 * sigma, sigma };
    pic.inject(Injection { high: false, population: k, pulse, inner: 0.0, radius: 0.0, energy: 1e10 * -ELECTRON.charge, per_step: 4 }).unwrap();
    let mut q = 0.0;
    for _ in 0..((10.0 * sigma + l / C0) / dt) as usize + 10 {
        q += pic.step().unwrap().emitted;
    }
    let got = pic.energy().0 / (q * q);
    let want = emcraft_theory::wake::Pillbox::new(a, l).loss_factor(1.0, sigma).unwrap();
    assert!((got / want - 1.0).abs() < 2e-3, "{got:e} V/C against {want:e}");
}

#[test]
fn a_ring_beam_sinks_by_its_space_charge_and_stops_at_the_limit() {
    // Bogdankevich & Rukhadze (Sov. Phys. Usp. 14:163, 1971): γ₀ = 1.25 on a 6 mm ring in a 10 mm
    // tube held by 10 T; at 0.7 of the limiting current it sinks to γ_b (2 % deeper here, the
    // ring 0.5 mm thick), and at 1.3 of it no more than about the limit passes. Faster beams
    // (γ₀ ≳ 1.5) drive the grid's numerical Cherenkov instability on these square cells (cells
    // finer along z: the next test).
    use emcraft_theory::plasma::AnnularBeam;
    let (r, rb, l, h) = (0.01f64, 0.006, 0.08, 0.5e-3);
    let beam = AnnularBeam { gamma0: 1.25, ratio: r / rb, charge: ELECTRON.charge, mass: ELECTRON.mass };
    let il = beam.limiting().unwrap();
    let run = |current: f64| {
        let g = AxiGrid::new((r / h).round() as usize, (l / h).round() as usize, h, h, Wall::Conductor).unwrap();
        let dt = g.time_step(0.95);
        let mut pic = AxiPic::new(g, dt, Pusher::Boris).unwrap();
        pic.set(5, |_, _| 10.0);
        let k = pic.add(Rings { species: ELECTRON, rings: Vec::new() }).unwrap();
        let pulse = Pulse::Steady { current, rise: 0.2e-9 };
        pic.inject(Injection {
            high: false,
            population: k,
            pulse,
            inner: rb - 0.25e-3,
            radius: rb + 0.25e-3,
            energy: 0.25 * ELECTRON.rest_energy(),
            per_step: 16,
        })
        .unwrap();
        let steps = (2e-9 / dt) as usize;
        let (mut out, mut sum, mut n) = (0.0, 0.0, 0usize);
        for s in 0..steps {
            let m = pic.step().unwrap();
            if s >= steps / 2 {
                out += m.collected[2][1];
                if s % 10 == 0 {
                    for p in pic.populations()[k].rings.iter().filter(|p| p.z > l / 3.0 && p.z < 2.0 * l / 3.0) {
                        sum += emcraft_particles::gamma(p.u);
                        n += 1;
                    }
                }
            }
        }
        (sum / n.max(1) as f64, out.abs() / ((steps - steps / 2) as f64 * dt))
    };
    let want = beam.depressed(0.7 * il).unwrap();
    let (g, passed) = run(0.7 * il);
    assert!(((1.25 - g) / (1.25 - want) - 1.0).abs() < 0.04, "γ_b {g} against {want}");
    assert!((passed / (0.7 * il) - 1.0).abs() < 0.02);
    let (_, over) = run(1.3 * il);
    assert!(over < 1.0 * il && over > 0.8 * il, "{over} A past the limit {il}");
}

#[test]
fn a_relativistic_ring_beam_keeps_its_depression_on_cells_fine_along_z() {
    // The ring beam of Bogdankevich & Rukhadze (Sov. Phys. Usp. 14:163, 1971) at γ₀ = 3, 0.7 of
    // its limiting current. On square cells it drives the Yee grid's numerical Cherenkov
    // instability (B. B. Godfrey, J.-L. Vay, J. Comput. Phys. 248:33, 2013): its depression comes
    // out a quarter deeper in this short tube. With cells 8 times finer along z the time step's
    // c Δt nears Δz, where the grid's waves along z travel at c (the magic time step, A. Taflove,
    // S. C. Hagness, *Computational Electrodynamics*, 3rd ed., 2005, §4.3), and the depression is
    // the closed form's to 3 % here (in an 80 mm tube 1–4 % shallower for γ₀ from 1.25 to 5
    // alike, and steady for 20 ns; on square cells 6–19 % deeper).
    use emcraft_theory::plasma::AnnularBeam;
    let (r, rb, l, h, gamma0) = (0.01f64, 0.006, 0.05, 1e-3, 3.0);
    let beam = AnnularBeam { gamma0, ratio: r / rb, charge: ELECTRON.charge, mass: ELECTRON.mass };
    let current = 0.7 * beam.limiting().unwrap();
    let want = gamma0 - beam.depressed(current).unwrap();
    let depression = |fine: usize| {
        let dz = h / fine as f64;
        let g = AxiGrid::new((r / h).round() as usize, (l / dz).round() as usize, h, dz, Wall::Conductor).unwrap();
        let dt = g.time_step(0.95);
        let mut pic = AxiPic::new(g, dt, Pusher::Boris).unwrap();
        pic.set(5, |_, _| 10.0);
        let k = pic.add(Rings { species: ELECTRON, rings: Vec::new() }).unwrap();
        let pulse = Pulse::Steady { current, rise: 0.2e-9 };
        pic.inject(Injection {
            high: false,
            population: k,
            pulse,
            inner: rb - 0.25e-3,
            radius: rb + 0.25e-3,
            energy: (gamma0 - 1.0) * ELECTRON.rest_energy(),
            per_step: 16,
        })
        .unwrap();
        let steps = (1.2e-9 / dt) as usize;
        let (mut sum, mut n) = (0.0, 0usize);
        for s in 0..steps {
            pic.step().unwrap();
            if s >= steps / 2 && s % 10 == 0 {
                for p in pic.populations()[k].rings.iter().filter(|p| p.z > l / 3.0 && p.z < 2.0 * l / 3.0) {
                    sum += emcraft_particles::gamma(p.u);
                    n += 1;
                }
            }
        }
        (gamma0 - sum / n.max(1) as f64) / want
    };
    let (square, fine) = std::thread::scope(|s| {
        let square = s.spawn(|| depression(1));
        (square.join().unwrap(), depression(8))
    });
    assert!(square > 1.15, "square cells: {square}");
    assert!((fine - 1.0).abs() < 0.035, "fine along z: {fine}");
}

#[test]
fn hostile_axisymmetric_settings_are_refused() {
    assert!(AxiGrid::new(0, 10, 1e-3, 1e-3, Wall::Conductor).is_err());
    assert!(AxiGrid::new(10, 10, f64::NAN, 1e-3, Wall::Conductor).is_err());
    assert!(AxiGrid::new(100_000, 100_000, 1e-3, 1e-3, Wall::Conductor).is_err());
    let g = AxiGrid::new(10, 20, 1e-3, 1e-3, Wall::Conductor).unwrap();
    for dt in [0.0, f64::NAN, g.time_step(1.2)] {
        assert!(AxiPic::new(g, dt, Pusher::Boris).is_err());
    }
    let mut pic = AxiPic::new(g, g.time_step(0.9), Pusher::Boris).unwrap();
    let one = |r: f64, z: f64, w: f64| Rings { species: ELECTRON, rings: vec![Ring { r, z, u: [0.0; 3], w }] };
    assert!(pic.add(one(0.02, 0.01, 1.0)).is_err() && pic.add(one(0.005, -1.0, 1.0)).is_err() && pic.add(one(0.005, 0.01, -1.0)).is_err());
    let k = pic.add(one(0.005, 0.01, 1.0)).unwrap();
    let base = Injection {
        high: false,
        population: k,
        pulse: Pulse::Steady { current: 1.0, rise: 0.0 },
        inner: 0.0,
        radius: 0.005,
        energy: 1e-15,
        per_step: 4,
    };
    for bad in [
        Injection { population: 7, ..base },
        Injection { inner: 0.006, ..base },
        Injection { radius: 0.5, ..base },
        Injection { per_step: 0, ..base },
    ] {
        assert!(pic.inject(bad).is_err(), "{bad:?}");
    }
    assert!(pic.absorb(true, 0).is_err() && pic.absorb(true, 25).is_err());
    let periodic = AxiGrid::new(10, 20, 1e-3, 1e-3, Wall::Periodic).unwrap();
    let mut p = AxiPic::new(periodic, periodic.time_step(0.9), Pusher::Boris).unwrap();
    assert!(p.absorb(false, 4).is_err());
    // A field beyond physics drops the ring rather than poisoning the grid.
    pic.set(2, |_, _| f64::INFINITY);
    assert_eq!(pic.step().unwrap().lost, 1);
}

/// The slow-wave structure of the BWO tests: a tube 10 mm inside its teeth, grooves 5 mm wide to
/// 13 mm, period 10 mm (grooves at the period's first half).
const TEETH: f64 = 0.010;
const GROOVES: f64 = 0.013;
const PERIOD: f64 = 0.010;

fn corrugated(r: f64, s: f64) -> bool {
    r > if (s.rem_euclid(1.0) - 0.25).abs() < 0.25 { GROOVES } else { TEETH }
}

/// A TM01 mode of the corrugated tube: its Bloch phase per period (rad), frequency (Hz), Pierce
/// impedance of its backward harmonic k = (2π − ψ)/d at each asked radius (Ω), group velocity.
struct ColdMode {
    psi: f64,
    f: f64,
    impedance: Vec<f64>,
    vg: f64,
}

/// The structure's cold TM01 band on a periodic stretch of `periods` (cells of `h`), at the
/// Bloch phases 2πm/periods for `modes`: E_z = J₀(x₀₁r/a) cos(kz) let ring, its strongest line
/// over 8–15 GHz found, then every component's phasor there accumulated; the stored energy
/// W = ¼∫(ε₀|Ê|² + |B̂|²/μ₀) dV, half of it each of the two waves ±ψ; the harmonic's amplitude E
/// the spatial transform of Ê_z along z; v_g = dω/dk from the neighbouring phases; then
/// K = E²/(2k²P), P = v_g (W/2)/L (Pierce 1950, ch. II).
fn cold_band(h: f64, periods: usize, modes: std::ops::RangeInclusive<usize>, radii: &[f64]) -> Vec<ColdMode> {
    use emcraft_model::units::{EPS0, MU0};
    use emcraft_theory::bessel::j;
    use num_complex::Complex64 as Z;
    let lz = periods as f64 * PERIOD;
    let nz = (lz / h).round() as usize;
    let g = AxiGrid::new((GROOVES / h).round() as usize, nz, h, lz / nz as f64, Wall::Periodic).unwrap();
    let dt = g.time_step(0.95);
    let x01 = 2.404_825_557_695_773;
    let zero = Z::new(0.0, 0.0);
    let mut out: Vec<(f64, f64, Vec<f64>, f64)> = Vec::new();
    for m in modes {
        let k = 2.0 * PI * m as f64 / lz;
        let start = || {
            let mut pic = AxiPic::new(g, dt, Pusher::Boris).unwrap();
            pic.set(2, |r, z| if r < TEETH { j(0, x01 * r / TEETH) * (k * z).cos() } else { 0.0 });
            pic.set_metal(|r, z| corrugated(r, z / PERIOD));
            pic
        };
        let mut pic = start();
        let freqs: Vec<f64> = (0..=700).map(|i| 8e9 + 0.01e9 * i as f64).collect();
        let mut sums = vec![zero; freqs.len()];
        for _ in 0..(30e-9 / dt) as usize {
            pic.step().unwrap();
            let (e, t) = (pic.field_at(0.009, 0.3 * PERIOD).0[2], pic.time());
            for (s, f) in sums.iter_mut().zip(&freqs) {
                *s += Z::from_polar(e, -2.0 * PI * f * t);
            }
        }
        let f = sums.iter().zip(&freqs).fold((0.0, 0.0), |a, (s, f)| if s.norm() > a.0 { (s.norm(), *f) } else { a }).1;
        let mut pic = start();
        let mut ph: [Vec<Z>; 6] = std::array::from_fn(|w| vec![zero; pic.lats[w].len()]);
        let steps = (40e-9 / dt) as usize;
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
        let mut w = 0.0;
        for (which, p) in ph.iter().enumerate() {
            let lat = pic.lats[which];
            let coefficient = if which < 3 { 0.25 * EPS0 } else { 0.25 / MU0 };
            for kk in 0..lat.nz {
                for i in 0..lat.nr {
                    let r = (i as f64 + if lat.r_node { 0.0 } else { 0.5 }) * h;
                    let ring = if lat.r_node && i == 0 { PI * 0.25 * h * h } else { 2.0 * PI * r * h };
                    w += coefficient * p[lat.at(i, kk)].norm_sqr() * ring * g.dz;
                }
            }
        }
        let kb = 2.0 * PI / PERIOD - k;
        let lez = pic.lats[EZ];
        let amplitudes = radii
            .iter()
            .map(|rb| {
                let i = (rb / h).round() as usize;
                (0..lez.nz).map(|kk| ph[EZ][lez.at(i, kk)] * Z::from_polar(g.dz / lz, -kb * (kk as f64 + 0.5) * g.dz)).sum::<Z>().norm()
            })
            .collect();
        out.push((k * PERIOD, f, amplitudes, w));
    }
    let n = out.len();
    (0..n)
        .map(|m| {
            let (psi, f, ref a, w) = out[m];
            let (lo, hi) = (m.saturating_sub(1), (m + 1).min(n - 1));
            let vg = 2.0 * PI * (out[hi].1 - out[lo].1) / ((out[hi].0 - out[lo].0) / PERIOD);
            let kb = (2.0 * PI - psi) / PERIOD;
            let p = vg.abs() * 0.5 * w / lz;
            ColdMode { psi, f, impedance: a.iter().map(|a| a * a / (2.0 * kb * kb * p)).collect(), vg }
        })
        .collect()
}

/// `periods` of the corrugated tube between 20 mm of smooth tube at the grooves' radius each end
/// (far above its cut-off) and a 10-cell absorber, on cells of `h`; and where the structure
/// begins (m).
fn tube(periods: usize, h: f64) -> (AxiPic, f64) {
    let (layer, buffer) = (10usize, 0.02f64);
    let cells = periods as f64;
    let lz = cells * PERIOD + 2.0 * (layer as f64 * h + buffer);
    let nz = (lz / h).round() as usize;
    let g = AxiGrid::new((GROOVES / h).round() as usize, nz, h, lz / nz as f64, Wall::Conductor).unwrap();
    let dt = g.time_step(0.95);
    let mut pic = AxiPic::new(g, dt, Pusher::Boris).unwrap();
    let z0 = layer as f64 * h + buffer;
    pic.set_metal(|r, z| {
        let s = (z - z0) / PERIOD;
        if s > 0.0 && s < cells { corrugated(r, s) } else { r > GROOVES }
    });
    pic.absorb(false, layer).unwrap();
    pic.absorb(true, layer).unwrap();
    (pic, z0)
}

/// A BWO in a body of revolution: the [`tube`] of `periods`; a ring beam (7.5–8.5 mm) of
/// `current` A at 100 kV held by 2 T through the cathode end. Each ns, the strongest line of E_z
/// within `band` (Hz; 21 lines) near the structure's cathode end (the window's mean, the beam's
/// own static field, taken out) and its frequency.
fn bwo(periods: usize, current: f64, time: f64, h: f64, band: (f64, f64)) -> Vec<(f64, f64)> {
    use num_complex::Complex64 as Z;
    let (mut pic, z0) = tube(periods, h);
    let dt = pic.dt();
    pic.set(5, |_, _| 2.0);
    let k = pic.add(Rings { species: ELECTRON, rings: Vec::new() }).unwrap();
    let pulse = Pulse::Steady { current, rise: 1e-9 };
    pic.inject(Injection { high: false, population: k, pulse, inner: 0.0075, radius: 0.0085, energy: 1e5 * -ELECTRON.charge, per_step: 8 }).unwrap();
    let (pr, pz) = (0.009, z0 + 0.25 * PERIOD);
    let window = (1e-9 / dt) as usize;
    let freqs: Vec<f64> = (0..=20).map(|i| band.0 + (band.1 - band.0) * i as f64 / 20.0).collect();
    let zero = Z::new(0.0, 0.0);
    let (mut sums, mut basis, mut mean) = (vec![zero; freqs.len()], vec![zero; freqs.len()], 0.0);
    let mut out = Vec::new();
    for n in 0..(time / dt) as usize {
        pic.step().unwrap();
        let (e, t) = (pic.field_at(pr, pz).0[2], pic.time());
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
    out
}

/// The round-trip reflection ρ = Γ₀Γ_L of the [`tube`]'s ends for its TM01 Bloch waves at each
/// of `freqs` (Hz), the band's phase `psi(f)` (rad per period) given: a pulse in the middle rung
/// down to nothing, E_z sampled at the same point s of `cells` neighbouring cells from `inset`
/// cells inside each end, and the waves along ±z fitted to them by least squares,
/// E_m = a e^{−jψm} + b e^{jψm}, m counted from the end plane. Left of the pulse a wave along +z
/// comes only from the cathode end's reflection, right of it one along −z only from the
/// collector's: a/b = Γ₀e^{−2jψs} there, b/a = Γ_L e^{2jψs} at the other end.
fn reflections(periods: usize, inset: usize, cells: usize, h: f64, freqs: &[f64], psi: impl Fn(f64) -> f64) -> Vec<num_complex::Complex64> {
    use emcraft_theory::bessel::j;
    use num_complex::Complex64 as Z;
    let (mut pic, z0) = tube(periods, h);
    let dt = pic.dt();
    let mid = z0 + 0.5 * periods as f64 * PERIOD;
    let x01 = 2.404_825_557_695_773;
    pic.set(2, |r, z| if r < TEETH { j(0, x01 * r / TEETH) * (-((z - mid) / 0.003).powi(2)).exp() } else { 0.0 });
    // Cells m from each end plane: inset.. on the left, −inset.. on the right.
    let ms: Vec<f64> = (inset..inset + cells).map(|m| m as f64).collect();
    let z_left = ms.iter().map(|m| z0 + (m + 0.3) * PERIOD);
    let z_right = ms.iter().map(|m| z0 + (periods as f64 - m + 0.3) * PERIOD);
    let points: Vec<f64> = z_left.chain(z_right).collect();
    let zero = Z::new(0.0, 0.0);
    let mut sums = vec![vec![zero; points.len()]; freqs.len()];
    for _ in 0..(80e-9 / dt) as usize {
        pic.step().unwrap();
        let t = pic.time();
        let e: Vec<f64> = points.iter().map(|z| pic.field_at(0.009, *z).0[2]).collect();
        for (s, f) in sums.iter_mut().zip(freqs) {
            let w = Z::from_polar(1.0, -2.0 * PI * f * t);
            for (a, v) in s.iter_mut().zip(&e) {
                *a += w * v;
            }
        }
    }
    // The least-squares (a, b) of E_m = a u_m + b v_m: the 2 × 2 normal equations.
    let fit = |e: &[Z], m: &[f64], psi: f64| {
        let u: Vec<Z> = m.iter().map(|m| Z::from_polar(1.0, -psi * m)).collect();
        let v: Vec<Z> = m.iter().map(|m| Z::from_polar(1.0, psi * m)).collect();
        let dot = |x: &[Z], y: &[Z]| x.iter().zip(y).map(|(x, y)| x.conj() * y).sum::<Z>();
        let (uu, uv, vv, ue, ve) = (dot(&u, &u), dot(&u, &v), dot(&v, &v), dot(&u, e), dot(&v, e));
        let det = uu * vv - uv * uv.conj();
        ((vv * ue - uv * ve) / det, (uu * ve - uv.conj() * ue) / det)
    };
    let right: Vec<f64> = ms.iter().map(|m| -m).collect();
    freqs
        .iter()
        .zip(&sums)
        .map(|(f, s)| {
            let p = psi(*f);
            let (a0, b0) = fit(&s[..cells], &ms, p);
            let (al, bl) = fit(&s[cells..], &right, p);
            a0 / b0 * (bl / al)
        })
        .collect()
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
    use emcraft_theory::bwo::{Reflecting, Ring, Sample};
    use num_complex::Complex64 as Z;
    let h = 0.5e-3;
    let band = cold_band(h, 36, 7..=13, &[0.0075, 0.008, 0.0085]);
    let between = |f: f64| {
        let w = band.windows(2).find(|w| f <= w[1].f).unwrap_or_else(|| &band[band.len() - 2..]);
        (&w[0], &w[1], (f - w[0].f) / (w[1].f - w[0].f))
    };
    // The impedance averaged over the ring's area (Simpson, r-weighted), interpolated in ln K.
    let mean = |m: &ColdMode| (7.5 * m.impedance[0] + 32.0 * m.impedance[1] + 8.5 * m.impedance[2]) / 48.0;
    let psi = |f: f64| {
        let (a, b, t) = between(f);
        a.psi + t * (b.psi - a.psi)
    };
    let impedance = |f: f64| {
        let (a, b, t) = between(f);
        (mean(a).ln() + t * (mean(b).ln() - mean(a).ln())).exp()
    };
    let freqs: Vec<f64> = (0..=56).map(|i| 11.0e9 + 0.025e9 * i as f64).collect();
    let rho = reflections(24, 2, 8, h, &freqs, psi);
    let circuit = |echo: bool| -> Vec<Sample> {
        let zero = Z::new(0.0, 0.0);
        freqs
            .iter()
            .zip(&rho)
            .map(|(f, r)| Sample { frequency: *f, phase: psi(*f), impedance: impedance(*f), reflection: if echo { *r } else { zero } })
            .collect()
    };
    let gamma0 = 1.0 + 1e5 / ELECTRON.rest_energy() * -ELECTRON.charge;
    let ring = Some(Ring { radius: 0.008, thickness: 0.001, tube: TEETH });
    let start = |periods: usize, echo: bool| {
        Reflecting { circuit: circuit(echo), period: PERIOD, periods, gamma0, charge: ELECTRON.charge, mass: ELECTRON.mass, ring }.start().unwrap()
    };
    let (short, long, matched) = (start(12, true), start(24, true), start(12, false));
    assert!(band.iter().all(|m| m.vg > 0.0) && rho.iter().all(|r| r.norm() > 0.1 && r.norm() < 0.3), "{rho:?}");
    assert!(matched.0 > 2.0 * short.0 && (60.0..120.0).contains(&short.0) && (5.0..9.0).contains(&long.0), "{short:?} {long:?} matched {matched:?}");
    // Each run watched within 2 % of its predicted line: the slow parasitic near 14.2 GHz and the
    // band's edges left out.
    let runs = [(12, 0.65, short), (12, 1.35, short), (24, 0.6, long), (24, 1.8, long)];
    let lines: Vec<Vec<(f64, f64)>> = std::thread::scope(|s| {
        let handles: Vec<_> = runs.iter().map(|&(n, x, (i, f))| s.spawn(move || bwo(n, x * i, 100e-9, h, (0.98 * f, 1.02 * f)))).collect();
        handles.into_iter().map(|t| t.join().unwrap()).collect()
    });
    // The strongest line over a stretch of ns, and its frequency.
    let peak = |r: &[(f64, f64)], from: usize, to: usize| r[from..to].iter().copied().fold((0.0, 0.0), |a, b| if b.0 > a.0 { b } else { a });
    for (pair, (_, f_start)) in lines.chunks(2).zip([short, long]) {
        let noise = peak(&pair[0], 10, 30).0;
        let every = |r: &[(f64, f64)]| r.iter().step_by(5).map(|(a, f)| format!("{a:.1e}@{:.2}", f * 1e-9)).collect::<Vec<_>>().join(" ");
        assert!(peak(&pair[0], 80, 100).0 < 3.0 * noise, "below the start: {}", every(&pair[0]));
        let (top, f) = peak(&pair[1], 0, 100);
        assert!(top > 100.0 * noise, "above the start: {}", every(&pair[1]));
        assert!((f / f_start - 1.0).abs() < 0.01, "{} GHz against {} GHz", f * 1e-9, f_start * 1e-9);
    }
}
