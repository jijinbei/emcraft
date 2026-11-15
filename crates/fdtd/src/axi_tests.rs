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

/// A ring beam of `current` A at `gamma0` (5.75–6.25 mm in radius) sent in through the low end
/// of a conducting tube 10 mm in radius and `l` long, held by 10 T, on cells of `h` across and
/// `h / fine` along z, for `time` (s): the mean γ in the tube's middle third and the current
/// reaching its far end (A), over the run's second half.
fn ring_beam(gamma0: f64, current: f64, l: f64, h: f64, fine: usize, time: f64) -> (f64, f64) {
    let (r, rb) = (0.01f64, 0.006);
    let dz = h / fine as f64;
    let g = AxiGrid::new((r / h).round() as usize, (l / dz).round() as usize, h, dz, Wall::Conductor).unwrap();
    let dt = g.time_step(0.95);
    let mut pic = AxiPic::new(g, dt, Pusher::Boris).unwrap();
    pic.set(5, |_, _| 10.0);
    let k = pic.add(Rings { species: ELECTRON, rings: Vec::new() }).unwrap();
    let pulse = Pulse::Steady { current, rise: 0.2e-9 };
    let energy = (gamma0 - 1.0) * ELECTRON.rest_energy();
    pic.inject(Injection { high: false, population: k, pulse, inner: rb - 0.25e-3, radius: rb + 0.25e-3, energy, per_step: 16 }).unwrap();
    let steps = (time / dt) as usize;
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
}

#[test]
fn a_ring_beam_sinks_by_its_space_charge_and_stops_at_the_limit() {
    // Bogdankevich & Rukhadze (Sov. Phys. Usp. 14:163, 1971): γ₀ = 1.25 on a 6 mm ring in a 10 mm
    // tube held by 10 T; at 0.7 of the limiting current it sinks to γ_b (2 % deeper here, the
    // ring 0.5 mm thick), and at 1.3 of it no more than about the limit passes. Faster beams
    // (γ₀ ≳ 1.5) drive the grid's numerical Cherenkov instability on these square cells (cells
    // finer along z: the next test).
    use emcraft_theory::plasma::AnnularBeam;
    let beam = AnnularBeam { gamma0: 1.25, ratio: 10.0 / 6.0, charge: ELECTRON.charge, mass: ELECTRON.mass };
    let il = beam.limiting().unwrap();
    let want = beam.depressed(0.7 * il).unwrap();
    let (g, passed) = ring_beam(1.25, 0.7 * il, 0.08, 0.5e-3, 1, 2e-9);
    assert!(((1.25 - g) / (1.25 - want) - 1.0).abs() < 0.04, "γ_b {g} against {want}");
    assert!((passed / (0.7 * il) - 1.0).abs() < 0.02);
    let (_, over) = ring_beam(1.25, 1.3 * il, 0.08, 0.5e-3, 1, 2e-9);
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
    let gamma0 = 3.0;
    let beam = AnnularBeam { gamma0, ratio: 10.0 / 6.0, charge: ELECTRON.charge, mass: ELECTRON.mass };
    let current = 0.7 * beam.limiting().unwrap();
    let want = gamma0 - beam.depressed(current).unwrap();
    let depression = |fine: usize| (gamma0 - ring_beam(gamma0, current, 0.05, 1e-3, fine, 1.2e-9).0) / want;
    let (square, fine) = std::thread::scope(|s| {
        let square = s.spawn(|| depression(1));
        (square.join().unwrap(), depression(8))
    });
    assert!(square > 1.15, "square cells: {square}");
    assert!((fine - 1.0).abs() < 0.035, "fine along z: {fine}");
}

#[test]
fn one_thread_and_seven_give_the_same_bits() {
    // A ring beam through a corrugated tube with an absorbing end, metal and an axial field: the
    // fields by z rows and the rings by chunks on the threads, their currents added in the rings'
    // order — the run is the same to the bit on any number of threads.
    let run = || {
        let g = AxiGrid::new(26, 120, 0.5e-3, 0.5e-3, Wall::Conductor).unwrap();
        let mut pic = AxiPic::new(g, g.time_step(0.95), Pusher::Boris).unwrap();
        pic.set_metal(|r, z| r > if ((z / 0.01).rem_euclid(1.0) - 0.25).abs() < 0.25 { 0.013 } else { 0.010 });
        pic.absorb(true, 10).unwrap();
        pic.set(5, |_, _| 2.0);
        let k = pic.add(Rings { species: ELECTRON, rings: Vec::new() }).unwrap();
        let pulse = Pulse::Steady { current: 200.0, rise: 0.1e-9 };
        pic.inject(Injection {
            high: false,
            population: k,
            pulse,
            inner: 0.0075,
            radius: 0.0085,
            energy: 0.3 * ELECTRON.rest_energy(),
            per_step: 64,
        })
        .unwrap();
        for _ in 0..400 {
            pic.step().unwrap();
        }
        let (field, kinetic) = pic.energy();
        let rings: Vec<u64> = pic.populations()[k].rings.iter().flat_map(|p| [p.r.to_bits(), p.z.to_bits(), p.u[2].to_bits()]).collect();
        (field.to_bits(), kinetic.to_bits(), rings, pic.gauss().iter().map(|g| g.to_bits()).collect::<Vec<_>>())
    };
    let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(run);
    let many = rayon::ThreadPoolBuilder::new().num_threads(7).build().unwrap().install(run);
    assert!(!one.2.is_empty() && one == many, "one thread and seven differ");
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
        // A pulse that is not a number would push rings of NaN weight (once accepted unchecked).
        Injection { pulse: Pulse::Steady { current: f64::NAN, rise: 0.0 }, ..base },
        Injection { pulse: Pulse::Gaussian { charge: 1e-9, t0: 1e-9, sigma: 0.0 }, ..base },
    ] {
        assert!(pic.inject(bad).is_err(), "{bad:?}");
    }
    assert!(pic.absorb(true, 0).is_err() && pic.absorb(true, 25).is_err());
    // A refused layer leaves none behind: the low end's layer of 12 then fits (once the 25 cells
    // were kept and 12 + 25 refused too).
    assert!(pic.absorb(false, 12).is_ok());
    let periodic = AxiGrid::new(10, 20, 1e-3, 1e-3, Wall::Periodic).unwrap();
    let mut p = AxiPic::new(periodic, periodic.time_step(0.9), Pusher::Boris).unwrap();
    assert!(p.absorb(false, 4).is_err());
    // A field beyond physics drops the ring rather than poisoning the grid.
    pic.set(2, |_, _| f64::INFINITY);
    assert_eq!(pic.step().unwrap().lost, 1);
}
