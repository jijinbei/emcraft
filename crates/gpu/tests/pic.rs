//! The Cartesian particle-in-cell run on a device against the CPU's threads: a beam sent in
//! through a conducting wall of a box periodic across it, past a metal iris and a dielectric
//! slab, in an axial magnetic field, its current smoothed, an absorbing layer at the far wall,
//! with a second population of heavier positive particles already moving; and two drifting
//! populations in a box periodic every way, pushed by Vay's pusher. Double precision to
//! rounding, single to its precision, the same bits twice; the state read back half way without
//! ending the run.

use emcraft_fdtd::pic::{Axis, Bunch, Emission, Emitter, Grid, Medium, Moved, Particle, Pic, Population, Pulse, Wall};
use emcraft_gpu::{Backend, DeviceStep, Gpu, Precision};
use emcraft_model::particles::{ELECTRON, Pusher, Species};
use emcraft_model::units::C0;

/// One device run at a time (CubeCL's CPU runtime keeps threads of its own).
static DEVICE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A positive particle ten times the electron's mass.
const HEAVY: Species = Species { charge: 1.602_176_634e-19, mass: 9.109_383_701_5e-30 };

/// Particles spread through the box (m) with momenta (m/s) from `u`, `n` of them.
fn spread(n: usize, lengths: [f64; 3], w: f64, u: impl Fn(f64) -> [f64; 3]) -> Vec<Particle> {
    (0..n)
        .map(|m| {
            let s = (m as f64 + 0.5) / n as f64;
            let x = [lengths[0] * (0.13 + 0.7 * s), lengths[1] * (0.71 - 0.5 * s), lengths[2] * (0.05 + 0.9 * s)];
            Particle { x, u: u(s), w }
        })
        .collect()
}

/// A box 6 × 6 cells of 0.5 mm across, periodic, and 28 of 0.5 mm along z between conducting
/// walls: a metal iris at z = 8 mm, a slab of εr = 2.5 below it, 0.3 T along z, a layer
/// absorbing at the far wall, one pass of smoothing; 10 A at 100 keV sent in through z− on a
/// disc of 1 mm about the axis, and heavier positive particles already moving.
fn tube() -> Pic {
    let across = Axis { cells: 6, h: 0.5e-3, wall: Wall::Periodic };
    let grid = Grid::new([across, across, Axis { cells: 28, h: 0.5e-3, wall: Wall::Conductor }]).unwrap();
    let mut pic = Pic::new(grid, grid.time_step(0.9), Pusher::Boris).unwrap();
    pic.set_media(|x| {
        let r = (x[0] - 1.5e-3).hypot(x[1] - 1.5e-3);
        if (x[2] - 8e-3).abs() < 0.5e-3 && r > 1.2e-3 {
            Medium::Metal
        } else if (4e-3..6e-3).contains(&x[2]) && x[1] < 1.5e-3 {
            Medium::Dielectric(2.5)
        } else {
            Medium::Dielectric(1.0)
        }
    });
    pic.absorb(2, true, 5).unwrap();
    pic.set_b(2, |_| 0.3);
    pic.smooth(1);
    let beam = pic.add(Population::new(ELECTRON)).unwrap();
    let lengths = grid.lengths();
    pic.add(Population { species: HEAVY, particles: spread(50, lengths, 3e4, |s| [0.1 * C0 * (s - 0.5), -0.05 * C0, 0.3 * C0 * (1.0 - s)]) })
        .unwrap();
    pic.inject(Bunch {
        axis: 2,
        high: false,
        population: beam,
        pulse: Pulse::Steady { current: 10.0, rise: 0.0 },
        centre: [1.5e-3, 1.5e-3],
        radius: 1e-3,
        inner: 0.0,
        energy: 100e3 * -ELECTRON.charge,
        per_step: 4,
    })
    .unwrap();
    pic
}

/// Two populations drifting through each other in a box periodic every way, pushed by Vay's
/// pusher.
fn streams() -> Pic {
    let a = Axis { cells: 5, h: 0.4e-3, wall: Wall::Periodic };
    let grid = Grid::new([a, a, Axis { cells: 16, h: 0.4e-3, wall: Wall::Periodic }]).unwrap();
    let mut pic = Pic::new(grid, grid.time_step(0.9), Pusher::Vay).unwrap();
    pic.set_b(0, |_| 0.1);
    let lengths = grid.lengths();
    pic.add(Population { species: ELECTRON, particles: spread(80, lengths, 2e5, |s| [0.02 * C0, 0.1 * C0 * s, 0.6 * C0]) }).unwrap();
    pic.add(Population { species: ELECTRON, particles: spread(60, lengths, 2e5, |s| [-0.03 * C0 * s, 0.0, -0.5 * C0]) }).unwrap();
    pic
}

const STEPS: usize = 240;

/// Probes (m).
const PROBES: [[f64; 3]; 2] = [[1.0e-3, 1.4e-3, 3e-3], [2.2e-3, 0.6e-3, 6.4e-3]];

/// The step the state is read back at.
const HALF: usize = 120;

/// What a run left: its fields, its particles (each with its population) and the sums of what
/// each step did, E at the probes each step, and its fields and particles after [`HALF`] steps.
struct Ran {
    fields: [Vec<f64>; 6],
    particles: Vec<(u32, Particle)>,
    moved: Moved,
    probes: Vec<[f64; 3]>,
    half: ([Vec<f64>; 6], Vec<(u32, Particle)>),
}

fn state(pic: &Pic) -> ([Vec<f64>; 6], Vec<(u32, Particle)>) {
    let p = pic.plan().unwrap();
    (p.fields.map(<[f64]>::to_vec), p.particles)
}

fn on_cpu(mut pic: Pic) -> Ran {
    let mut moved = Moved::default();
    let (mut probes, mut half) = (Vec::new(), Default::default());
    for n in 0..STEPS {
        moved += &pic.step().unwrap();
        probes.extend(PROBES.map(|x| pic.e_at(x)));
        if n + 1 == HALF {
            half = state(&pic);
        }
    }
    let (fields, particles) = state(&pic);
    Ran { fields, particles, moved, probes, half }
}

/// The run on `backend`, advanced in spans of `span` steps.
fn on_device(mut pic: Pic, backend: Backend, precision: Precision, span: usize) -> Ran {
    let _one = DEVICE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut dev = Gpu { backend, precision }.pic(&mut pic, &PROBES).unwrap();
    let mut steps: Vec<DeviceStep> = Vec::new();
    let mut half = Default::default();
    while steps.len() < STEPS {
        // Spans that stop at HALF, where the state is read back and the run goes on.
        let to = if steps.len() < HALF { HALF } else { STEPS };
        steps.extend(dev.advance(&mut pic, span.min(to - steps.len())).unwrap());
        if steps.len() == HALF {
            dev.fields(&mut pic).unwrap();
            // In their slots' order: population by population, as the run holds them.
            let mut particles = dev.particles().unwrap();
            particles.sort_by_key(|(k, _)| *k);
            half = (state(&pic).0, particles);
        }
    }
    let held = dev.held();
    dev.finish(&mut pic).unwrap();
    let (fields, particles) = state(&pic);
    assert_eq!(held, particles.len());
    let mut moved = Moved::default();
    for s in &steps {
        moved += &s.moved;
    }
    Ran { fields, particles, moved, probes: steps.into_iter().flat_map(|s| s.probes).collect(), half }
}

/// The largest difference of `a` from `b` over the largest |b| of each component, the
/// particles' places over 4 mm and their momenta over c, the probes over their largest.
fn worst(a: &Ran, b: &Ran) -> f64 {
    let rel = |x: &[f64], y: &[f64]| {
        let top = y.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
        x.iter().zip(y).map(|(p, q)| (p - q).abs() / top).fold(0.0, f64::max)
    };
    let particles = |x: &[(u32, Particle)], y: &[(u32, Particle)]| {
        assert_eq!(x.len(), y.len(), "particles");
        x.iter()
            .zip(y)
            .map(|((k, p), (l, q))| {
                assert_eq!(k, l);
                let place = p.x.iter().zip(&q.x).map(|(a, b)| (a - b).abs() / 4e-3).fold(0.0, f64::max);
                p.u.iter().zip(&q.u).map(|(a, b)| (a - b).abs() / C0).fold(place, f64::max)
            })
            .fold(0.0, f64::max)
    };
    let fields = |x: &[Vec<f64>; 6], y: &[Vec<f64>; 6]| x.iter().zip(y).map(|(p, q)| rel(p, q)).fold(0.0, f64::max);
    let flat = |p: &[[f64; 3]]| p.iter().flatten().copied().collect::<Vec<_>>();
    [
        fields(&a.fields, &b.fields),
        particles(&a.particles, &b.particles),
        rel(&flat(&a.probes), &flat(&b.probes)),
        fields(&a.half.0, &b.half.0),
        particles(&a.half.1, &b.half.1),
    ]
    .into_iter()
    .fold(0.0, f64::max)
}

#[test]
fn a_beam_through_a_box_on_the_cpu_backend_matches_the_threads() {
    let cpu = on_cpu(tube());
    let double = on_device(tube(), Backend::Cpu, Precision::Double, 37);
    assert_eq!(double.moved.lost, cpu.moved.lost);
    assert!((double.moved.emitted / cpu.moved.emitted - 1.0).abs() < 1e-12);
    for (a, b) in [(double.moved.collected[2][1], cpu.moved.collected[2][1]), (double.moved.metal, cpu.moved.metal)] {
        assert!((a - b).abs() <= 1e-9 * cpu.moved.emitted.abs(), "{a} against {b}");
    }
    assert!(cpu.moved.metal != 0.0 && cpu.particles.len() > 100, "the beam reaches the iris: {:?}", cpu.moved);
    assert!(worst(&double, &cpu) < 1e-9, "{:e}", worst(&double, &cpu));
}

#[test]
fn streams_in_a_periodic_box_on_the_cpu_backend_match_the_threads() {
    let cpu = on_cpu(streams());
    let double = on_device(streams(), Backend::Cpu, Precision::Double, STEPS);
    assert!(worst(&double, &cpu) < 1e-8, "{:e}", worst(&double, &cpu));
}

#[test]
fn single_precision_matches_the_threads_to_its_precision() {
    let cpu = on_cpu(tube());
    let single = on_device(tube(), Backend::Cpu, Precision::Single, 100);
    assert!(worst(&single, &cpu) < 1e-3, "{:e}", worst(&single, &cpu));
}

#[test]
fn a_run_on_the_device_repeats_bit_for_bit() {
    let [a, b] = [0, 1].map(|_| on_device(tube(), Backend::Cpu, Precision::Single, 64));
    assert!(a.fields == b.fields && a.particles == b.particles && a.moved == b.moved && a.probes == b.probes);
}

#[test]
fn emitting_walls_are_refused_on_a_device() {
    let mut pic = tube();
    let emitter = Emitter { axis: 2, high: false, population: 0, emission: Emission::Current(1e3), energy: 0.0, per_node: 1 };
    pic.emit(emitter).unwrap();
    let _one = DEVICE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    assert!(Gpu { backend: Backend::Cpu, precision: Precision::Double }.pic(&mut pic, &[]).is_err());
}

#[test]
#[ignore = "needs an NVIDIA GPU: cargo test -p emcraft-gpu --features cuda -- --ignored"]
fn cuda_matches_the_threads_in_a_box() {
    let cpu = on_cpu(tube());
    let run = |precision| on_device(tube(), Backend::Cuda, precision, STEPS);
    let double = run(Precision::Double);
    assert!(worst(&double, &cpu) < 1e-9, "{:e}", worst(&double, &cpu));
    let [a, b] = [0, 1].map(|_| run(Precision::Single));
    assert!(worst(&a, &cpu) < 1e-3, "{:e}", worst(&a, &cpu));
    assert!(a.fields == b.fields && a.particles == b.particles && a.moved == b.moved);
}
