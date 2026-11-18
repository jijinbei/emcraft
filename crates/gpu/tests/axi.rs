//! The axisymmetric particle-in-cell run on a device against the CPU's threads: a ring beam
//! injected through a conducting end into a tube with a metal iris, an axial magnetic field and
//! an absorbing layer, and moving rings between periodic ends; double precision to rounding,
//! single to its precision, the same bits twice. With them the power through a plane across the
//! tube each step (the plane's fields tapped on the device) and the fields and rings read back
//! half way without ending the run.

use emcraft_fdtd::axi::{AxiGrid, AxiPic, Component, Injection, Ring, Rings};
use emcraft_fdtd::pic::{Axis, Moved, Pulse, Wall};
use emcraft_gpu::{Backend, DeviceStep, Gpu, Precision};
use emcraft_model::particles::{ELECTRON, Pusher};
use emcraft_model::units::C0;

/// One device run at a time (CubeCL's CPU runtime keeps threads of its own).
static DEVICE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A tube of 8 × 0.5 mm cells across and 48 × 0.25 mm along, closed by conducting ends: a metal
/// iris half way, 0.4 T along z, a layer absorbing at the far end, 20 A at 100 keV sent in on a
/// ring 1–2.5 mm from the axis, and some rings already moving across it.
fn tube() -> AxiPic {
    let grid = AxiGrid::new(8, 0.5e-3, Axis { cells: 48, h: 0.25e-3, wall: Wall::Conductor }).unwrap();
    let mut pic = AxiPic::new(grid, grid.time_step(0.9), Pusher::Boris).unwrap();
    pic.set_metal(|r, z| r > 3e-3 && (z - 6e-3).abs() < 0.6e-3);
    pic.absorb(true, 6).unwrap();
    pic.set(Component::Bz, |_, _| 0.4);
    let moving = (0..40)
        .map(|n| {
            let s = n as f64 / 40.0;
            Ring { r: 0.3e-3 + 3.2e-3 * s, z: 1e-3 + 4e-3 * s, u: [0.2 * C0 * (s - 0.5), 0.1 * C0, 0.4 * C0], w: 2e5 }
        })
        .collect();
    let k = pic.add(Rings { species: ELECTRON, rings: moving }).unwrap();
    let energy = 100e3 * -ELECTRON.charge;
    pic.inject(Injection {
        high: false,
        population: k,
        pulse: Pulse::Steady { current: 20.0, rise: 0.0 },
        inner: 1e-3,
        radius: 2.5e-3,
        energy,
        per_step: 4,
    })
    .unwrap();
    pic
}

/// Rings between periodic ends, pushed by Vay's pusher: no walls along z.
fn periodic() -> AxiPic {
    let grid = AxiGrid::new(6, 0.5e-3, Axis { cells: 24, h: 0.5e-3, wall: Wall::Periodic }).unwrap();
    let mut pic = AxiPic::new(grid, grid.time_step(0.9), Pusher::Vay).unwrap();
    pic.set(Component::Bz, |_, _| 0.2);
    let rings = (0..60)
        .map(|n| {
            let s = n as f64 / 60.0;
            Ring { r: 0.2e-3 + 2.5e-3 * s, z: 12e-3 * s, u: [0.05 * C0, -0.3 * C0 * s, 0.8 * C0 * (1.0 - 2.0 * s)], w: 5e5 }
        })
        .collect();
    pic.add(Rings { species: ELECTRON, rings }).unwrap();
    pic
}

const STEPS: usize = 300;

/// Probes (r, z; m).
const PROBES: [[f64; 2]; 2] = [[0.5e-3, 4e-3], [2.0e-3, 9e-3]];

/// The plane across z whose power is taken each step (m), and the step the state is read at.
const PLANE: f64 = 9e-3;
const HALF: usize = 150;

/// What a run left: its fields, its rings and the sums of what each step did, the power through
/// [`PLANE`] each step (W), and its fields and rings after [`HALF`] steps.
struct Ran {
    fields: [Vec<f64>; 6],
    rings: Vec<Ring>,
    moved: Moved,
    probes: Vec<[f64; 3]>,
    power: Vec<f64>,
    half: ([Vec<f64>; 6], Vec<Ring>),
}

fn fields(pic: &AxiPic) -> [Vec<f64>; 6] {
    pic.plan().unwrap().fields.map(<[f64]>::to_vec)
}

fn on_cpu(mut pic: AxiPic) -> Ran {
    let mut moved = Moved::default();
    let (mut probes, mut power, mut half) = (Vec::new(), Vec::new(), Default::default());
    for n in 0..STEPS {
        moved += &pic.step().unwrap();
        probes.extend(PROBES.map(|[r, z]| pic.field_at(r, z).0));
        power.push(pic.plane(PLANE).flux());
        if n + 1 == HALF {
            half = (fields(&pic), pic.plan().unwrap().rings.to_vec());
        }
    }
    Ran { fields: fields(&pic), rings: pic.plan().unwrap().rings.to_vec(), moved, probes, power, half }
}

/// The run on a device, advanced in spans of `span` steps.
fn on_device(pic: AxiPic, precision: Precision, span: usize) -> Ran {
    on_device_of(pic, Backend::Cpu, precision, span)
}

/// The run on `backend`, advanced in spans of `span` steps.
fn on_device_of(mut pic: AxiPic, backend: Backend, precision: Precision, span: usize) -> Ran {
    let _one = DEVICE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let taps = pic.plane_taps(PLANE);
    let mut dev = Gpu { backend, precision }.axi(&mut pic, &PROBES).unwrap();
    dev.tap(&taps.taps).unwrap();
    let mut steps: Vec<DeviceStep> = Vec::new();
    let mut half = Default::default();
    while steps.len() < STEPS {
        // Spans that stop at HALF, where the state is read back and the run goes on.
        let to = if steps.len() < HALF { HALF } else { STEPS };
        steps.extend(dev.advance(&mut pic, span.min(to - steps.len())).unwrap());
        if steps.len() == HALF {
            dev.fields(&mut pic).unwrap();
            half = (fields(&pic), dev.rings().unwrap());
        }
    }
    let held = dev.held();
    dev.finish(&mut pic).unwrap();
    let rings = pic.plan().unwrap().rings.to_vec();
    assert_eq!(held, rings.len());
    let mut moved = Moved::default();
    for s in &steps {
        moved += &s.moved;
    }
    let power = steps.iter().map(|s| taps.plane(&s.taps).flux()).collect();
    Ran { fields: fields(&pic), rings, moved, probes: steps.into_iter().flat_map(|s| s.probes).collect(), power, half }
}

/// The largest difference of `a` from `b` over the largest |b| of each component, the rings'
/// places over the tube's radius and their momenta over c, the probes over their largest.
fn worst(a: &Ran, b: &Ran) -> f64 {
    let rel = |x: &[f64], y: &[f64]| {
        let top = y.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
        x.iter().zip(y).map(|(p, q)| (p - q).abs() / top).fold(0.0, f64::max)
    };
    assert_eq!(a.rings.len(), b.rings.len(), "rings");
    let fields = a.fields.iter().zip(&b.fields).map(|(x, y)| rel(x, y)).fold(0.0, f64::max);
    let rings = a
        .rings
        .iter()
        .zip(&b.rings)
        .map(|(p, q)| {
            ((p.r - q.r).abs().max((p.z - q.z).abs()) / 4e-3).max(p.u.iter().zip(&q.u).map(|(x, y)| (x - y).abs() / C0).fold(0.0, f64::max))
        })
        .fold(0.0, f64::max);
    let flat = |p: &[[f64; 3]]| p.iter().flatten().copied().collect::<Vec<_>>();
    let probes = rel(&flat(&a.probes), &flat(&b.probes));
    let power = rel(&a.power, &b.power);
    let (fa, fb) = (&a.half.0, &b.half.0);
    assert_eq!(a.half.1.len(), b.half.1.len(), "rings half way");
    let half = fa.iter().zip(fb).map(|(x, y)| rel(x, y)).fold(0.0, f64::max);
    let half_rings = a
        .half
        .1
        .iter()
        .zip(&b.half.1)
        .map(|(p, q)| {
            ((p.r - q.r).abs().max((p.z - q.z).abs()) / 4e-3).max(p.u.iter().zip(&q.u).map(|(x, y)| (x - y).abs() / C0).fold(0.0, f64::max))
        })
        .fold(0.0, f64::max);
    fields.max(rings).max(probes).max(power).max(half).max(half_rings)
}

#[test]
fn a_beam_through_a_tube_on_the_cpu_backend_matches_the_threads() {
    let cpu = on_cpu(tube());
    let double = on_device(tube(), Precision::Double, 37);
    // The same rings in, out and through: the arithmetic in another order (the current's
    // fixed-point sums, fused multiply–adds) rounding apart.
    assert_eq!(double.moved.lost, cpu.moved.lost);
    assert!((double.moved.emitted / cpu.moved.emitted - 1.0).abs() < 1e-12);
    for (a, b) in [(double.moved.collected[2][1], cpu.moved.collected[2][1]), (double.moved.metal, cpu.moved.metal)] {
        assert!((a - b).abs() <= 1e-9 * cpu.moved.emitted.abs(), "{a} against {b}");
    }
    assert!(cpu.moved.metal != 0.0 && cpu.rings.len() > 100, "the beam reaches the iris: {:?}", cpu.moved);
    // The beam's own field crosses the plane (with the steady B_r of none here, B_θ's).
    assert!(cpu.power.iter().any(|p| p.abs() > 0.0) && !cpu.half.1.is_empty());
    assert!(worst(&double, &cpu) < 1e-9, "{:e}", worst(&double, &cpu));
}

#[test]
fn rings_between_periodic_ends_on_the_cpu_backend_match_the_threads() {
    let cpu = on_cpu(periodic());
    let double = on_device(periodic(), Precision::Double, STEPS);
    // E_θ and B_r, small here (the rings' turning alone drives them), carry the current's
    // fixed-point rounding relative to their own size: 1e-9 of it after 300 steps.
    assert!(worst(&double, &cpu) < 1e-8, "{:e}", worst(&double, &cpu));
}

#[test]
fn single_precision_matches_the_threads_to_its_precision() {
    let cpu = on_cpu(tube());
    let single = on_device(tube(), Precision::Single, 100);
    assert!(worst(&single, &cpu) < 1e-3, "{:e}", worst(&single, &cpu));
}

#[test]
fn a_run_on_the_device_repeats_bit_for_bit() {
    // The current's sums are integers and the rings that left are sorted before they are
    // tallied: the same bits however the device's units interleave.
    let [a, b] = [0, 1].map(|_| on_device(tube(), Precision::Single, 64));
    assert!(a.fields == b.fields && a.rings == b.rings && a.moved == b.moved && a.probes == b.probes && a.power == b.power);
}

#[test]
#[ignore = "needs an NVIDIA GPU: cargo test -p emcraft-gpu --features cuda -- --ignored"]
fn cuda_matches_the_threads_in_a_body_of_revolution() {
    let cpu = on_cpu(tube());
    let run = |precision| on_device_of(tube(), Backend::Cuda, precision, STEPS);
    let double = run(Precision::Double);
    assert!(worst(&double, &cpu) < 1e-9, "{:e}", worst(&double, &cpu));
    let [a, b] = [0, 1].map(|_| run(Precision::Single));
    assert!(worst(&a, &cpu) < 1e-3, "{:e}", worst(&a, &cpu));
    assert!(a.fields == b.fields && a.rings == b.rings && a.moved == b.moved);
}
