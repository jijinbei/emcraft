//! The axisymmetric particle-in-cell run (`emcraft_fdtd::axi`) on a device: the fields, the
//! layers' ψ and the rings held there across the run, five kernels a step (the rings pushed,
//! B's half step, E's step, B's half step, the probes and the taps), the host reading back what
//! left the run, the probes and the taps a batch at a time. The taps are the field's values at
//! points the run names ([`AxiDevice::tap`]: an output plane's, `emcraft_fdtd::axi::PlaneTaps`);
//! [`AxiDevice::fields`] and [`AxiDevice::rings`] read the whole state back at a moment of the
//! run without ending it.
//!
//! **What runs where.** The host makes the injections' rings ([`AxiPic::births_and_tick`], the
//! same quasi-random places as on the CPU) and places them in the device's arrays a batch ahead,
//! each with the step it is born at; it tallies the rings that left (their step, index, wall,
//! weight and energy, which the push writes to a list) in their order, so a step's
//! [`Moved`] is summed as the CPU sums it. A ring that leaves stays in its slot, marked; the
//! slots are compacted (in order) on the host when they run out.
//!
//! **Determinism.** Each ring and each field value is written by one unit; the current's sums
//! are fixed-point integers (`kernels::axi`), the same in any order; the list of the rings that
//! left is sorted by step and index before it is tallied; the kinetic energy is summed in a fixed
//! tree. A run gives the same bits every time on a device and precision. Devices and precisions
//! round differently: their results agree with the CPU's to the precision, not bit for bit.

use cubecl::prelude::*;
use emcraft_fdtd::axi::device::{Plan, State};
use emcraft_fdtd::axi::{AxiPic, Ring};
use emcraft_fdtd::pic::{Energy, Moved};
use emcraft_fdtd::{FdtdError, Result, invalid, to_u32};
use emcraft_model::particles::Pusher;
use emcraft_model::units::{C0, EPS0};

use crate::buffer::{Buf, Ints, Real, Words};
use crate::device::{cube_units, guarded};
use crate::kernels::axi::{
    self as kernels, CurrentLaunch, DeathsLaunch, FieldsLaunch, GradingLaunch, GridLaunch, MetalLaunch, PsiLaunch, RingsLaunch, SpeciesLaunch,
    SteadyLaunch,
};
use crate::lattice::line;
use crate::{BATCH, Gpu, Precision};

/// What a step on the device did: what entered and left the run, E (E_r, E_θ, E_z; V/m) at
/// each probe after it and the field at each tap (V/m or T; B with the steady field).
#[derive(Clone, Debug, PartialEq)]
pub struct AxiStep {
    pub moved: Moved,
    pub probes: Vec<[f64; 3]>,
    pub taps: Vec<f64>,
}

/// An axisymmetric run held on a device: made by [`Gpu::axi`], stepped by
/// [`AxiDevice::advance`], its state given back to the run by [`AxiDevice::finish`].
pub struct AxiDevice {
    run: Box<dyn Ops>,
}

/// A run's work on the device, in the float type its precision picks.
trait Ops {
    fn advance(&mut self, pic: &mut AxiPic, steps: usize) -> Result<Vec<AxiStep>>;
    fn energy(&mut self, pic: &mut AxiPic) -> Result<Energy>;
    fn finish(&mut self, pic: &mut AxiPic) -> Result<()>;
    fn held(&self) -> usize;
    fn tap(&mut self, taps: &[Option<(usize, usize)>]) -> Result<()>;
    fn fields(&self) -> Result<[Vec<f64>; 6]>;
    fn rings(&self) -> Result<Vec<Ring>>;
}

impl Gpu {
    /// `pic` on this device, E watched at `probes` ((r, z), m): its fields, layers and rings
    /// moved there (the run's own rings taken out until [`AxiDevice::finish`]). Refused with
    /// other than one population of rings.
    pub fn axi(self, pic: &mut AxiPic, probes: &[[f64; 2]]) -> Result<AxiDevice> {
        let device = self.backend.device()?;
        guarded(|| {
            let client = device.client();
            let run: Box<dyn Ops> = match self.precision {
                Precision::Single => Box::new(Run::<f32>::new(client, pic, probes)?),
                Precision::Double => Box::new(Run::<f64>::new(client, pic, probes)?),
            };
            pic.take_rings();
            Ok(AxiDevice { run })
        })
    }
}

impl AxiDevice {
    /// `steps` steps of `pic` (its clock and injections kept on the host), what each did.
    pub fn advance(&mut self, pic: &mut AxiPic, steps: usize) -> Result<Vec<AxiStep>> {
        guarded(|| self.run.advance(pic, steps))
    }

    /// The energy on the grid (J): the field's (the fields read back into `pic`) and the rings'.
    pub fn energy(&mut self, pic: &mut AxiPic) -> Result<Energy> {
        guarded(|| self.run.energy(pic))
    }

    /// The rings the device holds.
    pub fn held(&self) -> usize {
        self.run.held()
    }

    /// The field watched from the next step on at `taps`: each its component (0–5 in the order
    /// of `emcraft_fdtd::axi::device::Plan`) and its point there, or none (read as zero).
    /// Refused for a point outside its lattice.
    pub fn tap(&mut self, taps: &[Option<(usize, usize)>]) -> Result<()> {
        guarded(|| self.run.tap(taps))
    }

    /// The fields as they stand, into `pic` (the run goes on on the device).
    pub fn fields(&self, pic: &mut AxiPic) -> Result<()> {
        let fields = guarded(|| self.run.fields())?;
        pic.put_fields(fields)
    }

    /// The rings the device holds, in order (the run goes on on the device).
    pub fn rings(&self) -> Result<Vec<Ring>> {
        guarded(|| self.run.rings())
    }

    /// The fields, the layers' ψ and the rings written back into `pic`.
    pub fn finish(mut self, pic: &mut AxiPic) -> Result<()> {
        guarded(|| self.run.finish(pic))
    }
}

/// The rings' arrays on the device: radius, z, momentum, weight; born, alive.
struct RingBufs<F> {
    f: [Buf<F>; 6],
    born: Ints,
    alive: Ints,
}

impl<F: Real> RingBufs<F> {
    /// `rings` born at `born`, in slots of `capacity` (the rest empty, not alive).
    fn new(client: &Client, rings: &[Ring], born: &[u32], capacity: usize) -> Result<RingBufs<F>> {
        let column = |f: &dyn Fn(&Ring) -> f64| -> Result<Vec<f64>> {
            let mut v = emcraft_solver::memory::with_capacity(capacity, "the rings on the device").map_err(|e| FdtdError::Device(e.to_string()))?;
            v.extend(rings.iter().map(f));
            v.resize(capacity, 0.0);
            Ok(v)
        };
        let cols = [column(&|p| p.r)?, column(&|p| p.z)?, column(&|p| p.u[0])?, column(&|p| p.u[1])?, column(&|p| p.u[2])?, column(&|p| p.w)?];
        let mut b = born.to_vec();
        b.resize(capacity, 0);
        let mut alive = vec![1u32; rings.len()];
        alive.resize(capacity, 0);
        Ok(RingBufs { f: cols.each_ref().map(|c| Buf::new(client, c)), born: Ints::new(client, &b), alive: Ints::new(client, &alive) })
    }

    fn arg(&self) -> RingsLaunch<F> {
        let [r, z, ur, ut, uz, w] = &self.f;
        RingsLaunch::new(r.arg(), z.arg(), ur.arg(), ut.arg(), uz.arg(), w.arg(), self.born.arg(), self.alive.arg())
    }

    /// The first `len` slots' rings still alive, in order.
    fn read(&self, client: &Client, len: usize) -> Result<(Vec<Ring>, Vec<u32>)> {
        let cols: Vec<Vec<f64>> = self.f.iter().map(|b| read_first(b, client, len)).collect::<Result<_>>()?;
        let alive = self.alive.read(client, len)?;
        let born = self.born.read(client, len)?;
        let mut out = emcraft_solver::memory::with_capacity(len, "the rings read back").map_err(|e| FdtdError::Device(e.to_string()))?;
        let mut kept_born = Vec::new();
        for m in (0..len).filter(|&m| alive.get(m) == Some(&1)) {
            let at = |c: usize| cols.get(c).and_then(|v| v.get(m)).copied().unwrap_or(0.0);
            out.push(Ring { r: at(0), z: at(1), u: [at(2), at(3), at(4)], w: at(5) });
            kept_born.push(born.get(m).copied().unwrap_or(0));
        }
        Ok((out, kept_born))
    }
}

/// The parts of E's curl a steady `b0` (B_r, B_z, B_θ; T) makes, as `emcraft_fdtd::axi`'s
/// Ampère takes them (in double precision): ∂z B_θ₀ at E_r's points, ∂z B_r₀ and ∂r B_z₀ at
/// E_θ's, (1/r) ∂r (r B_θ₀) at E_z's (4B_θ₀(r_½)/Δr on the axis); zero where E is not updated.
fn steady_curl(p: &Plan<'_>, b0: &[Vec<f64>; 3]) -> [Vec<f64>; 4] {
    let g = p.grid;
    let (nr, cz) = (g.nr, g.z.cells);
    let [ser, sez, set] = [p.shapes[0], p.shapes[1], p.shapes[2]];
    let [br, bz, bt] = b0;
    let at = |v: &[f64], i: usize, k: usize, n: usize| v.get(i + n * k).copied().unwrap_or(0.0);
    let below = |k: usize| -> Option<usize> {
        match g.z.wall {
            emcraft_fdtd::pic::Wall::Periodic => Some((k + cz - 1) % cz),
            emcraft_fdtd::pic::Wall::Conductor => k.checked_sub(1),
        }
    };
    let on_wall = |k: usize| g.z.wall == emcraft_fdtd::pic::Wall::Conductor && (k == 0 || k == cz);
    let grid_of = |shape: emcraft_fdtd::axi::device::Shape, f: &dyn Fn(usize, usize) -> f64| -> Vec<f64> {
        (0..shape.nz).flat_map(|k| (0..shape.nr).map(move |i| (i, k))).map(|(i, k)| f(i, k)).collect()
    };
    let dz_of = |v: &[f64], n: usize, i: usize, k: usize| match below(k) {
        Some(km) if !on_wall(k) => (at(v, i, k.min(cz - 1), n) - at(v, i, km, n)) / g.z.h,
        _ => 0.0,
    };
    let er_dz = grid_of(ser, &|i, k| dz_of(bt, nr, i, k));
    let et_dz = grid_of(set, &|i, k| dz_of(br, nr + 1, i, k));
    let et_dr = grid_of(set, &|i, k| if i >= 1 && i < nr { (at(bz, i, k, nr) - at(bz, i - 1, k, nr)) / g.dr } else { 0.0 });
    let ez_curl = grid_of(sez, &|i, k| match i {
        0 => 4.0 * at(bt, 0, k, nr) / g.dr,
        i if i < nr => {
            let (rp, rm, r) = ((i as f64 + 0.5) * g.dr, (i as f64 - 0.5) * g.dr, i as f64 * g.dr);
            (rp * at(bt, i, k, nr) - rm * at(bt, i - 1, k, nr)) / (r * g.dr)
        }
        _ => 0.0,
    });
    [er_dz, et_dz, et_dr, ez_curl]
}

/// The first `len` values of `b`.
fn read_first<F: Real>(b: &Buf<F>, client: &Client, len: usize) -> Result<Vec<f64>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let bytes = b.read_bytes(client, 0..len)?;
    Ok(F::from_bytes(&bytes).iter().map(|x| x.wide()).collect())
}

/// The list of rings that left, room for `capacity`.
struct DeathBufs<F> {
    step: Ints,
    slot: Ints,
    how: Ints,
    w: Buf<F>,
    kinetic: Buf<F>,
    count: Ints,
    capacity: usize,
}

impl<F: Real> DeathBufs<F> {
    fn new(client: &Client, capacity: usize) -> DeathBufs<F> {
        let n = capacity.max(1);
        DeathBufs {
            step: Ints::zeros(client, n),
            slot: Ints::zeros(client, n),
            how: Ints::zeros(client, n),
            w: Buf::zeros(client, n),
            kinetic: Buf::zeros(client, n),
            count: Ints::zeros(client, 1),
            capacity: n,
        }
    }

    fn arg(&self) -> DeathsLaunch<F> {
        DeathsLaunch::new(self.step.arg(), self.slot.arg(), self.how.arg(), self.w.arg(), self.kinetic.arg(), self.count.arg())
    }
}

/// A ring that left: its step, slot, how (`kernels::axi::Deaths`), weight and kinetic energy.
#[derive(Clone, Copy)]
struct Death {
    step: u32,
    slot: u32,
    how: u32,
    w: f64,
    kinetic: f64,
}

/// The run's arrays on the device.
struct Run<F> {
    client: Client,
    grid: [f64; 3],
    cells: [u32; 3],
    fields: [Buf<F>; 6],
    current: [Words; 3],
    psi: [Buf<F>; 4],
    ba: [Buf<F>; 2],
    absorbing: bool,
    metal: [Ints; 4],
    any_metal: bool,
    /// The steady B the run started with (B_r, B_z, B_θ; T), held apart on the host, and on
    /// the device with its curl's parts (`kernels::axi::Steady`; one-element stand-ins
    /// without).
    b0: Option<[Vec<f64>; 3]>,
    steady: [Buf<F>; 7],
    rings: RingBufs<F>,
    /// Slots used and their number.
    len: usize,
    capacity: usize,
    deaths: DeathBufs<F>,
    points: Buf<F>,
    probes: usize,
    record: Buf<F>,
    /// The taps' components and points, a mask of those that exist, what the steady field adds
    /// at each, and their values a batch at a time.
    tap_which: Ints,
    tap_at: Ints,
    tap_mask: Vec<bool>,
    tap_steady: Vec<f64>,
    tap_record: Buf<F>,
    out: Buf<F>,
    species: [f64; 3],
    vay: bool,
    /// The largest |charge| (C) a ring has carried.
    qmax: f64,
    held: usize,
    /// Steps run on the device.
    done: u32,
    units: u32,
}

/// The fewest ring slots the device holds.
const MIN_SLOTS: usize = 1024;

/// The fixed-point sums' range: no point's sum reaches 2^60 (of i64's 2^63).
const RANGE: f64 = (1u64 << 60) as f64;

/// The most units a kernel's cube uses.
const UNITS: u32 = 256;

impl<F: Real> Run<F> {
    fn new(client: Client, pic: &AxiPic, probes: &[[f64; 2]]) -> Result<Run<F>> {
        let p: Plan<'_> = pic.plan()?;
        let g = p.grid;
        let cells = [to_u32(g.nr, "cells along r")?, to_u32(g.z.cells, "cells along z")?, u32::from(g.z.wall == emcraft_fdtd::pic::Wall::Periodic)];
        // Indices must fit the kernels' 32 bits.
        to_u32((g.nr + 2).saturating_mul(g.z.cells + 2), "points")?;
        let b0 =
            p.fields[3..].iter().any(|b| b.iter().any(|&x| x != 0.0)).then(|| [p.fields[3].to_vec(), p.fields[4].to_vec(), p.fields[5].to_vec()]);
        let fields: [Buf<F>; 6] = std::array::from_fn(|c| match (&b0, p.fields.get(c)) {
            (Some(_), Some(f)) if c >= 3 => Buf::zeros(&client, f.len()),
            (_, Some(f)) => Buf::new(&client, f),
            (_, None) => Buf::zeros(&client, 1),
        });
        let steady = match &b0 {
            Some(b) => {
                let [er_dz, et_dz, et_dr, ez_curl] = steady_curl(&p, b);
                [&b[0], &b[1], &b[2], &er_dz, &et_dz, &et_dr, &ez_curl].map(|v| Buf::new(&client, v))
            }
            None => std::array::from_fn(|_| Buf::zeros(&client, 1)),
        };
        let current = [p.shapes[0].len(), p.shapes[1].len(), p.shapes[2].len()].map(|n| Words::zeros(&client, n.max(1)));
        let (psi, ba, absorbing) = match &p.layers {
            Some(l) => {
                let flat = |c: &[emcraft_fdtd::Cpml]| -> Vec<f64> { c.iter().flat_map(|&c| <[f64; 2]>::from(c)).collect() };
                (l.psi.map(|v| Buf::new(&client, v)), [Buf::new(&client, &flat(l.e[0])), Buf::new(&client, &flat(l.b[1]))], true)
            }
            None => (std::array::from_fn(|_| Buf::zeros(&client, 1)), [Buf::new(&client, &[1.0, 0.0]), Buf::new(&client, &[1.0, 0.0])], false),
        };
        let flags = |b: &[bool]| -> Vec<u32> { if b.is_empty() { vec![0] } else { b.iter().map(|&x| u32::from(x)).collect() } };
        let any_metal = !p.metal.is_empty();
        let metal = [flags(p.held[0]), flags(p.held[1]), flags(p.held[2]), flags(p.metal)].map(|v| Ints::new(&client, &v));
        let capacity = (2 * p.rings.len()).max(MIN_SLOTS);
        let rings = RingBufs::new(&client, p.rings, &vec![0; p.rings.len()], capacity)?;
        let qmax = p.rings.iter().map(|r| (p.species.charge * r.w).abs()).fold(0.0, f64::max);
        let points: Vec<f64> = probes.iter().flatten().copied().collect();
        let ok = |x: f64| x.is_finite();
        if !points.iter().all(|&x| ok(x)) {
            return Err(invalid("probe", "a point of finite r and z"));
        }
        let n = probes.len();
        Ok(Run {
            grid: [g.dr, g.z.h, p.dt],
            cells,
            fields,
            current,
            psi,
            ba,
            absorbing,
            metal,
            any_metal,
            b0,
            steady,
            len: p.rings.len(),
            capacity,
            deaths: DeathBufs::new(&client, capacity),
            rings,
            points: if n == 0 { Buf::zeros(&client, 2) } else { Buf::new(&client, &points) },
            probes: n,
            record: Buf::zeros(&client, (BATCH * n * 3).max(1)),
            tap_which: Ints::zeros(&client, 1),
            tap_at: Ints::zeros(&client, 1),
            tap_mask: Vec::new(),
            tap_steady: Vec::new(),
            tap_record: Buf::zeros(&client, 1),
            out: Buf::zeros(&client, 1),
            species: [p.species.charge * p.dt / p.species.mass, p.species.charge, p.species.rest_energy()],
            vay: p.pusher == Pusher::Vay,
            qmax,
            held: p.rings.len(),
            done: 0,
            units: cube_units(&client, UNITS),
            client,
        })
    }

    fn grid_arg(&self) -> GridLaunch<F> {
        let [dr, dz, dt] = self.grid;
        let [nr, nz, periodic] = self.cells;
        GridLaunch::new(nr, nz, periodic, F::of(dr), F::of(dz), F::of(dt), F::of(C0), F::of(std::f64::consts::PI))
    }

    fn fields_arg(&self) -> FieldsLaunch<F> {
        let [er, ez, et, br, bz, bt] = &self.fields;
        FieldsLaunch::new(er.arg(), ez.arg(), et.arg(), br.arg(), bz.arg(), bt.arg())
    }

    fn psi_arg(&self) -> PsiLaunch<F> {
        let [a, b, c, d] = &self.psi;
        PsiLaunch::new(a.arg(), b.arg(), c.arg(), d.arg())
    }

    fn grading_arg(&self) -> GradingLaunch<F> {
        GradingLaunch::new(self.ba[0].arg(), self.ba[1].arg())
    }

    fn metal_arg(&self) -> MetalLaunch {
        let [a, b, c, d] = &self.metal;
        MetalLaunch::new(a.arg(), b.arg(), c.arg(), d.arg())
    }

    fn steady_arg(&self) -> SteadyLaunch<F> {
        let [a, b, c, d, e, f, g] = &self.steady;
        SteadyLaunch::new(a.arg(), b.arg(), c.arg(), d.arg(), e.arg(), f.arg(), g.arg())
    }

    fn species_arg(&self) -> SpeciesLaunch<F> {
        let [kick, charge, rest] = self.species;
        SpeciesLaunch::new(F::of(kick), F::of(charge), F::of(rest))
    }

    /// B advanced by `dt` (s).
    fn faraday(&self, cubes: CubeCount, dim: CubeDim, dt: F) {
        kernels::faraday::launch::<F>(
            &self.client,
            cubes,
            dim,
            self.fields_arg(),
            self.psi_arg(),
            self.grading_arg(),
            self.grid_arg(),
            dt,
            self.absorbing,
        );
    }

    /// The fixed-point scales of J_r, J_z, J_θ: no point's sum, of at most 4 additions a ring in
    /// each of `slots`, each at most the largest charge's through the smallest face, reaches
    /// [`RANGE`].
    fn scales(&self, slots: usize) -> [f64; 3] {
        let [dr, dz, dt] = self.grid;
        let pi = std::f64::consts::PI;
        let most = 4.0 * slots.max(1) as f64 * self.qmax / dt;
        let faces = [2.0 * pi * 0.5 * dr * dz, pi * 0.25 * dr * dr, 2.0 * pi * dr * dr * dz / (C0 * dt)];
        faces.map(|face| {
            let bound = most / face;
            if bound > 0.0 && bound.is_finite() { RANGE / bound } else { 1.0 }
        })
    }

    fn current_arg(&self, scales: [f64; 3]) -> CurrentLaunch<F> {
        let [r, z, t] = &self.current;
        CurrentLaunch::new(r.arg(), z.arg(), t.arg(), F::of(scales[0]), F::of(scales[1]), F::of(scales[2]))
    }

    /// The slots compacted, alive rings in order, into room for `need` more.
    fn compact(&mut self, need: usize) -> Result<()> {
        let (rings, born) = self.rings.read(&self.client, self.len)?;
        let capacity = (2 * (rings.len() + need)).max(MIN_SLOTS);
        self.rings = RingBufs::new(&self.client, &rings, &born, capacity)?;
        self.len = rings.len();
        self.capacity = capacity;
        self.deaths = DeathBufs::new(&self.client, capacity);
        Ok(())
    }

    /// `rings` placed after the used slots, born at `born`.
    fn place(&mut self, rings: &[Ring], born: &[u32]) -> Result<()> {
        if rings.is_empty() {
            return Ok(());
        }
        let stage = RingBufs::<F>::new(&self.client, rings, born, rings.len())?;
        let count = to_u32(rings.len(), "rings")?;
        let (cubes, dim) = line(count, self.units);
        kernels::place::launch::<F>(&self.client, cubes, dim, self.rings.arg(), stage.arg(), to_u32(self.len, "rings")?, count);
        self.len += rings.len();
        Ok(())
    }

    /// One batch of `steps` (at most [`BATCH`]).
    fn batch(&mut self, pic: &mut AxiPic, steps: usize) -> Result<Vec<AxiStep>> {
        // The injections' rings, made on the host a step at a time.
        let mut entered = Vec::with_capacity(steps);
        let (mut born, mut at) = (Vec::new(), Vec::new());
        for b in 0..steps {
            let (moved, new) = pic.births_and_tick(self.held + born.len())?;
            let step = self.done.checked_add(to_u32(b, "steps")?).ok_or_else(|| invalid("steps", "more than 32 bits count"))?;
            at.extend(std::iter::repeat_n(step, new.len()));
            born.extend(new);
            entered.push(moved);
        }
        if self.len + born.len() > self.capacity {
            self.compact(born.len())?;
        }
        let charge = self.species[1];
        self.qmax = born.iter().map(|r| (charge * r.w).abs()).fold(self.qmax, f64::max);
        self.place(&born, &at)?;
        let scales = self.scales(self.capacity);
        let len = to_u32(self.len, "rings")?;
        let [nr, nz, _] = self.cells;
        let points = (nr + 1) * (nz + 1);
        let (over_points, dim) = line(points, self.units);
        let (over_rings, ring_dim) = line(len.max(1), self.units);
        let (over_probes, probe_dim) = line(to_u32(self.probes.max(1), "probes")?, self.units);
        let taps = to_u32(self.tap_mask.len(), "taps")?;
        let (over_taps, tap_dim) = line(taps.max(1), self.units);
        let half = F::of(0.5 * self.grid[2]);
        for b in 0..steps {
            let step = self.done + b as u32;
            kernels::push_rings::launch::<F>(
                &self.client,
                over_rings.clone(),
                ring_dim,
                self.rings.arg(),
                self.fields_arg(),
                self.steady_arg(),
                self.grid_arg(),
                self.species_arg(),
                self.metal_arg(),
                self.current_arg(scales),
                self.deaths.arg(),
                step,
                len,
                self.vay,
                self.any_metal,
                self.b0.is_some(),
            );
            self.faraday(over_points.clone(), dim, half);
            kernels::ampere::launch::<F>(
                &self.client,
                over_points.clone(),
                dim,
                self.fields_arg(),
                self.psi_arg(),
                self.grading_arg(),
                self.current_arg(scales),
                self.metal_arg(),
                self.steady_arg(),
                self.grid_arg(),
                F::of(C0 * C0),
                F::of(EPS0),
                self.absorbing,
                self.any_metal,
                self.b0.is_some(),
            );
            self.faraday(over_points.clone(), dim, half);
            if self.probes > 0 {
                kernels::probe::launch::<F>(
                    &self.client,
                    over_probes.clone(),
                    probe_dim,
                    self.fields_arg(),
                    self.grid_arg(),
                    self.points.arg(),
                    self.record.arg(),
                    b as u32,
                    self.probes as u32,
                );
            }
            if taps > 0 {
                kernels::tap::launch::<F>(
                    &self.client,
                    over_taps.clone(),
                    tap_dim,
                    self.fields_arg(),
                    self.tap_which.arg(),
                    self.tap_at.arg(),
                    self.tap_record.arg(),
                    b as u32,
                    taps,
                );
            }
        }
        self.done = self.done.checked_add(to_u32(steps, "steps")?).ok_or_else(|| invalid("steps", "more than 32 bits count"))?;
        let deaths = self.deaths()?;
        let record = if self.probes > 0 { read_first(&self.record, &self.client, steps * self.probes * 3)? } else { Vec::new() };
        let tapped = if taps > 0 { read_first(&self.tap_record, &self.client, steps * self.tap_mask.len())? } else { Vec::new() };
        self.held += born.len();
        self.held = self.held.saturating_sub(deaths.len());
        Ok(self.tally(entered, &deaths, &record, &tapped))
    }

    /// The rings that left this batch, by step and slot, the list cleared.
    fn deaths(&mut self) -> Result<Vec<Death>> {
        let d = &self.deaths;
        let count = d.count.read(&self.client, 1)?.first().copied().unwrap_or(0) as usize;
        if count > d.capacity {
            return Err(FdtdError::Device(format!("{count} rings left, more than the list holds")));
        }
        let (step, slot, how) = (d.step.read(&self.client, count)?, d.slot.read(&self.client, count)?, d.how.read(&self.client, count)?);
        let (w, kinetic) = (read_first(&d.w, &self.client, count)?, read_first(&d.kinetic, &self.client, count)?);
        let mut out: Vec<Death> = (0..count)
            .map(|m| Death {
                step: step.get(m).copied().unwrap_or(0),
                slot: slot.get(m).copied().unwrap_or(0),
                how: how.get(m).copied().unwrap_or(0),
                w: w.get(m).copied().unwrap_or(0.0),
                kinetic: kinetic.get(m).copied().unwrap_or(0.0),
            })
            .collect();
        out.sort_unstable_by_key(|d| (d.step, d.slot));
        self.deaths.count = Ints::zeros(&self.client, 1);
        Ok(out)
    }

    /// Each step's [`AxiStep`]: what entered, the rings that left tallied in their order (as
    /// `emcraft_fdtd::pic::Moved::tally`), the probes, the taps (zero where there is no point,
    /// B with the steady field).
    fn tally(&self, entered: Vec<Moved>, deaths: &[Death], record: &[f64], tapped: &[f64]) -> Vec<AxiStep> {
        let n = self.tap_mask.len();
        let charge = self.species[1];
        let first = self.done - entered.len() as u32;
        let mut deaths = deaths.iter().peekable();
        entered
            .into_iter()
            .enumerate()
            .map(|(b, mut moved)| {
                let step = first + b as u32;
                while let Some(d) = deaths.next_if(|d| d.step == step) {
                    let q = charge * d.w;
                    match d.how {
                        0 => moved.lost += 1,
                        1 => moved.collected[0][1] += q,
                        2 => moved.collected[2][0] += q,
                        3 => moved.collected[2][1] += q,
                        _ => moved.metal += q,
                    }
                    if d.how != 0 {
                        moved.kinetic_out += d.kinetic;
                    }
                }
                let probes = (0..self.probes)
                    .map(|p| std::array::from_fn(|c| record.get(3 * (b * self.probes + p) + c).copied().unwrap_or(f64::NAN)))
                    .collect();
                let taps = (0..n)
                    .map(|t| match self.tap_mask.get(t) {
                        Some(true) => tapped.get(b * n + t).copied().unwrap_or(f64::NAN) + self.tap_steady.get(t).copied().unwrap_or(0.0),
                        _ => 0.0,
                    })
                    .collect();
                AxiStep { moved, probes, taps }
            })
            .collect()
    }

    /// The fields read back, B the steady field and its changes.
    fn read_fields(&self) -> Result<[Vec<f64>; 6]> {
        let mut out: [Vec<f64>; 6] = Default::default();
        for (o, b) in out.iter_mut().zip(&self.fields) {
            *o = b.read(&self.client)?;
        }
        if let Some(b0) = &self.b0 {
            for (o, b) in out[3..].iter_mut().zip(b0) {
                o.iter_mut().zip(b).for_each(|(x, y)| *x += y);
            }
        }
        Ok(out)
    }
}

impl<F: Real> Ops for Run<F> {
    fn advance(&mut self, pic: &mut AxiPic, steps: usize) -> Result<Vec<AxiStep>> {
        let mut out = Vec::with_capacity(steps.min(emcraft_fdtd::MAX_STEPS));
        let mut left = steps;
        while left > 0 {
            let n = left.min(BATCH);
            out.extend(self.batch(pic, n)?);
            left -= n;
        }
        Ok(out)
    }

    fn energy(&mut self, pic: &mut AxiPic) -> Result<Energy> {
        pic.put_fields(self.read_fields()?)?;
        let field = pic.energy();
        let width = self.units as usize;
        let len = to_u32(self.len, "rings")?;
        kernels::kinetic::launch::<F>(
            &self.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(self.units),
            self.rings.arg(),
            self.species_arg(),
            self.grid_arg(),
            self.out.arg(),
            len,
            self.done,
            width,
        );
        let kinetic = self.out.read(&self.client)?.first().copied().unwrap_or(f64::NAN);
        Ok(Energy { kinetic, ..field })
    }

    fn finish(&mut self, pic: &mut AxiPic) -> Result<()> {
        let fields = self.read_fields()?;
        let psi = if self.absorbing {
            let [a, b, c, d] = &self.psi;
            Some([a.read(&self.client)?, b.read(&self.client)?, c.read(&self.client)?, d.read(&self.client)?])
        } else {
            None
        };
        let (rings, _) = self.rings.read(&self.client, self.len)?;
        pic.put_back(State { fields, psi, rings })
    }

    fn held(&self) -> usize {
        self.held
    }

    fn tap(&mut self, taps: &[Option<(usize, usize)>]) -> Result<()> {
        let lens: Vec<usize> = self.fields.iter().map(Buf::len).collect();
        if taps.iter().flatten().any(|&(which, at)| lens.get(which).is_none_or(|&n| at >= n)) {
            return Err(invalid("taps", "points on the fields' lattices"));
        }
        let (which, at): (Vec<u32>, Vec<u32>) = taps.iter().map(|t| t.map_or((0, 0), |(w, a)| (w as u32, a as u32))).unzip();
        self.tap_mask = taps.iter().map(Option::is_some).collect();
        self.tap_steady = taps
            .iter()
            .map(|t| match (t, &self.b0) {
                (Some((w, a)), Some(b0)) if *w >= 3 => b0.get(w - 3).and_then(|b| b.get(*a)).copied().unwrap_or(0.0),
                _ => 0.0,
            })
            .collect();
        let n = taps.len().max(1);
        self.tap_which = Ints::new(&self.client, if which.is_empty() { &[0] } else { &which });
        self.tap_at = Ints::new(&self.client, if at.is_empty() { &[0] } else { &at });
        self.tap_record = Buf::zeros(&self.client, BATCH * n);
        Ok(())
    }

    fn fields(&self) -> Result<[Vec<f64>; 6]> {
        self.read_fields()
    }

    fn rings(&self) -> Result<Vec<Ring>> {
        Ok(self.rings.read(&self.client, self.len)?.0)
    }
}
