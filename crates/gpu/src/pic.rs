//! The Cartesian particle-in-cell run (`emcraft_fdtd::pic`) on a device: the fields, the
//! layers' ψ and the particles held there across the run, the kernels of a step (the particles
//! pushed, the current turned back from fixed point and smoothed, B's half step, E's step, B's
//! half step, the probes), the host reading back what left the run and the probes a batch at a
//! time. [`PicDevice::fields`] and [`PicDevice::particles`] read the state back at a moment of
//! the run without ending it.
//!
//! **What runs where.** The host makes the bunches' and beams' particles
//! ([`Pic::births_and_tick`], the same quasi-random places as on the CPU); the slots they go
//! into, the list of those that left and its tally, and the probes' records are
//! `crate::particles`', as for every grid. Emitting walls and a held voltage are refused
//! (`emcraft_fdtd::pic::device`), and the charge particles leave on the walls is not kept (it
//! only checks Gauss's law on the CPU).
//!
//! **Determinism.** Each particle and each field value is written by one unit; the current's
//! sums are fixed-point integers (`kernels::particles`), the same in any order; the list of
//! those that left is sorted by step and slot before it is tallied; the kinetic energy is summed
//! in a fixed tree. A run gives the same bits every time on a device and precision. Devices and
//! precisions round differently: their results agree with the CPU's to the precision, not bit
//! for bit.

use cubecl::prelude::*;
use emcraft_fdtd::pic::device::{Plan, State};
use emcraft_fdtd::pic::{Energy, Particle, Pic, Wall};
use emcraft_fdtd::{Cpml, Result, invalid, to_u32};
use emcraft_model::particles::Pusher;
use emcraft_model::units::{C0, EPS0, MU0};

use crate::buffer::{Buf, Ints, Real, Words};
use crate::device::{cube_units, guarded};
use crate::kernels::pic::{self as kernels, CurrentLaunch, FieldsLaunch, GradingLaunch, GridLaunch, KindsLaunch, ParticlesLaunch, SteadyLaunch};
use crate::lattice::line;
use crate::particles::{DeviceStep, Held, Item, scale};
use crate::{Gpu, Precision};

/// A Cartesian run held on a device: made by [`Gpu::pic`], stepped by [`PicDevice::advance`],
/// its state given back to the run by [`PicDevice::finish`].
pub struct PicDevice {
    run: Box<dyn Ops>,
}

/// A run's work on the device, in the float type its precision picks.
trait Ops {
    fn advance(&mut self, pic: &mut Pic, steps: usize) -> Result<Vec<DeviceStep>>;
    fn energy(&mut self, pic: &mut Pic) -> Result<Energy>;
    fn finish(&mut self, pic: &mut Pic) -> Result<()>;
    fn held(&self) -> usize;
    fn fields(&self) -> Result<[Vec<f64>; 6]>;
    fn particles(&self) -> Result<Vec<(u32, Particle)>>;
}

impl Gpu {
    /// `pic` on this device, E watched at `probes` (m): its fields, layers and particles moved
    /// there (the run's own particles taken out until [`PicDevice::finish`]). Refused with
    /// emitting walls or a held voltage.
    pub fn pic(self, pic: &mut Pic, probes: &[[f64; 3]]) -> Result<PicDevice> {
        let device = self.backend.device()?;
        guarded(|| {
            let client = device.client();
            let run: Box<dyn Ops> = match self.precision {
                Precision::Single => Box::new(Run::<f32>::new(client, pic, probes)?),
                Precision::Double => Box::new(Run::<f64>::new(client, pic, probes)?),
            };
            pic.take_particles();
            Ok(PicDevice { run })
        })
    }
}

impl PicDevice {
    /// `steps` steps of `pic` (its clock and bunches kept on the host), what each did: E at the
    /// probes (V/m) after it; no taps.
    pub fn advance(&mut self, pic: &mut Pic, steps: usize) -> Result<Vec<DeviceStep>> {
        guarded(|| self.run.advance(pic, steps))
    }

    /// The energy on the grid (J): the field's and the particles', summed on the device.
    pub fn energy(&mut self, pic: &mut Pic) -> Result<Energy> {
        guarded(|| self.run.energy(pic))
    }

    /// The particles the device holds.
    pub fn held(&self) -> usize {
        self.run.held()
    }

    /// The fields as they stand, into `pic` (the run goes on on the device).
    pub fn fields(&self, pic: &mut Pic) -> Result<()> {
        let fields = guarded(|| self.run.fields())?;
        pic.put_fields(fields)
    }

    /// The particles the device holds, each with its population, in order (the run goes on on
    /// the device).
    pub fn particles(&self) -> Result<Vec<(u32, Particle)>> {
        guarded(|| self.run.particles())
    }

    /// The fields, the layers' ψ and the particles written back into `pic`.
    pub fn finish(mut self, pic: &mut Pic) -> Result<()> {
        guarded(|| self.run.finish(pic))
    }
}

/// A particle's values in its slot: x, y, z (m), u_x, u_y, u_z (m/s), weight.
fn item(&(kind, p): &(u32, Particle)) -> Item<7> {
    Item { values: [p.x[0], p.x[1], p.x[2], p.u[0], p.u[1], p.u[2], p.w], kind }
}

/// The particle in a slot, with its population.
fn particle(i: &Item<7>) -> (u32, Particle) {
    let [x, y, z, ux, uy, uz, w] = i.values;
    (i.kind, Particle { x: [x, y, z], u: [ux, uy, uz], w })
}

/// The layers' coefficients of one set (E's or B's) laid end to end, each axis's nodes then
/// centres, and where each of the six starts (in pairs).
fn grading_of(set: &[[&[Cpml]; 2]; 3]) -> (Vec<f64>, Vec<u32>) {
    let mut ba = Vec::new();
    let mut start = Vec::with_capacity(6);
    for c in set.iter().flatten() {
        start.push((ba.len() / 2) as u32);
        ba.extend(c.iter().flat_map(|&c| <[f64; 2]>::from(c)));
    }
    if ba.is_empty() {
        ba = vec![1.0, 0.0];
    }
    (ba, start)
}

/// The binomial stencil of `passes` passes, C(2k, k + j)/4^k over j = −k…k
/// (`emcraft_fdtd::pic`'s smoothing).
fn binomial(passes: usize) -> Vec<f64> {
    let mut weights = vec![1.0f64];
    for _ in 0..2 * passes {
        let mut next = vec![0.0; weights.len() + 1];
        for (i, w) in weights.iter().enumerate() {
            next[i] += 0.5 * w;
            next[i + 1] += 0.5 * w;
        }
        weights = next;
    }
    weights
}

/// The run's arrays on the device.
struct Run<F> {
    client: Client,
    cells: [u32; 3],
    periodic: [u32; 3],
    h: [f64; 3],
    dt: f64,
    fields: [Buf<F>; 6],
    current: [Words; 3],
    /// J as E's step takes it, and the smoothing's other half.
    j: [Buf<F>; 3],
    spare: [Buf<F>; 3],
    /// The smoothing's stencil and its width (passes: none, 0).
    weights: Buf<F>,
    passes: usize,
    held_edges: [Ints; 3],
    eps: [Buf<F>; 3],
    metal: Ints,
    any_media: bool,
    any_metal: bool,
    /// E's and B's layers' coefficients and starts, and ψ (E's then B's, per component its two
    /// differences); one-element stand-ins without layers.
    grading: [(Buf<F>, Ints); 2],
    psi: [[[Buf<F>; 2]; 3]; 2],
    absorbing: bool,
    /// The steady B the run started with, held apart on the host, and on the device with its
    /// curl's part of E's step (`kernels::pic::Steady`; one-element stand-ins without).
    b0: Option<[Vec<f64>; 3]>,
    steady: [Buf<F>; 3],
    curl: [Buf<F>; 3],
    /// The particles (x, y, z, u_x, u_y, u_z, w) and what is recorded.
    held: Held<F, 7>,
    /// Each kind's q Δt/m, charge (C) and rest energy (J).
    kinds: [Buf<F>; 3],
    charges: Vec<f64>,
    points: Buf<F>,
    out: Buf<F>,
    /// The field energy's partial sums, one a cube.
    partials: Buf<F>,
    vay: bool,
    /// The largest |charge| (C) a particle has carried.
    qmax: f64,
}

/// The most units a kernel's cube uses.
const UNITS: u32 = 256;

/// The most cubes a sum over the fields is split into.
const PARTS: usize = 1024;

/// The most additions a particle makes to one point of a component's current: Esirkepov's
/// window of 3 × 4 × 4 folded onto itself round the smallest periodic axes.
const ADDS: f64 = 48.0;

impl<F: Real> Run<F> {
    fn new(client: Client, pic: &Pic, probes: &[[f64; 3]]) -> Result<Run<F>> {
        let p: Plan<'_> = pic.plan()?;
        let axes = p.grid.axes;
        let cells = [to_u32(axes[0].cells, "cells along x")?, to_u32(axes[1].cells, "cells along y")?, to_u32(axes[2].cells, "cells along z")?];
        // Indices must fit the kernels' 32 bits.
        axes.iter()
            .try_fold(1u32, |n, a| n.checked_mul(to_u32(a.cells + 1, "cells")?).ok_or_else(|| invalid("grid", "more points than 32 bits count")))?;
        let lens: [usize; 6] = p.fields.map(<[f64]>::len);
        let b0 =
            p.fields[3..].iter().any(|b| b.iter().any(|&x| x != 0.0)).then(|| [p.fields[3].to_vec(), p.fields[4].to_vec(), p.fields[5].to_vec()]);
        let fields: [Buf<F>; 6] = std::array::from_fn(|c| match &b0 {
            Some(_) if c >= 3 => Buf::zeros(&client, lens[c]),
            _ => Buf::new(&client, p.fields[c]),
        });
        let one = || Buf::<F>::zeros(&client, 1);
        let (steady, curl) = match &b0 {
            Some(b) => {
                let c = pic.steady_curl(b)?;
                (b.each_ref().map(|v| Buf::new(&client, v)), c.each_ref().map(|v| Buf::new(&client, v)))
            }
            None => (std::array::from_fn(|_| one()), std::array::from_fn(|_| one())),
        };
        let e_lens = [lens[0], lens[1], lens[2]];
        let (held_edges, eps, metal, any_media, any_metal) = match &p.media {
            Some(m) => {
                let flags = |b: &[bool]| -> Vec<u32> { b.iter().map(|&x| u32::from(x)).collect() };
                let held_edges = m.held.map(|h| Ints::new(&client, &flags(h)));
                let eps = m.eps.map(|e| Buf::new(&client, e));
                let any_metal = m.metal.contains(&true);
                let metal = Ints::new(&client, &if any_metal { flags(m.metal) } else { vec![0] });
                (held_edges, eps, metal, true, any_metal)
            }
            None => (
                std::array::from_fn(|_| Ints::new(&client, &[0])),
                std::array::from_fn(|_| Buf::new(&client, &[1.0])),
                Ints::new(&client, &[0]),
                false,
                false,
            ),
        };
        let (grading, psi, absorbing) = match &p.layers {
            Some(l) => {
                let make = |set: &[[&[Cpml]; 2]; 3]| {
                    let (ba, start) = grading_of(set);
                    (Buf::new(&client, &ba), Ints::new(&client, &start))
                };
                let psis = |s: &[[&[f64]; 2]; 3]| -> [[Buf<F>; 2]; 3] { s.map(|pair| pair.map(|v| Buf::new(&client, v))) };
                ([make(&l.e), make(&l.b)], [psis(&l.psi_e), psis(&l.psi_b)], true)
            }
            None => {
                let none = || (Buf::new(&client, &[1.0, 0.0]), Ints::new(&client, &[0; 6]));
                ([none(), none()], std::array::from_fn(|_| std::array::from_fn(|_| [one(), one()])), false)
            }
        };
        let items: Vec<Item<7>> = p.particles.iter().map(item).collect();
        let held = Held::new(client.clone(), &items, probes.len(), cube_units(&client, UNITS))?;
        let charges: Vec<f64> = p.species.iter().map(|s| s.charge).collect();
        let qmax = p.particles.iter().map(|(k, q)| (charges.get(*k as usize).copied().unwrap_or(0.0) * q.w).abs()).fold(0.0, f64::max);
        let kind_column = |f: &dyn Fn(&emcraft_model::particles::Species) -> f64| -> Buf<F> {
            let v: Vec<f64> = p.species.iter().map(f).collect();
            Buf::new(&client, if v.is_empty() { &[0.0] } else { &v })
        };
        let kinds = [kind_column(&|s| s.charge * p.dt / s.mass), kind_column(&|s| s.charge), kind_column(&|s| s.rest_energy())];
        let points: Vec<f64> = probes.iter().flatten().copied().collect();
        if !points.iter().all(|x| x.is_finite()) {
            return Err(invalid("probe", "a point of finite x, y and z"));
        }
        let passes = p.smoothing;
        Ok(Run {
            cells,
            periodic: axes.map(|a| u32::from(a.wall == Wall::Periodic)),
            h: axes.map(|a| a.h),
            dt: p.dt,
            fields,
            current: e_lens.map(|n| Words::zeros(&client, n.max(1))),
            j: e_lens.map(|n| Buf::zeros(&client, n.max(1))),
            spare: e_lens.map(|n| Buf::zeros(&client, if passes > 0 { n.max(1) } else { 1 })),
            weights: Buf::new(&client, &binomial(passes)),
            passes,
            held_edges,
            eps,
            metal,
            any_media,
            any_metal,
            grading,
            psi,
            absorbing,
            b0,
            steady,
            curl,
            held,
            kinds,
            charges,
            points: if probes.is_empty() { Buf::zeros(&client, 3) } else { Buf::new(&client, &points) },
            out: Buf::zeros(&client, 1),
            partials: Buf::zeros(&client, PARTS),
            vay: p.pusher == Pusher::Vay,
            qmax,
            client,
        })
    }

    fn grid_arg(&self) -> GridLaunch<F> {
        let [nx, ny, nz] = self.cells;
        let [px, py, pz] = self.periodic;
        let [hx, hy, hz] = self.h.map(F::of);
        GridLaunch::new(nx, ny, nz, px, py, pz, hx, hy, hz, F::of(self.dt), F::of(C0))
    }

    fn fields_arg(&self) -> FieldsLaunch<F> {
        let [ex, ey, ez, bx, by, bz] = &self.fields;
        FieldsLaunch::new(ex.arg(), ey.arg(), ez.arg(), bx.arg(), by.arg(), bz.arg())
    }

    fn particles_arg(&self) -> ParticlesLaunch<F> {
        let s = &self.held.slots;
        let [x, y, z, ux, uy, uz, w] = &s.f;
        ParticlesLaunch::new(x.arg(), y.arg(), z.arg(), ux.arg(), uy.arg(), uz.arg(), w.arg(), s.born.arg(), s.alive.arg(), s.kind.arg())
    }

    fn kinds_arg(&self) -> KindsLaunch<F> {
        let [kick, charge, rest] = &self.kinds;
        KindsLaunch::new(kick.arg(), charge.arg(), rest.arg())
    }

    fn steady_arg(&self) -> SteadyLaunch<F> {
        let [x, y, z] = &self.steady;
        SteadyLaunch::new(x.arg(), y.arg(), z.arg())
    }

    fn grading_arg(&self, set: usize) -> GradingLaunch<F> {
        let (ba, start) = &self.grading[set];
        GradingLaunch::new(ba.arg(), start.arg())
    }

    /// The fixed-point scales of J_x, J_y, J_z: no point's sum, of at most [`ADDS`] additions a
    /// particle in each of `slots`, each at most the largest charge's through a cell's face in a
    /// step, reaches `crate::particles::RANGE`.
    fn scales(&self, slots: usize) -> [f64; 3] {
        let [hx, hy, hz] = self.h;
        let most = ADDS * slots.max(1) as f64 * self.qmax / self.dt;
        [hy * hz, hz * hx, hx * hy].map(|face| scale(most / face))
    }

    fn current_arg(&self, scales: [f64; 3]) -> CurrentLaunch<F> {
        let [x, y, z] = &self.current;
        CurrentLaunch::new(x.arg(), y.arg(), z.arg(), F::of(scales[0]), F::of(scales[1]), F::of(scales[2]))
    }

    /// The launch over E_a's lattice (`set` 0) or B_a's (1).
    fn over(&self, set: usize, a: usize) -> (CubeCount, CubeDim) {
        line(self.fields[3 * set + a].len() as u32, self.held.units)
    }

    /// B advanced by `dt` (s), a component at a time.
    fn faraday(&self, dt: F) {
        for a in 0..3 {
            let (q, r) = ((a + 1) % 3, (a + 2) % 3);
            let (cubes, dim) = self.over(1, a);
            let [pq, pr] = &self.psi[1][a];
            kernels::faraday::launch::<F>(
                &self.client,
                cubes,
                dim,
                self.fields[3 + a].arg(),
                self.fields[q].arg(),
                self.fields[r].arg(),
                pq.arg(),
                pr.arg(),
                self.grading_arg(1),
                self.grid_arg(),
                dt,
                a as u32,
                q as u32,
                r as u32,
                self.absorbing,
            );
        }
    }

    /// E advanced a step by J (`j`), a component at a time.
    fn ampere(&self, j: &[Buf<F>; 3]) {
        for a in 0..3 {
            let (q, r) = ((a + 1) % 3, (a + 2) % 3);
            let (cubes, dim) = self.over(0, a);
            let [pq, pr] = &self.psi[0][a];
            kernels::ampere::launch::<F>(
                &self.client,
                cubes,
                dim,
                self.fields[a].arg(),
                self.fields[3 + q].arg(),
                self.fields[3 + r].arg(),
                j[a].arg(),
                self.curl[a].arg(),
                self.held_edges[a].arg(),
                self.eps[a].arg(),
                pq.arg(),
                pr.arg(),
                self.grading_arg(0),
                self.grid_arg(),
                F::of(C0 * C0),
                F::of(EPS0),
                a as u32,
                q as u32,
                r as u32,
                self.absorbing,
                self.any_media,
                self.b0.is_some(),
            );
        }
    }

    /// The step's current, from its fixed-point sums (cleared), smoothed along each axis: in
    /// `j`, or in `spare` with smoothing (the one returned).
    fn current(&self, scales: [f64; 3]) -> &[Buf<F>; 3] {
        for a in 0..3 {
            let (cubes, dim) = self.over(0, a);
            let n = self.fields[a].len() as u32;
            kernels::settle::launch::<F>(&self.client, cubes, dim, self.current[a].arg(), self.j[a].arg(), F::of(scales[a]), n);
        }
        if self.passes == 0 {
            return &self.j;
        }
        let width = (2 * self.passes + 1) as u32;
        for a in 0..3 {
            let (cubes, dim) = self.over(0, a);
            let (j, s) = (&self.j[a], &self.spare[a]);
            for (d, (from, to)) in [(j, s), (s, j), (j, s)].into_iter().enumerate() {
                kernels::smooth::launch::<F>(
                    &self.client,
                    cubes.clone(),
                    dim,
                    from.arg(),
                    to.arg(),
                    self.weights.arg(),
                    width,
                    self.grid_arg(),
                    a as u32,
                    d as u32,
                );
            }
        }
        &self.spare
    }

    /// E at the probes after step `slot` of the batch, into their record.
    fn probe(&self, slot: usize) {
        let probes = self.held.probes as u32;
        if probes > 0 {
            let (cubes, dim) = line(probes, self.held.units);
            let record = self.held.record.arg();
            kernels::probe::launch::<F>(&self.client, cubes, dim, self.fields_arg(), self.grid_arg(), self.points.arg(), record, slot as u32, probes);
        }
    }

    /// One batch of `steps` (at most `crate::BATCH`).
    fn batch(&mut self, pic: &mut Pic, steps: usize) -> Result<Vec<DeviceStep>> {
        // The bunches' particles, made on the host a step at a time.
        let mut entered = Vec::with_capacity(steps);
        let (mut born, mut at) = (Vec::new(), Vec::new());
        for b in 0..steps {
            let (moved, new) = pic.births_and_tick(self.held.held + born.len())?;
            let step = self.held.done.checked_add(to_u32(b, "steps")?).ok_or_else(|| invalid("steps", "more than 32 bits count"))?;
            at.extend(std::iter::repeat_n(step, new.len()));
            born.extend(new.iter().map(item));
            entered.push(moved);
        }
        let charges = &self.charges;
        self.qmax = born.iter().map(|i| (charges.get(i.kind as usize).copied().unwrap_or(0.0) * i.values[6]).abs()).fold(self.qmax, f64::max);
        self.held.admit(&born, &at)?;
        let scales = self.scales(self.held.capacity);
        let units = self.held.units;
        let len = to_u32(self.held.len, "particles")?;
        let (over_particles, particle_dim) = line(len.max(1), units);
        let half = F::of(0.5 * self.dt);
        for b in 0..steps {
            let step = self.held.done + b as u32;
            kernels::push_particles::launch::<F>(
                &self.client,
                over_particles.clone(),
                particle_dim,
                self.particles_arg(),
                self.fields_arg(),
                self.steady_arg(),
                self.grid_arg(),
                self.kinds_arg(),
                self.metal.arg(),
                self.current_arg(scales),
                self.held.deaths_arg(),
                step,
                len,
                self.vay,
                self.any_metal,
                self.b0.is_some(),
            );
            let j = self.current(scales);
            self.faraday(half);
            self.ampere(j);
            self.faraday(half);
            self.probe(b);
        }
        let readings = self.held.read_batch(steps)?;
        // How a particle left (`kernels::pic::push_particles`): 1 + 2 × axis + (1 at the high
        // end) at a wall, 7 in the metal; its charge written as its weight.
        Ok(self.held.tally(entered, &readings, |d, moved| match d.how {
            how @ 1..=6 => {
                let wall = (how - 1) as usize;
                moved.collected[wall / 2][wall % 2] += d.w;
            }
            _ => moved.metal += d.w,
        }))
    }

    /// Σ wᵢ (vᵢ + sᵢ)² over `values` (the weights `w`, the steady part `s` where given): a
    /// partial sum a cube on the device, the partials added in their order on the host.
    fn squares(&self, values: &Buf<F>, w: Option<&Buf<F>>, s: Option<&Buf<F>>) -> Result<f64> {
        let units = self.held.units;
        let count = values.len() as u32;
        let cubes = count.div_ceil(units).clamp(1, PARTS as u32);
        kernels::squares::launch::<F>(
            &self.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(units),
            values.arg(),
            w.unwrap_or(values).arg(),
            s.unwrap_or(values).arg(),
            self.partials.arg(),
            count,
            w.is_some(),
            s.is_some(),
            units as usize,
        );
        Ok(crate::particles::read_first(&self.partials, &self.client, cubes as usize)?.iter().sum())
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
    fn advance(&mut self, pic: &mut Pic, steps: usize) -> Result<Vec<DeviceStep>> {
        let mut out = Vec::with_capacity(steps.min(emcraft_fdtd::MAX_STEPS));
        let mut left = steps;
        while left > 0 {
            let n = left.min(crate::BATCH);
            out.extend(self.batch(pic, n)?);
            left -= n;
        }
        Ok(out)
    }

    fn energy(&mut self, pic: &mut Pic) -> Result<Energy> {
        let electric: f64 = (0..3).map(|a| self.squares(&self.fields[a], self.any_media.then_some(&self.eps[a]), None)).sum::<Result<f64>>()?;
        let magnetic: f64 =
            (0..3).map(|a| self.squares(&self.fields[3 + a], None, self.b0.is_some().then_some(&self.steady[a]))).sum::<Result<f64>>()?;
        let v = pic.grid().cell_volume();
        let width = self.held.units as usize;
        let len = to_u32(self.held.len, "particles")?;
        kernels::kinetic::launch::<F>(
            &self.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(self.held.units),
            self.particles_arg(),
            self.kinds_arg(),
            self.grid_arg(),
            self.out.arg(),
            len,
            self.held.done,
            width,
        );
        let kinetic = self.out.read(&self.client)?.first().copied().unwrap_or(f64::NAN);
        Ok(Energy { field: v * (0.5 * EPS0 * electric + 0.5 / MU0 * magnetic), electric: v * 0.5 * EPS0 * electric, kinetic })
    }

    fn finish(&mut self, pic: &mut Pic) -> Result<()> {
        let fields = self.read_fields()?;
        let psi = if self.absorbing {
            let mut out: [[[Vec<f64>; 2]; 3]; 2] = Default::default();
            for (o, b) in out.iter_mut().flatten().flatten().zip(self.psi.iter().flatten().flatten()) {
                *o = b.read(&self.client)?;
            }
            Some(out)
        } else {
            None
        };
        let particles = self.held.items()?.iter().map(particle).collect();
        pic.put_back(State { fields, psi, particles })
    }

    fn held(&self) -> usize {
        self.held.held
    }

    fn fields(&self) -> Result<[Vec<f64>; 6]> {
        self.read_fields()
    }

    fn particles(&self) -> Result<Vec<(u32, Particle)>> {
        Ok(self.held.items()?.iter().map(particle).collect())
    }
}
