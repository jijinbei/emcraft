//! The axisymmetric particle-in-cell run (`emcraft_fdtd::axi`) on a device: the fields, the
//! layers' ψ and the rings held there across the run, five kernels a step (the rings pushed,
//! B's half step, E's step, B's half step, the probes and the taps), the host reading back what
//! left the run, the probes and the taps a batch at a time. The taps are the field's values at
//! points the run names ([`AxiDevice::tap`]: an output plane's, `emcraft_fdtd::axi::PlaneTaps`);
//! [`AxiDevice::fields`] and [`AxiDevice::rings`] read the whole state back at a moment of the
//! run without ending it.
//!
//! **What runs where.** The host makes the injections' rings ([`AxiPic::births_and_tick`], the
//! same quasi-random places as on the CPU); the slots they go into, the list of those that left
//! and its tally, and the probes' and taps' records are `crate::particles`', as for every grid.
//!
//! **Determinism.** Each ring and each field value is written by one unit; the current's sums
//! are fixed-point integers (`kernels::particles`), the same in any order; the list of the rings
//! that left is sorted by step and index before it is tallied; the kinetic energy is summed in a
//! fixed tree. A run gives the same bits every time on a device and precision. Devices and
//! precisions round differently: their results agree with the CPU's to the precision, not bit
//! for bit.

use cubecl::prelude::*;
use emcraft_fdtd::axi::device::{Plan, State};
use emcraft_fdtd::axi::{AxiPic, Ring};
use emcraft_fdtd::pic::Energy;
use emcraft_fdtd::{Result, invalid, to_u32};
use emcraft_model::particles::Pusher;
use emcraft_model::units::{C0, EPS0};

use crate::buffer::{Buf, Ints, Real, Words};
use crate::device::{cube_units, guarded};
use crate::kernels::axi::{
    self as kernels, CurrentLaunch, FieldsLaunch, GradingLaunch, GridLaunch, MetalLaunch, PsiLaunch, RingsLaunch, SpeciesLaunch, SteadyLaunch,
};
use crate::lattice::line;
use crate::particles::{DeviceStep, Held, Item, scale};
use crate::{Gpu, Precision};

/// An axisymmetric run held on a device: made by [`Gpu::axi`], stepped by
/// [`AxiDevice::advance`], its state given back to the run by [`AxiDevice::finish`].
pub struct AxiDevice {
    run: Box<dyn Ops>,
}

/// A run's work on the device, in the float type its precision picks.
trait Ops {
    fn advance(&mut self, pic: &mut AxiPic, steps: usize) -> Result<Vec<DeviceStep>>;
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
    /// `steps` steps of `pic` (its clock and injections kept on the host), what each did: E at
    /// the probes as (E_r, E_θ, E_z), V/m; the taps' fields, V/m or T (B with the steady field).
    pub fn advance(&mut self, pic: &mut AxiPic, steps: usize) -> Result<Vec<DeviceStep>> {
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

/// A ring's values in its slot: r, z (m), u_r, u_θ, u_z (m/s), weight.
fn item(r: &Ring) -> Item<6> {
    Item { values: [r.r, r.z, r.u[0], r.u[1], r.u[2], r.w], kind: 0 }
}

/// The ring in a slot.
fn ring(i: &Item<6>) -> Ring {
    let [r, z, ur, ut, uz, w] = i.values;
    Ring { r, z, u: [ur, ut, uz], w }
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
    /// The rings (r, z, u_r, u_θ, u_z, w) and what is recorded.
    held: Held<F, 6>,
    points: Buf<F>,
    out: Buf<F>,
    species: [f64; 3],
    vay: bool,
    /// The largest |charge| (C) a ring has carried.
    qmax: f64,
}

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
        let items: Vec<Item<6>> = p.rings.iter().map(item).collect();
        let held = Held::new(client.clone(), &items, probes.len(), cube_units(&client, UNITS))?;
        let qmax = p.rings.iter().map(|r| (p.species.charge * r.w).abs()).fold(0.0, f64::max);
        let points: Vec<f64> = probes.iter().flatten().copied().collect();
        if !points.iter().all(|x| x.is_finite()) {
            return Err(invalid("probe", "a point of finite r and z"));
        }
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
            held,
            points: if probes.is_empty() { Buf::zeros(&client, 2) } else { Buf::new(&client, &points) },
            out: Buf::zeros(&client, 1),
            species: [p.species.charge * p.dt / p.species.mass, p.species.charge, p.species.rest_energy()],
            vay: p.pusher == Pusher::Vay,
            qmax,
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

    fn rings_arg(&self) -> RingsLaunch<F> {
        let s = &self.held.slots;
        let [r, z, ur, ut, uz, w] = &s.f;
        RingsLaunch::new(r.arg(), z.arg(), ur.arg(), ut.arg(), uz.arg(), w.arg(), s.born.arg(), s.alive.arg())
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
    /// `crate::particles::RANGE`.
    fn scales(&self, slots: usize) -> [f64; 3] {
        let [dr, dz, dt] = self.grid;
        let pi = std::f64::consts::PI;
        let most = 4.0 * slots.max(1) as f64 * self.qmax / dt;
        let faces = [2.0 * pi * 0.5 * dr * dz, pi * 0.25 * dr * dr, 2.0 * pi * dr * dr * dz / (C0 * dt)];
        faces.map(|face| scale(most / face))
    }

    fn current_arg(&self, scales: [f64; 3]) -> CurrentLaunch<F> {
        let [r, z, t] = &self.current;
        CurrentLaunch::new(r.arg(), z.arg(), t.arg(), F::of(scales[0]), F::of(scales[1]), F::of(scales[2]))
    }

    /// One batch of `steps` (at most `crate::BATCH`).
    fn batch(&mut self, pic: &mut AxiPic, steps: usize) -> Result<Vec<DeviceStep>> {
        // The injections' rings, made on the host a step at a time.
        let mut entered = Vec::with_capacity(steps);
        let (mut born, mut at) = (Vec::new(), Vec::new());
        for b in 0..steps {
            let (moved, new) = pic.births_and_tick(self.held.held + born.len())?;
            let step = self.held.done.checked_add(to_u32(b, "steps")?).ok_or_else(|| invalid("steps", "more than 32 bits count"))?;
            at.extend(std::iter::repeat_n(step, new.len()));
            born.extend(new.iter().map(item));
            entered.push(moved);
        }
        let charge = self.species[1];
        self.qmax = born.iter().map(|r| (charge * r.values[5]).abs()).fold(self.qmax, f64::max);
        self.held.admit(&born, &at)?;
        let scales = self.scales(self.held.capacity);
        let units = self.held.units;
        let len = to_u32(self.held.len, "rings")?;
        let [nr, nz, _] = self.cells;
        let (over_points, dim) = line((nr + 1) * (nz + 1), units);
        let (over_rings, ring_dim) = line(len.max(1), units);
        let probes = self.held.probes;
        let (over_probes, probe_dim) = line(to_u32(probes.max(1), "probes")?, units);
        let taps = self.held.taps_arg().3;
        let (over_taps, tap_dim) = line(taps.max(1), units);
        let half = F::of(0.5 * self.grid[2]);
        for b in 0..steps {
            let step = self.held.done + b as u32;
            kernels::push_rings::launch::<F>(
                &self.client,
                over_rings.clone(),
                ring_dim,
                self.rings_arg(),
                self.fields_arg(),
                self.steady_arg(),
                self.grid_arg(),
                self.species_arg(),
                self.metal_arg(),
                self.current_arg(scales),
                self.held.deaths_arg(),
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
            if probes > 0 {
                kernels::probe::launch::<F>(
                    &self.client,
                    over_probes.clone(),
                    probe_dim,
                    self.fields_arg(),
                    self.grid_arg(),
                    self.points.arg(),
                    self.held.record.arg(),
                    b as u32,
                    probes as u32,
                );
            }
            if taps > 0 {
                let (which, at, record, _) = self.held.taps_arg();
                kernels::tap::launch::<F>(&self.client, over_taps.clone(), tap_dim, self.fields_arg(), which, at, record, b as u32, taps);
            }
        }
        let readings = self.held.read_batch(steps)?;
        // How a ring left (`kernels::axi::push_rings`): 1 the outer wall, 2 the low end, 3 the
        // high end, 4 the metal.
        Ok(self.held.tally(entered, &readings, |d, moved| {
            let q = charge * d.w;
            match d.how {
                1 => moved.collected[0][1] += q,
                2 => moved.collected[2][0] += q,
                3 => moved.collected[2][1] += q,
                _ => moved.metal += q,
            }
        }))
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
    fn advance(&mut self, pic: &mut AxiPic, steps: usize) -> Result<Vec<DeviceStep>> {
        let mut out = Vec::with_capacity(steps.min(emcraft_fdtd::MAX_STEPS));
        let mut left = steps;
        while left > 0 {
            let n = left.min(crate::BATCH);
            out.extend(self.batch(pic, n)?);
            left -= n;
        }
        Ok(out)
    }

    fn energy(&mut self, pic: &mut AxiPic) -> Result<Energy> {
        pic.put_fields(self.read_fields()?)?;
        let field = pic.energy();
        let width = self.held.units as usize;
        let len = to_u32(self.held.len, "rings")?;
        kernels::kinetic::launch::<F>(
            &self.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(self.held.units),
            self.rings_arg(),
            self.species_arg(),
            self.grid_arg(),
            self.out.arg(),
            len,
            self.held.done,
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
        let rings = self.held.items()?.iter().map(ring).collect();
        pic.put_back(State { fields, psi, rings })
    }

    fn held(&self) -> usize {
        self.held.held
    }

    fn tap(&mut self, taps: &[Option<(usize, usize)>]) -> Result<()> {
        self.held.set_taps(taps, &self.fields, self.b0.as_ref())
    }

    fn fields(&self) -> Result<[Vec<f64>; 6]> {
        self.read_fields()
    }

    fn rings(&self) -> Result<Vec<Ring>> {
        Ok(self.held.items()?.iter().map(ring).collect())
    }
}
