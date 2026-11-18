//! What a particle-in-cell run on a device keeps on the host's side, whatever its grid (the
//! rings about an axis of `axi`, the particles in a box of `pic`): the particles' slots and their
//! compaction, the list of those that left and its tally in order, the batches, the probes and
//! the taps recorded on the device and read back a batch at a time.
//!
//! **Slots.** Each particle has a slot: its `N` values (position, momentum, weight), the step it
//! is born at, whether it is still in the run, and its kind (its population). The host makes the
//! particles born in a batch (the same quasi-random places as on the CPU) and places them after
//! the slots in use, each with its step, before the batch runs; one that leaves stays in its slot,
//! marked; the slots are compacted, in order, when they run out.
//!
//! **What left.** The push writes each particle that leaves to a list (its step, slot, how, weight
//! or charge and kinetic energy) in whatever order the device runs it; the host sorts the list by
//! step and slot and tallies it in that order, so a step's [`Moved`] is summed the same way every
//! run.

use emcraft_fdtd::pic::Moved;
use emcraft_fdtd::{FdtdError, Result, invalid, to_u32};

use crate::BATCH;
use crate::buffer::{Buf, Ints, Real};
use crate::kernels::particles::{self as kernels, DeathsLaunch};
use crate::lattice::line;
use cubecl::prelude::*;

/// What a step on the device did: what entered and left the run, E (V/m, three components in
/// the grid's own frame) at each probe after it, and the field at each tap.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceStep {
    pub moved: Moved,
    pub probes: Vec<[f64; 3]>,
    pub taps: Vec<f64>,
}

/// The fewest slots the device holds.
const MIN_SLOTS: usize = 1024;

/// The fixed-point sums' range: no point's sum reaches 2^60 (of i64's 2^63).
pub(crate) const RANGE: f64 = (1u64 << 60) as f64;

/// The fixed-point scale of a sum that never exceeds `bound` in size: none reaches [`RANGE`].
pub(crate) fn scale(bound: f64) -> f64 {
    if bound > 0.0 && bound.is_finite() { RANGE / bound } else { 1.0 }
}

/// A particle as the host hands it over and reads it back: its values and its kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Item<const N: usize> {
    pub(crate) values: [f64; N],
    pub(crate) kind: u32,
}

/// The particles' arrays on the device: each of their `N` values; the step each is born at,
/// whether it is still in the run (1 or 0) and its kind.
pub(crate) struct Slots<F, const N: usize> {
    pub(crate) f: [Buf<F>; N],
    pub(crate) born: Ints,
    pub(crate) alive: Ints,
    pub(crate) kind: Ints,
}

impl<F: Real, const N: usize> Slots<F, N> {
    /// `items` born at `born`, in slots of `capacity` (the rest empty, not alive).
    fn new(client: &Client, items: &[Item<N>], born: &[u32], capacity: usize) -> Result<Slots<F, N>> {
        let mut columns: [Vec<f64>; N] = std::array::from_fn(|_| Vec::new());
        for c in &mut columns {
            *c = emcraft_solver::memory::with_capacity(capacity, "the particles on the device").map_err(|e| FdtdError::Device(e.to_string()))?;
        }
        for item in items {
            for (c, v) in columns.iter_mut().zip(item.values) {
                c.push(v);
            }
        }
        columns.iter_mut().for_each(|c| c.resize(capacity, 0.0));
        let ints = |f: &dyn Fn(usize) -> u32| -> Vec<u32> { (0..capacity).map(f).collect() };
        Ok(Slots {
            f: columns.each_ref().map(|c| Buf::new(client, c)),
            born: Ints::new(client, &ints(&|m| born.get(m).copied().unwrap_or(0))),
            alive: Ints::new(client, &ints(&|m| u32::from(m < items.len()))),
            kind: Ints::new(client, &ints(&|m| items.get(m).map_or(0, |p| p.kind))),
        })
    }

    /// The first `len` slots' particles still in the run, in order, and the steps they were
    /// born at.
    fn read(&self, client: &Client, len: usize) -> Result<(Vec<Item<N>>, Vec<u32>)> {
        let cols: Vec<Vec<f64>> = self.f.iter().map(|b| read_first(b, client, len)).collect::<Result<_>>()?;
        let alive = self.alive.read(client, len)?;
        let born = self.born.read(client, len)?;
        let kind = self.kind.read(client, len)?;
        let mut out = emcraft_solver::memory::with_capacity(len, "the particles read back").map_err(|e| FdtdError::Device(e.to_string()))?;
        let mut kept_born = Vec::new();
        for m in (0..len).filter(|&m| alive.get(m) == Some(&1)) {
            let values = std::array::from_fn(|c| cols.get(c).and_then(|v| v.get(m)).copied().unwrap_or(0.0));
            out.push(Item { values, kind: kind.get(m).copied().unwrap_or(0) });
            kept_born.push(born.get(m).copied().unwrap_or(0));
        }
        Ok((out, kept_born))
    }

    /// `stage`'s first `count` slots copied into these from `start` on.
    fn place(&self, client: &Client, stage: &Slots<F, N>, start: u32, count: u32, units: u32) {
        let (cubes, dim) = line(count, units);
        for (to, from) in self.f.iter().zip(&stage.f) {
            kernels::copy_values::launch::<F>(client, cubes.clone(), dim, to.arg(), from.arg(), start, count);
        }
        for (to, from) in [(&self.born, &stage.born), (&self.alive, &stage.alive), (&self.kind, &stage.kind)] {
            kernels::copy_ints::launch(client, cubes.clone(), dim, to.arg(), from.arg(), start, count);
        }
    }
}

/// The first `len` values of `b`.
pub(crate) fn read_first<F: Real>(b: &Buf<F>, client: &Client, len: usize) -> Result<Vec<f64>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let bytes = b.read_bytes(client, 0..len)?;
    Ok(F::from_bytes(&bytes).iter().map(|x| x.wide()).collect())
}

/// The list of particles that left, room for `capacity`.
struct DeathList<F> {
    step: Ints,
    slot: Ints,
    how: Ints,
    w: Buf<F>,
    kinetic: Buf<F>,
    count: Ints,
    capacity: usize,
}

impl<F: Real> DeathList<F> {
    fn new(client: &Client, capacity: usize) -> DeathList<F> {
        let n = capacity.max(1);
        DeathList {
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

/// A particle that left: its step, slot, how (the grid's numbering, 0 dropped), weight or
/// charge (as its kernel wrote it) and kinetic energy (J).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Death {
    pub(crate) step: u32,
    pub(crate) slot: u32,
    pub(crate) how: u32,
    pub(crate) w: f64,
    pub(crate) kinetic: f64,
}

/// The taps: each one's array (its index in the grid's own order) and point there, a mask of
/// those that exist, what a steady field adds at each, and their values a batch at a time.
struct Taps<F> {
    which: Ints,
    at: Ints,
    mask: Vec<bool>,
    steady: Vec<f64>,
    record: Buf<F>,
}

/// A batch's readings: what left, the probes' E and the taps' values, step by step.
pub(crate) struct Readings {
    deaths: Vec<Death>,
    record: Vec<f64>,
    tapped: Vec<f64>,
}

/// The particles a run holds on a device and what it records there.
pub(crate) struct Held<F, const N: usize> {
    pub(crate) client: Client,
    pub(crate) slots: Slots<F, N>,
    deaths: DeathList<F>,
    /// Slots used, and their number.
    pub(crate) len: usize,
    pub(crate) capacity: usize,
    /// Particles in the run.
    pub(crate) held: usize,
    /// Steps run on the device.
    pub(crate) done: u32,
    /// The probes, and their E (three components) a batch at a time.
    pub(crate) probes: usize,
    pub(crate) record: Buf<F>,
    taps: Taps<F>,
    /// Units per cube.
    pub(crate) units: u32,
}

impl<F: Real, const N: usize> Held<F, N> {
    /// `items` in the run from step 0, E recorded at `probes` probes.
    pub(crate) fn new(client: Client, items: &[Item<N>], probes: usize, units: u32) -> Result<Held<F, N>> {
        let capacity = (2 * items.len()).max(MIN_SLOTS);
        Ok(Held {
            slots: Slots::new(&client, items, &vec![0; items.len()], capacity)?,
            deaths: DeathList::new(&client, capacity),
            len: items.len(),
            capacity,
            held: items.len(),
            done: 0,
            probes,
            record: Buf::zeros(&client, (BATCH * probes * 3).max(1)),
            taps: Taps {
                which: Ints::zeros(&client, 1),
                at: Ints::zeros(&client, 1),
                mask: Vec::new(),
                steady: Vec::new(),
                record: Buf::zeros(&client, 1),
            },
            units,
            client,
        })
    }

    /// The list of those that left, as a kernel argument.
    pub(crate) fn deaths_arg(&self) -> DeathsLaunch<F> {
        self.deaths.arg()
    }

    /// The taps' arrays, points and values, as kernel arguments, and how many there are.
    pub(crate) fn taps_arg(&self) -> (BufferArg, BufferArg, BufferArg, u32) {
        (self.taps.which.arg(), self.taps.at.arg(), self.taps.record.arg(), self.taps.mask.len() as u32)
    }

    /// `born` placed after the slots in use, each born at its step in `at`, the slots compacted
    /// first if they would run out.
    pub(crate) fn admit(&mut self, born: &[Item<N>], at: &[u32]) -> Result<()> {
        if self.len + born.len() > self.capacity {
            self.compact(born.len())?;
        }
        if !born.is_empty() {
            let stage = Slots::<F, N>::new(&self.client, born, at, born.len())?;
            let count = to_u32(born.len(), "particles")?;
            self.slots.place(&self.client, &stage, to_u32(self.len, "particles")?, count, self.units);
            self.len += born.len();
            self.held += born.len();
        }
        Ok(())
    }

    /// The slots compacted, those in the run in order, into room for `need` more.
    fn compact(&mut self, need: usize) -> Result<()> {
        let (items, born) = self.slots.read(&self.client, self.len)?;
        let capacity = (2 * (items.len() + need)).max(MIN_SLOTS);
        self.slots = Slots::new(&self.client, &items, &born, capacity)?;
        self.len = items.len();
        self.capacity = capacity;
        self.deaths = DeathList::new(&self.client, capacity);
        Ok(())
    }

    /// The particles in the run, in order.
    pub(crate) fn items(&self) -> Result<Vec<Item<N>>> {
        Ok(self.slots.read(&self.client, self.len)?.0)
    }

    /// The batch of `steps` just run: the clock moved on, what left (by step and slot, the list
    /// cleared), the probes and the taps read back.
    pub(crate) fn read_batch(&mut self, steps: usize) -> Result<Readings> {
        self.done = self.done.checked_add(to_u32(steps, "steps")?).ok_or_else(|| invalid("steps", "more than 32 bits count"))?;
        let d = &self.deaths;
        let count = d.count.read(&self.client, 1)?.first().copied().unwrap_or(0) as usize;
        if count > d.capacity {
            return Err(FdtdError::Device(format!("{count} particles left, more than the list holds")));
        }
        let (step, slot, how) = (d.step.read(&self.client, count)?, d.slot.read(&self.client, count)?, d.how.read(&self.client, count)?);
        let (w, kinetic) = (read_first(&d.w, &self.client, count)?, read_first(&d.kinetic, &self.client, count)?);
        let mut deaths: Vec<Death> = (0..count)
            .map(|m| Death {
                step: step.get(m).copied().unwrap_or(0),
                slot: slot.get(m).copied().unwrap_or(0),
                how: how.get(m).copied().unwrap_or(0),
                w: w.get(m).copied().unwrap_or(0.0),
                kinetic: kinetic.get(m).copied().unwrap_or(0.0),
            })
            .collect();
        deaths.sort_unstable_by_key(|d| (d.step, d.slot));
        self.deaths.count = Ints::zeros(&self.client, 1);
        self.held = self.held.saturating_sub(deaths.len());
        let record = if self.probes > 0 { read_first(&self.record, &self.client, steps * self.probes * 3)? } else { Vec::new() };
        let n = self.taps.mask.len();
        let tapped = if n > 0 { read_first(&self.taps.record, &self.client, steps * n)? } else { Vec::new() };
        Ok(Readings { deaths, record, tapped })
    }

    /// Each step's [`DeviceStep`]: what entered (`entered`, a step each), what left tallied in
    /// its order by `left` (each death added to its step's [`Moved`]), the probes, the taps (zero
    /// where there is no point, the steady field added).
    pub(crate) fn tally(&self, entered: Vec<Moved>, readings: &Readings, left: impl Fn(&Death, &mut Moved)) -> Vec<DeviceStep> {
        let n = self.taps.mask.len();
        let first = self.done - entered.len() as u32;
        let mut deaths = readings.deaths.iter().peekable();
        entered
            .into_iter()
            .enumerate()
            .map(|(b, mut moved)| {
                let step = first + b as u32;
                while let Some(d) = deaths.next_if(|d| d.step == step) {
                    if d.how == 0 {
                        moved.lost += 1;
                    } else {
                        left(d, &mut moved);
                        moved.kinetic_out += d.kinetic;
                    }
                }
                let probes = (0..self.probes)
                    .map(|p| std::array::from_fn(|c| readings.record.get(3 * (b * self.probes + p) + c).copied().unwrap_or(f64::NAN)))
                    .collect();
                let taps = (0..n)
                    .map(|t| match self.taps.mask.get(t) {
                        Some(true) => readings.tapped.get(b * n + t).copied().unwrap_or(f64::NAN) + self.taps.steady.get(t).copied().unwrap_or(0.0),
                        _ => 0.0,
                    })
                    .collect();
                DeviceStep { moved, probes, taps }
            })
            .collect()
    }

    /// The taps from the next step on: each its array and point there (`lens` the arrays'
    /// lengths) or none, and what a steady field adds at each. Refused for a point outside its
    /// array.
    pub(crate) fn set_taps(&mut self, taps: &[Option<(usize, usize)>], lens: &[usize], steady: Vec<f64>) -> Result<()> {
        if taps.iter().flatten().any(|&(which, at)| lens.get(which).is_none_or(|&n| at >= n)) {
            return Err(invalid("taps", "points on the fields' lattices"));
        }
        let (which, at): (Vec<u32>, Vec<u32>) = taps.iter().map(|t| t.map_or((0, 0), |(w, a)| (w as u32, a as u32))).unzip();
        let n = taps.len().max(1);
        self.taps = Taps {
            which: Ints::new(&self.client, if which.is_empty() { &[0] } else { &which }),
            at: Ints::new(&self.client, if at.is_empty() { &[0] } else { &at }),
            mask: taps.iter().map(Option::is_some).collect(),
            steady,
            record: Buf::zeros(&self.client, BATCH * n),
        };
        Ok(())
    }
}
