//! One particle-in-cell step for both grids (#16): the Cartesian box ([`super::Pic`]) and the body
//! of revolution (`crate::axi::AxiPic`) each say what their particles are (points or rings), how
//! one is pushed and deposits its current, where their births come from and how their fields
//! leapfrog ([`Scheme`]); [`step`] runs the step over them in the one order the scheme takes
//! (Birdsall & Langdon §15-4): births, every population pushed on the threads with its current
//! added in its order ([`super::advance_on_threads`]), what left tallied in order, the current
//! made ready, B's half step, E's step, B's half step.
//!
//! The stencils and the deposition stay each grid's own: the axis's rules (E_z's 4B_θ/Δr, the odd
//! components, the rings' 2πr volumes) and the box's three dimensions are written and verified
//! apart, against their closed forms.

use emcraft_particles::{Species, V3};

use super::{Addition, Fate, Moved, advance_on_threads, keep_kept};
use crate::Result;

/// What a grid does in a step.
pub(crate) trait Scheme {
    /// A particle: a point in a box, a ring about an axis.
    type Item: Send;

    /// About how many additions to the current one item makes.
    const ADDITIONS: usize;

    /// The step's births added to their populations; what they bring in added to `moved`.
    fn births_now(&mut self, moved: &mut Moved) -> Result<()>;

    /// The current's three components (on its lattices), zeroed, taken out for the push.
    fn take_current(&mut self) -> [Vec<f64>; 3];

    /// The current put back after the push.
    fn put_current(&mut self, j: [Vec<f64>; 3]);

    /// How many populations there are.
    fn populations(&self) -> usize;

    /// Population `k`'s species and items, taken out for the push.
    fn take(&mut self, k: usize) -> (Species, Vec<Self::Item>);

    /// Population `k`'s items put back.
    fn put(&mut self, k: usize, items: Vec<Self::Item>);

    /// Item `p` of `species` pushed and moved a step, its current's additions handed to `adds`
    /// in order; its fate.
    fn advance(&self, species: Species, p: &mut Self::Item, adds: &mut Vec<Addition>) -> Fate;

    /// Its weight (particles it stands for).
    fn weight(p: &Self::Item) -> f64;

    /// The charge `q` (C) an absorbed item left at `at` (m): the walls' and bodies' own, where the
    /// grid keeps it.
    fn absorbed(&mut self, at: V3, q: f64);

    /// The pushed current made ready for E's step (smoothed, a held voltage's mean taken out).
    fn ready_current(&mut self) {}

    /// B advanced by `dt` (s).
    fn faraday(&mut self, dt: f64);

    /// E advanced a step by the current.
    fn ampere(&mut self);

    /// The step (s).
    fn dt(&self) -> f64;

    /// The clock moved on a step.
    fn tick(&mut self);
}

/// One step of `run`: births, the particles pushed and moved, their current, the fields.
pub(crate) fn step<S: Scheme + Sync>(run: &mut S) -> Result<Moved> {
    let mut moved = Moved::default();
    run.births_now(&mut moved)?;
    let mut j = run.take_current();
    for k in 0..run.populations() {
        let (species, mut items) = run.take(k);
        let fates = {
            let this = &*run;
            let [jx, jy, jz] = &mut j;
            advance_on_threads(
                &mut items,
                S::ADDITIONS,
                |p, adds| this.advance(species, p, adds),
                [jx.as_mut_slice(), jy.as_mut_slice(), jz.as_mut_slice()],
            )
        };
        moved.tally(species.charge, items.iter().map(S::weight), &fates, |at, q| run.absorbed(at, q));
        keep_kept(&mut items, &fates);
        run.put(k, items);
    }
    run.put_current(j);
    run.ready_current();
    let dt = run.dt();
    run.faraday(0.5 * dt);
    run.ampere();
    run.faraday(0.5 * dt);
    run.tick();
    Ok(moved)
}
