//! `solve.particles` in a body of revolution (`axisymmetric`): the model's geometry taken as a
//! body about the z axis (x = y = 0, as solve.revolution takes it), run on
//! `emcraft_fdtd::axi`'s r–z grid (the fields that do not vary round the axis, the particles
//! rings about it).
//!
//! **The grid.** The cylinder's radius is as far from the axis as the box reaches; its cells about `cell`
//! across and `fine` times finer along z (8 for beams faster than about 0.6c: the numerical
//! Cherenkov instability, `emcraft_fdtd::axi`'s docs); the time step 0.95 of the grid's limit.
//! The ends conducting or periodic as particles.walls sets z; an open end a CPML of
//! [`super::LAYER`] × `fine` cells beyond it, the geometry carried on into it as it is at the
//! face. Whatever lies outside the solid in the half plane θ = 0 is a conductor.
//!
//! **The sources.** Beams and bunches through z− or z+ on the axis, and an axial magnetic field;
//! emitters, plasmas, a held voltage, a field across the axis and dielectrics are refused (not
//! supported yet in a body of revolution).
//!
//! **The result.** As the Cartesian run's: the box the cylinder's, the probes' fields along x, y
//! and z, the rings at the end spread round the axis by the golden angle (each a particle where
//! its ring passes), so the window draws them as the beam they are.

use std::cell::Cell;

use emcraft_fdtd::axi::{AxiGrid, AxiPic, Injection, Rings};
use emcraft_fdtd::pic;
use emcraft_model::particles::Wall;
use emcraft_particles::ELECTRON;
use serde_json::Value;

use super::{COURANT, EV, End, Frame, LAYER, Run, SNAPSHOT, Settings, Snap, bunch_pulse, fdtd, finish, record, steps_of};
use crate::{EngineError, Params, Result, Session};

/// A run in a body of revolution and its radius R (m): the grid's corner sits at (−R, −R).
struct Axi {
    pic: AxiPic,
    radius: f64,
}

impl Axi {
    /// (r, z) of a point (m from the grid's corner), and the cosine and sine of its angle.
    fn polar(&self, at: [f64; 3]) -> (f64, f64, f64, f64) {
        let (x, y) = (at[0] - self.radius, at[1] - self.radius);
        let r = x.hypot(y);
        let (c, s) = if r > 0.0 { (x / r, y / r) } else { (1.0, 0.0) };
        (r, at[2], c, s)
    }
}

impl Run for Axi {
    fn advance(&mut self) -> Result<pic::Moved> {
        self.pic.step().map_err(fdtd)
    }
    fn energies(&self) -> (f64, f64) {
        self.pic.energy()
    }
    fn held(&self) -> usize {
        self.pic.rings()
    }
    fn e_at_point(&self, at: [f64; 3]) -> [f64; 3] {
        let (r, z, c, s) = self.polar(at);
        let [er, et, ez] = self.pic.field_at(r, z).0;
        [er * c - et * s, er * s + et * c, ez]
    }
    fn now(&self) -> f64 {
        self.pic.time()
    }
}

/// What a body of revolution cannot run yet, if the model asks for it.
fn unsupported(s: &Session) -> Option<String> {
    let m = &s.model.particles;
    let what = if !m.emitters.is_empty() {
        "emitting walls"
    } else if !m.plasmas.is_empty() {
        "plasmas"
    } else if m.voltage.is_some() {
        "a held voltage"
    } else if m.magnetic[0] != 0.0 || m.magnetic[1] != 0.0 {
        "a magnetic field across the axis"
    } else if m.open.iter().any(|f| f.axis != 2) || m.beams.iter().map(|b| b.face).chain(m.bunches.iter().map(|b| b.face)).any(|f| f.axis != 2) {
        "faces other than z− and z+"
    } else if m.beams.iter().map(|b| b.at).chain(m.bunches.iter().map(|b| b.at)).any(|at| at.is_some()) {
        "beams or bunches off the axis (`at`)"
    } else {
        return None;
    };
    Some(format!("not supported yet in a body of revolution: {what} (beams and bunches through z− or z+ on the axis, and a magnetic field along z)"))
}

/// The run in a body of revolution.
pub(super) fn solve(s: &mut Session, p: &Params, set: &Settings) -> Result<Value> {
    if let Some(why) = unsupported(s) {
        return Err(EngineError::failed(why));
    }
    let b =
        s.model.bounds().ok_or_else(|| EngineError::failed("no geometry: the body of revolution particles run in is its bounds (geometry.solid)"))?;
    let m = s.model.particles.clone();
    let radius = [b.min.x, b.max.x, b.min.y, b.max.y].iter().fold(0.0f64, |r, c| r.max(c.abs()));
    if radius <= 0.0 || !radius.is_finite() {
        return Err(EngineError::failed("the geometry has no extent about the z axis: a body of revolution turns about x = y = 0"));
    }
    let (z_lo, z_hi) = (b.min.z, b.max.z);
    let nr = (radius / set.cell).round().clamp(2.0, 1e9) as usize;
    let dr = radius / nr as f64;
    let nz = ((z_hi - z_lo) / (set.cell / set.fine as f64)).round().clamp(2.0, 1e9) as usize;
    let dz = (z_hi - z_lo) / nz as f64;
    let opened = |high: bool| m.open.iter().any(|f| f.axis == 2 && f.high == high);
    let layer = LAYER * set.fine;
    let (below, above) = (usize::from(opened(false)) * layer, usize::from(opened(true)) * layer);
    let wall = match m.walls[2] {
        Wall::Periodic => pic::Wall::Periodic,
        Wall::Conductor => pic::Wall::Conductor,
    };
    if wall == pic::Wall::Periodic && (below + above > 0) {
        return Err(EngineError::failed("an open end on a periodic axis: open faces are on axes closed by walls"));
    }
    let grid = AxiGrid::new(nr, nz + below + above, dr, dz, wall).map_err(fdtd)?;
    let z_origin = z_lo - below as f64 * dz;
    let dt = grid.time_step(COURANT);
    let steps = steps_of(p, set.time, dt)?;
    let mut run = AxiPic::new(grid, dt, set.pusher).map_err(fdtd)?;
    for high in [false, true].into_iter().filter(|h| opened(*h)) {
        run.absorb(high, layer).map_err(fdtd)?;
    }
    if s.model.domain.solid.is_some() {
        // The half plane θ = 0, just inside the box (a solid's own faces are not inside it).
        let dielectric = Cell::new(None);
        let inset = 1e-6 * radius;
        run.set_metal(|r, z| {
            let point = emcraft_model::Point3::new(r.min(radius - inset), 0.0, (z + z_origin).clamp(z_lo + inset, z_hi - inset));
            match s.model.material_at(point) {
                None => true,
                Some(i) => {
                    if s.model.materials.get(i).is_some_and(|m| m.eps_r.get() != 1.0) {
                        dielectric.set(Some(i));
                    }
                    false
                }
            }
        });
        if let Some(i) = dielectric.get() {
            let name = s.model.materials.get(i).map_or("?", |m| m.name.as_str());
            return Err(EngineError::failed(format!(
                "not supported yet in a body of revolution: a dielectric (`{name}`): vacuum and conductors only"
            )));
        }
    }
    if m.magnetic[2] != 0.0 {
        run.set(5, |_, _| m.magnetic[2]);
    }
    let population = run.add(Rings { species: ELECTRON, rings: Vec::new() }).map_err(fdtd)?;
    let beams = m.beams.iter().map(|b| (b.face, pic::Pulse::Steady { current: b.current, rise: b.rise }, b.inner, b.radius, b.energy, b.per_step));
    let bunches = m.bunches.iter().map(|b| (b.face, bunch_pulse(b), 0.0, b.radius, b.energy, b.per_step));
    for (face, pulse, inner, outer, energy, per_step) in beams.chain(bunches) {
        let injection = Injection { high: face.high, population, pulse, inner, radius: outer, energy: energy * EV, per_step };
        run.inject(injection).map_err(|e| EngineError::failed(format!("the beam through {}: {e}", face.name())))?;
    }
    let origin = [-radius, -radius, z_origin];
    let probes: Vec<[f64; 3]> = m.probes.iter().map(|x| std::array::from_fn(|d| x[d] - origin[d])).collect();
    let mut axi = Axi { pic: run, radius };
    let before = axi.energies().0;
    let rec = record(&mut axi, steps, dt, &probes)?;
    let frame = Frame { origin, cells: [2 * nr, 2 * nr, nz + below + above], h: [dr, dr, dz], dt, steps, axisymmetric: true };
    let end = End { fields: (before, axi.energies().0), snapshot: snapshot(&axi.pic, z_origin), particles: axi.pic.rings() };
    finish(s, frame, rec, &probes, end)
}

/// At most [`SNAPSHOT`] of the run's rings, evenly chosen, each a particle where its ring passes
/// the golden angle times its index round the axis; `z0` the grid's z corner (m).
fn snapshot(run: &AxiPic, z0: f64) -> Vec<Snap> {
    let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    let stride = run.rings().div_ceil(SNAPSHOT).max(1);
    run.populations()
        .iter()
        .enumerate()
        .flat_map(|(k, pop)| pop.rings.iter().map(move |q| (k, pop.species, q)))
        .step_by(stride)
        .enumerate()
        .map(|(n, (population, species, q))| {
            let (s, c) = (golden * n as f64).sin_cos();
            Snap {
                x: [q.r * c, q.r * s, z0 + q.z],
                u: [q.u[0] * c - q.u[1] * s, q.u[0] * s + q.u[1] * c, q.u[2]],
                energy: species.kinetic(q.u) / EV,
                population,
            }
        })
        .collect()
}
