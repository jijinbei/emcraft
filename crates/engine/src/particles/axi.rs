//! `solve.particles` in a body of revolution (`axisymmetric`): the model's geometry taken as a
//! body about the z axis (x = y = 0, as solve.revolution takes it), run on
//! `emcraft_fdtd::axi`'s r–z grid (the fields that do not vary round the axis, the particles
//! rings about it).
//!
//! **The grid.** The cylinder's radius is as far from the axis as the box reaches; its cells about `cell`
//! across and `fine` times finer along z (8 for beams faster than about 0.6c: the numerical
//! Cherenkov instability, `emcraft_fdtd::axi`'s docs); the time step 0.95 of the grid's limit.
//! The ends conducting or periodic as particles.walls sets z; an open end a CPML of
//! [`super::LAYER`] × `fine` cells (at most `emcraft_fdtd::pic::MAX_LAYER`) beyond it, the geometry carried on into it as it is at the
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
use emcraft_gpu::{AxiDevice, Gpu};
use emcraft_model::ParticleRun;
use emcraft_particles::ELECTRON;
use serde_json::Value;

use super::{
    COURANT, EV, FieldMap, Frame, INSET, LAYER, MAP_POINTS, MAX_FRAMES, Run, SNAPSHOT, Snap, cells_along, fdtd, injections, open_on_periodic,
    run_to_end, steps_of,
};
use crate::{EngineError, Params, Result, Session};

/// A run in a body of revolution, its radius R (m) and the z of its grid's low end (m): the
/// grid's corner sits at (−R, −R, z₀).
struct Axi {
    pic: AxiPic,
    radius: f64,
    z_origin: f64,
}

/// A point about the axis: its radius and z (m), and the cosine and sine of its angle.
struct Polar {
    r: f64,
    z: f64,
    cos: f64,
    sin: f64,
}

impl Axi {
    /// A point (m from the grid's corner) about the axis.
    fn polar(&self, at: [f64; 3]) -> Polar {
        let (x, y) = (at[0] - self.radius, at[1] - self.radius);
        let r = x.hypot(y);
        let (cos, sin) = if r > 0.0 { (x / r, y / r) } else { (1.0, 0.0) };
        Polar { r, z: at[2], cos, sin }
    }
}

impl Run for Axi {
    fn advance(&mut self) -> Result<pic::Moved> {
        self.pic.step().map_err(fdtd)
    }
    fn energy(&mut self) -> Result<pic::Energy> {
        Ok(self.pic.energy())
    }
    fn held(&self) -> usize {
        self.pic.rings()
    }
    fn e_at_point(&self, at: [f64; 3]) -> [f64; 3] {
        let Polar { r, z, cos, sin } = self.polar(at);
        let [er, et, ez] = self.pic.field_at(r, z).0;
        [er * cos - et * sin, er * sin + et * cos, ez]
    }
    fn plane(&self, z: f64) -> Option<emcraft_fdtd::axi::Plane> {
        Some(self.pic.plane(z))
    }
    /// E_z on the r–z plane, every few points so it holds at most [`MAP_POINTS`]: at most 60
    /// across, the rest along z.
    fn field_map(&self) -> Option<FieldMap> {
        let g = self.pic.grid();
        let across = g.nr + 1;
        let every_r = across.div_ceil(60).max(1);
        let along = (MAP_POINTS / across.div_ceil(every_r)).max(1);
        let every_z = g.z.cells.div_ceil(along).max(1);
        let m = self.pic.sampled(emcraft_fdtd::axi::Component::Ez, every_r, every_z);
        Some(FieldMap { r: m.r, z: m.z.iter().map(|z| z + self.z_origin).collect(), ez: m.values })
    }
    fn now(&self) -> f64 {
        self.pic.time()
    }
    /// The rings, each a particle where its ring passes the golden angle times its index round
    /// the axis.
    fn snapshot(&self) -> Vec<Snap> {
        let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        let stride = self.pic.rings().div_ceil(SNAPSHOT).max(1);
        self.pic
            .populations()
            .iter()
            .enumerate()
            .flat_map(|(k, pop)| pop.rings.iter().map(move |q| (k, pop.species, q)))
            .step_by(stride)
            .enumerate()
            .map(|(n, (population, species, q))| {
                let (s, c) = (golden * n as f64).sin_cos();
                Snap {
                    x: [q.r * c, q.r * s, self.z_origin + q.z],
                    u: [q.u[0] * c - q.u[1] * s, q.u[0] * s + q.u[1] * c, q.u[2]],
                    energy: species.kinetic(q.u) / EV,
                    population,
                }
            })
            .collect()
    }
}

/// A run in a body of revolution on a device (`emcraft_gpu::AxiDevice`): its clock and
/// injections on the host, its fields and rings on the device until [`Run::end`] brings them
/// back; E at the probes (in the half plane θ = 0 at each one's r and z) recorded there each
/// step.
struct OnDevice {
    axi: Axi,
    device: Option<AxiDevice>,
    probes: Vec<Polar>,
}

impl OnDevice {
    /// `axi` moved onto `gpu`, E watched at `probes` (m from the grid's corner).
    fn new(mut axi: Axi, gpu: Gpu, probes: &[[f64; 3]]) -> Result<OnDevice> {
        let probes: Vec<Polar> = probes.iter().map(|at| axi.polar(*at)).collect();
        let points: Vec<[f64; 2]> = probes.iter().map(|q| [q.r, q.z]).collect();
        let device = gpu.axi(&mut axi.pic, &points).map_err(fdtd)?;
        Ok(OnDevice { axi, device: Some(device), probes })
    }
}

impl Run for OnDevice {
    fn advance(&mut self) -> Result<pic::Moved> {
        Ok(self.span(1, &[], &[], &|_| false)?.into_iter().next().map_or_else(pic::Moved::default, |s| s.0))
    }
    fn energy(&mut self) -> Result<pic::Energy> {
        match self.device.as_mut() {
            Some(d) => d.energy(&mut self.axi.pic).map_err(fdtd),
            None => self.axi.energy(),
        }
    }
    fn held(&self) -> usize {
        self.device.as_ref().map_or_else(|| self.axi.held(), AxiDevice::held)
    }
    /// The field as the host last read it back (the device's own is recorded at the probes).
    fn e_at_point(&self, at: [f64; 3]) -> [f64; 3] {
        self.axi.e_at_point(at)
    }
    fn now(&self) -> f64 {
        self.axi.now()
    }
    fn snapshot(&self) -> Vec<Snap> {
        self.axi.snapshot()
    }
    /// The steps on the device, E at the probes (the ones it was made with) turned from the
    /// half plane θ = 0 to x and y; no output planes (refused before a run on a device).
    fn span(&mut self, steps: usize, _: &[[f64; 3]], _: &[f64], sampled: &dyn Fn(usize) -> bool) -> Result<Vec<super::Stepped>> {
        let Some(device) = self.device.as_mut() else {
            return (0..steps).map(|_| Ok((self.axi.advance()?, None, Vec::new()))).collect();
        };
        let done = device.advance(&mut self.axi.pic, steps).map_err(fdtd)?;
        Ok(done
            .into_iter()
            .enumerate()
            .map(|(b, step)| {
                let probes = sampled(b).then(|| {
                    step.probes.iter().zip(&self.probes).map(|([er, et, ez], q)| [er * q.cos - et * q.sin, er * q.sin + et * q.cos, *ez]).collect()
                });
                (step.moved, probes, Vec::new())
            })
            .collect())
    }
    fn end(&mut self) -> Result<()> {
        match self.device.take() {
            Some(d) => d.finish(&mut self.axi.pic).map_err(fdtd),
            None => Ok(()),
        }
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

/// The run in a body of revolution, on `gpu` if given (else the CPU's threads).
pub(super) fn solve(s: &mut Session, p: &Params, set: &ParticleRun, fine: usize, gpu: Option<Gpu>) -> Result<Value> {
    if let Some(why) = unsupported(s) {
        return Err(EngineError::failed(why));
    }
    let frames = p.integer("frames", 0..=MAX_FRAMES).or(0)?;
    if gpu.is_some() && (!s.model.particles.outputs.is_empty() || frames > 0) {
        return Err(EngineError::failed(
            "not supported yet: output planes (particles.output) or moments (`frames`) on a device; run on the CPU's threads (`device`: cpu)",
        ));
    }
    let b =
        s.model.bounds().ok_or_else(|| EngineError::failed("no geometry: the body of revolution particles run in is its bounds (geometry.solid)"))?;
    let m = s.model.particles.clone();
    let radius = [b.min.x, b.max.x, b.min.y, b.max.y].iter().fold(0.0f64, |r, c| r.max(c.abs()));
    if radius <= 0.0 || !radius.is_finite() {
        return Err(EngineError::failed("the geometry has no extent about the z axis: a body of revolution turns about x = y = 0"));
    }
    let (z_lo, z_hi) = (b.min.z, b.max.z);
    // At least two cells each way: a ring between the axis and the wall, and one between the ends.
    let (nr, dr) = cells_along(radius, set.cell.get(), 2);
    let (nz, dz) = cells_along(z_hi - z_lo, set.cell.get() / fine as f64, 2);
    open_on_periodic(&m)?;
    let opened = |high: bool| m.open.iter().any(|f| f.axis == 2 && f.high == high);
    let layer = (LAYER * fine).min(emcraft_fdtd::pic::MAX_LAYER);
    let (below, above) = (usize::from(opened(false)) * layer, usize::from(opened(true)) * layer);
    let grid = AxiGrid::new(nr, dr, pic::Axis { cells: nz + below + above, h: dz, wall: m.walls[2].into() }).map_err(fdtd)?;
    let z_origin = z_lo - below as f64 * dz;
    let dt = grid.time_step(COURANT);
    let steps = steps_of(p, set.time.get(), dt)?;
    let mut run = AxiPic::new(grid, dt, set.pusher).map_err(fdtd)?;
    for high in [false, true].into_iter().filter(|h| opened(*h)) {
        run.absorb(high, layer).map_err(fdtd)?;
    }
    if s.model.domain.solid.is_some() {
        // The half plane θ = 0, just inside the box (a solid's own faces are not inside it).
        let dielectric = Cell::new(None);
        let inset = INSET * radius;
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
        // Uniform, or falling away past `end` with the B_r that keeps it free of divergence.
        let b0 = m.magnetic[2];
        match m.magnetic_end {
            None => run.set(emcraft_fdtd::axi::Component::Bz, |_, _| b0),
            Some(end) => {
                run.set(emcraft_fdtd::axi::Component::Bz, |_, z| b0 * end.share(z + z_origin).0);
                run.set(emcraft_fdtd::axi::Component::Br, |r, z| -0.5 * r * b0 * end.share(z + z_origin).1);
            }
        }
    }
    let population = run.add(Rings { species: ELECTRON, rings: Vec::new() }).map_err(fdtd)?;
    for i in injections(&m) {
        let injection =
            Injection { high: i.face.high, population, pulse: i.pulse, inner: i.inner, radius: i.radius, energy: i.energy, per_step: i.per_step };
        run.inject(injection).map_err(|e| EngineError::failed(format!("the {} through {}: {e}", i.kind, i.face.name())))?;
    }
    let origin = [-radius, -radius, z_origin];
    let probes: Vec<[f64; 3]> = m.probes.iter().map(|x| std::array::from_fn(|d| x[d] - origin[d])).collect();
    let frame = Frame { origin, cells: [2 * nr, 2 * nr, nz + below + above], h: [dr, dr, dz], dt, steps, axisymmetric: true };
    let mut axi = Axi { pic: run, radius, z_origin };
    match gpu {
        Some(gpu) => run_to_end(s, &mut OnDevice::new(axi, gpu, &probes)?, frame, &probes, frames),
        None => run_to_end(s, &mut axi, frame, &probes, frames),
    }
}
