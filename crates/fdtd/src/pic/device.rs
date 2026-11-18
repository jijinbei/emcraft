//! A Cartesian particle-in-cell run handed to a device and back (`emcraft_gpu`): the run's
//! arrays as they stand, the bunches' particles made on the host step by step (the same
//! quasi-random places as on the CPU), and the state the device leaves written back.
//!
//! Emitting walls (whose emission reads E at the wall each step) and a held voltage (the mean
//! current taken out each step) are not run on a device yet.
//!
//! ```
//! use emcraft_fdtd::pic::{Axis, Grid, Particle, Pic, Population, Wall};
//! use emcraft_particles::{ELECTRON, Pusher};
//! let a = Axis { cells: 4, h: 1e-3, wall: Wall::Periodic };
//! let mut run = Pic::new(Grid::new([a, a, a])?, 1e-12, Pusher::Boris)?;
//! run.add(Population { species: ELECTRON, particles: vec![Particle { x: [1e-3; 3], u: [0.0; 3], w: 1e3 }] })?;
//! let plan = run.plan()?;
//! assert_eq!((plan.particles.len(), plan.fields[0].len()), (1, 4 * 4 * 4));
//! # Ok::<(), emcraft_fdtd::FdtdError>(())
//! ```

use emcraft_particles::{Pusher, Species};

use super::{Grid, Moved, Particle, Pic, fields, zeros};
use crate::yee::Cpml;
use crate::{FdtdError, Result, invalid};

/// What a device needs of a run. The six components in the order E_x, E_y, E_z, B_x, B_y, B_z,
/// each on its lattice x fastest: E_a at the cells' centres along a and the nodes across, B_a at
/// the nodes along a and the centres across (a periodic axis has as many nodes as cells).
pub struct Plan<'a> {
    pub grid: Grid,
    pub dt: f64,
    pub pusher: Pusher,
    /// Each population's species: a particle's kind is its population's index.
    pub species: Vec<Species>,
    pub fields: [&'a [f64]; 6],
    pub media: Option<Media<'a>>,
    pub layers: Option<Layers<'a>>,
    /// Binomial passes over the current along each axis each step.
    pub smoothing: usize,
    /// The particles, each with its kind, population by population.
    pub particles: Vec<(u32, Particle)>,
}

/// What fills the cells: whether each E_a edge is held at zero (it touches metal), its relative
/// permittivity, and whether each cell (x fastest) is metal.
pub struct Media<'a> {
    pub held: [&'a [bool]; 3],
    pub eps: [&'a [f64]; 3],
    pub metal: &'a [bool],
}

/// The absorbing layers: per axis, the coefficients at its nodes and at its cells' centres for
/// E's whole steps (`e`) and B's half steps (`b`); the ψ of E_a's two differences (along a + 1
/// and a + 2) and of B_a's, on their lattices.
pub struct Layers<'a> {
    pub e: [[&'a [Cpml]; 2]; 3],
    pub b: [[&'a [Cpml]; 2]; 3],
    pub psi_e: [[&'a [f64]; 2]; 3],
    pub psi_b: [[&'a [f64]; 2]; 3],
}

/// The state a device leaves: the six components (in [`Plan`]'s order), the layers' ψ (E's then
/// B's; none without layers) and the particles, each with its kind.
pub struct State {
    pub fields: [Vec<f64>; 6],
    pub psi: Option<[[[Vec<f64>; 2]; 3]; 2]>,
    pub particles: Vec<(u32, Particle)>,
}

/// Each axis's pair of arrays as slices.
fn slices<T>(s: &[[Vec<T>; 2]; 3]) -> [[&[T]; 2]; 3] {
    std::array::from_fn(|d| [s[d][0].as_slice(), s[d][1].as_slice()])
}

impl Pic {
    /// The run as a device takes it; refused with emitting walls or a held voltage (not
    /// supported yet on a device).
    pub fn plan(&self) -> Result<Plan<'_>> {
        if !self.emitters.is_empty() {
            return Err(FdtdError::Unsupported("emitting walls on a device".into()));
        }
        if self.hold.is_some() {
            return Err(FdtdError::Unsupported("a held voltage on a device".into()));
        }
        let [ex, ey, ez] = &self.e;
        let [bx, by, bz] = &self.b;
        Ok(Plan {
            grid: self.grid,
            dt: self.dt,
            pusher: self.pusher,
            species: self.populations.iter().map(|p| p.species).collect(),
            fields: [ex, ey, ez, bx, by, bz],
            media: self.media.as_ref().map(|m| Media {
                held: [&m.edges[0], &m.edges[1], &m.edges[2]],
                eps: [&m.eps[0], &m.eps[1], &m.eps[2]],
                metal: &m.metal,
            }),
            layers: self.absorber.as_ref().map(|a| Layers { e: slices(&a.e), b: slices(&a.b), psi_e: slices(&a.psi_e), psi_b: slices(&a.psi_b) }),
            smoothing: self.smoothing,
            particles: self.populations.iter().zip(0u32..).flat_map(|(p, k)| p.particles.iter().map(move |q| (k, *q))).collect(),
        })
    }

    /// The part of E's step a steady B₀ (B_x, B_y, B_z on their lattices; T) makes, c²∇ × B₀
    /// (V/(m·s)) at E's points as Ampère takes its differences, without the layers' stretching;
    /// zero where E is not updated.
    pub fn steady_curl(&self, b0: &[Vec<f64>; 3]) -> Result<[Vec<f64>; 3]> {
        if b0.iter().zip(&self.b).any(|(a, b)| a.len() != b.len()) {
            return Err(invalid("device", "a steady field of other sizes than the grid's"));
        }
        let [x, y, z] = [0, 1, 2].map(|a| zeros(self.grid.e(a).len()));
        let mut curl = [x?, y?, z?];
        let [jx, jy, jz] = [0, 1, 2].map(|a| zeros(self.grid.e(a).len()));
        fields::ampere(&self.grid, &mut curl, (b0, &[jx?, jy?, jz?]), 1.0, (None, None));
        Ok(curl)
    }

    /// The particles born this step (each with its kind; a device holding `held`) and what they
    /// brought in; the run's clock moved on a step, as [`Pic::step`] moves it.
    pub fn births_and_tick(&mut self, held: usize) -> Result<(Moved, Vec<(u32, Particle)>)> {
        let mut moved = Moved::default();
        let born = self.births(held, &mut moved)?.into_iter().flat_map(|(k, new)| new.into_iter().map(move |p| (k as u32, p))).collect();
        self.steps += 1;
        Ok((moved, born))
    }

    /// The particles taken out, the device holding them now (the run's energy then the
    /// field's alone).
    pub fn take_particles(&mut self) {
        self.populations.iter_mut().for_each(|p| p.particles.clear());
    }

    /// The fields as a device has them, checked against the lattices.
    pub fn put_fields(&mut self, fields: [Vec<f64>; 6]) -> Result<()> {
        let lens = [0, 1, 2].map(|a| self.grid.e(a).len()).into_iter().chain([0, 1, 2].map(|a| self.grid.b(a).len()));
        if fields.iter().zip(lens).any(|(v, n)| v.len() != n) {
            return Err(invalid("device", "fields of other sizes than the grid's"));
        }
        let [ex, ey, ez, bx, by, bz] = fields;
        self.e = [ex, ey, ez];
        self.b = [bx, by, bz];
        Ok(())
    }

    /// The state a device leaves written back: its fields, the layers' ψ and the particles
    /// (each into its population).
    pub fn put_back(&mut self, state: State) -> Result<()> {
        let State { fields, psi, particles } = state;
        match (self.absorber.as_mut(), psi) {
            (Some(a), Some([e, b])) => {
                let fits = |new: &[[Vec<f64>; 2]; 3], old: &[[Vec<f64>; 2]; 3]| {
                    new.iter().flatten().zip(old.iter().flatten()).all(|(x, y)| x.len() == y.len())
                };
                if !(fits(&e, &a.psi_e) && fits(&b, &a.psi_b)) {
                    return Err(invalid("device", "the layers' state of other sizes than the run's"));
                }
                a.psi_e = e;
                a.psi_b = b;
            }
            (None, None) => {}
            _ => return Err(invalid("device", "the layers' state of other sizes than the run's")),
        }
        self.put_fields(fields)?;
        self.take_particles();
        for (k, p) in particles {
            let pop = self.populations.get_mut(k as usize).ok_or_else(|| invalid("device", format!("a particle of population {k}")))?;
            pop.particles.push(p);
        }
        Ok(())
    }
}
