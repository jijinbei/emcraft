//! An axisymmetric run handed to a device and back (`emcraft_gpu`): the run's arrays as they
//! stand, the injections' rings made on the host step by step (the same quasi-random places as
//! on the CPU), and the state the device leaves written back.
//!
//! ```
//! use emcraft_fdtd::axi::{AxiGrid, AxiPic, Ring, Rings};
//! use emcraft_fdtd::pic::{Axis, Wall};
//! use emcraft_particles::{ELECTRON, Pusher};
//! let z = Axis { cells: 16, h: 0.5e-3, wall: Wall::Conductor };
//! let grid = AxiGrid::new(8, 0.5e-3, z)?;
//! let mut run = AxiPic::new(grid, grid.time_step(0.9), Pusher::Boris)?;
//! run.add(Rings { species: ELECTRON, rings: vec![Ring { r: 1e-3, z: 4e-3, u: [0.0; 3], w: 1e3 }] })?;
//! let plan = run.plan()?;
//! assert_eq!((plan.rings.len(), plan.shapes[0].len()), (1, 8 * 17));
//! // Two populations are not run on a device yet.
//! run.add(Rings { species: ELECTRON, rings: Vec::new() })?;
//! assert!(run.plan().is_err());
//! # Ok::<(), emcraft_fdtd::FdtdError>(())
//! ```

use emcraft_particles::{Pusher, Species};

use super::{AxiGrid, AxiPic, BR, BT, BZ, ER, ET, EZ, Lat, Ring};
use crate::pic::Moved;
use crate::yee::Cpml;
use crate::{Result, invalid};

/// A lattice's points along r and z, r running fastest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub nr: usize,
    pub nz: usize,
}

impl Shape {
    pub fn len(self) -> usize {
        self.nr * self.nz
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

impl From<Lat> for Shape {
    fn from(l: Lat) -> Shape {
        Shape { nr: l.nr, nz: l.nz }
    }
}

/// What a device needs of a run. The six components in the order E_r, E_z, E_θ, B_r, B_z, B_θ
/// (their lattices `shapes`).
pub struct Plan<'a> {
    pub grid: AxiGrid,
    pub dt: f64,
    pub pusher: Pusher,
    pub species: Species,
    pub shapes: [Shape; 6],
    pub fields: [&'a [f64]; 6],
    /// Whether each E_r, E_z, E_θ point touches metal (all empty without metal).
    pub held: [&'a [bool]; 3],
    /// Whether each cell (r fastest) is metal (empty without metal).
    pub metal: &'a [bool],
    pub layers: Option<Layers<'a>>,
    pub rings: &'a [Ring],
}

/// The absorbing layers along z: the coefficients at z's nodes and centres for E's steps (`e`)
/// and B's half steps (`b`), and the ψ of E_r's, E_θ's, B_θ's and B_r's z differences.
pub struct Layers<'a> {
    pub e: [&'a [Cpml]; 2],
    pub b: [&'a [Cpml]; 2],
    pub psi: [&'a [f64]; 4],
}

/// The state a device leaves: the six components (in [`Plan`]'s order), the layers' ψ (none
/// without layers) and the rings.
pub struct State {
    pub fields: [Vec<f64>; 6],
    pub psi: Option<[Vec<f64>; 4]>,
    pub rings: Vec<Ring>,
}

impl AxiPic {
    /// The run as a device takes it; refused with other than one population (not supported yet
    /// on a device).
    pub fn plan(&self) -> Result<Plan<'_>> {
        let [pop] = self.populations.as_slice() else {
            return Err(invalid("device", "not supported yet: other than one population of rings on a device"));
        };
        let f = &self.f;
        Ok(Plan {
            grid: self.grid,
            dt: self.dt,
            pusher: self.pusher,
            species: pop.species,
            shapes: [ER, EZ, ET, BR, BZ, BT].map(|w| Shape::from(self.lats[w])),
            fields: [&f.er, &f.ez, &f.et, &f.br, &f.bz, &f.bt],
            held: [&self.held[0], &self.held[1], &self.held[2]],
            metal: self.metal.as_deref().unwrap_or(&[]),
            layers: self.layers.as_ref().map(|l| Layers {
                e: [&l.e[0], &l.e[1]],
                b: [&l.b[0], &l.b[1]],
                psi: [&l.psi[0], &l.psi[1], &l.psi[2], &l.psi[3]],
            }),
            rings: &pop.rings,
        })
    }

    /// The rings born this step (a device holding `held`) and what they brought in; the
    /// run's clock moved on a step, as [`AxiPic::step`] moves it.
    pub fn births_and_tick(&mut self, held: usize) -> Result<(Moved, Vec<Ring>)> {
        let mut moved = Moved::default();
        let born = self.births(held, &mut moved)?.into_iter().flat_map(|(_, rings)| rings).collect();
        self.steps += 1;
        Ok((moved, born))
    }

    /// The rings taken out, the device holding them now (the run's energy then the field's
    /// alone).
    pub fn take_rings(&mut self) -> Vec<Ring> {
        self.populations.iter_mut().flat_map(|p| std::mem::take(&mut p.rings)).collect()
    }

    /// The fields as a device has them, checked against the lattices.
    pub fn put_fields(&mut self, fields: [Vec<f64>; 6]) -> Result<()> {
        let lats = [ER, EZ, ET, BR, BZ, BT].map(|w| self.lats[w]);
        if fields.iter().zip(&lats).any(|(v, l)| v.len() != l.len()) {
            return Err(invalid("device", "fields of other sizes than the grid's"));
        }
        let [er, ez, et, br, bz, bt] = fields;
        self.f = super::Fields { er, ez, et, br, bz, bt };
        Ok(())
    }

    /// The state a device leaves written back: its fields, the layers' ψ and the rings (into
    /// the one population).
    pub fn put_back(&mut self, state: State) -> Result<()> {
        let State { fields, psi, rings } = state;
        match (self.layers.as_mut(), psi) {
            (Some(l), Some(psi)) if psi.iter().zip(&l.psi).all(|(a, b)| a.len() == b.len()) => l.psi = psi,
            (None, None) => {}
            _ => return Err(invalid("device", "the layers' state of other sizes than the run's")),
        }
        self.put_fields(fields)?;
        let [pop] = self.populations.as_mut_slice() else {
            return Err(invalid("device", "not supported yet: other than one population of rings on a device"));
        };
        pop.rings = rings;
        Ok(())
    }
}
