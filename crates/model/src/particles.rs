//! A particle-in-cell run's sources (docs/adr/0012-particles.md, stage B): how the box's walls
//! end each axis, the walls that emit electrons, the plasmas loaded at the start, a voltage
//! held across an axis, a uniform static magnetic field and the points whose field is recorded.
//! The box is the model's geometry's bounds; the engine's `solve.particles` runs it.
//!
//! Lengths are in metres, densities in m⁻³, velocities in m/s, energies in eV, voltages in V,
//! current densities in A/m², magnetic fields in T.

use serde::{Deserialize, Serialize};

use crate::error::{Result, invalid};
use crate::model::Model;
use crate::units::C0;

/// The most emitters, plasmas and probes a run holds.
pub const MAX_EMITTERS: usize = 6;
pub const MAX_PLASMAS: usize = 8;
pub const MAX_PROBES: usize = 16;

/// The most macro-particles a plasma loads in a cell along an axis, and in a cell in all.
pub const MAX_PER_AXIS: usize = 64;
pub const MAX_PER_CELL: usize = 512;

/// The largest density (m⁻³), energy (eV), voltage (V), current density (A/m²), speed (m/s)
/// and magnetic field (T) a run takes.
pub const MAX_DENSITY: f64 = 1e32;
pub const MAX_ENERGY: f64 = 1e10;
pub const MAX_VOLTAGE: f64 = 1e10;
pub const MAX_CURRENT_DENSITY: f64 = 1e14;
pub const MAX_SPEED: f64 = 3e10;
pub const MAX_MAGNETIC: f64 = 1e4;

/// A kind of particle: its charge (C) and rest mass (kg).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Species {
    pub charge: f64,
    pub mass: f64,
}

/// The electron (CODATA 2018: the elementary charge exact, mₑ).
pub const ELECTRON: Species = Species { charge: -1.602_176_634e-19, mass: 9.109_383_701_5e-31 };

/// u·u for a vector u.
fn norm2(u: [f64; 3]) -> f64 {
    u[0] * u[0] + u[1] * u[1] + u[2] * u[2]
}

impl Species {
    /// Its rest energy mc² (J).
    pub fn rest_energy(self) -> f64 {
        self.mass * C0 * C0
    }

    /// The momentum u = γv (m/s) of kinetic energy `kinetic` (J) along unit vector `along`.
    ///
    /// ```
    /// use emcraft_model::particles::{ELECTRON, gamma};
    /// let mev = 1e6 * -ELECTRON.charge;
    /// let u = ELECTRON.momentum(mev, [0.0, 0.0, 1.0]);
    /// // γ = 1 + T/(mc²): 1 MeV on 0.511 MeV of rest energy.
    /// assert!((gamma(u) - (1.0 + mev / ELECTRON.rest_energy())).abs() < 1e-12);
    /// ```
    pub fn momentum(self, kinetic: f64, along: [f64; 3]) -> [f64; 3] {
        let g = 1.0 + kinetic / self.rest_energy();
        let speed = C0 * (g * g - 1.0).max(0.0).sqrt();
        let n = norm2(along).sqrt();
        if n > 0.0 { along.map(|c| c * (speed / n)) } else { [0.0; 3] }
    }

    /// The kinetic energy (γ − 1)mc² (J) of momentum `u` (m/s), accurate for small speeds too.
    pub fn kinetic(self, u: [f64; 3]) -> f64 {
        let u2 = norm2(u) / (C0 * C0);
        // γ − 1 = u²/(γ + 1): no cancellation at small u.
        self.rest_energy() * u2 / (gamma(u) + 1.0)
    }
}

/// γ of momentum `u` = γv (m/s).
#[inline]
pub fn gamma(u: [f64; 3]) -> f64 {
    (1.0 + norm2(u) / (C0 * C0)).sqrt()
}

/// The most macro-particles a wall's node emits a step.
pub const MAX_PER_NODE: usize = 64;

/// The most cells along z per cell across in a body of revolution.
pub const MAX_FINE: usize = 16;

emcraft_macros::named_enum! {
    /// How a step turns a particle's momentum in the fields.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum Pusher {
        /// Boris's rotation (J. P. Boris, 1970): the usual one.
        #[default]
        Boris = "boris",
        /// Vay's (Phys. Plasmas 15:056701, 2008): right for relativistic particles whose E and
        /// v × B cancel.
        Vay = "vay",
    }
}

/// The grid a particle run is on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RunGrid {
    /// The box: the geometry's bounds in cubic cells.
    #[default]
    Box,
    /// A body of revolution about z, its cells `fine` times finer along z than across.
    Revolution { fine: std::num::NonZeroUsize },
}

impl RunGrid {
    /// The grid a run's flags ask for: `fine` (1 to [`MAX_FINE`]) other than 1 only in a body
    /// of revolution.
    ///
    /// ```
    /// use emcraft_model::particles::RunGrid;
    /// assert!(matches!(RunGrid::new(true, 4), Ok(RunGrid::Revolution { .. })));
    /// assert_eq!(RunGrid::new(false, 1).unwrap(), RunGrid::Box);
    /// assert!(RunGrid::new(false, 4).is_err() && RunGrid::new(true, 0).is_err() && RunGrid::new(true, 17).is_err());
    /// ```
    pub fn new(axisymmetric: bool, fine: usize) -> std::result::Result<RunGrid, String> {
        let fine = std::num::NonZeroUsize::new(fine)
            .filter(|f| f.get() <= MAX_FINE)
            .ok_or_else(|| format!("1 to {MAX_FINE} cells along z per cell across"))?;
        match (axisymmetric, fine.get()) {
            (true, _) => Ok(RunGrid::Revolution { fine }),
            (false, 1) => Ok(RunGrid::Box),
            (false, _) => Err("cells finer along z are for a body of revolution (axisymmetric)".to_owned()),
        }
    }

    /// Cells along z per cell across: 1 in the box.
    pub fn fine(self) -> usize {
        match self {
            RunGrid::Box => 1,
            RunGrid::Revolution { fine } => fine.get(),
        }
    }
}

emcraft_macros::named_enum! {
    /// How the box ends an axis.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum Wall {
        /// Perfectly conducting walls at both ends: particles reaching them are absorbed.
        #[default]
        Conductor = "conductor",
        /// What leaves through one end comes back through the other.
        Periodic = "periodic",
    }
}

emcraft_macros::named_serde!(Wall);

/// A face of the box: normal to `axis` (0 to 2), at its high end or its low one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Face {
    pub axis: usize,
    pub high: bool,
}

impl Face {
    /// The face named `x-`, `x+`, `y-`, `y+`, `z-` or `z+`.
    ///
    /// ```
    /// use emcraft_model::particles::Face;
    /// assert_eq!(Face::parse("z+"), Some(Face { axis: 2, high: true }));
    /// assert_eq!(Face::parse("z+").map(|f| f.name()), Some("z+".to_owned()));
    /// assert_eq!(Face::parse("w-"), None);
    /// ```
    pub fn parse(s: &str) -> Option<Face> {
        let mut c = s.chars();
        let axis = match c.next()? {
            'x' => 0,
            'y' => 1,
            'z' => 2,
            _ => return None,
        };
        let high = match (c.next()?, c.next()) {
            ('+', None) => true,
            ('-', None) => false,
            _ => return None,
        };
        Some(Face { axis, high })
    }

    /// The faces' names, as [`Face::parse`] reads them.
    pub const IDS: [&'static str; 6] = ["x-", "x+", "y-", "y+", "z-", "z+"];

    /// Its name, as [`Face::parse`] reads it.
    pub fn name(self) -> String {
        format!("{}{}", ["x", "y", "z"].get(self.axis).copied().unwrap_or("?"), if self.high { '+' } else { '-' })
    }
}

impl TryFrom<String> for Face {
    type Error = String;
    fn try_from(s: String) -> std::result::Result<Face, String> {
        Face::parse(&s).ok_or_else(|| format!("a face is x-, x+, y-, y+, z- or z+, not `{s}`"))
    }
}

impl From<Face> for String {
    fn from(f: Face) -> String {
        f.name()
    }
}

/// How much a wall emits.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Emission {
    /// As much as the field at the wall draws (space-charge limited).
    Limited,
    /// A current density (A/m²).
    Current { density: f64 },
}

/// A wall emitting electrons.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Emitter {
    pub face: Face,
    pub emission: Emission,
    /// The electrons' kinetic energy as they leave (eV).
    #[serde(default)]
    pub energy: f64,
    /// Macro-particles from each node of the wall a step.
    pub per_node: usize,
}

/// A wave on a plasma's velocities: `amplitude` (m/s) along `axis`, `modes` wavelengths across
/// the box along it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Wave {
    pub axis: usize,
    pub amplitude: f64,
    pub modes: usize,
}

/// Electrons of `density` filling the box at the start on a neutralising background of
/// immobile ions, `per_cell` macro-particles a cell along each axis, moving at `drift` with
/// `wave` on it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Plasma {
    pub density: f64,
    pub per_cell: [usize; 3],
    #[serde(default)]
    pub drift: [f64; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wave: Option<Wave>,
}

/// A Gaussian bunch of electrons sent in through a conducting face: `charge` (C, its size), rms
/// length `length` (m) along the face's normal at its speed, `energy` (eV), spread over a disc
/// of `radius` (m) about `at` (m, a point of the face; none: its centre), its peak entering
/// `delay` (s) after the start (none: five of its rms durations), `per_step` macro-particles a
/// step.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bunch {
    pub face: Face,
    pub charge: f64,
    pub length: f64,
    pub energy: f64,
    #[serde(default)]
    pub radius: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delay: Option<f64>,
    pub per_step: usize,
}

/// A steady beam of electrons sent in through a conducting face: `current` (A, its size) at
/// `energy` (eV) along the face's normal, spread evenly over the ring between `inner` and `radius`
/// (m; a disc when `inner` is 0) about `at` (m, a point of the face; none: its centre), reached
/// linearly over `rise` (s), `per_step` macro-particles a step.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Beam {
    pub face: Face,
    pub current: f64,
    pub energy: f64,
    pub radius: f64,
    #[serde(default)]
    pub inner: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<[f64; 3]>,
    #[serde(default)]
    pub rise: f64,
    pub per_step: usize,
}

/// The largest beam current (A).
pub const MAX_BEAM_CURRENT: f64 = 1e8;

/// The largest bunch charge (C) and length (m).
pub const MAX_BUNCH_CHARGE: f64 = 1e3;
pub const MAX_LENGTH: f64 = 1e3;

/// A voltage held across the walls of `axis` by the outer circuit: the high wall `volts` above
/// the low one.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Voltage {
    pub axis: usize,
    pub volts: f64,
}

/// The run's sources.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Particles {
    #[serde(default)]
    pub walls: [Wall; 3],
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub emitters: Vec<Emitter>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub plasmas: Vec<Plasma>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bunches: Vec<Bunch>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub beams: Vec<Beam>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voltage: Option<Voltage>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub magnetic: [f64; 3],
    /// Faces left open: what reaches them leaves the box (an absorbing layer beyond them, the
    /// geometry carried on into it).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub open: Vec<Face>,
    /// Points whose field is recorded (m, the model's coordinates).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub probes: Vec<[f64; 3]>,
    /// Planes across z whose power through them (towards +z) is recorded: their z (m, the
    /// model's coordinates). A body of revolution's run only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<f64>,
}

fn is_zero(b: &[f64; 3]) -> bool {
    b.iter().all(|x| *x == 0.0)
}

/// Whether `x` is a finite number of at most `max` in size (and ≥ 0 when `signed` is false).
fn within(x: f64, max: f64, signed: bool) -> bool {
    x.is_finite() && x.abs() <= max && (signed || x >= 0.0)
}

impl Particles {
    /// Whether it holds anything to run: an emitter or a plasma.
    pub fn is_empty(&self) -> bool {
        *self == Particles::default()
    }

    /// Whether something puts particles in the box: an emitter, a plasma, a bunch or a beam.
    pub fn has_sources(&self) -> bool {
        !(self.emitters.is_empty() && self.plasmas.is_empty() && self.bunches.is_empty() && self.beams.is_empty())
    }

    /// The sources checked: their counts and every number within its bounds.
    ///
    /// ```
    /// use emcraft_model::particles::{Face, Emission, Emitter, Particles};
    /// let mut p = Particles::default();
    /// p.emitters.push(Emitter { face: Face { axis: 2, high: false }, emission: Emission::Limited, energy: 0.0, per_node: 4 });
    /// assert!(p.check().is_ok());
    /// p.emitters[0].per_node = 0;
    /// assert!(p.check().is_err());
    /// ```
    pub fn check(&self) -> Result<()> {
        if self.emitters.len() > MAX_EMITTERS
            || self.plasmas.len() > MAX_PLASMAS
            || self.probes.len() > MAX_PROBES
            || self.outputs.len() > MAX_PROBES
            || self.bunches.len() > MAX_EMITTERS
        {
            return Err(invalid(
                "particles",
                format!("at most {MAX_EMITTERS} emitters and bunches, {MAX_PLASMAS} plasmas, {MAX_PROBES} probes and {MAX_PROBES} output planes"),
            ));
        }
        if self.beams.len() > MAX_EMITTERS {
            return Err(invalid("particles", format!("at most {MAX_EMITTERS} beams")));
        }
        for b in &self.beams {
            let at = b.at.is_none_or(|x| x.iter().all(|c| c.is_finite()));
            let ok = b.face.axis <= 2
                && within(b.current, MAX_BEAM_CURRENT, false)
                && within(b.energy, MAX_ENERGY, false)
                && b.energy > 0.0
                && within(b.radius, MAX_LENGTH, false)
                && within(b.inner, b.radius, false)
                && within(b.rise, 1.0, false)
                && (1..=MAX_PER_NODE).contains(&b.per_step);
            if !(ok && at) {
                return Err(invalid(
                    "beam",
                    format!("{b:?}: a face, a current ≥ 0, an energy > 0, radii 0 ≤ inner ≤ outer, a rise ≥ 0, 1 to {MAX_PER_NODE} particles a step"),
                ));
            }
        }
        for b in &self.bunches {
            let at = b.at.is_none_or(|x| x.iter().all(|c| c.is_finite()));
            let delay = b.delay.is_none_or(|d| within(d, 1.0, false));
            let ok = b.face.axis <= 2
                && within(b.charge, MAX_BUNCH_CHARGE, false)
                && within(b.length, MAX_LENGTH, false)
                && b.length > 0.0
                && within(b.energy, MAX_ENERGY, false)
                && b.energy > 0.0
                && within(b.radius, MAX_LENGTH, false)
                && (1..=MAX_PER_NODE).contains(&b.per_step);
            if !(ok && at && delay) {
                return Err(invalid(
                    "bunch",
                    format!("{b:?}: a face, a charge ≥ 0, a length and an energy > 0, a radius ≥ 0, 1 to {MAX_PER_NODE} particles a step"),
                ));
            }
        }
        for e in &self.emitters {
            let current_ok = match e.emission {
                Emission::Limited => true,
                Emission::Current { density } => within(density, MAX_CURRENT_DENSITY, false),
            };
            if e.face.axis > 2 || !current_ok || !within(e.energy, MAX_ENERGY, false) || !(1..=MAX_PER_NODE).contains(&e.per_node) {
                return Err(invalid(
                    "emitter",
                    format!("{e:?}: a face, a current density ≥ 0, an energy ≥ 0 and 1 to {MAX_PER_NODE} particles a node"),
                ));
            }
        }
        for p in &self.plasmas {
            let per = p.per_cell.iter().all(|n| (1..=MAX_PER_AXIS).contains(n)) && p.per_cell.iter().product::<usize>() <= MAX_PER_CELL;
            let wave = p.wave.is_none_or(|w| w.axis <= 2 && within(w.amplitude, MAX_SPEED, true) && (1..=1024).contains(&w.modes));
            if !(within(p.density, MAX_DENSITY, false) && per && p.drift.iter().all(|v| within(*v, MAX_SPEED, true)) && wave) {
                return Err(invalid(
                    "plasma",
                    format!("{p:?}: a density ≥ 0, 1 to {MAX_PER_AXIS} particles a cell along each axis ({MAX_PER_CELL} in all), finite speeds"),
                ));
            }
        }
        if self.open.iter().any(|f| f.axis > 2) || self.open.len() > 6 {
            return Err(invalid("particles", "open faces are among x-, x+, y-, y+, z- and z+"));
        }
        if let Some(v) = self.voltage
            && (v.axis > 2 || !within(v.volts, MAX_VOLTAGE, true))
        {
            return Err(invalid("voltage", format!("{v:?}: an axis and a finite voltage")));
        }
        if !self.magnetic.iter().all(|b| within(*b, MAX_MAGNETIC, true)) || !self.probes.iter().flatten().chain(&self.outputs).all(|x| x.is_finite())
        {
            return Err(invalid("particles", "the magnetic field and the probes must be finite"));
        }
        Ok(())
    }
}

impl Model {
    /// Sets the particle-in-cell run's sources after checking them.
    pub fn set_particles(&mut self, particles: Particles) -> Result<()> {
        particles.check()?;
        self.particles = particles;
        Ok(())
    }
}
