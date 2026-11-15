//! Particle-in-cell in a body of revolution (docs/adr/0012-particles.md): the fields that do not
//! vary round the axis (m = 0) on an r–z grid, the particles rings about it.
//!
//! **Fields.** Both sets of an m = 0 field: E_r, E_z, B_θ (TM) and E_θ, B_r, B_z (TE, for an
//! axial magnetic field and the particles' turning). On the r–z Yee grid (as
//! C. K. Birdsall, A. B. Langdon, *Plasma Physics via Computer Simulation*, 1991, §14-11, and
//! the codes that model BWOs this way) E_r lies at (r_{i+½}, z_k), E_z at (r_i, z_{k+½}), E_θ at
//! (r_i, z_k), B_r at (r_i, z_{k+½}), B_z at (r_{i+½}, z_k), B_θ at (r_{i+½}, z_{k+½}); the
//! curls are written as circulations over the rings' faces (finite integration), so E_z on the
//! axis takes 4B_θ(r_½)/Δr and B_z next to it r_1 E_θ(r_1) over r_½Δr. ε₀ ∂E/∂t = ∇×B/μ₀ − J,
//! ∂B/∂t = −∇×E, leapfrogged as the Cartesian grid's are.
//!
//! **Rings.** A macro-particle is a ring of charge at (r, z) with momentum (u_r, u_θ, u_z): it
//! is pushed in the plane θ = 0 as a particle at (r, 0, z) (Boris or Vay), moved straight there,
//! and its position and momentum turned back onto that plane (the angle of its move round the
//! axis). Fields are gathered linearly from their lattices, odd about the axis for E_r, E_θ, B_θ
//! and even for B_z.
//!
//! **Charge.** A ring's charge goes to the four nodes about it with linear weights in r and z;
//! a node holds it in its ring of volume 2πr_iΔrΔz (π(Δr/2)²Δz on the axis). Its current is
//! Esirkepov's (Comput. Phys. Commun. 135:144, 2001) in the indices: the change of its weights
//! split into the flux through each node ring's faces, I_r through 2πr_{i+½}Δz and I_z through
//! the ring's area, so that the flux form of Gauss's law, ε₀∮E·dA = Q, holds at every node to
//! rounding. Its azimuthal current goes to E_θ's nodes at its move's middle.
//!
//! **Walls.** A conductor at r = R; along z conducting walls (which emit and absorb) or
//! periodic ends; conducting bodies as staircases of cells (`set_metal`, as the Cartesian
//! grid's); an absorbing layer (CPML along z, Roden & Gedney 2000) inside a conducting end.
//! One step is the Cartesian grid's: emission, push and deposit, B half, E, B half.
//!
//! **Relativistic beams.** On square cells a beam faster than about 0.6c drives the Yee grid's
//! numerical Cherenkov instability (Godfrey & Vay, J. Comput. Phys. 248:33, 2013): a ring beam's
//! space-charge depression comes out 6–19 % deep at γ₀ from 1.25 to 3. Cells 8 times finer along
//! z than along r bring c Δt near Δz at the usual Courant number (0.95), the magic time step
//! along z (Taflove & Hagness §4.3), and the depression back to 1–4 % of the closed form up to
//! γ₀ = 5; nearer still (0.999) the leapfrog with particles is unstable.

use std::f64::consts::PI;

use emcraft_model::units::{C0, EPS0, MU0};
use emcraft_particles::{Pusher, Species, V3, push, velocity};

use crate::pic::deposit::Window;
use crate::pic::{Axis, Energy, Fate, Moved, Pulse, Wall, advance_on_threads, keep_kept, zeros};
use crate::yee::Cpml;
use crate::{FdtdError, MAX_CELLS, Result, invalid};

/// The most rings a run holds: as many as the Cartesian grid's particles.
pub const MAX_RINGS: usize = crate::pic::MAX_PARTICLES;

/// The r–z grid: `nr` cells of `dr` (m) from the axis to a conducting wall, and the axis `z`:
/// its cells between conducting walls or periodic ends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxiGrid {
    pub nr: usize,
    pub dr: f64,
    pub z: Axis,
}

/// One of the six components' lattices: its points along r and z.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Lat {
    nr: usize,
    nz: usize,
    r_node: bool,
    z_node: bool,
}

impl Lat {
    fn len(self) -> usize {
        self.nr * self.nz
    }

    #[inline]
    fn at(self, i: usize, k: usize) -> usize {
        i + self.nr * k
    }
}

impl AxiGrid {
    /// The grid checked: cells of finite sizes > 0, at most [`MAX_CELLS`].
    ///
    /// ```
    /// use emcraft_fdtd::axi::AxiGrid;
    /// use emcraft_fdtd::pic::{Axis, Wall};
    /// let z = Axis { cells: 40, h: 0.5e-3, wall: Wall::Conductor };
    /// let g = AxiGrid::new(20, 0.5e-3, z)?;
    /// assert!(g.time_step(0.9) < 0.5e-3 / 299_792_458.0);
    /// assert!(AxiGrid::new(0, 0.5e-3, z).is_err());
    /// # Ok::<(), emcraft_fdtd::FdtdError>(())
    /// ```
    pub fn new(nr: usize, dr: f64, z: Axis) -> Result<AxiGrid> {
        let ok = |h: f64| h.is_finite() && h > 0.0;
        if nr == 0 || z.cells == 0 || !ok(dr) || !ok(z.h) {
            return Err(invalid("axisymmetric grid", "at least a cell of a finite size > 0 along r and z"));
        }
        let cells = nr.saturating_mul(z.cells);
        if cells > MAX_CELLS {
            return Err(FdtdError::TooLarge { cells, limit: MAX_CELLS });
        }
        Ok(AxiGrid { nr, dr, z })
    }

    /// The largest stable step (s) times `courant`.
    pub fn time_step(&self, courant: f64) -> f64 {
        courant / (C0 * (1.0 / (self.dr * self.dr) + 1.0 / (self.z.h * self.z.h)).sqrt())
    }

    /// The radius of its conducting wall (m).
    pub fn radius(&self) -> f64 {
        self.nr as f64 * self.dr
    }

    /// Its length (m).
    pub fn length(&self) -> f64 {
        self.z.length()
    }

    fn lat(&self, r_node: bool, z_node: bool) -> Lat {
        Lat { nr: if r_node { self.nr + 1 } else { self.nr }, nz: self.z.points(z_node), r_node, z_node }
    }

    /// The area of node ring `i`'s faces normal to z (m²).
    fn area(&self, i: usize) -> f64 {
        if i == 0 { PI * 0.25 * self.dr * self.dr } else { 2.0 * PI * i as f64 * self.dr * self.dr }
    }

    /// The volume of node `i`'s ring (m³).
    fn volume(&self, i: usize) -> f64 {
        self.area(i) * self.z.h
    }
}

/// A ring of particles: its radius and z (m), momentum γv (m/s) along r, θ and z, and how many
/// particles it is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ring {
    pub r: f64,
    pub z: f64,
    pub u: V3,
    pub w: f64,
}

/// Rings of one species.
#[derive(Clone, Debug, PartialEq)]
pub struct Rings {
    pub species: Species,
    pub rings: Vec<Ring>,
}

/// A beam or bunch sent in through a conducting end, as the Cartesian grid's [`crate::pic::Bunch`]:
/// `pulse` in time, spread evenly over the area between `inner` and `radius` (m), at `energy` (J)
/// along z, `per_step` rings a step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Injection {
    pub high: bool,
    pub population: usize,
    pub pulse: Pulse,
    pub inner: f64,
    pub radius: f64,
    pub energy: f64,
    pub per_step: usize,
}

/// The six components.
struct Fields {
    er: Vec<f64>,
    ez: Vec<f64>,
    et: Vec<f64>,
    br: Vec<f64>,
    bz: Vec<f64>,
    bt: Vec<f64>,
}

impl Fields {
    fn of(&mut self, c: Component) -> &mut Vec<f64> {
        match c {
            Component::Er => &mut self.er,
            Component::Etheta => &mut self.et,
            Component::Ez => &mut self.ez,
            Component::Br => &mut self.br,
            Component::Btheta => &mut self.bt,
            Component::Bz => &mut self.bz,
        }
    }
}

/// The absorbing layers along z: the coefficients at z's nodes and centres for E's steps and B's
/// half steps, and ψ of each z difference (Er's ∂zBθ, Eθ's ∂zBr, Bθ's ∂zEr, Br's ∂zEθ).
struct Layers {
    cells: [usize; 2],
    e: [Vec<Cpml>; 2],
    b: [Vec<Cpml>; 2],
    psi: [Vec<f64>; 4],
}

/// An axisymmetric particle-in-cell run.
pub struct AxiPic {
    grid: AxiGrid,
    dt: f64,
    pusher: Pusher,
    f: Fields,
    jr: Vec<f64>,
    jz: Vec<f64>,
    jt: Vec<f64>,
    lats: [Lat; 6],
    populations: Vec<Rings>,
    injections: Vec<Injection>,
    metal: Option<Vec<bool>>,
    /// Whether each E_r, E_z, E_θ point touches metal (empty without metal).
    held: [Vec<bool>; 3],
    layers: Option<Layers>,
    count: u64,
    steps: usize,
}

/// About how many additions a ring's current makes (its 4 × 4 window and θ's).
const ADDITIONS: usize = 24;

/// A wall a ring meets: the conductor at r = R, or a conducting end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RingWall {
    Outer,
    End { high: bool },
}

impl RingWall {
    /// Where [`Moved::collected`] counts what it absorbs: the outer wall as the high end of the
    /// first axis (r), the ends as the third's (z).
    fn collected(self) -> (usize, bool) {
        match self {
            RingWall::Outer => (0, true),
            RingWall::End { high } => (2, high),
        }
    }
}

/// A component of the m = 0 field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Component {
    Er,
    Etheta,
    Ez,
    Br,
    Btheta,
    Bz,
}

impl Component {
    /// Its lattice's place in `lats`.
    fn lattice(self) -> usize {
        match self {
            Component::Er => ER,
            Component::Ez => EZ,
            Component::Etheta => ET,
            Component::Br => BR,
            Component::Bz => BZ,
            Component::Btheta => BT,
        }
    }

    /// An E tangential to the conductor at r = R (held at zero on its nodes there).
    fn tangential_to_outer_wall(self) -> bool {
        matches!(self, Component::Etheta | Component::Ez)
    }

    /// An E tangential to the conducting ends.
    fn tangential_to_ends(self) -> bool {
        matches!(self, Component::Er | Component::Etheta)
    }

    /// Odd about the axis (zero on it): the radial and azimuthal E and the radial B.
    fn odd_on_axis(self) -> bool {
        matches!(self, Component::Er | Component::Etheta | Component::Br)
    }
}

/// Lattices' order in `lats`.
const ER: usize = 0;
const EZ: usize = 1;
const ET: usize = 2;
const BR: usize = 3;
const BZ: usize = 4;
const BT: usize = 5;

/// The step of the low-discrepancy sequence that places an injection's rings over its annulus
/// (the first of Roberts's R₃ steps, 1/φ₃).
const RING_STEP: f64 = 0.819_172_513_396_164_4;

impl AxiPic {
    /// A run with no field and no rings, stepped by `dt` (s) within the grid's limit.
    pub fn new(grid: AxiGrid, dt: f64, pusher: Pusher) -> Result<AxiPic> {
        if !(dt.is_finite() && dt > 0.0 && dt < grid.time_step(1.0)) {
            return Err(invalid("time step", format!("{dt} s: must be > 0 and below the stability limit {} s", grid.time_step(1.0))));
        }
        let lats = [
            grid.lat(false, true),
            grid.lat(true, false),
            grid.lat(true, true),
            grid.lat(true, false),
            grid.lat(false, true),
            grid.lat(false, false),
        ];
        let f = Fields {
            er: zeros(lats[ER].len())?,
            ez: zeros(lats[EZ].len())?,
            et: zeros(lats[ET].len())?,
            br: zeros(lats[BR].len())?,
            bz: zeros(lats[BZ].len())?,
            bt: zeros(lats[BT].len())?,
        };
        Ok(AxiPic {
            grid,
            dt,
            pusher,
            f,
            jr: zeros(lats[ER].len())?,
            jz: zeros(lats[EZ].len())?,
            jt: zeros(lats[ET].len())?,
            lats,
            populations: Vec::new(),
            injections: Vec::new(),
            metal: None,
            held: [Vec::new(), Vec::new(), Vec::new()],
            layers: None,
            count: 0,
            steps: 0,
        })
    }

    pub fn grid(&self) -> &AxiGrid {
        &self.grid
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    pub fn time(&self) -> f64 {
        self.steps as f64 * self.dt
    }

    pub fn populations(&self) -> &[Rings] {
        &self.populations
    }

    pub fn rings(&self) -> usize {
        self.populations.iter().map(|p| p.rings.len()).sum()
    }

    /// Adds `rings`, checked (finite, inside, weights ≥ 0); returns its index.
    pub fn add(&mut self, mut rings: Rings) -> Result<usize> {
        if self.rings().saturating_add(rings.rings.len()) > MAX_RINGS {
            return Err(invalid("rings", format!("at most {MAX_RINGS}")));
        }
        let (rmax, l) = (self.grid.radius(), self.grid.length());
        for p in &mut rings.rings {
            let ok = p.r.is_finite()
                && p.z.is_finite()
                && p.u.iter().all(|v| v.is_finite())
                && p.w.is_finite()
                && p.w >= 0.0
                && (0.0..=rmax).contains(&p.r);
            if !ok {
                return Err(invalid("ring", format!("{p:?}")));
            }
            match self.grid.z.wall {
                Wall::Periodic => p.z = p.z.rem_euclid(l),
                Wall::Conductor if !(0.0..=l).contains(&p.z) => return Err(invalid("ring", format!("{p:?}: outside"))),
                Wall::Conductor => {}
            }
        }
        self.populations.push(rings);
        Ok(self.populations.len() - 1)
    }

    /// Adds `injection`, through a conducting end.
    pub fn inject(&mut self, injection: Injection) -> Result<()> {
        let ok = |x: f64| x.is_finite() && x >= 0.0;
        if self.grid.z.wall != Wall::Conductor
            || injection.population >= self.populations.len()
            || !(ok(injection.inner)
                && ok(injection.radius)
                && injection.inner <= injection.radius
                && injection.radius <= self.grid.radius()
                && ok(injection.energy))
            || !injection.pulse.valid()
            || injection.per_step == 0
            || injection.per_step > crate::pic::MAX_PER_NODE
        {
            let most = crate::pic::MAX_PER_NODE;
            return Err(invalid(
                "injection",
                format!(
                    "{injection:?}: through a conducting end, finite charges and times (a Gaussian's length > 0), a ring inside the wall, 1 to {most} a step"
                ),
            ));
        }
        self.injections.push(injection);
        Ok(())
    }

    /// Field `component` set to `f` of each point's (r, z) (m) (V/m for E, T for B), off the
    /// walls it is tangential to and the axis where it is odd.
    pub fn set(&mut self, component: Component, f: impl Fn(f64, f64) -> f64) {
        let lat = self.lats[component.lattice()];
        let (dr, dz, g) = (self.grid.dr, self.grid.z.h, self.grid);
        let values = self.f.of(component);
        for k in 0..lat.nz {
            for i in 0..lat.nr {
                let r = (i as f64 + if lat.r_node { 0.0 } else { 0.5 }) * dr;
                let z = (k as f64 + if lat.z_node { 0.0 } else { 0.5 }) * dz;
                let held = (component.tangential_to_outer_wall() && lat.r_node && i == g.nr)
                    || (component.tangential_to_ends() && lat.z_node && g.z.on_wall(k))
                    || (component.odd_on_axis() && lat.r_node && i == 0);
                if let Some(v) = values.get_mut(lat.at(i, k)) {
                    *v = if held { 0.0 } else { f(r, z) };
                }
            }
        }
    }

    /// Conducting bodies: each cell whose centre (r, z) `metal` says is in a conductor. Returns
    /// whether any is.
    pub fn set_metal(&mut self, metal: impl Fn(f64, f64) -> bool) -> bool {
        let g = self.grid;
        let cells: Vec<bool> = (0..g.z.cells)
            .flat_map(|k| (0..g.nr).map(move |i| (i, k)))
            .map(|(i, k)| metal((i as f64 + 0.5) * g.dr, (k as f64 + 0.5) * g.z.h))
            .collect();
        let any = cells.contains(&true);
        self.metal = any.then_some(cells);
        self.held = [ER, EZ, ET].map(|which| {
            let lat = self.lats[which];
            if !any {
                return Vec::new();
            }
            (0..lat.nz).flat_map(|k| (0..lat.nr).map(move |i| (i, k))).map(|(i, k)| self.held(which, i, k)).collect()
        });
        // E on edges touching metal starts at zero.
        for (which, lat) in [(ER, self.lats[ER]), (EZ, self.lats[EZ]), (ET, self.lats[ET])] {
            for k in 0..lat.nz {
                for i in 0..lat.nr {
                    if self.held(which, i, k) {
                        let v = match which {
                            ER => &mut self.f.er,
                            EZ => &mut self.f.ez,
                            _ => &mut self.f.et,
                        };
                        if let Some(x) = v.get_mut(lat.at(i, k)) {
                            *x = 0.0;
                        }
                    }
                }
            }
        }
        any
    }

    /// Whether cell (i, k) is metal (none outside the grid).
    fn metal_cell(&self, i: isize, k: isize) -> bool {
        let Some(m) = &self.metal else { return false };
        let g = self.grid;
        if i < 0 || i >= g.nr as isize {
            return false;
        }
        let Some(k) = g.z.wrap(k, false) else { return false };
        m.get(i as usize + g.nr * k).copied().unwrap_or(false)
    }

    /// Whether E component `which` at (i, k) touches a metal cell.
    fn held(&self, which: usize, i: usize, k: usize) -> bool {
        let (i, k) = (i as isize, k as isize);
        match which {
            ER => self.metal_cell(i, k - 1) || self.metal_cell(i, k),
            EZ => self.metal_cell(i - 1, k) || self.metal_cell(i, k),
            _ => (i - 1..=i).any(|a| (k - 1..=k).any(|b| self.metal_cell(a, b))),
        }
    }

    /// Opens the conducting end at `high` (or the low one): a CPML of `cells` inside it.
    pub fn absorb(&mut self, high: bool, cells: usize) -> Result<()> {
        let g = self.grid;
        if g.z.wall != Wall::Conductor || cells == 0 || cells > crate::pic::MAX_LAYER {
            return Err(invalid("absorber", format!("a layer of 1 to {} cells inside a conducting end", crate::pic::MAX_LAYER)));
        }
        let other = self.layers.as_ref().map_or(0, |l| l.cells[usize::from(!high)]);
        if cells + other >= g.z.cells {
            return Err(invalid("absorber", "the layers longer than the grid"));
        }
        let mut layers = match self.layers.take() {
            Some(l) => l,
            None => Layers {
                cells: [0; 2],
                e: [vec![Cpml::NONE; g.z.cells + 1], vec![Cpml::NONE; g.z.cells]],
                b: [vec![Cpml::NONE; g.z.cells + 1], vec![Cpml::NONE; g.z.cells]],
                psi: [zeros(self.lats[ER].len())?, zeros(self.lats[ET].len())?, zeros(self.lats[BT].len())?, zeros(self.lats[BR].len())?],
            },
        };
        layers.cells[usize::from(high)] = cells;
        let graded = crate::pic::absorb::Graded::along(g.z, layers.cells, self.dt);
        layers.e = graded.e;
        layers.b = graded.b;
        self.layers = Some(layers);
        Ok(())
    }

    /// The field at (r, z): (E_r, E_θ, E_z) (V/m) and (B_r, B_θ, B_z) (T).
    pub fn field_at(&self, r: f64, z: f64) -> (V3, V3) {
        let g = |which: usize, values: &[f64], odd: bool| self.interpolate(self.lats[which], values, r, z, odd);
        (
            [g(ER, &self.f.er, true), g(ET, &self.f.et, true), g(EZ, &self.f.ez, false)],
            [g(BR, &self.f.br, true), g(BT, &self.f.bt, true), g(BZ, &self.f.bz, false)],
        )
    }

    /// `values` on `lat` at (r, z), linearly; across the axis odd (`odd`) or even, past the outer
    /// wall and a conducting end the nearest point's.
    fn interpolate(&self, lat: Lat, values: &[f64], r: f64, z: f64, odd: bool) -> f64 {
        let g = self.grid;
        let sr = r / g.dr - if lat.r_node { 0.0 } else { 0.5 };
        let sz = z / g.z.h - if lat.z_node { 0.0 } else { 0.5 };
        let (i0, k0) = (sr.floor(), sz.floor());
        let (fr, fz) = (sr - i0, sz - k0);
        let at = |i: isize, k: isize| -> f64 {
            let (i, sign) = if i < 0 { (-i - if lat.r_node { 0 } else { 1 }, if odd { -1.0 } else { 1.0 }) } else { (i, 1.0) };
            let i = (i as usize).min(lat.nr - 1);
            let k = g.z.clamp(k, lat.z_node);
            sign * values.get(lat.at(i, k)).copied().unwrap_or(0.0)
        };
        let (i0, k0) = (i0 as isize, k0 as isize);
        (1.0 - fr) * (1.0 - fz) * at(i0, k0) + fr * (1.0 - fz) * at(i0 + 1, k0) + (1.0 - fr) * fz * at(i0, k0 + 1) + fr * fz * at(i0 + 1, k0 + 1)
    }

    /// The energy in the field and in the rings (J).
    pub fn energy(&self) -> Energy {
        let g = self.grid;
        let sum = |lat: Lat, values: &[f64], coefficient: f64| -> f64 {
            let mut s = 0.0;
            for k in 0..lat.nz {
                for i in 0..lat.nr {
                    let r = (i as f64 + if lat.r_node { 0.0 } else { 0.5 }) * g.dr;
                    let ring = if lat.r_node && i == 0 { PI * 0.25 * g.dr * g.dr } else { 2.0 * PI * r * g.dr };
                    let x = values.get(lat.at(i, k)).copied().unwrap_or(0.0);
                    s += coefficient * x * x * ring * g.z.h;
                }
            }
            s
        };
        let electric =
            sum(self.lats[ER], &self.f.er, 0.5 * EPS0) + sum(self.lats[EZ], &self.f.ez, 0.5 * EPS0) + sum(self.lats[ET], &self.f.et, 0.5 * EPS0);
        let field = electric
            + sum(self.lats[BR], &self.f.br, 0.5 / MU0)
            + sum(self.lats[BZ], &self.f.bz, 0.5 / MU0)
            + sum(self.lats[BT], &self.f.bt, 0.5 / MU0);
        let kinetic = self.populations.iter().map(|p| p.rings.iter().map(|q| q.w * p.species.kinetic(q.u)).sum::<f64>()).sum();
        Energy { field, electric, kinetic }
    }

    /// ε₀∮E·dA − Q (C) about each node off the walls, the axis' too, the metal and the layers:
    /// zero where Gauss's law holds.
    pub fn gauss(&self) -> Vec<f64> {
        let g = self.grid;
        let nodes = g.lat(true, true);
        let mut q = vec![0.0; nodes.len()];
        for pop in &self.populations {
            for p in &pop.rings {
                self.charge(&mut q, nodes, p.r, p.z, pop.species.charge * p.w);
            }
        }
        let mut out = Vec::new();
        for k in 0..nodes.nz {
            if g.z.on_wall(k)
                || self.layers.as_ref().is_some_and(|l| (l.cells[0] > 0 && k <= l.cells[0]) || (l.cells[1] > 0 && k + l.cells[1] >= g.z.cells))
            {
                continue;
            }
            let km = g.z.wrap(k as isize - 1, false);
            for i in 0..g.nr {
                if (i as isize - 1..=i as isize).any(|a| (k as isize - 1..=k as isize).any(|b| self.metal_cell(a, b))) {
                    continue;
                }
                let er = |ii: usize| self.f.er.get(self.lats[ER].at(ii, k)).copied().unwrap_or(0.0);
                let radial = 2.0 * PI * g.z.h * ((i as f64 + 0.5) * g.dr * er(i) - if i > 0 { (i as f64 - 0.5) * g.dr * er(i - 1) } else { 0.0 });
                let ez = |kk: usize| self.f.ez.get(self.lats[EZ].at(i, kk)).copied().unwrap_or(0.0);
                let axial = g.area(i) * (ez(k.min(g.z.cells - 1)) * if k < g.z.cells { 1.0 } else { 0.0 } - km.map_or(0.0, ez));
                out.push(EPS0 * (radial + axial) - q[nodes.at(i, k)]);
            }
        }
        out
    }

    /// Adds `charge` (C) at (r, z) to the nodes' charge `q`.
    fn charge(&self, q: &mut [f64], nodes: Lat, r: f64, z: f64, charge: f64) {
        let g = self.grid;
        let (sr, sz) = (r / g.dr, z / g.z.h);
        let (i0, k0) = (sr.floor(), sz.floor());
        for (di, wr) in [(0, 1.0 - (sr - i0)), (1, sr - i0)] {
            for (dk, wz) in [(0, 1.0 - (sz - k0)), (1, sz - k0)] {
                let i = i0 as isize + di;
                if i < 0 || i > g.nr as isize {
                    continue;
                }
                let Some(k) = g.z.wrap(k0 as isize + dk, true) else { continue };
                if let Some(v) = q.get_mut(nodes.at(i as usize, k)) {
                    *v += charge * wr * wz;
                }
            }
        }
    }

    /// One step.
    pub fn step(&mut self) -> Result<Moved> {
        let mut moved = Moved::default();
        self.jr.fill(0.0);
        self.jz.fill(0.0);
        self.jt.fill(0.0);
        self.emit(&mut moved)?;
        let (mut jr, mut jz, mut jt) = (std::mem::take(&mut self.jr), std::mem::take(&mut self.jz), std::mem::take(&mut self.jt));
        for k in 0..self.populations.len() {
            let species = self.populations[k].species;
            let mut rings = std::mem::take(&mut self.populations[k].rings);
            let this = &*self;
            let fates = advance_on_threads(
                &mut rings,
                ADDITIONS,
                |p, adds| this.advance(species, p, &mut |w, id, v| adds.push((w, id as u32, v))),
                [jr.as_mut_slice(), jz.as_mut_slice(), jt.as_mut_slice()],
            );
            moved.tally(species.charge, rings.iter().map(|p| p.w), &fates, |_, _| {});
            keep_kept(&mut rings, &fates);
            self.populations[k].rings = rings;
        }
        (self.jr, self.jz, self.jt) = (jr, jz, jt);
        self.faraday(0.5 * self.dt);
        self.ampere();
        self.faraday(0.5 * self.dt);
        self.steps += 1;
        Ok(moved)
    }

    /// The injections' rings for this step, on their ends.
    fn emit(&mut self, moved: &mut Moved) -> Result<()> {
        let t = self.time();
        for inj in self.injections.clone() {
            let Some(pop) = self.populations.get(inj.population) else { continue };
            let species = pop.species;
            let q = inj.pulse.charge(t, self.dt);
            if q <= 0.0 {
                continue;
            }
            let w = q / (species.charge.abs() * inj.per_step as f64);
            let sign = if inj.high { -1.0 } else { 1.0 };
            let u = species.momentum(inj.energy, [0.0, 0.0, sign]);
            let z = if inj.high { self.grid.length() } else { 0.0 };
            let mut new = Vec::with_capacity(inj.per_step);
            for _ in 0..inj.per_step {
                self.count += 1;
                let s = (0.5 + self.count as f64 * RING_STEP).fract();
                let r = crate::pic::emit::annulus_radius(inj.inner, inj.radius, s);
                if self.metal_cell((r / self.grid.dr).floor() as isize, if inj.high { self.grid.z.cells as isize - 1 } else { 0 }) {
                    continue;
                }
                new.push(Ring { r, z, u, w });
            }
            if self.rings().saturating_add(new.len()) > MAX_RINGS {
                return Err(invalid("rings", format!("the injection would hold more than {MAX_RINGS}")));
            }
            moved.emitted += new.iter().map(|p| species.charge * p.w).sum::<f64>();
            moved.kinetic_in += new.iter().map(|p| p.w * species.kinetic(p.u)).sum::<f64>();
            if let Some(pop) = self.populations.get_mut(inj.population) {
                pop.rings.extend(new);
            }
        }
        Ok(())
    }

    /// Pushes and moves ring `p`, its current handed to `add(component, id, value)` (0: J_r,
    /// 1: J_z, 2: J_θ; A/m²); its fate.
    fn advance(&self, species: Species, p: &mut Ring, add: &mut impl FnMut(u8, usize, f64)) -> Fate {
        let g = self.grid;
        let (e, b) = self.field_at(p.r, p.z);
        let u = push(self.pusher, species, p.u, e, b, self.dt);
        if !u.iter().all(|v| v.is_finite()) {
            return Fate::Lost;
        }
        let v = velocity(u);
        // Straight in the plane θ = 0, then turned back onto it.
        let (x, y) = (p.r + v[0] * self.dt, v[1] * self.dt);
        let mut r1 = x.hypot(y);
        let mut z1 = p.z + v[2] * self.dt;
        let (c, s) = if r1 > 0.0 { (x / r1, y / r1) } else { (1.0, 0.0) };
        let u1 = [c * u[0] + s * u[1], -s * u[0] + c * u[1], u[2]];
        // Walls: the conductor at R and conducting ends.
        let mut hit = None;
        let mut f = 1.0f64;
        if r1 > g.radius() {
            f = ((g.radius() - p.r) / (r1 - p.r)).clamp(0.0, 1.0);
            hit = Some(RingWall::Outer);
        }
        if g.z.wall == Wall::Conductor {
            for (high, beyond) in [(false, z1 < 0.0), (true, z1 > g.length())] {
                if beyond {
                    let wall = if high { g.length() } else { 0.0 };
                    let fe = ((wall - p.z) / (z1 - p.z)).clamp(0.0, 1.0);
                    if fe < f {
                        (f, hit) = (fe, Some(RingWall::End { high }));
                    }
                }
            }
        }
        if hit.is_some() {
            r1 = (p.r + f * (r1 - p.r)).clamp(0.0, g.radius());
            z1 = (p.z + f * (z1 - p.z)).clamp(0.0, g.length());
        }
        let q = species.charge * p.w;
        self.deposit((p.r, p.z), (r1, z1), q, add);
        self.deposit_theta(0.5 * (p.r + r1), 0.5 * (p.z + z1), q * v[1], add);
        let kinetic = p.w * species.kinetic(u1);
        let at = [r1, 0.0, z1];
        if let Some(wall) = hit {
            return Fate::Absorbed { wall: Some(wall.collected()), at, kinetic };
        }
        if self.metal_cell((r1 / g.dr).floor() as isize, (z1 / g.z.h).floor() as isize) {
            return Fate::Absorbed { wall: None, at, kinetic };
        }
        p.r = r1;
        p.z = if g.z.wall == Wall::Periodic { z1.rem_euclid(g.length()) } else { z1 };
        p.u = u1;
        Fate::Kept
    }

    /// The current of charge `q` (C) moving from (r0, z0) to (r1, z1) in a step (z unwrapped):
    /// Esirkepov's flux through each node ring's faces, over their areas.
    fn deposit(&self, (r0, z0): (f64, f64), (r1, z1): (f64, f64), q: f64, add: &mut impl FnMut(u8, usize, f64)) {
        let g = self.grid;
        let (wr, wz) = (Window::of(r0, r1, g.dr), Window::of(z0, z1, g.z.h));
        let (fr, s0r, dsr) = (wr.first, wr.before, wr.change());
        let (fz, s0z, dsz) = (wz.first, wz.before, wz.change());
        let (ler, lez) = (self.lats[ER], self.lats[EZ]);
        // I_r through r_{i+½}: the running sum over i.
        for mz in 0..4 {
            let Some(k) = g.z.wrap(fz + mz as isize, true) else { continue };
            let across = s0z[mz] + 0.5 * dsz[mz];
            if across == 0.0 {
                continue;
            }
            let mut flux = 0.0;
            for mr in 0..3 {
                flux -= q * dsr[mr] * across / self.dt;
                let i = fr + mr as isize;
                if i < 0 || i >= g.nr as isize {
                    continue;
                }
                let face = 2.0 * PI * (i as f64 + 0.5) * g.dr * g.z.h;
                add(0, ler.at(i as usize, k), flux / face);
            }
        }
        // I_z through the node rings' areas: the running sum over k.
        for mr in 0..4 {
            let i = fr + mr as isize;
            if i < 0 || i > g.nr as isize {
                continue;
            }
            let across = s0r[mr] + 0.5 * dsr[mr];
            if across == 0.0 {
                continue;
            }
            let area = g.area(i as usize);
            let mut flux = 0.0;
            for mz in 0..3 {
                flux -= q * dsz[mz] * across / self.dt;
                let Some(k) = g.z.wrap(fz + mz as isize, false) else { continue };
                add(1, lez.at(i as usize, k), flux / area);
            }
        }
    }

    /// The azimuthal current q v_θ (A·m) at (r, z) on E_θ's nodes, over their rings' volumes.
    fn deposit_theta(&self, r: f64, z: f64, qv: f64, add: &mut impl FnMut(u8, usize, f64)) {
        let g = self.grid;
        let let_ = self.lats[ET];
        let (sr, sz) = (r / g.dr, z / g.z.h);
        let (i0, k0) = (sr.floor(), sz.floor());
        for (di, wr) in [(0, 1.0 - (sr - i0)), (1, sr - i0)] {
            let i = i0 as isize + di;
            if i < 1 || i > g.nr as isize {
                continue;
            }
            for (dk, wz) in [(0, 1.0 - (sz - k0)), (1, sz - k0)] {
                let Some(k) = g.z.wrap(k0 as isize + dk, true) else { continue };
                let volume = g.volume(i as usize);
                add(2, let_.at(i as usize, k), qv * wr * wz / volume);
            }
        }
    }

    /// A z difference through the layers: d + ψ, ψ advanced (`coefficients` B's or E's, z point
    /// `k`, a node or a centre).
    #[inline]
    fn stretch(coefficients: Option<&[Vec<Cpml>; 2]>, (node, k): (bool, usize), psi: Option<&mut f64>, d: f64) -> f64 {
        let (Some(set), Some(psi)) = (coefficients, psi) else { return d };
        let c = set[usize::from(!node)].get(k).copied().unwrap_or(Cpml::NONE);
        if c == Cpml::NONE {
            return d;
        }
        *psi = c.psi(*psi, d);
        d + *psi
    }

    /// B advanced by `dt` (s): ∂B/∂t = −∇ × E, each z row on the threads.
    fn faraday(&mut self, dt: f64) {
        let g = self.grid;
        let [ler, lez, let_, lbr, lbz, lbt] = self.lats;
        let Fields { er, ez, et, br, bz, bt } = &mut self.f;
        let (er, ez, et) = (&*er, &*ez, &*et);
        let (coefficients, psi) = match self.layers.as_mut() {
            Some(l) => (Some(&l.b), Some(&mut l.psi)),
            None => (None, None),
        };
        let [_, _, psi_bt, psi_br] = match psi {
            Some([a, b, c, d]) => [Some(a), Some(b), Some(c), Some(d)],
            None => [None, None, None, None],
        };
        // B_θ: −(∂z E_r − ∂r E_z).
        by_rows(bt, psi_bt, lbt.nr, |k, row, mut psi| {
            let Some(kp) = g.z.wrap(k as isize + 1, true) else { return };
            for (i, v) in row.iter_mut().enumerate() {
                let dz = (er[ler.at(i, kp)] - er[ler.at(i, k)]) / g.z.h;
                let dz = Self::stretch(coefficients, (false, k), psi.as_deref_mut().and_then(|p| p.get_mut(i)), dz);
                let dr = (ez[lez.at(i + 1, k)] - ez[lez.at(i, k)]) / g.dr;
                *v -= dt * (dz - dr);
            }
        });
        // B_r: +∂z E_θ.
        by_rows(br, psi_br, lbr.nr, |k, row, mut psi| {
            let Some(kp) = g.z.wrap(k as isize + 1, true) else { return };
            for (i, v) in row.iter_mut().enumerate() {
                let dz = (et[let_.at(i, kp)] - et[let_.at(i, k)]) / g.z.h;
                *v += dt * Self::stretch(coefficients, (false, k), psi.as_deref_mut().and_then(|p| p.get_mut(i)), dz);
            }
        });
        // B_z: −(1/r) ∂r (r E_θ).
        by_rows(bz, None, lbz.nr, |k, row, _| {
            for (i, v) in row.iter_mut().enumerate() {
                let (r0, r1, rm) = (i as f64 * g.dr, (i + 1) as f64 * g.dr, (i as f64 + 0.5) * g.dr);
                *v -= dt * (r1 * et[let_.at(i + 1, k)] - r0 * et[let_.at(i, k)]) / (rm * g.dr);
            }
        });
    }

    /// E advanced by a step: ε₀ ∂E/∂t = ∇ × B/μ₀ − J, off the walls and metal it is tangential
    /// to, each z row on the threads.
    fn ampere(&mut self) {
        let g = self.grid;
        let dt = self.dt;
        let c2 = C0 * C0;
        let [ler, lez, let_, lbr, lbz, lbt] = self.lats;
        let held = &self.held;
        let is_held = |w: usize, id: usize| held[w].get(id).copied().unwrap_or(false);
        let (jr, jz, jt) = (&self.jr, &self.jz, &self.jt);
        let Fields { er, ez, et, br, bz, bt } = &mut self.f;
        let (br, bz, bt) = (&*br, &*bz, &*bt);
        let (coefficients, psi) = match self.layers.as_mut() {
            Some(l) => (Some(&l.e), Some(&mut l.psi)),
            None => (None, None),
        };
        let [psi_er, psi_et, _, _] = match psi {
            Some([a, b, c, d]) => [Some(a), Some(b), Some(c), Some(d)],
            None => [None, None, None, None],
        };
        // E_r: −c² ∂z B_θ − J_r/ε₀, off the conducting ends.
        by_rows(er, psi_er, ler.nr, |k, row, mut psi| {
            if g.z.on_wall(k) {
                return;
            }
            let Some(km) = g.z.wrap(k as isize - 1, false) else { return };
            for (i, v) in row.iter_mut().enumerate() {
                let id = ler.at(i, k);
                let dz = (bt[lbt.at(i, k.min(lbt.nz - 1))] - bt[lbt.at(i, km)]) / g.z.h;
                let dz = Self::stretch(coefficients, (true, k), psi.as_deref_mut().and_then(|p| p.get_mut(i)), dz);
                if !is_held(0, id) {
                    *v += dt * (-c2 * dz - jr[id] / EPS0);
                }
            }
        });
        // E_θ: c²(∂z B_r − ∂r B_z) − J_θ/ε₀, off the axis, the wall and the ends.
        by_rows(et, psi_et, let_.nr, |k, row, mut psi| {
            if g.z.on_wall(k) {
                return;
            }
            let Some(km) = g.z.wrap(k as isize - 1, false) else { return };
            for (i, v) in row.iter_mut().enumerate().take(g.nr).skip(1) {
                let id = let_.at(i, k);
                let dz = (br[lbr.at(i, k.min(lbr.nz - 1))] - br[lbr.at(i, km)]) / g.z.h;
                let dz = Self::stretch(coefficients, (true, k), psi.as_deref_mut().and_then(|p| p.get_mut(i)), dz);
                let dr = (bz[lbz.at(i, k)] - bz[lbz.at(i - 1, k)]) / g.dr;
                if !is_held(2, id) {
                    *v += dt * (c2 * (dz - dr) - jt[id] / EPS0);
                }
            }
        });
        // E_z: c² (1/r) ∂r (r B_θ) − J_z/ε₀, off the wall; on the axis 4B_θ(r_½)/Δr.
        by_rows(ez, None, lez.nr, |k, row, _| {
            for (i, v) in row.iter_mut().enumerate().take(g.nr) {
                let id = lez.at(i, k);
                let curl = if i == 0 {
                    4.0 * bt[lbt.at(0, k)] / g.dr
                } else {
                    let (rp, rm, r) = ((i as f64 + 0.5) * g.dr, (i as f64 - 0.5) * g.dr, i as f64 * g.dr);
                    (rp * bt[lbt.at(i, k)] - rm * bt[lbt.at(i - 1, k)]) / (r * g.dr)
                };
                if !is_held(1, id) {
                    *v += dt * (c2 * curl - jz[id] / EPS0);
                }
            }
        });
    }
}

/// `row(k, values of row k, the layers' ψ of row k)` for each z row (of `nr` points) of
/// `values`, on the threads: each value written by one of them, from values the update does not
/// change (the same bits on any number of threads).
fn by_rows(values: &mut [f64], psi: Option<&mut Vec<f64>>, nr: usize, row: impl Fn(usize, &mut [f64], Option<&mut [f64]>) + Sync + Send) {
    use rayon::prelude::*;
    match psi {
        Some(psi) if psi.len() == values.len() => {
            values.par_chunks_mut(nr).zip(psi.par_chunks_mut(nr)).enumerate().for_each(|(k, (v, p))| row(k, v, Some(p)));
        }
        _ => values.par_chunks_mut(nr).enumerate().for_each(|(k, v)| row(k, v, None)),
    }
}

#[cfg(test)]
#[path = "axi_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "axi_bwo_tests.rs"]
mod bwo_tests;
