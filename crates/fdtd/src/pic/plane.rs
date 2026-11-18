//! The fields on a plane across z that carry power along it, on either grid: the Cartesian
//! box's (`Pic::plane`) and the body of revolution's (`crate::axi::AxiPic::plane`). Across the
//! plane two directions, 1 and 2, right-handed with z (x and y in a box, r and θ about an axis);
//! E₁ and B₂ lie at one set of points of the plane, E₂ and B₁ at another (as the Yee lattices
//! place them), each point with the area it stands for.
//!
//! **Taps.** A run that holds its fields elsewhere (a device) reads the plane's values where they
//! lie on its lattices each step ([`PlaneTaps`]): E on the plane's nodes, B on the half cells
//! either side, averaged, so E and B are taken at the same time (the end of a step).
//!
//! ```
//! use emcraft_fdtd::pic::Plane;
//! // E₁ = 1 V/m and B₂ = 1 mT over 1 m²: 1e-3/μ₀ W along +z.
//! let p = Plane { e1: vec![1.0], b2: vec![1e-3], area1: vec![1.0], ..Plane::default() };
//! assert!((p.flux() - 1e-3 / emcraft_model::units::MU0).abs() < 1e-9);
//! ```

use emcraft_model::units::MU0;

/// The fields on a plane across z (V/m, T): E₁ and B₂ at points of areas `area1` (m²), E₂ and
/// B₁ at points of areas `area2`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plane {
    pub e1: Vec<f64>,
    pub b2: Vec<f64>,
    pub area1: Vec<f64>,
    pub e2: Vec<f64>,
    pub b1: Vec<f64>,
    pub area2: Vec<f64>,
}

impl Plane {
    /// The power through it towards +z (W): Poynting's (E × H)·ẑ = (E₁ B₂ − E₂ B₁)/μ₀ over its
    /// points, as the Yee lattice holds the fields — the energy balance's own flux.
    pub fn flux(&self) -> f64 {
        let along = |e: &[f64], b: &[f64], area: &[f64]| e.iter().zip(b).zip(area).map(|((e, b), a)| e * b * a).sum::<f64>();
        (along(&self.e1, &self.b2, &self.area1) - along(&self.e2, &self.b1, &self.area2)) / MU0
    }

    /// `other`'s fields added to its own (a sum towards a mean); an empty plane takes them.
    pub fn add(&mut self, other: &Plane) {
        if self.area1.is_empty() && self.area2.is_empty() {
            *self = other.clone();
            return;
        }
        for (mine, theirs) in [(&mut self.e1, &other.e1), (&mut self.b2, &other.b2), (&mut self.e2, &other.e2), (&mut self.b1, &other.b1)] {
            mine.iter_mut().zip(theirs).for_each(|(m, t)| *m += t);
        }
    }

    /// Its fields times `k` (a sum of `n` planes times 1/n is their mean).
    pub fn scaled(mut self, k: f64) -> Plane {
        for f in [&mut self.e1, &mut self.b2, &mut self.e2, &mut self.b1] {
            f.iter_mut().for_each(|x| *x *= k);
        }
        self
    }
}

/// A tap: a lattice (its index in the run's own order of components) and a point on it, or none
/// where the lattice has no such point (read as zero).
pub type Tap = Option<(usize, usize)>;

/// Where a plane's fields lie on the lattices: three taps for each point of the first set (E₁,
/// B₂ below, B₂ above), then three for each of the second (E₂, B₁ below, B₁ above); with the
/// points' areas.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlaneTaps {
    pub taps: Vec<Tap>,
    area1: Vec<f64>,
    area2: Vec<f64>,
}

impl PlaneTaps {
    /// The taps of the points `first` (each E₁'s tap and B₂'s two, with its area) and `second`
    /// (E₂, B₁'s two, area).
    pub(crate) fn new(first: impl Iterator<Item = ([Tap; 3], f64)>, second: impl Iterator<Item = ([Tap; 3], f64)>) -> PlaneTaps {
        let mut taps = Vec::new();
        let (mut area1, mut area2) = (Vec::new(), Vec::new());
        for (t, a) in first {
            taps.extend(t);
            area1.push(a);
        }
        for (t, a) in second {
            taps.extend(t);
            area2.push(a);
        }
        PlaneTaps { taps, area1, area2 }
    }

    /// The plane of the taps' `values` (in their order): B's two sides averaged.
    pub fn plane(&self, values: &[f64]) -> Plane {
        let v = |n: usize| values.get(n).copied().unwrap_or(0.0);
        let (n1, n2) = (self.area1.len(), self.area2.len());
        let at = |base: usize, n: usize| -> [Vec<f64>; 2] {
            [(0..n).map(|i| v(base + 3 * i)).collect(), (0..n).map(|i| 0.5 * (v(base + 3 * i + 1) + v(base + 3 * i + 2))).collect()]
        };
        let [e1, b2] = at(0, n1);
        let [e2, b1] = at(3 * n1, n2);
        Plane { e1, b2, area1: self.area1.clone(), e2, b1, area2: self.area2.clone() }
    }

    /// The plane of the values `value(lattice, point)` gives at the taps.
    pub(crate) fn read(&self, value: impl Fn(usize, usize) -> Option<f64>) -> Plane {
        let values: Vec<f64> = self.taps.iter().map(|t| t.and_then(|(which, at)| value(which, at)).unwrap_or(0.0)).collect();
        self.plane(&values)
    }
}
