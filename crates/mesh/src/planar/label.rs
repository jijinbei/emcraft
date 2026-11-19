//! Which copper covers each triangle, layer by layer: the union of a layer's outlines without
//! computing it as polygons.
//!
//! Every outline's edges are edges of the triangulation, so no triangle straddles one: the
//! triangles joined across free edges (a component) are covered alike, and one point inside the
//! component decides for all of them. That point is the centroid of the component's roundest
//! triangle, its covering decided by the winding number about each outline near it (exact
//! `orient2d`; odd is inside, the even–odd rule — a pour's holes joined to its outline by a
//! bridge are holes whichever way they turn: K. Hormann, A. Agathos, *The point in polygon
//! problem for arbitrary polygons*, Computational Geometry 20(3), 2001).

use emcraft_tetmesh::predicates::{ExactPoint2 as Point2, Sign, orient2d};

use super::NO_NET;
use crate::{MeshError, Result};

/// The most cells of the outlines' grid, and of cell visits while filling it.
const MAX_CELLS: usize = 1 << 22;

/// A closed copper outline on one layer (lattice coordinates).
pub(super) struct Ring {
    pub layer: usize,
    pub net: u32,
    pub points: Vec<Point2>,
}

/// The outlines with a grid of the cells their boxes cover.
pub(super) struct Rings {
    rings: Vec<Ring>,
    lo: [f64; 2],
    side: f64,
    cells: [usize; 2],
    grid: Vec<Vec<u32>>,
}

fn fail(why: String) -> MeshError {
    MeshError::invalid("board copper", why)
}

fn bounds(points: &[Point2]) -> ([f64; 2], [f64; 2]) {
    points
        .iter()
        .map(Point2::coords)
        .fold(([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]), |(lo, hi), p| ([lo[0].min(p[0]), lo[1].min(p[1])], [hi[0].max(p[0]), hi[1].max(p[1])]))
}

impl Rings {
    /// Indexes `rings` over the box `lo`..`hi` (lattice coordinates).
    pub fn new(rings: Vec<Ring>, lo: [f64; 2], hi: [f64; 2]) -> Result<Rings> {
        let span = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1.0);
        let side = (span / (rings.len() as f64).sqrt().max(1.0)).max(1.0);
        let cells = [((hi[0] - lo[0]) / side).floor() as usize + 1, ((hi[1] - lo[1]) / side).floor() as usize + 1];
        let mut index = Rings { rings: Vec::new(), lo, side, cells, grid: vec![Vec::new(); cells[0].saturating_mul(cells[1]).min(MAX_CELLS)] };
        let mut visits = 0usize;
        for (k, ring) in rings.iter().enumerate() {
            let (a, b) = bounds(&ring.points);
            let ([i0, j0], [i1, j1]) = (index.cell(a), index.cell(b));
            for i in i0..=i1 {
                for j in j0..=j1 {
                    visits += 1;
                    if visits > MAX_CELLS {
                        return Err(fail("too many copper outlines cover the region; use a smaller region".into()));
                    }
                    if let Some(c) = index.grid.get_mut(i * cells[1] + j) {
                        c.push(k as u32);
                    }
                }
            }
        }
        index.rings = rings;
        Ok(index)
    }

    fn cell(&self, p: [f64; 2]) -> [usize; 2] {
        let at = |d: usize| (((p[d] - self.lo[d]) / self.side).floor().max(0.0) as usize).min(self.cells[d] - 1);
        [at(0), at(1)]
    }

    /// The outlines that hold `p`: an odd winding number about it (the even–odd rule).
    pub fn holding(&self, p: [f64; 2]) -> Result<impl Iterator<Item = &Ring>> {
        let exact = Point2::new(p[0], p[1]).map_err(|e| fail(e.to_string()))?;
        Ok(self.near(p).filter(move |ring| winding(&ring.points, &exact) % 2 != 0))
    }

    /// The outlines whose boxes may hold `p`.
    fn near(&self, p: [f64; 2]) -> impl Iterator<Item = &Ring> {
        let [i, j] = self.cell(p);
        self.grid.get(i * self.cells[1] + j).into_iter().flatten().filter_map(|&k| self.rings.get(k as usize))
    }
}

/// The winding number of `ring` about `p` (exact).
fn winding(ring: &[Point2], p: &Point2) -> i32 {
    let y = p.coords()[1];
    let mut w = 0;
    for (i, a) in ring.iter().enumerate() {
        let b = &ring[(i + 1) % ring.len()];
        let (ay, by) = (a.coords()[1], b.coords()[1]);
        if ay <= y {
            if by > y && orient2d(a, b, p) == Sign::Positive {
                w += 1;
            }
        } else if by <= y && orient2d(a, b, p) == Sign::Negative {
            w -= 1;
        }
    }
    w
}

/// The components of a triangulation — triangles joined across edges that are not in
/// `constrained` (each as `[low id, high id]`, sorted) — and a point inside each: each triangle's
/// component (numbered in the order of their first triangles), and for each component the
/// centroid of its roundest triangle (largest inradius, 2 area / perimeter; ties by index),
/// where its covering is decided.
pub(super) fn components(points: &[[f64; 2]], triangles: &[[u32; 3]], constrained: &[[u32; 2]]) -> (Vec<u32>, Vec<[f64; 2]>) {
    let mut edges: Vec<([u32; 2], u32)> = triangles
        .iter()
        .enumerate()
        .flat_map(|(t, v)| (0..3).map(move |k| ([v[k].min(v[(k + 1) % 3]), v[k].max(v[(k + 1) % 3])], t as u32)))
        .collect();
    edges.sort_unstable();
    let mut parent: Vec<u32> = (0..triangles.len() as u32).collect();
    for w in edges.windows(2) {
        if w[0].0 == w[1].0 && constrained.binary_search(&w[0].0).is_err() {
            let (a, b) = (root(&mut parent, w[0].1), root(&mut parent, w[1].1));
            parent[a.max(b) as usize] = a.min(b);
        }
    }
    let corner = |v: u32| points.get(v as usize).copied().unwrap_or([0.0; 2]);
    let mut best: Vec<(f64, u32)> = vec![(-1.0, 0); triangles.len()];
    let mut of: Vec<u32> = Vec::with_capacity(triangles.len());
    for (t, v) in triangles.iter().enumerate() {
        let [a, b, c] = v.map(corner);
        let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        let perimeter = (b[0] - a[0]).hypot(b[1] - a[1]) + (c[0] - b[0]).hypot(c[1] - b[1]) + (a[0] - c[0]).hypot(a[1] - c[1]);
        let k = root(&mut parent, t as u32);
        of.push(k);
        if let Some(b) = best.get_mut(k as usize).filter(|b| area / perimeter > b.0) {
            *b = (area / perimeter, t as u32);
        }
    }
    let mut number = vec![u32::MAX; triangles.len()];
    let mut middles: Vec<[f64; 2]> = Vec::new();
    for k in of.iter_mut() {
        let r = *k as usize;
        if number[r] == u32::MAX {
            number[r] = middles.len() as u32;
            let [a, b, c] = triangles.get(best[r].1 as usize).map_or([[0.0; 2]; 3], |v| v.map(corner));
            middles.push([(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0]);
        }
        *k = number[r];
    }
    (of, middles)
}

/// The root of `t` in the union–find forest `parent`, halving the path on the way.
pub(super) fn root(parent: &mut [u32], mut t: u32) -> u32 {
    while parent[t as usize] != t {
        let up = parent[parent[t as usize] as usize];
        parent[t as usize] = up;
        t = up;
    }
    t
}

/// Each layer's net over each triangle ([`NO_NET`] where there is no copper). `constrained`
/// holds the outlines' edges (each as `[low id, high id]`, sorted); `names` names the layers and
/// nets for the error when two nets cover the same place on a layer.
pub(super) fn label(
    points: &[[f64; 2]],
    triangles: &[[u32; 3]],
    constrained: &[[u32; 2]],
    rings: &Rings,
    names: (&[String], &[String]),
) -> Result<Vec<Vec<u32>>> {
    let (of, middles) = components(points, triangles, constrained);
    let decided: Vec<Vec<u32>> = middles.iter().map(|&p| covering(rings, p, names)).collect::<Result<_>>()?;
    let mut nets = vec![vec![NO_NET; triangles.len()]; names.0.len()];
    for (t, k) in of.iter().enumerate() {
        for (layer, &net) in decided.get(*k as usize).into_iter().flatten().enumerate() {
            if let Some(x) = nets.get_mut(layer).and_then(|n| n.get_mut(t)) {
                *x = net;
            }
        }
    }
    Ok(nets)
}

/// The net on each layer at `p`, refusing two nets in one place.
fn covering(rings: &Rings, p: [f64; 2], (layers, nets): (&[String], &[String])) -> Result<Vec<u32>> {
    let mut on = vec![NO_NET; layers.len()];
    for ring in rings.holding(p)? {
        let Some(x) = on.get_mut(ring.layer) else { continue };
        if *x != NO_NET && *x != ring.net {
            let name = |n: u32| nets.get(n as usize).map_or_else(|| n.to_string(), Clone::clone);
            let (a, b) = (name((*x).min(ring.net)), name((*x).max(ring.net)));
            let layer = layers.get(ring.layer).map_or_else(|| (ring.layer + 1).to_string(), Clone::clone);
            return Err(fail(format!("nets {a} and {b} overlap on layer {layer} (a short circuit)")));
        }
        *x = ring.net;
    }
    Ok(on)
}
