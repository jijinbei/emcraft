//! Copper pours filled: a zone's outline less the copper it must keep clear of, as polygons.
//!
//! The Boolean is found on a triangulation, as the planar mesher finds the union of a layer's
//! copper ([`super`]): every outline — the zone's, the keep-outs (other nets' copper grown by the
//! clearance), the reliefs round the net's own pads, the spokes across them and the net's own
//! copper — on the lattice of [`super::lattice`], split where they meet ([`super::split`]) and put
//! in one constrained triangulation ([`super::cdt`]). Each component of triangles between the
//! outlines is covered alike, decided at one point inside it by the even–odd rule
//! ([`super::label`]):
//!
//! filled = zone ∧ ¬keep-out ∧ (¬relief ∨ spoke ∨ own).
//!
//! The filled triangles joined edge to edge are the pour's islands. An island holding none of
//! the net's own copper and smaller than `min_island` is dropped. Each island's boundary — the
//! edges between it and the rest, in the triangles' counterclockwise turn, the points along
//! straight runs left out — comes back as one even–odd polygon: its loops (the outline and the
//! holes) joined by bridges out from its first point and back along the same line, so the
//! bridges cancel under the even–odd rule and the polygon is read as copper is everywhere else.
//!
//! The same input gives the same polygons bit for bit (the lattice, exact predicates, no hash
//! order). Every count is capped.
//!
//! ```
//! use emcraft_mesh::planar::fill::{Pour, fill};
//!
//! // A 10 mm square pour round a 2 mm square of another net, kept 0.5 mm clear of it.
//! let square = |c: f64, h: f64| vec![[c - h, c - h], [c + h, c - h], [c + h, c + h], [c - h, c + h]];
//! let pour = Pour { outline: &square(5e-3, 5e-3), keepout: &[square(5e-3, 1.5e-3)], ..Pour::default() };
//! let islands = fill(&pour)?;
//! assert_eq!(islands.len(), 1);
//! // The square with its 3 mm square hole (the loops turn opposite ways): 100 − 9 mm².
//! let p = &islands[0];
//! let area = (0..p.len()).map(|k| p[k][0] * p[(k + 1) % p.len()][1] - p[(k + 1) % p.len()][0] * p[k][1]).sum::<f64>() / 2.0;
//! assert!((area - 91e-6).abs() < 1e-12);
//! // An outline that is not finite is refused.
//! assert!(fill(&Pour { outline: &[[f64::NAN, 0.0], [1e-3, 0.0], [0.0, 1e-3]], ..Pour::default() }).is_err());
//! # Ok::<(), emcraft_mesh::MeshError>(())
//! ```

use std::collections::BTreeMap;

use emcraft_tetmesh::predicates::{Sign, orient2d};

use super::cdt::Cdt;
use super::label::{self, Ring, Rings, root};
use super::lattice::{self, Lattice, MAX_SIDE};
use super::split::{self, Points};
use super::{MAX_POINTS, MAX_TRIANGLES, exact, invalid, offset, pieces};
use crate::Result;

/// A pour to fill (metres; outlines either way round, each closed implicitly).
#[derive(Clone, Copy, Debug, Default)]
pub struct Pour<'a> {
    /// The zone's outline.
    pub outline: &'a [[f64; 2]],
    /// Where the pour may not go: other nets' copper grown by the clearance.
    pub keepout: &'a [Vec<[f64; 2]>],
    /// The gaps round the net's own pads that only spokes cross.
    pub relief: &'a [Vec<[f64; 2]>],
    /// The spokes across the reliefs.
    pub spokes: &'a [Vec<[f64; 2]>],
    /// The net's own copper: filled wherever the zone holds it, and an island holding some is kept.
    pub own: &'a [Vec<[f64; 2]>],
    /// The smallest island (m²) kept that holds none of the net's copper.
    pub min_island: f64,
}

/// The roles of the outlines, as the nets of the triangulation's rings.
const ZONE: u32 = 0;
const KEEPOUT: u32 = 1;
const RELIEF: u32 = 2;
const SPOKE: u32 = 3;
const OWN: u32 = 4;

/// The pour filled: one even–odd polygon (metres) per island, in the order of their first
/// triangles. Refused: an outline of fewer than 3 points or not finite, a zone wider than
/// [`MAX_SIDE`], a minimum island that is not a number ≥ 0, more points or triangles than the
/// planar mesher's caps.
pub fn fill(p: &Pour) -> Result<Vec<Vec<[f64; 2]>>> {
    let finite = |r: &[[f64; 2]]| r.iter().flatten().all(|v| v.is_finite());
    let zone = [p.outline.to_vec()];
    let groups: [(u32, &[Vec<[f64; 2]>]); 5] = [(ZONE, &zone), (KEEPOUT, p.keepout), (RELIEF, p.relief), (SPOKE, p.spokes), (OWN, p.own)];
    if p.outline.len() < 3 || !groups.iter().all(|(_, rs)| rs.iter().all(|r| finite(r))) {
        return Err(invalid("a pour's outlines need 3 or more finite points"));
    }
    if !(p.min_island.is_finite() && p.min_island >= 0.0) {
        return Err(invalid("a pour's smallest island must be a number ≥ 0"));
    }
    let total: usize = groups.iter().flat_map(|(_, rs)| rs.iter().map(Vec::len)).sum();
    if total > MAX_POINTS {
        return Err(invalid(format!("a pour's outlines have {total} points, more than the limit of {MAX_POINTS}")));
    }
    let (lo, hi) = bounds(p.outline);
    if hi[0] - lo[0] > MAX_SIDE || hi[1] - lo[1] > MAX_SIDE || hi[0] <= lo[0] || hi[1] <= lo[1] {
        return Err(invalid(format!("a pour needs an area and at most {MAX_SIDE} m across")));
    }
    // A box a micrometre round the zone, so no outline runs along its sides.
    let (lo, hi) = ([lo[0] - 1e-6, lo[1] - 1e-6], [hi[0] + 1e-6, hi[1] + 1e-6]);
    let lattice = Lattice::new(lo);
    let top = lattice.on(hi);
    let corners = [[0.0, 0.0], [top[0], 0.0], top, [0.0, top[1]]];
    let mut points = Points::default();
    let ids: Vec<u32> = corners.iter().map(|&c| points.id(c)).collect();
    let mut edges: Vec<[u32; 2]> = (0..4).map(|k| [ids[k], ids[(k + 1) % 4]]).collect();
    let mut owners: Vec<(u32, usize)> = Vec::new();
    for &(role, rings) in &groups {
        for ring in rings {
            let (a, b) = bounds(ring);
            if a[0] > hi[0] || a[1] > hi[1] || b[0] < lo[0] || b[1] < lo[1] {
                continue;
            }
            let mut ids: Vec<u32> = offset::clipped(ring, lo, hi).into_iter().map(|q| points.id(lattice.on(q))).collect();
            ids.dedup();
            while ids.len() > 1 && ids.first() == ids.last() {
                ids.pop();
            }
            if ids.len() < 3 {
                continue;
            }
            owners.push((role, edges.len()));
            edges.extend((0..ids.len()).map(|k| [ids[k], ids[(k + 1) % ids.len()]]));
        }
    }
    if points.xy.len() > MAX_POINTS {
        return Err(invalid(format!("a pour's outlines need more than {MAX_POINTS} points")));
    }
    let chains = split::split(&mut points, &edges)?;
    let rings: Vec<Ring> = owners
        .iter()
        .enumerate()
        .map(|(k, &(role, first))| {
            let end = owners.get(k + 1).map_or(edges.len(), |o| o.1);
            let ids: Vec<u32> =
                chains.get(first..end).into_iter().flatten().flat_map(|c| c.split_last().map_or(&[][..], |(_, rest)| rest)).copied().collect();
            let xy: Vec<[f64; 2]> = ids.iter().filter_map(|&v| points.xy.get(v as usize).copied()).collect();
            Ok(Ring { layer: 0, net: role, points: exact(&xy)? })
        })
        .collect::<Result<_>>()?;
    let rings = Rings::new(rings, [0.0, 0.0], top)?;
    let (segments, constrained) = pieces(&chains, 4);
    let tri = Cdt::build(&exact(&points.xy)?, &segments, MAX_TRIANGLES)?.into_triangulation();
    let (of, middles) = label::components(&tri.points, &tri.triangles, &constrained);
    // Each component's roles, and whether it is filled and holds the net's copper.
    let decided: Vec<(bool, bool)> = middles
        .iter()
        .map(|&m| {
            let mut on = [false; 5];
            for ring in rings.holding(m)? {
                if let Some(x) = on.get_mut(ring.net as usize) {
                    *x = true;
                }
            }
            let filled = on[ZONE as usize] && !on[KEEPOUT as usize] && (!on[RELIEF as usize] || on[SPOKE as usize] || on[OWN as usize]);
            Ok((filled, on[OWN as usize]))
        })
        .collect::<Result<_>>()?;
    let state = |t: usize| of.get(t).and_then(|&k| decided.get(k as usize)).copied().unwrap_or((false, false));
    let islands = islands(&tri.triangles, &state);
    let corner = |v: u32| tri.points.get(v as usize).copied().unwrap_or([0.0; 2]);
    let area = |t: &[u32; 3]| {
        let [a, b, c] = t.map(corner);
        0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) * lattice::STEP * lattice::STEP
    };
    let count = islands.iter().filter_map(|i| *i).max().map_or(0, |n| n as usize + 1);
    let mut sizes = vec![(0.0, false); count];
    for (t, island) in islands.iter().enumerate() {
        if let (Some(i), Some(v)) = (island, tri.triangles.get(t))
            && let Some(s) = sizes.get_mut(*i as usize)
        {
            s.0 += area(v);
            s.1 |= state(t).1;
        }
    }
    let kept: Vec<bool> = sizes.iter().map(|&(a, own)| own || a >= p.min_island).collect();
    let mut out = Vec::new();
    for (i, _) in kept.iter().enumerate().filter(|(_, k)| **k) {
        let loops = boundary(&tri.triangles, &islands, i as u32, &tri.points)?;
        out.push(bridged(&loops).into_iter().map(|q| lattice.back(q)).collect());
    }
    Ok(out)
}

fn bounds(ring: &[[f64; 2]]) -> ([f64; 2], [f64; 2]) {
    ring.iter()
        .fold(([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]), |(lo, hi), p| ([lo[0].min(p[0]), lo[1].min(p[1])], [hi[0].max(p[0]), hi[1].max(p[1])]))
}

/// Each triangle's island (the filled triangles joined across their edges, numbered in the order
/// of their first triangles), `None` where it is not filled.
fn islands(triangles: &[[u32; 3]], state: &impl Fn(usize) -> (bool, bool)) -> Vec<Option<u32>> {
    let mut edges: Vec<([u32; 2], u32)> = triangles
        .iter()
        .enumerate()
        .filter(|(t, _)| state(*t).0)
        .flat_map(|(t, v)| (0..3).map(move |k| ([v[k].min(v[(k + 1) % 3]), v[k].max(v[(k + 1) % 3])], t as u32)))
        .collect();
    edges.sort_unstable();
    let mut parent: Vec<u32> = (0..triangles.len() as u32).collect();
    for w in edges.windows(2) {
        if w[0].0 == w[1].0 {
            let (a, b) = (root(&mut parent, w[0].1), root(&mut parent, w[1].1));
            parent[a.max(b) as usize] = a.min(b);
        }
    }
    let mut number: BTreeMap<u32, u32> = BTreeMap::new();
    (0..triangles.len())
        .map(|t| {
            state(t).0.then(|| {
                let r = root(&mut parent, t as u32);
                let n = number.len() as u32;
                *number.entry(r).or_insert(n)
            })
        })
        .collect()
}

/// The loops bounding island `island` (lattice points): its triangles' edges whose other side is
/// not in it, each in its triangle's counterclockwise turn, chained end to start; the points
/// along a straight run left out.
fn boundary(triangles: &[[u32; 3]], islands: &[Option<u32>], island: u32, points: &[[f64; 2]]) -> Result<Vec<Vec<[f64; 2]>>> {
    let inside: Vec<&[u32; 3]> = triangles.iter().zip(islands).filter(|(_, i)| **i == Some(island)).map(|(t, _)| t).collect();
    let mut directed: Vec<[u32; 2]> = inside.iter().flat_map(|v| (0..3).map(move |k| [v[k], v[(k + 1) % 3]])).collect();
    directed.sort_unstable();
    // An edge bounds the island when its reverse is not one of the island's edges.
    let mut out: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for &[a, b] in &directed {
        if directed.binary_search(&[b, a]).is_err() {
            out.entry(a).or_default().push(b);
        }
    }
    let at = |v: u32| points.get(v as usize).copied().ok_or_else(|| invalid("a pour's point out of range"));
    let mut loops = Vec::new();
    while let Some((&start, _)) = out.iter().find(|(_, ends)| !ends.is_empty()) {
        let mut ring = vec![start];
        let mut v = start;
        loop {
            let Some(next) = out.get_mut(&v).and_then(Vec::pop) else {
                return Err(invalid("a pour's boundary does not close"));
            };
            if next == start {
                break;
            }
            ring.push(next);
            v = next;
            if ring.len() > MAX_POINTS {
                return Err(invalid("a pour's boundary has too many points"));
            }
        }
        let xy: Vec<[f64; 2]> = ring.iter().map(|&v| at(v)).collect::<Result<_>>()?;
        loops.push(straightened(&xy)?);
    }
    Ok(loops)
}

/// A loop without the points that lie on the straight line between their neighbours (exact).
fn straightened(ring: &[[f64; 2]]) -> Result<Vec<[f64; 2]>> {
    let e = exact(ring)?;
    let n = e.len();
    Ok((0..n).filter(|&k| orient2d(&e[(k + n - 1) % n], &e[k], &e[(k + 1) % n]) != Sign::Zero).filter_map(|k| ring.get(k).copied()).collect())
}

/// The loops as one polygon: each joined to the first loop's first point by a bridge out and back
/// along the same line (the bridges cancel under the even–odd rule).
fn bridged(loops: &[Vec<[f64; 2]>]) -> Vec<[f64; 2]> {
    let mut polygon: Vec<[f64; 2]> = loops.first().cloned().unwrap_or_default();
    let Some(&anchor) = polygon.first() else { return polygon };
    for ring in loops.iter().skip(1).filter(|r| !r.is_empty()) {
        polygon.push(anchor);
        polygon.extend(ring.iter().copied());
        polygon.extend(ring.first().copied());
    }
    polygon
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(c: [f64; 2], h: f64) -> Vec<[f64; 2]> {
        vec![[c[0] - h, c[1] - h], [c[0] + h, c[1] - h], [c[0] + h, c[1] + h], [c[0] - h, c[1] + h]]
    }

    /// The even–odd area of a polygon (m²).
    fn area(p: &[[f64; 2]]) -> f64 {
        (0..p.len()).map(|k| p[k][0] * p[(k + 1) % p.len()][1] - p[(k + 1) % p.len()][0] * p[k][1]).sum::<f64>().abs() * 0.5
    }

    /// Whether `q` is inside `p` by the even–odd rule.
    fn inside(p: &[[f64; 2]], q: [f64; 2]) -> bool {
        let mut odd = false;
        for k in 0..p.len() {
            let (a, b) = (p[k], p[(k + 1) % p.len()]);
            if (a[1] > q[1]) != (b[1] > q[1]) && q[0] < a[0] + (q[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]) {
                odd = !odd;
            }
        }
        odd
    }

    /// A keep-out across the pour splits it in two islands; the one without the net's copper is
    /// dropped when smaller than `min_island`, kept otherwise.
    #[test]
    fn a_keepout_across_splits_the_pour_and_small_islands_go() {
        let outline = square([5e-3, 5e-3], 5e-3);
        let band = vec![[-1e-3, 6e-3], [11e-3, 6e-3], [11e-3, 7e-3], [-1e-3, 7e-3]];
        let own = vec![square([5e-3, 2e-3], 0.5e-3)];
        let keepout = [band];
        let pour = Pour { outline: &outline, keepout: &keepout, own: &own, ..Pour::default() };
        let islands = fill(&pour).unwrap();
        assert_eq!(islands.len(), 2);
        let mut areas: Vec<f64> = islands.iter().map(|i| area(i)).collect();
        areas.sort_by(f64::total_cmp);
        assert!((areas[0] - 30e-6).abs() < 1e-12 && (areas[1] - 60e-6).abs() < 1e-12, "{areas:?}");
        // The upper island (30 mm², none of the net's copper) dropped under a 40 mm² minimum.
        let islands = fill(&Pour { min_island: 40e-6, ..pour }).unwrap();
        assert!(islands.len() == 1 && inside(&islands[0], [5e-3, 2e-3]) && !inside(&islands[0], [5e-3, 9e-3]));
    }

    /// A pad of the net inside a relief joined by its spokes: the gap empty but where they cross.
    #[test]
    fn spokes_cross_a_relief() {
        let outline = square([0.0, 0.0], 5e-3);
        let pad = square([0.0, 0.0], 0.5e-3);
        let relief = [square([0.0, 0.0], 0.8e-3)];
        let spokes = [vec![[-1e-3, -0.1e-3], [1e-3, -0.1e-3], [1e-3, 0.1e-3], [-1e-3, 0.1e-3]]];
        let own = [pad];
        let islands = fill(&Pour { outline: &outline, relief: &relief, spokes: &spokes, own: &own, ..Pour::default() }).unwrap();
        assert_eq!(islands.len(), 1, "the pad joined to the pour");
        let p = &islands[0];
        assert!(inside(p, [0.1e-3, 0.03e-3]) && inside(p, [0.65e-3, 0.03e-3]) && !inside(p, [0.1e-3, 0.65e-3]) && inside(p, [2e-3, 2e-3]));
        // 100 mm² less the gap (1.6² − 1² mm²) but the two spokes across it (2 × 0.3 × 0.2 mm²).
        assert!((area(p) - (100e-6 - (2.56e-6 - 1e-6) + 2.0 * 0.3e-3 * 0.2e-3)).abs() < 1e-12, "{}", area(p));
        // Without spokes the pad is an island of its own, the pour round it another.
        let islands = fill(&Pour { outline: &outline, relief: &relief, own: &own, ..Pour::default() }).unwrap();
        assert_eq!(islands.len(), 2);
    }

    /// The same pour twice gives the same polygons, bit for bit; keep-outs that overlap and
    /// reach outside the zone are united and clipped.
    #[test]
    fn overlapping_keepouts_unite_and_the_fill_repeats() {
        let outline = vec![[0.0, 0.0], [8e-3, 0.0], [8e-3, 4e-3], [4e-3, 6e-3], [0.0, 4e-3]];
        let keepout: Vec<Vec<[f64; 2]>> =
            (0..6).map(|k| offset::trace([k as f64 * 1.5e-3, -1e-3], [k as f64 * 1.5e-3 + 1e-3, 7e-3], 0.6e-3, 1e-6)).collect();
        let pour = Pour { outline: &outline, keepout: &keepout, ..Pour::default() };
        let a = fill(&pour).unwrap();
        assert!(a.len() >= 5, "{}", a.len());
        assert_eq!(a, fill(&pour).unwrap());
        // On a grid of points: copper where the zone is and no keep-out, nowhere else.
        for i in 0..80 {
            for j in 0..60 {
                let q = [0.0503e-3 + i as f64 * 0.1e-3, 0.0507e-3 + j as f64 * 0.1e-3];
                let copper = a.iter().any(|island| inside(island, q));
                assert_eq!(copper, inside(&outline, q) && !keepout.iter().any(|k| inside(k, q)), "{q:?}");
            }
        }
    }

    #[test]
    fn hostile_pours_are_refused() {
        let outline = square([0.0, 0.0], 1e-3);
        assert!(fill(&Pour { outline: &outline[..2], ..Pour::default() }).is_err());
        assert!(fill(&Pour { outline: &outline, min_island: -1.0, ..Pour::default() }).is_err());
        assert!(fill(&Pour { outline: &square([0.0, 0.0], 2.0), ..Pour::default() }).is_err());
        let bad = [vec![[0.0, 0.0], [f64::INFINITY, 0.0], [0.0, 1.0]]];
        assert!(fill(&Pour { outline: &outline, keepout: &bad, ..Pour::default() }).is_err());
        // A zone with all of it kept out has no islands.
        let all = [square([0.0, 0.0], 2e-3)];
        assert!(fill(&Pour { outline: &outline, keepout: &all, ..Pour::default() }).unwrap().is_empty());
    }
}
