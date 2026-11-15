//! An imported extrusion stopped at a plane across its axis (a symmetry plane halving it):
//! everything past the plane brought down onto it, the vertices that meet welded, the flattened
//! triangles dropped — the surface shortened as clean as it was, where a Boolean with a box
//! would leave slivers.

use emcraft_model::{FaceSurface, Point3, Shape, Vec3};

use crate::shared::{closed, corners};

/// A plane a surface is stopped at: across axis `d`, at `at` (m), within `tol` (m).
#[derive(Clone, Copy)]
pub(super) struct Cut {
    pub(super) d: usize,
    pub(super) at: f64,
    pub(super) tol: f64,
}

/// How far from upright (the share of its normal along the cut's axis) a triangle across the
/// cut may lean: a design's sides are upright to rounding.
const UPRIGHT: f64 = 1e-7;

/// An imported surface stopped at the cut, if it is an extrusion along the cut's axis: every
/// triangle across the plane upright, and every vertex at or past it on a line along the axis
/// through a vertex of the far end (a design's tall sides may be split at heights of their own).
/// Everything past the plane comes down onto it, the vertices that meet there are one and the
/// triangles flattened between them go; none is folded over, and the result is closed.
pub(super) fn stopped_surface(shape: &Shape, cut: Cut) -> Option<Shape> {
    let Shape::Surface { name, positions, triangles, tags, surfaces } = shape else { return None };
    let Cut { d, at, tol } = cut;
    let hi = positions.iter().map(|p| p[d]).fold(f64::NEG_INFINITY, f64::max);
    let lo = positions.iter().map(|p| p[d]).fold(f64::INFINITY, f64::min);
    if !(at > lo + tol && at < hi - tol && upright(positions, triangles, cut)) {
        return None;
    }
    let down = brought_down(positions, cut, hi)?;
    let kept = unfolded(positions, triangles, &down, tol)?;
    let triangles: Vec<[u32; 3]> = kept.iter().map(|(_, t)| *t).collect();
    if triangles.is_empty() || !closed(&triangles) {
        return None;
    }
    // A plane on the far end comes down with it.
    let surfaces = surfaces
        .iter()
        .map(|s| match s {
            Some(FaceSurface::Plane { normal, point }) if (point[d] - hi).abs() <= tol => {
                Some(FaceSurface::Plane { normal: *normal, point: point.with(d, at) })
            }
            other => other.clone(),
        })
        .collect();
    let tags = kept.iter().map(|(k, _)| tags.get(*k).copied().unwrap_or(0)).collect();
    Some(Shape::Surface { name: name.clone(), positions: down.points, triangles, tags, surfaces })
}

/// A triangle's normal, its length twice the area.
fn normal([a, b, c]: [Point3; 3]) -> Vec3 {
    (b - a).cross(c - a)
}

/// Whether every triangle across the cut's plane is upright: its plane holds the cut's axis.
fn upright(positions: &[Point3], triangles: &[[u32; 3]], Cut { d, at, tol }: Cut) -> bool {
    triangles.iter().all(|t| {
        corners(positions, t).is_some_and(|c| {
            let crosses = c.iter().any(|p| p[d] < at - tol) && c.iter().any(|p| p[d] > at + tol);
            let n = normal(c);
            !crosses || n[d].abs() <= UPRIGHT * n.norm()
        })
    })
}

/// The vertices with everything past the cut brought down onto it and those that meet made one,
/// if every vertex at or past the plane meets one from the far end `hi` (else the sides'
/// vertices there would cut the moved end's triangles).
fn brought_down(positions: &[Point3], Cut { d, at, tol }: Cut, hi: f64) -> Option<Welded> {
    let moved: Vec<Point3> = positions.iter().map(|p| if p[d] > at { p.with(d, at) } else { *p }).collect();
    let down = welded(&moved, tol)?;
    let far: std::collections::BTreeSet<u32> = positions.iter().zip(&down.index).filter(|(p, _)| (p[d] - hi).abs() <= tol).map(|(_, i)| *i).collect();
    let met = positions.iter().zip(&down.index).all(|(p, i)| p[d] < at - tol || far.contains(i));
    met.then_some(down)
}

/// The triangles left once the vertices are brought `down`, each with its index among the
/// triangles: the flattened ones gone; none if one is folded over (turned against its side).
fn unfolded(positions: &[Point3], triangles: &[[u32; 3]], down: &Welded, tol: f64) -> Option<Vec<(usize, [u32; 3])>> {
    let moved = |t: &[u32; 3]| t.map(|i| down.index[i as usize]);
    let flattened = |w: [u32; 3]| w[0] == w[1] || w[1] == w[2] || w[0] == w[2] || normal(w.map(|i| down.points[i as usize])).norm() <= tol * tol;
    let kept: Vec<(usize, [u32; 3])> = triangles.iter().enumerate().map(|(k, t)| (k, moved(t))).filter(|(_, w)| !flattened(*w)).collect();
    let turned = |(k, w): &(usize, [u32; 3])| {
        let before = triangles.get(*k).and_then(|t| corners(positions, t)).map(normal);
        before.is_some_and(|n| n.dot(normal(w.map(|i| down.points[i as usize]))) > 0.0)
    };
    kept.iter().all(turned).then_some(kept)
}

/// Points made one where they lie within a tolerance of one another: those kept, and each
/// original point's index among them.
struct Welded {
    points: Vec<Point3>,
    index: Vec<u32>,
}

/// `points` with each one within `tol` (m) of a point kept before it made that one; none for a
/// `tol` that is not a positive number (every point would fall into one cell).
fn welded(points: &[Point3], tol: f64) -> Option<Welded> {
    if !(tol > 0.0 && tol.is_finite()) {
        return None;
    }
    let key = |p: Point3| p.to_array().map(|c| (c / tol).round() as i64);
    let mut seen: std::collections::BTreeMap<[i64; 3], Vec<u32>> = std::collections::BTreeMap::new();
    let mut kept: Vec<Point3> = Vec::new();
    let index = points
        .iter()
        .map(|p| {
            let k = key(*p);
            // A point within `tol` may round into a neighbouring cell: those count too.
            let near = |n: i64| [k[0] + n % 3 - 1, k[1] + n / 3 % 3 - 1, k[2] + n / 9 - 1];
            let found = (0..27).flat_map(|n| seen.get(&near(n)).into_iter().flatten()).find(|&&i| (kept[i as usize] - *p).norm() <= tol).copied();
            found.unwrap_or_else(|| {
                let i = kept.len() as u32;
                kept.push(*p);
                seen.entry(k).or_default().push(i);
                i
            })
        })
        .collect();
    Some(Welded { points: kept, index })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Points weld only within the tolerance: two in neighbouring cells 1.4 tol apart stay two
    /// (the neighbours' cells were once taken whole, merging points up to 3.5 tol apart).
    #[test]
    fn points_weld_only_within_the_tolerance() {
        let p = |x: f64| Point3::new(x, 0.0, 0.0);
        let w = welded(&[p(0.0), p(0.4), p(1.4), p(2.2)], 1.0).unwrap();
        assert_eq!(w.index, vec![0, 0, 1, 1]);
        assert_eq!(w.points.len(), 2);
        let far = welded(&[p(0.0), p(1.4)], 1.0).unwrap();
        assert_eq!(far.index, vec![0, 1]);
        assert!(welded(&[p(0.0)], 0.0).is_none() && welded(&[p(0.0)], f64::NAN).is_none());
    }
}
