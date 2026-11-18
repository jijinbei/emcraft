//! Inclusions that touch along a cylinder: a rod in its sleeve (the rod's side against the
//! sleeve's bore, facing each other), a disc that fills the tube it lies in (its side in the
//! tube's, facing the same way). The bodies' faces on one cylinder — each tessellated its own
//! way, their corners on the cylinder — are triangulated again together on the cylinder
//! unrolled: the plane of the arc length R·θ and the height z along the axis, cut along one
//! straight line of the cylinder (the seam) where no corner lies.
//!
//! As in a plane ([`super`]), the faces' outlines, cut where they cross, are the constraints of
//! one constrained Delaunay triangulation refined to the size (Sloan 1993, Ruppert 1995,
//! `crate::planar::pslg_keep`); here the seam's two sides are constraints too, cut at the same
//! heights, so the triangles on either side of it meet. A triangle is put back on the cylinder
//! by its corners (a flat facet, its corners on the cylinder).
//!
//! A face's rim was a polygon of chords between its own corners; with the other body's points on
//! it, it has more. The triangle beyond each rim edge (a cap's, in the same body) is split into a
//! fan to them, so the body's surface stays closed.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;

use emcraft_model::{Point3, Vec3};

use super::{Body, MERGE, Nesting, Piece, Points, crossing, fail, on_segment, put};
use crate::Result;
use crate::shared::DisjointSets;
use crate::solid::narrow::{overlap, polygon_area};

/// A cylinder: a point of its axis, the axis (a unit vector) and its radius (m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::solid) struct Round {
    pub(in crate::solid) origin: Point3,
    pub(in crate::solid) axis: Vec3,
    pub(in crate::solid) radius: f64,
}

impl Round {
    /// Whether `o` is this cylinder: its axis along this one's (either way), a point of it on
    /// this one's and its radius the same, to `tol` (m).
    fn same(&self, o: &Round, tol: f64) -> bool {
        let off = (o.origin - self.origin).tangential(self.axis).norm();
        self.axis.dot(o.axis).abs() > 1.0 - 1e-9 && (self.radius - o.radius).abs() <= tol && off <= tol
    }
}

/// The number of cylinder `c` among `cylinders`, added when it is none of them.
pub(super) fn number(cylinders: &mut Vec<Round>, c: Round, tol: f64) -> usize {
    cylinders.iter().position(|o| o.same(&c, tol)).unwrap_or_else(|| {
        cylinders.push(c);
        cylinders.len() - 1
    })
}

/// A cylinder unrolled: arc length from the seam (the line at `e1`) and height along the axis.
struct Chart {
    origin: Point3,
    axis: Vec3,
    e1: Vec3,
    e2: Vec3,
    radius: f64,
}

impl Chart {
    /// The chart of `c` with its seam in the widest gap between the `points`' angles.
    fn new(c: &Round, points: impl Iterator<Item = Point3>) -> Chart {
        let k = c.axis.dominant_axis();
        let e1 = Vec3::unit((k + 1) % 3).tangential(c.axis).normalized().unwrap_or_else(|| Vec3::unit((k + 1) % 3));
        let mut chart = Chart { origin: c.origin, axis: c.axis, e1, e2: c.axis.cross(e1), radius: c.radius };
        let mut angles: Vec<f64> = points.map(|p| chart.flat(p)[0] / c.radius).collect();
        angles.sort_by(f64::total_cmp);
        let gaps = angles.windows(2).map(|w| (w[1] - w[0], w[0])).chain(angles.first().zip(angles.last()).map(|(a, b)| (a + 2.0 * PI - b, *b)));
        if let Some((gap, from)) = gaps.max_by(|a, b| a.0.total_cmp(&b.0).then(b.1.total_cmp(&a.1))) {
            let (sin, cos) = (from + 0.5 * gap).sin_cos();
            (chart.e1, chart.e2) = (e1 * cos + chart.e2 * sin, chart.e2 * cos - e1 * sin);
        }
        chart
    }

    /// The seam to the seam again: the circle's length.
    fn around(&self) -> f64 {
        2.0 * PI * self.radius
    }

    /// `p` on the chart: its arc length from the seam in `[0, around)` and its height.
    fn flat(&self, p: Point3) -> [f64; 2] {
        let d = p - self.origin;
        let angle = d.dot(self.e2).atan2(d.dot(self.e1));
        [self.radius * if angle < 0.0 { angle + 2.0 * PI } else { angle }, d.dot(self.axis)]
    }

    /// The cylinder's point at `q`.
    fn lift(&self, q: [f64; 2]) -> Point3 {
        let (sin, cos) = (q[0] / self.radius).sin_cos();
        self.origin + (self.e1 * cos + self.e2 * sin) * self.radius + self.axis * q[1]
    }

    /// `q` with its arc length the nearest to `near`'s (a turn more or less).
    fn beside(&self, q: [f64; 2], near: [f64; 2]) -> [f64; 2] {
        let turn = self.around();
        [q[0] + turn * ((near[0] - q[0]) / turn).round(), q[1]]
    }
}

/// A body's face on the cylinder: its triangles on the chart (each beside its first corner, so
/// one across the seam reaches past it), whether it bulges outwards (a rod's side; a bore's does
/// not), and its rim as the body's vertices.
struct Face {
    body: usize,
    convex: bool,
    flat: Vec<[[f64; 2]; 3]>,
    rim: Vec<[u32; 2]>,
}

impl Face {
    fn new(bodies: &[Body], k: usize, cylinder: usize, chart: &Chart) -> Option<Face> {
        let body = bodies.get(k)?;
        let own: Vec<usize> = (0..body.triangles.len()).filter(|t| body.on.get(*t) == Some(&Some(cylinder))).collect();
        let first = body.corners(*own.first()?)?;
        let out = (first[1] - first[0]).cross(first[2] - first[0]);
        let from_axis = (Point3::centroid(first)? - chart.origin).tangential(chart.axis);
        let flat = own
            .iter()
            .filter_map(|t| body.corners(*t))
            .map(|q| {
                let a = chart.flat(q[0]);
                [a, chart.beside(chart.flat(q[1]), a), chart.beside(chart.flat(q[2]), a)]
            })
            .collect();
        let edges = |t: &usize| body.triangles.get(*t).into_iter().flat_map(|q| [[q[0], q[1]], [q[1], q[2]], [q[2], q[0]]]);
        let held: BTreeSet<[u32; 2]> = own.iter().flat_map(edges).collect();
        let rim = held.iter().filter(|e| !held.contains(&[e[1], e[0]])).copied().collect();
        Some(Face { body: k, convex: out.dot(from_axis) > 0.0, flat, rim })
    }

    /// Whether the face holds chart point `c` (in one of its triangles, a turn more or less).
    fn holds(&self, c: [f64; 2], turn: f64) -> bool {
        let cross = |o: [f64; 2], p: [f64; 2], q: [f64; 2]| (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0]);
        let within = |t: &[[f64; 2]; 3], c: [f64; 2]| {
            let d = [cross(t[0], t[1], c), cross(t[1], t[2], c), cross(t[2], t[0], c)];
            d.iter().all(|x| *x >= 0.0) || d.iter().all(|x| *x <= 0.0)
        };
        [-turn, 0.0, turn].iter().any(|shift| self.flat.iter().any(|t| within(t, [c[0] + shift, c[1]])))
    }

    /// Whether a triangle of this face overlaps one of `o`'s by more than `least` (m²).
    fn overlaps(&self, o: &Face, turn: f64, least: f64) -> bool {
        [-turn, 0.0, turn].iter().any(|shift| {
            let moved = |t: &[[f64; 2]; 3]| t.map(|p| [p[0] + shift, p[1]]);
            self.flat.iter().any(|a| o.flat.iter().any(|b| polygon_area(&overlap(*a, moved(b))) > least))
        })
    }
}

/// A rim edge on the chart: the pieces it is drawn as (two when it crosses the seam: up to the
/// seam's one side, on from its other side), each between two of the chart's points.
struct Edge {
    ends: [u32; 2],
    pieces: Vec<[u32; 2]>,
}

/// Draws rim edge `e` of `body`: one piece, or two with a point on either side of the seam at
/// the height it crosses at (noted in `seam`).
fn draw(chart: &Chart, body: &Body, e: [u32; 2], points: &mut Points, seam: &mut Vec<f64>) -> Option<(Edge, [u32; 2])> {
    let turn = chart.around();
    let a = chart.flat(*body.positions.get(e[0] as usize)?);
    let b = chart.beside(chart.flat(*body.positions.get(e[1] as usize)?), a);
    let ids = [points.id(a), points.id([b[0].rem_euclid(turn), b[1]])];
    let pieces = if (0.0..turn).contains(&b[0]) {
        vec![ids]
    } else {
        let (near, far) = if b[0] < 0.0 { (0.0, turn) } else { (turn, 0.0) };
        let z = a[1] + (near - a[0]) / (b[0] - a[0]) * (b[1] - a[1]);
        seam.push(z);
        vec![[ids[0], points.id([near, z])], [points.id([far, z]), ids[1]]]
    };
    Some((Edge { ends: e, pieces }, ids))
}

/// Splits the triangle of `body` beyond rim edge `ends` (the one holding it the other way
/// round) into a fan to the `chain` of vertices now along that edge.
fn fan(body: &mut Body, chain: &[u32]) {
    let (Some(&u), Some(&v)) = (chain.first(), chain.last()) else { return };
    let beyond = body.triangles.iter().position(|t| (0..3).any(|i| t[i] == v && t[(i + 1) % 3] == u));
    let Some(t) = beyond else { return };
    let Some(&apex) = body.triangles.get(t).and_then(|q| q.iter().find(|x| **x != u && **x != v)) else { return };
    let (label, on) = (body.beyond.get(t).copied(), body.on.get(t).copied().flatten());
    let Some(label) = label else { return };
    let mut pieces = chain.windows(2).map(|w| [w[1], w[0], apex]);
    if let (Some(first), Some(slot)) = (pieces.next(), body.triangles.get_mut(t)) {
        *slot = first;
    }
    for piece in pieces {
        body.triangles.push(piece);
        body.beyond.push(label);
        body.on.push(on);
    }
}

/// The faces on cylinder `cylinder` that mate: those of bodies beside each other facing one
/// another, or of a body and the one it lies in facing the same way, with triangles that
/// overlap; joined into the sets that touch.
fn mating(bodies: &[Body], cylinder: usize, chart: &Chart, nesting: &Nesting, least: f64) -> Vec<Vec<Face>> {
    let faces: Vec<Face> = (0..bodies.len()).filter_map(|k| Face::new(bodies, k, cylinder, chart)).collect();
    let mut sets = DisjointSets::new(faces.len());
    for (i, f) in faces.iter().enumerate() {
        for (j, g) in faces.iter().enumerate().skip(i + 1) {
            let facing = (nesting.within(f.body, g.body) && f.convex == g.convex) || (nesting.beside(f.body, g.body) && f.convex != g.convex);
            if facing && f.overlaps(g, chart.around(), least) {
                sets.union(i, j);
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<Face>> = BTreeMap::new();
    for (i, f) in faces.into_iter().enumerate() {
        groups.entry(sets.find(i)).or_default().push(f);
    }
    groups.into_values().filter(|g| g.len() > 1).collect()
}

/// Triangulates the mating `faces` on cylinder `cylinder` together and puts the triangles in
/// their bodies in place of the faces' ([`put`]), the triangles beyond the faces' rims split
/// to the rims' new points.
fn retriangulate(bodies: &mut [Body], faces: &[Face], (cylinder, chart, lines): (usize, &Chart, &[f64]), nesting: &Nesting, size: f64) -> Result<()> {
    let turn = chart.around();
    let mut points = Points::default();
    let mut seam: Vec<f64> = Vec::new();
    let mut vertex_of: BTreeMap<usize, BTreeMap<u32, u32>> = BTreeMap::new();
    let mut edges: Vec<Vec<Edge>> = Vec::with_capacity(faces.len());
    for face in faces {
        let Some(body) = bodies.get(face.body) else { return Err(fail("a face of no body")) };
        let mut list = Vec::with_capacity(face.rim.len());
        for e in &face.rim {
            let Some((edge, ids)) = draw(chart, body, *e, &mut points, &mut seam) else { continue };
            let of = vertex_of.entry(face.body).or_default();
            of.insert(ids[0], e[0]);
            of.insert(ids[1], e[1]);
            list.push(edge);
        }
        edges.push(list);
    }
    let pieces = |list: &[Edge]| -> Vec<[u32; 2]> { list.iter().flat_map(|e| e.pieces.iter().copied()).filter(|s| s[0] != s[1]).collect() };
    let drawn: Vec<Vec<[u32; 2]>> = edges.iter().map(|list| pieces(list)).collect();
    if drawn.iter().map(Vec::len).sum::<usize>() > super::MAX_SEGMENTS {
        return Err(fail(format!("faces in contact with more than {} rim edges: a larger element size", super::MAX_SEGMENTS)));
    }
    // Where two faces' rims cross: a point of both.
    let ends = |points: &Points, s: [u32; 2]| s.map(|v| points.xy.get(v as usize).copied().unwrap_or_default());
    for f in 0..drawn.len() {
        for g in f + 1..drawn.len() {
            for a in &drawn[f] {
                for b in &drawn[g] {
                    if let Some(c) = crossing(ends(&points, *a), ends(&points, *b)) {
                        points.id(c);
                    }
                }
            }
        }
    }
    // The seam's two sides where a face lies across it, cut at the same heights.
    seam.sort_by(f64::total_cmp);
    seam.dedup_by(|a, b| (*a - *b).abs() <= MERGE);
    let mut segments: BTreeSet<[u32; 2]> = BTreeSet::new();
    for w in seam.windows(2) {
        if faces.iter().any(|f| f.holds([0.0, 0.5 * (w[0] + w[1])], turn)) {
            for s in [0.0, turn] {
                segments.insert([points.id([s, w[0]]), points.id([s, w[1]])]);
            }
        }
    }
    // The straight lines the triangles' edges are to follow (a sector's cut planes), where a
    // face lies: from each of a line's crossings of the rims to the next.
    let height = points.xy.iter().fold([f64::INFINITY, f64::NEG_INFINITY], |r, p| [r[0].min(p[1]), r[1].max(p[1])]);
    for &s in lines.iter().filter(|s| **s > MERGE && **s < turn - MERGE) {
        let line = [[s, height[0] - size], [s, height[1] + size]];
        let mut at: Vec<f64> = drawn.iter().flatten().filter_map(|piece| crossing(line, ends(&points, *piece))).map(|c| c[1]).collect();
        at.sort_by(f64::total_cmp);
        at.dedup_by(|a, b| (*a - *b).abs() <= MERGE);
        for w in at.windows(2) {
            if faces.iter().any(|f| f.holds([s, 0.5 * (w[0] + w[1])], turn)) {
                segments.insert([points.id([s, w[0]]), points.id([s, w[1]])]);
            }
        }
    }
    segments.extend(drawn.iter().flatten().copied());
    // Each cut at the points on it.
    let cut: BTreeSet<[u32; 2]> = segments
        .iter()
        .flat_map(|s| {
            let chain: Vec<u32> = std::iter::once(s[0]).chain(on_segment(&points.xy, *s)).chain([s[1]]).collect();
            chain.windows(2).map(|w| [w[0].min(w[1]), w[0].max(w[1])]).collect::<Vec<_>>()
        })
        .collect();
    let holders = |c: [f64; 2]| -> Vec<(usize, bool)> { faces.iter().filter(|f| f.holds(c, turn)).map(|f| (f.body, f.convex)).collect() };
    let (out, triangles) = crate::planar::pslg_keep(&points.xy, &cut.into_iter().collect::<Vec<_>>(), size, &|c| !holders(c).is_empty())?;
    // The faces' own triangles go.
    for face in faces {
        if let Some(body) = bodies.get_mut(face.body) {
            let on = body.on.clone();
            body.retain(|t| on.get(t) != Some(&Some(cylinder)));
        }
    }
    for tri in &triangles {
        let q = tri.map(|v| out.get(v as usize).copied().unwrap_or_default());
        let c = [(q[0][0] + q[1][0] + q[2][0]) / 3.0, (q[0][1] + q[1][1] + q[2][1]) / 3.0];
        put(bodies, nesting, &mut vertex_of, Piece { corners: *tri, at: q.map(|p| chart.lift(p)), on: Some(cylinder) }, &holders(c));
    }
    // The rims' new points: the triangles beyond them split to them.
    for (face, list) in faces.iter().zip(&edges) {
        let (Some(body), Some(of)) = (bodies.get_mut(face.body), vertex_of.get_mut(&face.body)) else { continue };
        for edge in list {
            let mut chain: Vec<u32> = vec![edge.ends[0]];
            for piece in &edge.pieces {
                for p in on_segment(&out, *piece).into_iter().chain([piece[1]]) {
                    let at = chart.lift(out.get(p as usize).copied().unwrap_or_default());
                    chain.push(*of.entry(p).or_insert_with(|| {
                        body.positions.push(at);
                        (body.positions.len() - 1) as u32
                    }));
                }
            }
            if let Some(last) = chain.last_mut() {
                *last = edge.ends[1];
            }
            chain.dedup();
            if chain.len() > 2 {
                fan(body, &chain);
            }
        }
    }
    Ok(())
}

/// The half planes through a line along z (through `centre`, at the `angles` from +x) that a
/// sector cuts the model by: a contact along a cylinder about that line is triangulated with
/// edges along them, so each cut runs straight down the cylinder and the two are alike.
#[derive(Clone, Debug, Default, PartialEq)]
pub(in crate::solid) struct Meridians {
    pub(in crate::solid) centre: [f64; 2],
    pub(in crate::solid) angles: Vec<f64>,
}

impl Meridians {
    /// Where the half planes meet the chart of cylinder `c` (their arc lengths from its seam);
    /// none for a cylinder not about their line.
    fn on(&self, c: &Round, chart: &Chart, tol: f64) -> Vec<f64> {
        let about = c.axis.z.abs() > 1.0 - 1e-9 && (c.origin.x - self.centre[0]).hypot(c.origin.y - self.centre[1]) <= tol;
        let at =
            |angle: &f64| chart.flat(Point3::new(self.centre[0] + c.radius * angle.cos(), self.centre[1] + c.radius * angle.sin(), c.origin.z))[0];
        if about { self.angles.iter().map(at).collect() } else { Vec::new() }
    }
}

/// Makes the contacts along the `cylinders`: on each, the faces that mate are triangulated
/// again together, to the smaller of their bodies' `sizes` and a sixteenth of the half circle,
/// with edges along the `meridians`. The bodies changed.
pub(super) fn mate(
    bodies: &mut [Body],
    (cylinders, meridians): (&[Round], &Meridians),
    nesting: &Nesting,
    sizes: &[f64],
    tol: f64,
) -> Result<BTreeSet<usize>> {
    let mut touched = BTreeSet::new();
    for (c, cylinder) in cylinders.iter().enumerate() {
        let on_it = |b: &Body| b.on.contains(&Some(c));
        if bodies.iter().filter(|b| on_it(b)).count() < 2 || !(cylinder.radius.is_finite() && cylinder.radius > tol) {
            continue;
        }
        let corners = bodies.iter().flat_map(|b| {
            b.triangles
                .iter()
                .zip(&b.on)
                .filter(|(_, on)| **on == Some(c))
                .flat_map(|(t, _)| t.iter().filter_map(|v| b.positions.get(*v as usize).copied()))
        });
        let chart = Chart::new(cylinder, corners);
        let size_of = |k: usize| sizes.get(k).copied().unwrap_or(f64::INFINITY);
        let least = (0..bodies.len()).filter(|k| bodies.get(*k).is_some_and(on_it)).map(size_of).fold(f64::INFINITY, f64::min);
        for faces in mating(bodies, c, &chart, nesting, (1e-4 * least).powi(2)) {
            let size = faces.iter().map(|f| size_of(f.body)).fold(cylinder.radius * PI / 16.0, f64::min);
            if !(size.is_finite() && size > 0.0) {
                return Err(fail("an element size that is not finite and > 0"));
            }
            retriangulate(bodies, &faces, (c, &chart, &meridians.on(cylinder, &chart, tol)), nesting, size)?;
            touched.extend(faces.iter().map(|f| f.body));
        }
    }
    Ok(touched)
}
