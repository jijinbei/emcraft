//! Inclusions that touch face to face: a liquid on the floor of its vessel, a resonator on its
//! support, a substrate under a cover. Where two bodies' flat faces lie in one plane and overlap,
//! both are triangulated again there with the same triangles, so the bodies' regions share that
//! patch as an interface of their own, and what is left of each face stays its body's interface
//! with the region around.
//!
//! In three steps:
//! 1. **Faces**: each body's triangles joined into flat faces (neighbours across an edge in one
//!    plane); two faces of different bodies are in contact when their planes are one, they face
//!    each other and a triangle of one overlaps (with area) a triangle of the other (the
//!    triangles clipped against each other, as `narrow` does for gaps: Sutherland and Hodgman
//!    1974).
//! 2. **Triangulation**: the faces in contact through one plane are triangulated together in
//!    that plane — their outlines, cut where they cross, the constraints of one constrained
//!    Delaunay triangulation refined to the bodies' size ([`crate::planar::pslg_keep`]: Sloan
//!    1993, Ruppert 1995). A triangle inside a face of each side is the contact's, in both
//!    bodies; one inside a single face is that body's alone.
//! 3. **Conformity**: a point the triangulation put on a face's outline (or one body's corner on
//!    another's edge) splits the triangles beyond that edge too, in every body that has it, so
//!    each body's surface stays closed and the bodies agree along the contact's rim.
//!
//! A body inside another may lie against that one's wall from inside (the air over a liquid
//! reaching its vessel's rim): there the two faces look the same way, the inner body's triangles
//! part it from what lies outside the vessel, and the vessel's own face has a hole.
//!
//! So a surface is kept as patches, each between two spaces — an inclusion on its inner side and,
//! beyond it, another inclusion or the region around ([`Patch`]); a space's boundary is the
//! patches on either side of which it lies.
//!
//! Flat faces here; faces on one cylinder — a rod in its sleeve, a disc filling a tube — in
//! [`round`], on the cylinder unrolled. Bodies that touch along another curved surface, along an
//! edge or at a point are left as they are, and the mesher refuses surfaces that meet.

mod round;

use std::collections::{BTreeMap, BTreeSet, HashMap};

use emcraft_model::{Aabb, Point3, Vec3};

pub(super) use round::{Meridians, Round};

use super::narrow::{overlap, polygon_area};
use super::sector::{between, inside};
use super::tessellation::Tessellation;
use crate::csg::Surface;
use crate::planar::lattice::STEP;
use crate::shared::{DisjointSets, closed};
use crate::{MeshError, Result};

/// A patch of the inclusions' surfaces: its triangles, oriented out of inclusion `inner` into
/// what lies beyond them there — inclusion `outer`, or the solid's own region (`None`) — one tag
/// per triangle.
#[derive(Clone, Debug)]
pub(super) struct Patch {
    pub(super) inner: usize,
    pub(super) outer: Option<usize>,
    pub(super) faces: Tessellation,
}

/// What lies beyond a triangle of a body's surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Beyond {
    /// Another body, against this one from outside: it holds the triangle too, turned over.
    Shared(usize),
    /// A space around: the body this one lies in, or (against that one's wall from inside) what
    /// lies outside it; `None` is the solid's own region.
    Outside(Option<usize>),
}

/// Points nearer than this are one (m): a few steps of the planar mesher's lattice.
const MERGE: f64 = 4.0 * STEP;
/// The most outline segments of one contact (their crossings are found pair by pair).
const MAX_SEGMENTS: usize = 20_000;
/// The most passes of splitting triangles at points on their edges.
const MAX_CONFORM_PASSES: usize = 16;

fn fail(why: impl Into<String>) -> MeshError {
    MeshError::invalid("inclusions in contact", why)
}

/// A body's surface while its contacts are made: each triangle with what lies beyond it and
/// the cylinder it lies on, if any (its number among the bodies' cylinders).
struct Body {
    positions: Vec<Point3>,
    triangles: Vec<[u32; 3]>,
    beyond: Vec<Beyond>,
    on: Vec<Option<usize>>,
}

impl Body {
    fn corners(&self, t: usize) -> Option<[Point3; 3]> {
        crate::shared::corners(&self.positions, self.triangles.get(t)?)
    }

    fn bounds(&self) -> Option<Aabb> {
        Aabb::around(self.positions.iter().copied())
    }

    /// Keeps the triangles `keep` holds (by their number).
    fn retain(&mut self, keep: impl Fn(usize) -> bool) {
        let mut t = 0..self.triangles.len();
        self.triangles.retain(|_| t.next().is_some_and(&keep));
        let mut t = 0..self.beyond.len();
        self.beyond.retain(|_| t.next().is_some_and(&keep));
        let mut t = 0..self.on.len();
        self.on.retain(|_| t.next().is_some_and(&keep));
    }

    /// Drops the points no triangle uses and makes those at the same place one (two contacts
    /// each put theirs on the edge they share); a triangle left with a corner twice goes.
    fn compact(&mut self) {
        let mut at: HashMap<[u64; 3], u32> = HashMap::new();
        let mut new = vec![u32::MAX; self.positions.len()];
        let mut kept = Vec::new();
        for v in self.triangles.iter_mut().flatten() {
            let (Some(slot), Some(p)) = (new.get_mut(*v as usize), self.positions.get(*v as usize)) else { continue };
            if *slot == u32::MAX {
                *slot = *at.entry([p.x, p.y, p.z].map(f64::to_bits)).or_insert_with(|| {
                    kept.push(*p);
                    (kept.len() - 1) as u32
                });
            }
            *v = *slot;
        }
        self.positions = kept;
        let whole: Vec<bool> = self.triangles.iter().map(|t| t[0] != t[1] && t[1] != t[2] && t[2] != t[0]).collect();
        self.retain(|t| whole.get(t).copied().unwrap_or(false));
    }
}

/// A flat face of a body: its triangles, its unit normal (outwards) and its plane's offset
/// along it.
struct Face {
    body: usize,
    normal: Vec3,
    offset: f64,
    triangles: Vec<usize>,
    bounds: Aabb,
}

/// A triangle's unit normal and its plane's offset along it; `None` for one without area.
fn plane(q: [Point3; 3]) -> Option<(Vec3, f64)> {
    let n = (q[1] - q[0]).cross(q[2] - q[0]).normalized()?;
    Some((n, n.dot(q[0].to_vec())))
}

/// The flat faces of body `k`: triangles joined across the edges where both lie in one plane
/// (normals alike to 1e-9, offsets to `tol`). The triangles on a cylinder are not faces of
/// their own (`round` mates those).
fn faces(body: &Body, k: usize, tol: f64) -> Vec<Face> {
    let planes: Vec<Option<(Vec3, f64)>> = (0..body.triangles.len()).map(|t| body.corners(t).and_then(plane)).collect();
    let mut sets = DisjointSets::new(body.triangles.len());
    let mut across: BTreeMap<[u32; 2], usize> = BTreeMap::new();
    for (t, tri) in body.triangles.iter().enumerate() {
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            let key = [tri[i].min(tri[j]), tri[i].max(tri[j])];
            let Some(&other) = across.get(&key) else {
                across.insert(key, t);
                continue;
            };
            if let (Some(Some((n, d))), Some(Some((m, e)))) = (planes.get(t), planes.get(other))
                && body.on.get(t) == body.on.get(other)
                && n.dot(*m) > 1.0 - 1e-9
                && (d - e).abs() <= tol
            {
                sets.union(t, other);
            }
        }
    }
    let mut by_root: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for t in 0..body.triangles.len() {
        by_root.entry(sets.find(t)).or_default().push(t);
    }
    by_root
        .into_iter()
        .filter(|(root, _)| body.on.get(*root).copied().flatten().is_none())
        .filter_map(|(root, triangles)| {
            let (normal, offset) = (*planes.get(root)?)?;
            let bounds = Aabb::around(triangles.iter().filter_map(|t| body.corners(*t)).flatten())?;
            Some(Face { body: k, normal, offset, triangles, bounds })
        })
        .collect()
}

/// A plane's frame: a point of it and two unit vectors along it, `u × v` its normal. Along the
/// coordinate axes when the normal is one, so a face in an axis plane stays exactly in it.
#[derive(Clone, Copy)]
struct Frame {
    origin: Point3,
    u: Vec3,
    v: Vec3,
    normal: Vec3,
}

impl Frame {
    fn new(origin: Point3, normal: Vec3) -> Frame {
        let k = normal.dominant_axis();
        let (u, v) = if normal[k].abs() > 1.0 - 1e-12 {
            let (a, b) = (Vec3::unit((k + 1) % 3), Vec3::unit((k + 2) % 3));
            if normal[k] > 0.0 { (a, b) } else { (b, a) }
        } else {
            let u = Vec3::unit((k + 1) % 3).tangential(normal).normalized().unwrap_or_else(|| Vec3::unit((k + 1) % 3));
            (u, normal.cross(u))
        };
        Frame { origin, u, v, normal }
    }

    fn flat(&self, p: Point3) -> [f64; 2] {
        let d = p - self.origin;
        [d.dot(self.u), d.dot(self.v)]
    }

    fn lift(&self, q: [f64; 2]) -> Point3 {
        self.origin + self.u * q[0] + self.v * q[1]
    }
}

/// Whether faces `f` and `g` (of different bodies) are in contact: one plane, facing each
/// other — or, with one body `within` the other, the same way — a triangle of each overlapping
/// the other's by more than `least` (m²).
fn touching(bodies: &[Body], f: &Face, g: &Face, within: bool, tol: f64, least: f64) -> bool {
    let one_plane = if within {
        f.normal.dot(g.normal) > 1.0 - 1e-9 && (f.offset - g.offset).abs() <= tol
    } else {
        f.normal.dot(g.normal) < -1.0 + 1e-9 && (f.offset + g.offset).abs() <= tol
    };
    if !one_plane || !f.bounds.grown(tol).overlaps(g.bounds) {
        return false;
    }
    let (Some(a), Some(b)) = (bodies.get(f.body), bodies.get(g.body)) else { return false };
    let Some(origin) = f.triangles.first().and_then(|t| a.corners(*t)).map(|q| q[0]) else { return false };
    let frame = Frame::new(origin, f.normal);
    let flat = |body: &Body, face: &Face| -> Vec<[[f64; 2]; 3]> {
        face.triangles.iter().filter_map(|t| body.corners(*t)).map(|q| q.map(|p| frame.flat(p))).collect()
    };
    let (ta, tb) = (flat(a, f), flat(b, g));
    let reach = |t: &[[f64; 2]; 3]| {
        t.iter().fold(([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]), |(lo, hi), p| {
            ([lo[0].min(p[0]), lo[1].min(p[1])], [hi[0].max(p[0]), hi[1].max(p[1])])
        })
    };
    let boxes: Vec<([f64; 2], [f64; 2])> = tb.iter().map(reach).collect();
    ta.iter().any(|x| {
        let (lo, hi) = reach(x);
        tb.iter()
            .zip(&boxes)
            .any(|(y, (l, h))| lo[0] <= h[0] && l[0] <= hi[0] && lo[1] <= h[1] && l[1] <= hi[1] && polygon_area(&overlap(*x, *y)) > least)
    })
}

/// Points of a plane, those within [`MERGE`] of one already there being that one (found through
/// a hash on cells of that size, checking the neighbours).
#[derive(Default)]
struct Points {
    xy: Vec<[f64; 2]>,
    cells: HashMap<[i64; 2], Vec<u32>>,
}

impl Points {
    fn id(&mut self, p: [f64; 2]) -> u32 {
        let k = p.map(|v| (v / MERGE).floor() as i64);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for &i in self.cells.get(&[k[0] + dx, k[1] + dy]).map_or(&[][..], Vec::as_slice) {
                    if self.xy.get(i as usize).is_some_and(|q| (q[0] - p[0]).hypot(q[1] - p[1]) <= MERGE) {
                        return i;
                    }
                }
            }
        }
        self.xy.push(p);
        let i = (self.xy.len() - 1) as u32;
        self.cells.entry(k).or_default().push(i);
        i
    }
}

/// Where segments `a` and `b` cross, if they do (not parallel, the point on both).
fn crossing(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> Option<[f64; 2]> {
    let (r, d) = ([a[1][0] - a[0][0], a[1][1] - a[0][1]], [b[1][0] - b[0][0], b[1][1] - b[0][1]]);
    let den = r[0] * d[1] - r[1] * d[0];
    if den.abs() <= 1e-12 * r[0].hypot(r[1]) * d[0].hypot(d[1]) {
        return None;
    }
    let w = [b[0][0] - a[0][0], b[0][1] - a[0][1]];
    let (t, s) = ((w[0] * d[1] - w[1] * d[0]) / den, (w[0] * r[1] - w[1] * r[0]) / den);
    ((0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&s)).then(|| [a[0][0] + t * r[0], a[0][1] + t * r[1]])
}

/// The points strictly inside segment `s` (within [`MERGE`] of it), in order along it.
fn on_segment(points: &[[f64; 2]], s: [u32; 2]) -> Vec<u32> {
    let at = |v: u32| points.get(v as usize).copied().unwrap_or_default();
    between(points, at(s[0]), at(s[1]), MERGE).into_iter().map(|k| k as u32).collect()
}

/// The faces in contact through one plane, drawn in it: the points of their outlines (and of
/// the outlines' crossings) and, per face, its outline between them and each point's vertex in
/// the face's body.
struct Drawing {
    frame: Frame,
    points: Vec<[f64; 2]>,
    outlines: Vec<Vec<[u32; 2]>>,
    vertices: Vec<BTreeMap<u32, u32>>,
}

/// A face's outline: the edges only one of its triangles holds, as the body's vertices.
fn outline(body: &Body, face: &Face) -> Vec<[u32; 2]> {
    let edges = |t: &usize| body.triangles.get(*t).into_iter().flat_map(|q| [[q[0], q[1]], [q[1], q[2]], [q[2], q[0]]]);
    let held: BTreeSet<[u32; 2]> = face.triangles.iter().flat_map(edges).collect();
    held.iter().filter(|e| !held.contains(&[e[1], e[0]])).copied().collect()
}

fn draw(bodies: &[Body], faces: &[&Face]) -> Result<Drawing> {
    let first = faces.first().ok_or_else(|| fail("a contact without faces"))?;
    let origin = bodies.get(first.body).and_then(|b| first.triangles.first().and_then(|t| b.corners(*t))).map(|q| q[0]);
    let frame = Frame::new(origin.ok_or_else(|| fail("a face without triangles"))?, first.normal);
    let mut points = Points::default();
    let mut vertices = Vec::with_capacity(faces.len());
    let mut rims: Vec<Vec<[u32; 2]>> = Vec::with_capacity(faces.len());
    for face in faces {
        let Some(body) = bodies.get(face.body) else { return Err(fail("a face of no body")) };
        let mut of: BTreeMap<u32, u32> = BTreeMap::new();
        let mut rim = Vec::new();
        for e in outline(body, face) {
            let ends = e.map(|v| {
                let id = points.id(frame.flat(body.positions.get(v as usize).copied().unwrap_or(frame.origin)));
                of.insert(id, v);
                id
            });
            if ends[0] != ends[1] {
                rim.push(ends);
            }
        }
        vertices.push(of);
        rims.push(rim);
    }
    if rims.iter().map(Vec::len).sum::<usize>() > MAX_SEGMENTS {
        return Err(fail(format!("faces in contact with more than {MAX_SEGMENTS} outline edges: a larger element size")));
    }
    // Where two faces' outlines cross: a point of both.
    let ends = |points: &Points, s: [u32; 2]| s.map(|v| points.xy.get(v as usize).copied().unwrap_or_default());
    for f in 0..rims.len() {
        for g in f + 1..rims.len() {
            for a in &rims[f] {
                for b in &rims[g] {
                    if let Some(c) = crossing(ends(&points, *a), ends(&points, *b)) {
                        points.id(c);
                    }
                }
            }
        }
    }
    // Each outline cut at the points on it.
    let outlines = rims
        .iter()
        .map(|rim| {
            rim.iter()
                .flat_map(|s| {
                    let chain: Vec<u32> = std::iter::once(s[0]).chain(on_segment(&points.xy, *s)).chain([s[1]]).collect();
                    chain.windows(2).map(|w| [w[0], w[1]]).collect::<Vec<_>>()
                })
                .collect()
        })
        .collect();
    Ok(Drawing { frame, points: points.xy, outlines, vertices })
}

/// Which body lies in which: each one's parent and how many bodies lie around it.
struct Nesting<'a> {
    parent: &'a [Option<usize>],
    depth: Vec<usize>,
}

impl Nesting<'_> {
    fn new(parent: &[Option<usize>]) -> Nesting<'_> {
        let up = |i: usize| parent.get(i).copied().flatten();
        // Bounded by the number of bodies: a list of parents that went round would not end.
        let depth = (0..parent.len()).map(|i| std::iter::successors(up(i), |j| up(*j)).take(parent.len()).count()).collect();
        Nesting { parent, depth }
    }

    fn parent(&self, i: usize) -> Option<usize> {
        self.parent.get(i).copied().flatten()
    }

    /// Whether one of the two lies directly in the other.
    fn within(&self, i: usize, j: usize) -> bool {
        self.parent(i) == Some(j) || self.parent(j) == Some(i)
    }

    /// Whether the two lie in the same body (or both in the solid's own region).
    fn beside(&self, i: usize, j: usize) -> bool {
        self.parent.get(i) == self.parent.get(j)
    }

    /// Of bodies one inside the next, the innermost and the outermost.
    fn ends(&self, bodies: &[usize]) -> Option<(usize, usize)> {
        let depth = |i: &&usize| self.depth.get(**i).copied().unwrap_or(0);
        Some((*bodies.iter().max_by_key(depth)?, *bodies.iter().min_by_key(depth)?))
    }
}

/// A triangle of a sheet triangulated for several bodies: its corners (the sheet's points), where
/// they are, counter-clockwise seen from the sheet's upper side, and the cylinder it lies on.
struct Piece {
    corners: [u32; 3],
    at: [Point3; 3],
    on: Option<usize>,
}

/// Puts a piece into the bodies that keep it. `holders`: the bodies whose faces hold it, each
/// with whether its face looks to the sheet's upper side. Of those on one side (one inside the
/// next), the innermost keeps it: beyond it lies the innermost of the other side's (which holds
/// it too, turned over), or what is outside this side's outermost. `vertex_of`: per body, the
/// sheet's points that are its vertices already; the others are added, and all put exactly at
/// the piece's corners.
fn put(bodies: &mut [Body], nesting: &Nesting, vertex_of: &mut BTreeMap<usize, BTreeMap<u32, u32>>, piece: Piece, holders: &[(usize, bool)]) {
    let side = |up: bool| -> Vec<usize> { holders.iter().filter(|h| h.1 == up).map(|h| h.0).collect() };
    let ends = [true, false].map(|up| nesting.ends(&side(up)));
    for (k, up) in [true, false].into_iter().enumerate() {
        let Some((inner, outer)) = ends[k] else { continue };
        let other = match ends[1 - k] {
            Some((theirs, _)) => Beyond::Shared(theirs),
            None => Beyond::Outside(nesting.parent(outer)),
        };
        let Some(body) = bodies.get_mut(inner) else { continue };
        let of = vertex_of.entry(inner).or_default();
        let mut ids = [0u32; 3];
        for ((id, v), at) in ids.iter_mut().zip(piece.corners).zip(piece.at) {
            *id = *of.entry(v).or_insert_with(|| {
                body.positions.push(at);
                (body.positions.len() - 1) as u32
            });
            if let Some(p) = body.positions.get_mut(*id as usize) {
                *p = at;
            }
        }
        body.triangles.push(if up { ids } else { [ids[0], ids[2], ids[1]] });
        body.beyond.push(other);
        body.on.push(piece.on);
    }
}

/// Triangulates the faces in contact through one plane together and puts the triangles in
/// their bodies in place of the faces' ([`put`]).
fn retriangulate(bodies: &mut [Body], faces: &[&Face], nesting: &Nesting, sizes: &[f64]) -> Result<()> {
    let Drawing { frame, points, outlines, vertices } = draw(bodies, faces)?;
    let segments: Vec<[u32; 2]> = outlines.iter().flatten().map(|s| [s[0].min(s[1]), s[0].max(s[1])]).collect::<BTreeSet<_>>().into_iter().collect();
    let within = |c: [f64; 2]| -> Vec<usize> { (0..faces.len()).filter(|f| outlines.get(*f).is_some_and(|o| inside(&points, o, c))).collect() };
    // As fine at each place as the finest of the bodies whose faces lie there: a small body's
    // foot on a large plate is divided for the small one, the plate around it for itself.
    let size_of = |f: usize| faces.get(f).and_then(|face| sizes.get(face.body)).copied().unwrap_or(f64::INFINITY);
    let finest = (0..faces.len()).map(size_of).fold(f64::INFINITY, f64::min);
    if !(finest.is_finite() && finest > 0.0) {
        return Err(fail("an element size that is not finite and > 0"));
    }
    let size = |c: [f64; 2]| within(c).into_iter().map(size_of).fold(f64::INFINITY, f64::min);
    let (out, triangles) =
        crate::planar::pslg_graded(&points, &segments, finest, &|c| if size(c).is_finite() { size(c) } else { finest }, &|c| !within(c).is_empty())?;
    // The faces' own triangles go; the points keep their vertices, now exactly on the
    // triangulation's.
    let mut gone: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for face in faces {
        gone.entry(face.body).or_default().extend(&face.triangles);
    }
    let mut vertex_of: BTreeMap<usize, BTreeMap<u32, u32>> = BTreeMap::new();
    for (face, of) in faces.iter().zip(&vertices) {
        vertex_of.entry(face.body).or_default().extend(of);
    }
    for (k, dead) in &gone {
        if let Some(body) = bodies.get_mut(*k) {
            body.retain(|t| !dead.contains(&t));
        }
    }
    for tri in &triangles {
        let q = tri.map(|v| out.get(v as usize).copied().unwrap_or_default());
        let c = [(q[0][0] + q[1][0] + q[2][0]) / 3.0, (q[0][1] + q[1][1] + q[2][1]) / 3.0];
        let holders: Vec<(usize, bool)> =
            within(c).into_iter().filter_map(|f| faces.get(f)).map(|f| (f.body, f.normal.dot(frame.normal) > 0.0)).collect();
        put(bodies, nesting, &mut vertex_of, Piece { corners: *tri, at: q.map(|p| frame.lift(p)), on: None }, &holders);
    }
    Ok(())
}

/// The points of the bodies in contact on a hash of cells of `cell` (m), for finding those on
/// an edge; those within [`MERGE`] of each other are one point, the first met.
struct Cloud {
    cell: f64,
    cells: HashMap<[i64; 3], Vec<Point3>>,
}

impl Cloud {
    fn new(cell: f64) -> Cloud {
        Cloud { cell, cells: HashMap::new() }
    }

    /// The cloud's point at `p`: one within [`MERGE`] already there, else `p`, added.
    fn at(&mut self, p: Point3) -> Point3 {
        let k = [p.x, p.y, p.z].map(|v| (v / self.cell).floor() as i64);
        // A neighbouring cell holds it only when `p` is within the distance of the cell's side.
        let reach = |v: f64, k: i64| {
            let low = v - k as f64 * self.cell;
            (if low <= MERGE { -1 } else { 0 })..=(if self.cell - low <= MERGE { 1 } else { 0 })
        };
        for x in reach(p.x, k[0]) {
            for y in reach(p.y, k[1]) {
                for z in reach(p.z, k[2]) {
                    let list = self.cells.get(&[k[0] + x, k[1] + y, k[2] + z]).map_or(&[][..], Vec::as_slice);
                    if let Some(q) = list.iter().find(|q| (**q - p).norm() <= MERGE) {
                        return *q;
                    }
                }
            }
        }
        self.cells.entry(k).or_default().push(p);
        p
    }

    /// The points strictly inside the edge from `a` to `b` (within [`MERGE`] of it, off its
    /// ends), in order from `a`.
    fn on_edge(&self, a: Point3, b: Point3) -> Vec<Point3> {
        let d = b - a;
        let len = d.norm();
        if len <= 2.0 * MERGE {
            return Vec::new();
        }
        let key = |p: Point3, shift: f64| [p.x, p.y, p.z].map(|v| ((v + shift) / self.cell).floor() as i64);
        let (lo, hi) = (key(a.min(b), -MERGE), key(a.max(b), MERGE));
        let mut on: Vec<(f64, Point3)> = Vec::new();
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    for p in self.cells.get(&[x, y, z]).map_or(&[][..], Vec::as_slice) {
                        let t = (*p - a).dot(d) / (len * len);
                        let off = (*p - a).cross(d).norm() / len;
                        if off <= MERGE && t * len > MERGE && (1.0 - t) * len > MERGE {
                            on.push((t, *p));
                        }
                    }
                }
            }
        }
        on.sort_by(|x, y| x.0.total_cmp(&y.0));
        on.dedup_by(|x, y| (x.1 - y.1).norm() <= MERGE);
        on.into_iter().map(|(_, p)| p).collect()
    }
}

/// Splits each triangle of `body` with points of the cloud on an edge into a fan about its
/// corner opposite, until none has any (at most [`MAX_CONFORM_PASSES`] passes).
fn conform(body: &mut Body, cloud: &Cloud) -> Result<()> {
    let mut index: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    let key = |p: Point3| [p.x, p.y, p.z].map(|v| (v / MERGE).floor() as i64);
    for (i, p) in body.positions.iter().enumerate() {
        index.entry(key(*p)).or_default().push(i as u32);
    }
    for _ in 0..MAX_CONFORM_PASSES {
        let mut split = false;
        let mut triangles = Vec::with_capacity(body.triangles.len());
        let mut beyond = Vec::with_capacity(body.beyond.len());
        let mut round = Vec::with_capacity(body.on.len());
        for ((tri, label), lies_on) in body.triangles.iter().zip(&body.beyond).zip(&body.on) {
            let at = |v: u32| body.positions.get(v as usize).copied();
            let cut = (0..3).find_map(|i| {
                let (a, b, c) = (tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3]);
                let on = cloud.on_edge(at(a)?, at(b)?);
                (!on.is_empty()).then_some((a, b, c, on))
            });
            let Some((a, b, c, on)) = cut else {
                triangles.push(*tri);
                beyond.push(*label);
                round.push(*lies_on);
                continue;
            };
            split = true;
            let mut chain = vec![a];
            for p in on {
                let k = key(p);
                let near = (-1..=1).flat_map(|dx| (-1..=1).flat_map(move |dy| (-1..=1).map(move |dz| [k[0] + dx, k[1] + dy, k[2] + dz])));
                let found = near
                    .filter_map(|cell| index.get(&cell))
                    .flatten()
                    .copied()
                    .find(|i| body.positions.get(*i as usize).is_some_and(|q| (*q - p).norm() <= MERGE));
                let id = found.unwrap_or_else(|| {
                    body.positions.push(p);
                    let id = (body.positions.len() - 1) as u32;
                    index.entry(k).or_default().push(id);
                    id
                });
                if let Some(q) = body.positions.get_mut(id as usize) {
                    *q = p;
                }
                chain.push(id);
            }
            chain.push(b);
            chain.dedup();
            for w in chain.windows(2) {
                triangles.push([w[0], w[1], c]);
                beyond.push(*label);
                round.push(*lies_on);
            }
        }
        body.triangles = triangles;
        body.beyond = beyond;
        body.on = round;
        if !split {
            return Ok(());
        }
    }
    Err(fail("their surfaces could not be made to agree along the contact's rim; try another element size"))
}

/// The triangles of `body` marked `beyond`, as a tessellation of their own (one tag each).
fn part(body: &Body, beyond: Beyond) -> Tessellation {
    let mut new: BTreeMap<u32, u32> = BTreeMap::new();
    let mut surface = Surface::default();
    for (tri, _) in body.triangles.iter().zip(&body.beyond).filter(|(_, b)| **b == beyond) {
        let ids = tri.map(|v| {
            *new.entry(v).or_insert_with(|| {
                surface.positions.push(body.positions.get(v as usize).copied().unwrap_or(Point3::ORIGIN));
                (surface.positions.len() - 1) as u32
            })
        });
        surface.tags.push(surface.triangles.len() as u32);
        surface.triangles.push(ids);
    }
    Tessellation { surface, ..Tessellation::default() }
}

/// Whether the patches around inclusion `space` close: every edge held once each way, by the
/// patches it is the inner side of as they are and those it lies beyond turned over.
fn closes(patches: &[Patch], space: usize) -> bool {
    let mut ids: HashMap<[u64; 3], u32> = HashMap::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    for patch in patches.iter().filter(|p| p.inner == space || p.outer == Some(space)) {
        for tri in &patch.faces.surface.triangles {
            let q = tri.map(|v| {
                let p = patch.faces.surface.positions.get(v as usize).copied().unwrap_or(Point3::ORIGIN);
                let next = ids.len() as u32;
                *ids.entry([p.x, p.y, p.z].map(f64::to_bits)).or_insert(next)
            });
            triangles.push(if patch.inner == space { q } else { [q[0], q[2], q[1]] });
        }
    }
    closed(&triangles)
}

/// The bodies' faces that may be in contact, and which are (joined into the sets that touch
/// through one plane).
fn contacts(bodies: &[Body], nesting: &Nesting, sizes: &[f64], tol: f64) -> (Vec<Face>, DisjointSets) {
    let boxes: Vec<Option<Aabb>> = bodies.iter().map(Body::bounds).collect();
    let near = |i: usize, j: usize| match (boxes.get(i).copied().flatten(), boxes.get(j).copied().flatten()) {
        (Some(a), Some(b)) => i != j && (nesting.beside(i, j) || nesting.within(i, j)) && a.grown(tol).overlaps(b),
        _ => false,
    };
    // Only the faces that reach another body's box can touch it; one a body lies in is reached
    // by all of that body's faces, so there only those near the box's own faces count.
    let reaches = |f: &Face| {
        (0..bodies.len()).any(|j| {
            let Some(b) = boxes.get(j).copied().flatten() else { return false };
            let at_wall =
                !nesting.within(f.body, j) || (0..3).any(|d| (f.bounds.min[d] - b.min[d]).abs() <= tol || (f.bounds.max[d] - b.max[d]).abs() <= tol);
            near(f.body, j) && at_wall && f.bounds.grown(tol).overlaps(b)
        })
    };
    let all: Vec<Face> = (0..bodies.len())
        .filter(|&i| (0..bodies.len()).any(|j| near(i, j)))
        .flat_map(|k| bodies.get(k).map_or_default(|b| faces(b, k, tol)))
        .filter(|f| reaches(f))
        .collect();
    let size_of = |k: usize| sizes.get(k).copied().unwrap_or(f64::INFINITY);
    let mut sets = DisjointSets::new(all.len());
    for (f, a) in all.iter().enumerate() {
        for (g, b) in all.iter().enumerate().skip(f + 1) {
            let least = (1e-4 * size_of(a.body).min(size_of(b.body))).powi(2);
            if near(a.body, b.body) && touching(bodies, a, b, nesting.within(a.body, b.body), tol, least) {
                sets.union(f, g);
            }
        }
    }
    (all, sets)
}

/// Makes the contacts between the inclusions' `surfaces` (closed, each oriented outwards):
/// where flat faces of two inclusions touch — two inside the same body from outside each other,
/// or one against the wall of the body it lies in (`parent`) from inside — they are
/// triangulated again together to the smaller of the bodies' `sizes` (m). The surfaces come
/// back as patches: first each inclusion's towards the body it lies in (what it keeps of its
/// surface; all of it without a contact), in the inclusions' order, then the contacts'. `tol`
/// (m): how far apart two faces' planes may be and still be one. `round`: per surface, the
/// cylinder each triangle lies on, and the half planes a sector will cut the bodies by.
pub(super) fn imprint(
    surfaces: Vec<Tessellation>,
    (round, meridians): (&[Vec<Option<Round>>], &Meridians),
    parent: &[Option<usize>],
    sizes: &[f64],
    tol: f64,
) -> Result<Vec<Patch>> {
    let nesting = Nesting::new(parent);
    let mut cylinders: Vec<Round> = Vec::new();
    let mut bodies: Vec<Body> = surfaces
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let n = t.surface.triangles.len();
            let mut on: Vec<Option<usize>> =
                round.get(i).map_or_default(|r| r.iter().map(|c| c.map(|c| round::number(&mut cylinders, c, tol))).collect());
            on.resize(n, None);
            Body {
                positions: t.surface.positions.clone(),
                triangles: t.surface.triangles.clone(),
                beyond: vec![Beyond::Outside(nesting.parent(i)); n],
                on,
            }
        })
        .collect();
    // The curved contacts first: they put points on the rims of the flat faces beside them.
    let mut touched = round::mate(&mut bodies, (&cylinders, meridians), &nesting, sizes, tol)?;
    let (all, mut sets) = contacts(&bodies, &nesting, sizes, tol);
    let size_of = |k: usize| sizes.get(k).copied().unwrap_or(f64::INFINITY);
    let mut clusters: BTreeMap<usize, Vec<&Face>> = BTreeMap::new();
    for (f, face) in all.iter().enumerate() {
        clusters.entry(sets.find(f)).or_default().push(face);
    }
    // A body's triangles are numbered anew as each contact is made: a face still to come is
    // found again by what was taken out before it.
    let mut done: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for cluster in clusters.values().filter(|c| c.len() > 1) {
        let shifted: Vec<Face> = cluster
            .iter()
            .map(|f| {
                let before = done.get(&f.body);
                let moved = |t: usize| t - before.map_or(0, |d| d.range(..t).count());
                Face {
                    body: f.body,
                    normal: f.normal,
                    offset: f.offset,
                    triangles: f.triangles.iter().map(|t| moved(*t)).collect(),
                    bounds: f.bounds,
                }
            })
            .collect();
        retriangulate(&mut bodies, &shifted.iter().collect::<Vec<_>>(), &nesting, sizes)?;
        for f in cluster {
            done.entry(f.body).or_default().extend(&f.triangles);
            touched.insert(f.body);
        }
    }
    let whole = |i: usize, t: Tessellation| Patch { inner: i, outer: nesting.parent(i), faces: t };
    if touched.is_empty() {
        return Ok(surfaces.into_iter().enumerate().map(|(i, t)| whole(i, t)).collect());
    }
    let least = touched.iter().map(|k| size_of(*k)).fold(f64::INFINITY, f64::min);
    // A point two bodies hold is one point: each contact put its own on its plane's lattice.
    let mut cloud = Cloud::new(least);
    for &k in &touched {
        let Some(body) = bodies.get_mut(k) else { continue };
        // Without the points of the triangles taken out: they are no longer the surface's.
        body.compact();
        for p in &mut body.positions {
            *p = cloud.at(*p);
        }
        body.compact();
    }
    for &k in &touched {
        if let Some(body) = bodies.get_mut(k) {
            conform(body, &cloud)?;
        }
    }
    let mut patches: Vec<Patch> = surfaces
        .into_iter()
        .enumerate()
        .map(|(i, t)| match bodies.get(i).filter(|_| touched.contains(&i)) {
            Some(body) => whole(i, part(body, Beyond::Outside(nesting.parent(i)))),
            None => whole(i, t),
        })
        .collect();
    for &a in &touched {
        let Some(body) = bodies.get(a) else { continue };
        let kinds: BTreeSet<Beyond> = body.beyond.iter().copied().filter(|b| *b != Beyond::Outside(nesting.parent(a))).collect();
        for kind in kinds {
            let faces = part(body, kind);
            match kind {
                Beyond::Outside(outer) => patches.push(Patch { inner: a, outer, faces }),
                // The two hold the contact: taken from the first.
                Beyond::Shared(b) if a < b => {
                    let theirs = bodies.get(b).map_or(0, |o| o.beyond.iter().filter(|x| **x == Beyond::Shared(a)).count());
                    if theirs != faces.surface.triangles.len() {
                        return Err(fail(format!("inclusions {} and {} do not agree on their contact; try another element size", a + 1, b + 1)));
                    }
                    patches.push(Patch { inner: a, outer: Some(b), faces });
                }
                Beyond::Shared(_) => {}
            }
        }
    }
    if let Some(k) = touched.iter().find(|k| !closes(&patches, **k)) {
        return Err(fail(format!("inclusion {}'s surface did not close again around its contacts; try another element size", k + 1)));
    }
    Ok(patches)
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use emcraft_model::{Metres, Shape};

    use super::*;
    use crate::solid::tessellation::{cylinders, placed, tessellate};

    fn area(t: &Tessellation) -> f64 {
        let corners = |tri: &[u32; 3]| crate::shared::corners(&t.surface.positions, tri);
        t.surface.triangles.iter().filter_map(corners).map(|q| 0.5 * (q[1] - q[0]).cross(q[2] - q[0]).norm()).sum()
    }

    /// A disc that fills the tube it lies in (a cylinder of the outer one's radius inside it,
    /// divided more finely): the disc's side is a patch between it and what is outside the tube,
    /// its caps stay between it and the tube, and the tube keeps the rest of its own surface.
    /// Evidence: the patches' areas against 2πrh and πr² (flat facets in the circle, so a little
    /// under), and both bodies' patches closing (the imprint checks it).
    #[test]
    fn a_disc_in_a_tube_parts_the_tube_s_side_with_it() {
        let (r, tube, disc) = (0.005, 0.010, 0.004);
        let body = |length: f64, z: f64, size: f64| {
            let t = tessellate(&Shape::Cylinder { radius: Metres::new(r).unwrap(), length: Metres::new(length).unwrap() }, size);
            let at = Point3::new(0.0, 0.0, z);
            let round = cylinders(&t, at);
            (placed(t, at), round)
        };
        let ((outer, on_outer), (inner, on_inner)) = (body(tube, 0.0, 0.002), body(disc, 0.003, 0.001));
        let patches = imprint(vec![outer, inner], (&[on_outer, on_inner], &Meridians::default()), &[None, Some(0)], &[0.002, 0.001], 1e-11).unwrap();
        let of = |inner: usize, outer: Option<usize>| -> f64 {
            patches.iter().filter(|p| p.inner == inner && p.outer == outer).map(|p| area(&p.faces)).sum()
        };
        let (circle, band) = (PI * r * r, 2.0 * PI * r);
        let near = |got: f64, exact: f64| got < exact && got > 0.99 * exact;
        assert!(near(of(1, None), band * disc), "{}", of(1, None));
        assert!(near(of(1, Some(0)), 2.0 * circle), "{}", of(1, Some(0)));
        assert!(near(of(0, None), 2.0 * circle + band * (tube - disc)), "{}", of(0, None));
        assert_eq!(patches.len(), 3);
    }
}
