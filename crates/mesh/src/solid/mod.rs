//! Unstructured tetrahedral meshes of primitive bodies (box, cylinder, sphere) by the ported
//! mesher ([`emcraft_tetmesh`]): each body is tessellated with a tag per face and that face's
//! exact surface, the tessellation tetrahedralized with its boundary recovered, then refined,
//! smoothed and improved to the requested size, the boundary vertices finally put on the exact
//! surfaces (so a cylinder's or a sphere's wall is approximated by flat triangles whose corners
//! lie on it).
//!
//! The bodies' tessellations are in `tessellation`, the crossing into the mesher in `mesher`, the
//! bending of edges onto curved surfaces in `curved`; here the regions are meshed until their
//! interfaces agree and joined.

mod combine;
mod contact;
mod curved;
mod grid;
mod mesher;
mod narrow;
mod sector;
mod stop;
mod tessellation;

use emcraft_model::{
    Aabb, Axis, Boundary, BoundaryFace, BoxSize, Buckets, Combine, Feature, Mesh, Metres, Model, Point3, Shape, SphericalLayer, Vec3, positive,
};
use emcraft_tetmesh::size::{LocalSize, Region};

use crate::GEOMETRY_TOLERANCE;
use crate::{MeshError, Result, too_large};
use combine::combined;
use curved::bend;
use mesher::{RegionFace, RegionMesh, mesher_point, region};
use tessellation::{
    Tessellation, append, body_planes, extract, grid_axes, interface_size, joined, merged_planes, placed, same_triangles, tessellate,
};

/// The model's solid meshed with tetrahedra of about `size`: its end faces (box z0/z1, the
/// cylinder's caps) tagged as ports 0 and 1, the rest as conducting walls. Inclusions make holes
/// in it (conductors) or regions of their own (dielectrics), each meshed on its own with the
/// shared interface's triangles kept, the regions then joined node for node.
pub fn solid(model: &Model, size: Metres) -> Result<Mesh> {
    solid_with(model, &SolidOptions::new(size))
}

/// How a solid is meshed: [`solid_with`].
#[derive(Clone, Debug, PartialEq)]
pub struct SolidOptions {
    /// The element sizes: each material region meshed at its own (the solid's own material
    /// outside the inclusions, each dielectric inclusion at its material's), the interfaces at
    /// the smaller of their two sides'.
    pub sizes: Sizes,
    /// Whether the edges on curved surfaces are bent onto them (`curved`) or left straight.
    pub curved: bool,
    /// Local refinements: the size field is at most each one's size in its ball, graded away
    /// from it.
    pub refine: Vec<Refinement>,
}

impl SolidOptions {
    /// One size everywhere, edges bent onto curved surfaces, no local refinement.
    pub fn new(size: Metres) -> SolidOptions {
        SolidOptions { sizes: Sizes { default: size, per_material: Vec::new() }, curved: true, refine: Vec::new() }
    }
}

/// Element sizes: a default, and one per material (by index into the model's materials) where
/// given — a fixed number of elements per wavelength in each material, say.
#[derive(Clone, Debug, PartialEq)]
pub struct Sizes {
    pub default: Metres,
    pub per_material: Vec<Option<Metres>>,
}

impl Sizes {
    pub fn of(&self, material: usize) -> Metres {
        self.per_material.get(material).copied().flatten().unwrap_or(self.default)
    }
}

/// The tetrahedra a body of `volume` meshed at `size` has, about.
fn tets_in(volume: f64, size: f64) -> f64 {
    volume / (size.powi(3) / 6.0) * 1.5
}

/// The tetrahedra local refinements add, about: the body's bounding box in voxels (at most 64
/// a side), each at the smallest size of the balls that reach it, counted against the uniform
/// `size`.
fn refined_estimate(points: &[Point3], size: f64, refine: &[Refinement]) -> f64 {
    let Some(Aabb { min: lo, max: hi }) = Aabb::around(points.iter().copied()) else { return 0.0 };
    let ext = hi - lo;
    let cell = (ext.x.max(ext.y).max(ext.z) / 64.0).max(f64::MIN_POSITIVE);
    let voxels = Voxels { lo, step: Vec3::new(cell, cell, cell), dims: [ext.x, ext.y, ext.z].map(|e| ((e / cell).ceil() as usize).clamp(1, 64)) };
    let dims = voxels.dims;
    let mut h = vec![size; dims[0] * dims[1] * dims[2]];
    for r in refine {
        let a = [0, 1, 2].map(|d| voxels.index(r.center[d] - r.radius, d));
        let b = [0, 1, 2].map(|d| voxels.index(r.center[d] + r.radius, d));
        for k in a[2]..=b[2] {
            for j in a[1]..=b[1] {
                for i in a[0]..=b[0] {
                    if let Some(v) = h.get_mut((k * dims[1] + j) * dims[0] + i) {
                        *v = v.min(r.size);
                    }
                }
            }
        }
    }
    let voxel = cell.powi(3);
    h.iter().map(|v| tets_in(voxel, *v) - tets_in(voxel, size)).sum::<f64>().max(0.0)
}

/// The most local refinements one mesh takes.
pub const MAX_REFINEMENTS: usize = 200_000;

/// A local refinement: elements of about `size` within `radius` of `center`, the size growing
/// away from it at the mesher's grading rate (adaptive refinement marks the tetrahedra of a
/// previous mesh this way, [`crate::adapt`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Refinement {
    pub center: Point3,
    pub radius: f64,
    pub size: f64,
}

/// The refinements as the mesher's local sizes (balls), checked.
fn local_sizes(refine: &[Refinement]) -> Result<Vec<LocalSize>> {
    if refine.len() > MAX_REFINEMENTS {
        return Err(MeshError::Invalid { what: "refinement".into(), why: format!("{} local sizes; the limit is {MAX_REFINEMENTS}", refine.len()) });
    }
    refine
        .iter()
        .map(|r| {
            if !r.center.is_finite() {
                return Err(MeshError::Invalid { what: "refinement".into(), why: "a centre is not finite".into() });
            }
            positive("refinement size", r.size)?;
            positive("refinement radius", r.radius)?;
            Ok(LocalSize { region: Region::Sphere { center: mesher_point(r.center), radius: r.radius }, size: r.size })
        })
        .collect()
}

/// The element size of each inclusion: its material's for a dielectric, the solid's `size`
/// for a conductor (a hole).
fn inclusion_sizes(model: &Model, sizes: &Sizes, size: f64) -> Result<Vec<f64>> {
    model
        .inclusions()
        .iter()
        .map(|inc| match &inc.material {
            Some(name) => Ok(sizes.of(model.material(name)?.0).get()),
            None => Ok(size),
        })
        .collect()
}

/// The model's spherical layer, if it has one, checked: no features with it, and a positive
/// inner radius.
fn spherical_layer(model: &Model) -> Result<Option<SphericalLayer>> {
    let layer = model.spherical_layer();
    if let Some(SphericalLayer { inner, .. }) = layer {
        if !model.features().is_empty() {
            return Err(MeshError::Invalid { what: "spherical layer".into(), why: "not supported yet: features with a spherical layer".into() });
        }
        if !(inner > 0.0 && inner.is_finite()) {
            return Err(MeshError::Invalid { what: "spherical layer".into(), why: "its inner radius must be positive".into() });
        }
    }
    Ok(layer)
}

/// The region's surface before it is meshed: the solid's shape tessellated at about `size` (m)
/// with its features combined as the mesher combines them (faces flush with others moved off,
/// creases collapsed) — the mesh's boundary to come, for drawing a model before meshing (an
/// imported part's metal less the air cut out of it). Symmetry planes and sectors are not cut.
pub fn region_surface(model: &Model, size: Metres) -> Result<crate::csg::Surface> {
    let s = model.domain.solid.as_ref().ok_or_else(|| MeshError::Invalid { what: "model".into(), why: "has no solid".into() })?;
    let size = size.get();
    let mut main = tessellate(&s.shape, size);
    let features = joined(model.features(), &s.shape, size)?;
    if !features.is_empty() {
        main = combine_all(main, &features, &s.shape, size)?;
    }
    Ok(main.surface)
}

/// [`solid`] with its [`SolidOptions`]: sizes per material, curved edges or straight, local
/// refinements.
pub fn solid_with(model: &Model, options: &SolidOptions) -> Result<Mesh> {
    // The half a symmetry plane keeps holds the inclusions below it; those wholly beyond it are
    // their images' (the model's check refuses one across it).
    let mirrored;
    let model = match beyond_planes(model) {
        gone if gone.is_empty() => model,
        gone => {
            let mut m = model.clone();
            let mut bodies = m.bodies_mut();
            for k in gone.into_iter().rev() {
                bodies.remove(k);
            }
            drop(bodies);
            mirrored = m;
            &mirrored
        }
    };
    // A sector keeps the inclusions inside it and those across its cut faces (cut with it);
    // those outside it are their images'.
    let kept;
    let mut across: Vec<usize> = Vec::new();
    let model = match &model.domain.rotation {
        Some(rot) if !model.inclusions().is_empty() => {
            let mut m = model.clone();
            let sides: Vec<Option<bool>> = m.inclusions().iter().map(|inc| rot.sector_side(&inc.shape, inc.at)).collect();
            // The inclusions wholly on the dropped side go (their bodies among the model's).
            let gone: Vec<usize> = m.mesh_view().inclusion_body.iter().zip(&sides).filter(|(_, s)| **s == Some(false)).map(|(k, _)| *k).collect();
            let mut bodies = m.bodies_mut();
            for k in gone.into_iter().rev() {
                bodies.remove(k);
            }
            drop(bodies);
            across = sides.iter().filter(|s| **s != Some(false)).enumerate().filter(|(_, s)| s.is_none()).map(|(k, _)| k).collect();
            kept = m;
            &kept
        }
        _ => model,
    };
    let SolidOptions { sizes, curved, refine } = options;
    let curved = *curved;
    let mut locals = local_sizes(refine)?;
    let s = model.domain.solid.as_ref().ok_or_else(|| MeshError::Invalid { what: "model".into(), why: "has no solid".into() })?;
    let (material, _) = model.material(&s.material)?;
    // The half a symmetry plane keeps: the shape itself shortened where it can be, else cut.
    let (shape, cuts) = halved(model, &s.shape)?;
    let shape = &shape;
    let size = sizes.of(material).get();
    let inner_size = inclusion_sizes(model, sizes, size)?;
    let estimate = model.inclusions().iter().zip(&inner_size).fold(tets_in(shape.volume(), size), |e, (inc, h)| e + tets_in(inc.shape.volume(), *h));
    too_large(estimate)?;
    // A spherical layer (model.open, shape sphere): the solid's region is the ball inside the
    // layer's inner sphere, its surface an interface (tagged per triangle and frozen, like an
    // inclusion's) with the layer's shell, meshed as a region of its own between it and the
    // solid's sphere — so the shell's tetrahedra are a layer of their own.
    let layer = spherical_layer(model)?;
    if model.domain.rotation.is_some() && (!model.domain.symmetry.is_empty() || layer.is_some()) {
        return Err(MeshError::invalid("rotational symmetry", "not supported yet with symmetry planes or a spherical layer"));
    }
    let mut main = match layer {
        Some(SphericalLayer { inner, .. }) => {
            let ball = Shape::Sphere { radius: Metres::new(inner)? };
            placed(tessellate(&ball, interface_size(&ball, size, false)), Point3::ORIGIN)
        }
        None => tessellate(shape, size),
    };
    if !refine.is_empty() {
        too_large(estimate + refined_estimate(&main.surface.positions, size, refine))?;
    }
    let mut features = joined(model.features(), shape, size)?;
    features.extend(cuts);
    if !features.is_empty() {
        main = combine_all(main, &features, shape, size)?;
    }
    // The sector a rotation keeps: cut by a wedge last, its two cut faces then matched.
    let mut frozen = Vec::new();
    let mut cuts: Vec<(usize, sector::Cut)> = Vec::new();
    if let (Some(rot), Some(b)) = (&model.domain.rotation, model.bounds()) {
        let bodies: Vec<Tessellation> = across
            .iter()
            .map(|&i| {
                let inc = &model.inclusions()[i];
                let h = inner_size.get(i).copied().unwrap_or(size);
                placed(tessellate(&inc.shape, interface_size(&inc.shape, size.min(h), inc.material.is_none())), inc.at)
            })
            .collect();
        let sector::Closed { tessellation: matched, base, cuts: c } = sector::sector(main, rot, size, (b.max - b.min).norm(), &bodies)?;
        main = matched;
        frozen.push(base);
        cuts = across.iter().copied().zip(c).collect();
    }
    // Gaps narrower than a tenth of the size filled with elements as small as they are.
    let (gaps, gap_tets) = narrow::gap_sizes(&main, size, combine::gaps(size).flush)?;
    if !gaps.is_empty() {
        too_large(estimate + gap_tets)?;
        locals.extend(gaps);
    }
    let body = Body { model, shape, material: material as u32, size, inner_size, locals, shell: layer.is_some(), frozen, cuts };
    let regions = body.regions(main)?;
    let mut mesh = body.join(&regions)?;
    tag_symmetry_planes(&mut mesh, model, size);
    if curved {
        bend(&mut mesh, model, shape, size);
    }
    crate::check_tetrahedra(&mesh)?;
    mesh.buckets = Some(buckets(&mesh));
    Ok(mesh)
}

/// The solid's shape for the half its symmetry planes keep (the side below each), and the cuts
/// left to make by intersection. A box, a cylinder along its axis or an extrusion (every vertex
/// on one of its two ends, a prism) is shortened to the plane, clean as it was; across any other
/// direction, or where a feature added to the solid crosses the plane, the solid is intersected
/// with a box stopping at it (a Boolean whose cut leaves slivers by the faces' grid lines, and so
/// more elements).
fn halved(model: &Model, shape: &Shape) -> Result<(Shape, Vec<Feature>)> {
    let Some(bounds) = model.bounds() else { return Ok((shape.clone(), Vec::new())) };
    let around = bounds.grown(0.1 * (bounds.max - bounds.min).norm());
    let tol = GEOMETRY_TOLERANCE * (bounds.max - bounds.min).norm();
    let mut shape = shape.clone();
    let mut cuts = Vec::new();
    for plane in &model.domain.symmetry {
        let d = plane.axis.index();
        let crossed =
            model.features().iter().any(|f| f.combine == Combine::Add && f.bounds().min[d] < plane.at - tol && f.bounds().max[d] > plane.at + tol);
        match shortened(&shape, d, plane.at, tol) {
            Some(short) if !crossed => shape = short,
            _ => {
                let kept = Aabb::new(around.min, around.max.with(d, plane.at));
                let size = BoxSize::new(kept.max - kept.min)?;
                cuts.push(Feature { combine: Combine::Intersect, shape: Shape::Box { size }, at: kept.min, axis: Axis::Z });
            }
        }
    }
    Ok((shape, cuts))
}

/// The bodies of the inclusions wholly beyond a symmetry plane (above it), ascending.
fn beyond_planes(model: &Model) -> Vec<usize> {
    let Some(bounds) = model.bounds() else { return Vec::new() };
    let tol = GEOMETRY_TOLERANCE * (bounds.max - bounds.min).norm();
    let view = model.mesh_view();
    let mut gone: Vec<usize> = view
        .inclusions
        .iter()
        .zip(&view.inclusion_body)
        .filter(|(inc, _)| model.domain.symmetry.iter().any(|p| inc.bounds().min[p.axis.index()] >= p.at - tol))
        .map(|(_, k)| *k)
        .collect();
    gone.sort_unstable();
    gone
}

/// `shape` stopped at `at` across axis `d`, if it is a box, a cylinder along z cut across z, or
/// an imported extrusion along `d` ([`stopped_surface`]).
fn shortened(shape: &Shape, d: usize, at: f64, tol: f64) -> Option<Shape> {
    match shape {
        Shape::Box { size } => {
            let size = BoxSize::new(size.with(d, at)).ok()?;
            Some(Shape::Box { size })
        }
        Shape::Cylinder { radius, .. } if d == 2 => Some(Shape::Cylinder { radius: *radius, length: Metres::new(at).ok()? }),
        Shape::Surface { .. } => stop::stopped_surface(shape, stop::Cut { d, at, tol }),
        _ => None,
    }
}

/// The boundary faces the symmetry planes cut (every corner on a plane, to a billionth of the
/// size) tagged as the plane's wall.
fn tag_symmetry_planes(mesh: &mut Mesh, model: &Model, size: f64) {
    let tol = GEOMETRY_TOLERANCE * size.max(model.bounds().map_or(0.0, |b| (b.max - b.min).norm()));
    for plane in &model.domain.symmetry {
        let d = plane.axis.index();
        let on = |n: &u32| mesh.nodes.get(*n as usize).is_some_and(|q| (q[d] - plane.at).abs() <= tol);
        let cut: Vec<usize> = mesh.boundary.iter().enumerate().filter(|(_, f)| f.nodes.iter().all(on)).map(|(i, _)| i).collect();
        for i in cut {
            if let Some(f) = mesh.boundary.get_mut(i) {
                f.tag = Boundary::Symmetry(plane.kind);
            }
        }
    }
}

/// The solid's tessellation combined with each feature in turn, every body's grid lines kept
/// off the other bodies' planes first (`grid`).
fn combine_all(mut main: Tessellation, features: &[emcraft_model::Feature], solid: &Shape, size: f64) -> Result<Tessellation> {
    let (own, planes) = body_planes(features, solid, size);
    grid::keep_off(&mut main.surface.positions, grid_axes(solid, emcraft_model::Axis::Z), &merged_planes(&planes), grid::NEAR * size);
    for (k, f) in features.iter().enumerate() {
        let others = merged_planes(std::iter::once(&own).chain(planes.iter().enumerate().filter(|(j, _)| *j != k).map(|(_, p)| p)));
        main = combined(main, f, k, solid, size, &others)?;
    }
    Ok(main)
}

/// What a solid is meshed from: the model (its inclusions), the solid's shape, material and
/// element size, each inclusion's size, the local sizes, and whether a spherical layer's shell
/// is a region of its own.
struct Body<'a> {
    model: &'a Model,
    shape: &'a Shape,
    material: u32,
    size: f64,
    inner_size: Vec<f64>,
    locals: Vec<LocalSize>,
    shell: bool,
    /// A sector's cut faces: the first of their tags (theirs are the solid's last; frozen).
    frozen: Vec<u32>,
    /// The inclusions across a sector's cut faces (by index): their surfaces in the sector and caps.
    cuts: Vec<(usize, sector::Cut)>,
}

/// A range of face tags: `first..first + count`.
#[derive(Clone, Copy, Debug)]
struct Tags {
    first: u32,
    count: u32,
}

impl Tags {
    fn contains(self, tag: u32) -> bool {
        (self.first..self.first + self.count).contains(&tag)
    }
}

/// The regions of a solid, meshed so that they agree on their interfaces: the outer region (the
/// solid around its inclusions, with a spherical layer the ball inside it), the layer's shell
/// when there is one, then each dielectric inclusion, each with its material.
struct Regions {
    meshes: Vec<(RegionMesh, u32)>,
    /// The solid's faces that are ports.
    start_face: Option<u32>,
    end_face: Option<u32>,
    /// The first tag of the inclusions' interfaces in the outer region (after the solid's own
    /// faces, with a spherical layer the inner sphere's triangles).
    first_tag: u32,
    /// Per region, the inclusions beyond its faces, each with its interface's tags there: its
    /// holes (the outer region's are the inclusions inside no other, a dielectric's those
    /// directly inside it) and, for a dielectric, the inclusions it touches.
    holes: Vec<Vec<(usize, Tags)>>,
    /// Per region, the first tag of its caps on a sector's cut faces (an inclusion across them):
    /// walls tied by the solver, not an interface.
    caps_from: Vec<Option<u32>>,
}

/// `seen` in place of `current` when its triangles differ (corners within 1e-9 of the size);
/// whether it did.
fn adopt(current: &mut Tessellation, seen: Tessellation, size: f64) -> bool {
    let differs = !same_triangles(&seen, current, size);
    if differs {
        *current = seen;
    }
    differs
}

/// The holes' interfaces as region `mesh` kept them, in place of those that differ; whether any did.
fn adopt_holes(interfaces: &mut [Tessellation], mesh: &RegionMesh, holes: &[(usize, Tags)], size: f64) -> bool {
    holes.iter().fold(false, |changed, (i, tags)| match interfaces.get_mut(*i) {
        Some(t) => adopt(t, extract(mesh, tags.first, tags.count, false), size) | changed,
        None => changed,
    })
}

/// Appends the interfaces `of` to `t` as holes, their tags from `next` on: each one's tags.
fn pierce(t: &mut Tessellation, interfaces: &[Tessellation], of: &[usize], mut next: u32) -> Vec<(usize, Tags)> {
    of.iter()
        .filter_map(|&i| {
            let hole = interfaces.get(i)?;
            let tags = Tags { first: next, count: hole.surface.triangles.len() as u32 };
            append(t, hole, next, false);
            next += tags.count;
            Some((i, tags))
        })
        .collect()
}

/// Each inclusion's parent: the innermost other inclusion it lies inside (`None` when it lies in
/// the solid itself), from points just inside its surface `surfaces[i]` (each triangle's middle,
/// moved in by a millionth of the body's extent: a body touching another from outside has none
/// of them in it). One with some of them inside another and some outside crosses it and is
/// refused, both named; so is one inside a conductor (nothing is solved there) and two that
/// fill each other (the same body twice).
fn parents(model: &Model, surfaces: &[Tessellation]) -> Result<Vec<Option<usize>>> {
    let incs = model.inclusions();
    let refuse = |why: String| Err(MeshError::invalid("inclusion", why));
    let mut around: Vec<Vec<usize>> = Vec::with_capacity(incs.len());
    // Pairs with some of one's points in the other: crossing, unless the other lies in the one
    // (against its wall from inside, where the one's points just under that wall are in it).
    let mut partly: Vec<(usize, usize)> = Vec::new();
    for (i, (inc, t)) in incs.iter().zip(surfaces).enumerate() {
        let mut list = Vec::new();
        let step = 1e-6 * (inc.bounds().max - inc.bounds().min).norm();
        let within: Vec<Point3> = t
            .surface
            .triangles
            .iter()
            .filter_map(|tri| crate::shared::corners(&t.surface.positions, tri))
            .filter_map(|q| Some(Point3::centroid(q)? - (q[1] - q[0]).cross(q[2] - q[0]).normalized()? * step))
            .collect();
        for (j, other) in incs.iter().enumerate() {
            if j == i || !other.bounds().overlaps_inside(inc.bounds()) {
                continue;
            }
            match within.iter().filter(|p| other.contains(**p)).count() {
                0 => {}
                n if n == within.len() => list.push(j),
                _ => partly.push((i, j)),
            }
        }
        around.push(list);
    }
    if let Some((i, j)) = partly.into_iter().find(|(i, j)| !around.get(*j).is_some_and(|theirs| theirs.contains(i))) {
        return refuse(format!("inclusions {} and {} cross: one must lie wholly inside the other, or apart from it", i + 1, j + 1));
    }
    let mut parent = Vec::with_capacity(incs.len());
    for (i, list) in around.iter().enumerate() {
        if let Some(&j) = list.iter().find(|&&j| around.get(j).is_some_and(|theirs| theirs.contains(&i))) {
            return refuse(format!("inclusions {} and {} fill the same place", i + 1, j + 1));
        }
        // The innermost of those around it: the one with the most around itself.
        let p = list.iter().copied().max_by_key(|&j| (around.get(j).map_or(0, Vec::len), std::cmp::Reverse(j)));
        if let Some(j) = p.filter(|&j| incs.get(j).is_some_and(|o| o.material.is_none())) {
            return refuse(format!("inclusion {} lies inside inclusion {}, a conductor: nothing is solved there", i + 1, j + 1));
        }
        // Against its wall from inside, the two surfaces would share a patch: not made yet.
        if let (Some(j), Some(inc)) = (p, incs.get(i))
            && incs.get(j).is_some_and(|o| !o.bounds().encloses_inside(inc.bounds()))
        {
            return refuse(format!(
                "not supported yet: inclusion {} against the wall of inclusion {}, which it lies in (keep it off that surface)",
                i + 1,
                j + 1
            ));
        }
        parent.push(p);
    }
    Ok(parent)
}

/// The most rounds of meshing the regions again until their interfaces agree.
const MAX_INTERFACE_ROUNDS: usize = 6;

impl Body<'_> {
    /// Meshes the regions: boundary recovery may add points on an interface in one region; that
    /// region's triangulation then becomes the other's input, until both keep the same (frozen)
    /// triangles (at most [`MAX_INTERFACE_ROUNDS`] rounds). `main` is the solid's tessellation
    /// (with a spherical layer, its inner sphere's).
    fn regions(&self, mut main: Tessellation) -> Result<Regions> {
        let (model, size) = (self.model, self.size);
        let incs = model.inclusions();
        // An interface is tessellated at the smaller of the sizes of the two regions it parts.
        let surface = |i: usize, inc: &emcraft_model::Inclusion, around: f64| match self.cuts.iter().find(|(k, _)| *k == i) {
            Some((_, c)) => c.open.clone(),
            None => {
                let h = around.min(self.inner_size.get(i).copied().unwrap_or(size));
                placed(tessellate(&inc.shape, interface_size(&inc.shape, h, inc.material.is_none())), inc.at)
            }
        };
        let first: Vec<Tessellation> = incs.iter().enumerate().map(|(i, inc)| surface(i, inc, size)).collect();
        let parent = parents(model, &first)?;
        if !self.frozen.is_empty() && parent.iter().any(Option::is_some) {
            return Err(MeshError::invalid("rotational symmetry", "not supported yet: an inclusion inside another"));
        }
        let children = |of: Option<usize>| -> Vec<usize> { (0..parent.len()).filter(|k| parent.get(*k) == Some(&of)).collect() };
        // The interfaces' current triangulations, oriented out of each inclusion.
        let around: Vec<f64> = parent.iter().map(|p| p.and_then(|p| self.inner_size.get(p)).copied().unwrap_or(size)).collect();
        let mut interfaces: Vec<Tessellation> = first
            .into_iter()
            .zip(incs.iter().zip(&around))
            .enumerate()
            .map(|(i, (t, (inc, around)))| if *around < size { surface(i, inc, *around) } else { t })
            .collect();
        // Inclusions that touch face to face share the contact's triangles: each keeps towards
        // the region around it what is left of its surface.
        let fine: Vec<f64> = around.iter().zip(&self.inner_size).map(|(a, h)| a.min(*h)).collect();
        let scale = model.bounds().map_or(size, |b| (b.max - b.min).norm());
        let mut contacts = contact::imprint(&mut interfaces, &parent, &fine, GEOMETRY_TOLERANCE * (1.0 + scale))?;
        if !self.frozen.is_empty() && !contacts.is_empty() {
            return Err(MeshError::invalid("rotational symmetry", "not supported yet: inclusions that touch"));
        }
        // Their caps on the cut faces, which only the inclusion's own region holds.
        let mut caps: Vec<Option<Tessellation>> =
            (0..model.inclusions().len()).map(|i| self.cuts.iter().find(|(k, _)| *k == i).map(|(_, c)| c.caps.clone())).collect();
        let mut out = Regions {
            meshes: Vec::new(),
            start_face: main.start_face,
            end_face: main.end_face,
            first_tag: 0,
            holes: Vec::new(),
            caps_from: Vec::new(),
        };
        // A sector's cut faces not yet each other's turn after the last round's meshing.
        let mut unmatched = false;
        for _ in 0..MAX_INTERFACE_ROUNDS {
            // Tags: the solid's faces (with a spherical layer, the inner sphere's triangles), then
            // each interface's triangles in turn.
            out.first_tag = main.surface.tags.iter().map(|f| f + 1).max().unwrap_or(0);
            let mut outer = main.clone();
            let outer_holes = pierce(&mut outer, &interfaces, &children(None), out.first_tag);
            let next = outer_holes.last().map_or(out.first_tag, |(_, t)| t.first + t.count);
            let frozen_from = if self.shell { 0 } else { out.first_tag };
            // A sector: its cut faces kept as triangulated (each other's turn); the solid's other
            // faces remeshed, bound to the cut outlines' points (thinned to the size).
            let cut = self.frozen.first().map_or(0..0, |&base| base..out.first_tag);
            let frozen: Vec<u32> = (frozen_from..next).chain(cut).collect();
            let outer_mesh = region(&outer, size, frozen, &self.locals)?;
            // A sector's cut faces with points the mesher added: matched again and meshed again.
            if let (Some(rot), Some(&base)) = (&model.domain.rotation, self.frozen.first()) {
                let scale = model.bounds().map_or(size, |bx| (bx.max - bx.min).norm());
                unmatched = false;
                let bodies: Vec<Tessellation> = self.cuts.iter().filter_map(|(k, _)| interfaces.get(*k).cloned()).collect();
                if let Some((t, again)) = sector::refreeze(&outer_mesh, &main, rot, base, scale, &bodies)? {
                    main = t;
                    for ((k, _), c) in self.cuts.iter().zip(again) {
                        if let (Some(i), Some(cap)) = (interfaces.get_mut(*k), caps.get_mut(*k)) {
                            *i = c.open;
                            *cap = Some(c.caps);
                        }
                    }
                    unmatched = true;
                    continue;
                }
            }
            let mut changed = false;
            changed |= adopt_holes(&mut interfaces, &outer_mesh, &outer_holes, size);
            // The layer's shell: the solid's sphere outside, the inner sphere as a hole (tags 1..).
            let shell = if self.shell {
                changed |= adopt(&mut main, extract(&outer_mesh, 0, out.first_tag, true), size);
                let mut t = tessellate(self.shape, size);
                append(&mut t, &main, 1, false);
                let n = main.surface.triangles.len() as u32;
                let shell = region(&t, size, (1..1 + n).collect(), &self.locals)?;
                changed |= adopt(&mut main, extract(&shell, 1, n, false), size);
                Some(shell)
            } else {
                None
            };
            out.meshes = vec![(outer_mesh, self.material)];
            out.meshes.extend(shell.map(|m| (m, self.material)));
            out.caps_from = vec![None; out.meshes.len()];
            out.holes = vec![Vec::new(); out.meshes.len()];
            if let Some(h) = out.holes.first_mut() {
                *h = outer_holes;
            }
            for (i, ((inc, h), cap)) in incs.iter().zip(&self.inner_size).zip(&caps).enumerate() {
                let Some(name) = &inc.material else { continue };
                let (m, _) = model.material(name)?;
                let Some(mut whole) = interfaces.get(i).cloned() else { continue };
                let n = whole.surface.triangles.len() as u32;
                // Across a sector's cut: its surface there closed by its caps.
                if let Some(c) = cap {
                    append(&mut whole, c, n, true);
                }
                // Its contacts with the inclusions it touches: the rest of its surface.
                let mut shared: Vec<(usize, Tags)> = Vec::new();
                for (k, c) in contacts.iter().enumerate().filter(|(_, c)| c.a == i || c.b == i) {
                    let tags = Tags { first: whole.surface.triangles.len() as u32, count: c.patch.surface.triangles.len() as u32 };
                    append(&mut whole, &c.patch, tags.first, c.a == i);
                    shared.push((k, tags));
                }
                // The inclusions directly inside it: holes in its region.
                let next = whole.surface.triangles.len() as u32;
                let mut holes = pierce(&mut whole, &interfaces, &children(Some(i)), next);
                let all = whole.surface.triangles.len() as u32;
                let inner = region(&whole, *h, (0..all).collect(), &self.locals)?;
                if let Some(t) = interfaces.get_mut(i) {
                    changed |= adopt(t, extract(&inner, 0, n, true), size);
                }
                changed |= adopt_holes(&mut interfaces, &inner, &holes, size);
                for (k, tags) in shared {
                    let Some(c) = contacts.get_mut(k) else { continue };
                    changed |= adopt(&mut c.patch, extract(&inner, tags.first, tags.count, c.a == i), size);
                    // Beyond these faces lies the other inclusion.
                    holes.push((if c.a == i { c.b } else { c.a }, tags));
                }
                out.meshes.push((inner, m as u32));
                out.caps_from.push(cap.as_ref().map(|_| n));
                out.holes.push(holes);
            }
            if !changed {
                break;
            }
        }
        if unmatched {
            return Err(MeshError::invalid(
                "rotational symmetry",
                "the mesher kept adding points to the sector's cut faces; try another element size",
            ));
        }
        Ok(out)
    }

    /// Whether face `face` of region `r` lies on an interface between two regions: a hole's
    /// when the inclusion there is a dielectric (a conductor's is a wall), else an inner region's
    /// own surface off its caps on a sector's cut faces. With a spherical layer, region 1 is its
    /// shell (tag 0 its outer wall, the rest the inner sphere) and the solid region's own faces
    /// are that inner sphere.
    fn on_interface(&self, regions: &Regions, r: usize, face: u32) -> bool {
        let hole = regions.holes.get(r).and_then(|holes| holes.iter().find(|(_, tags)| tags.contains(face)));
        if let Some((i, _)) = hole {
            return self.model.inclusions().get(*i).is_some_and(|inc| inc.material.is_some());
        }
        match r {
            0 => self.shell && face < regions.first_tag,
            1 if self.shell => face >= 1,
            _ => !regions.caps_from.get(r).copied().flatten().is_some_and(|from| face >= from),
        }
    }

    /// The regions joined into one mesh: nodes with the same coordinates (the frozen interfaces'
    /// vertices) become one; the outer region's faces are walls or ports, the shell's outer wall
    /// a wall, and every interface triangle must be seen from both sides.
    fn join(&self, regions: &Regions) -> Result<Mesh> {
        let mut nodes = Nodes::new(GEOMETRY_TOLERANCE * self.size);
        let mut tets = Vec::new();
        let mut tet_material = Vec::new();
        let mut boundary = Vec::new();
        let mut interface_sides: std::collections::HashMap<[u32; 3], u32> = std::collections::HashMap::new();
        for (r, (mesh, mat)) in regions.meshes.iter().enumerate() {
            let map: Vec<u32> = mesh.points.iter().map(|p| nodes.id(*p)).collect();
            let first = tets.len() as u32;
            tets.extend(mesh.tets.iter().map(|t| t.map(|v| map[v as usize])));
            tet_material.extend(std::iter::repeat_n(*mat, mesh.tets.len()));
            for &RegionFace { nodes, tet, tag: face } in &mesh.boundary {
                let n = nodes.map(|v| map[v as usize]);
                if self.on_interface(regions, r, face) {
                    let mut key = n;
                    key.sort_unstable();
                    *interface_sides.entry(key).or_default() += 1;
                    continue;
                }
                // What is left is a wall: the solid's faces off its ports, the shell's outer
                // wall, a conductor's surface, a cap on a sector's cut face.
                let tag = match r {
                    0 if Some(face) == regions.start_face => Boundary::Port(0),
                    0 if Some(face) == regions.end_face => Boundary::Port(1),
                    _ => Boundary::Pec,
                };
                boundary.push(BoundaryFace { nodes: n, tet: first + tet, tag });
            }
        }
        // Every interface triangle seen from both sides: the regions conform.
        if interface_sides.values().any(|&c| c != 2) {
            return Err(MeshError::Invalid {
                what: "tetrahedral meshing".into(),
                why: "an interface between two materials was not kept the same from both sides (boundary recovery added points on it); try another element size".into(),
            });
        }
        Ok(Mesh { tet_material, nodes: nodes.points, tets, boundary, grid: None, buckets: None, curved: Vec::new() })
    }
}

/// The nodes of a joined mesh: a point within `tol` of one already there (the regions' snapping
/// to the exact surfaces may differ in the last bits) is that node, found through a hash on
/// cells of that size, checking the neighbours.
struct Nodes {
    tol: f64,
    index: std::collections::HashMap<[i64; 3], u32>,
    points: Vec<Point3>,
}

impl Nodes {
    fn new(tol: f64) -> Nodes {
        Nodes { tol, index: std::collections::HashMap::new(), points: Vec::new() }
    }

    /// The node at `p`, added when there is none.
    fn id(&mut self, p: Point3) -> u32 {
        let tol = self.tol;
        let k = [p.x, p.y, p.z].map(|v| (v / tol).floor() as i64);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(&i) = self.index.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) {
                        let q = self.points[i as usize];
                        if (q - p).iter().all(|d| d.abs() <= tol) {
                            return i;
                        }
                    }
                }
            }
        }
        self.points.push(p);
        let i = (self.points.len() - 1) as u32;
        self.index.insert(k, i);
        i
    }
}

/// A uniform grid of boxes over space: `dims` boxes of `step` along each axis from `lo`.
struct Voxels {
    lo: Point3,
    step: Vec3,
    dims: [usize; 3],
}

impl Voxels {
    fn of(b: &Buckets) -> Voxels {
        Voxels { lo: b.lo, step: b.step, dims: b.dims }
    }

    /// The box along axis `d` holding coordinate `v`, clamped to the grid.
    fn index(&self, v: f64, d: usize) -> usize {
        (((v - self.lo[d]) / self.step[d]).floor().max(0.0) as usize).min(self.dims[d].saturating_sub(1))
    }
}

/// The bucket along axis `d` of a mesh's buckets holding coordinate `v`, clamped to the grid
/// (for point location and box queries, [`crate::locate`], [`crate::tets_near`]).
pub(crate) fn bucket_index(b: &Buckets, v: f64, d: usize) -> usize {
    Voxels::of(b).index(v, d)
}

/// The tetrahedra binned on a uniform grid of about one tetrahedron per bucket.
pub fn buckets(mesh: &Mesh) -> Buckets {
    let Aabb { min: lo, max: hi } = Aabb::around(mesh.nodes.iter().copied()).unwrap_or_default();
    let ext = (hi - lo).map(|e| e.max(f64::MIN_POSITIVE));
    let per_side = (mesh.tets.len().max(1) as f64).cbrt().ceil() as usize;
    let longest = ext.iter().fold(0.0, f64::max);
    let dims = [0, 1, 2].map(|d| ((ext[d] / longest * per_side as f64).ceil() as usize).clamp(1, 512));
    let step = Vec3::new(ext.x / dims[0] as f64, ext.y / dims[1] as f64, ext.z / dims[2] as f64);
    let grid = Voxels { lo, step, dims };
    let nb = dims[0] * dims[1] * dims[2];
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); nb];
    for (t, tet) in mesh.tets.iter().enumerate() {
        let mut a = [usize::MAX; 3];
        let mut b = [0usize; 3];
        for &n in tet {
            let p = mesh.nodes[n as usize];
            for d in 0..3 {
                a[d] = a[d].min(grid.index(p[d], d));
                b[d] = b[d].max(grid.index(p[d], d));
            }
        }
        for k in a[2]..=b[2] {
            for j in a[1]..=b[1] {
                for i in a[0]..=b[0] {
                    lists[i + dims[0] * (j + dims[1] * k)].push(t as u32);
                }
            }
        }
    }
    let mut start = Vec::with_capacity(nb + 1);
    let mut items = Vec::new();
    start.push(0);
    for l in lists {
        items.extend(l);
        start.push(items.len() as u32);
    }
    Buckets { lo, step, dims, start, items }
}

#[cfg(test)]
mod tests {
    use super::*;
    use emcraft_model::Solid;
    use emcraft_model::{Axis, Combine};
    use std::f64::consts::PI;

    /// A length (m), checked.
    fn metres(v: f64) -> emcraft_model::Metres {
        emcraft_model::Metres::new(v).unwrap()
    }

    /// A box's sides (m), checked.
    fn box_size(v: emcraft_model::Vec3) -> emcraft_model::BoxSize {
        emcraft_model::BoxSize::new(v).unwrap()
    }

    fn mesh_of(shape: Shape, size: f64) -> Mesh {
        let mut m = Model::new();
        m.set_solid(Solid { shape, material: "vacuum".into() }).unwrap();
        solid(&m, metres(size)).unwrap()
    }

    fn area(mesh: &Mesh, f: &BoundaryFace) -> f64 {
        let p = f.nodes.map(|n| mesh.nodes[n as usize]);
        let u = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
        let v = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
        let c = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        0.5 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
    }

    #[test]
    fn a_box_is_filled_exactly_with_its_ports_tagged() {
        let mesh = mesh_of(Shape::Box { size: box_size(Vec3::new(0.02, 0.01, 0.03)) }, 0.004);
        let v = crate::volumes(&mesh);
        assert!(v.iter().all(|x| *x > 0.0));
        approx::assert_abs_diff_eq!(v.iter().sum::<f64>(), 6e-6, epsilon = 1e-12 * 6e-6 + 1e-15);
        let port = |k: u32| mesh.boundary.iter().filter(|f| f.tag == Boundary::Port(k)).map(|f| area(&mesh, f)).sum::<f64>();
        assert!((port(0) - 2e-4).abs() < 1e-12 && (port(1) - 2e-4).abs() < 1e-12);
        assert!(
            mesh.boundary.iter().filter(|f| f.tag == Boundary::Port(0)).all(|f| f.nodes.iter().all(|n| mesh.nodes[*n as usize][2].abs() < 1e-12))
        );
    }

    #[test]
    fn a_cylinder_and_a_sphere_approach_their_volumes() {
        let (r, l) = (0.01, 0.02);
        let mesh = mesh_of(Shape::Cylinder { radius: metres(r), length: metres(l) }, 0.0025);
        let v: f64 = crate::volumes(&mesh).iter().sum();
        let exact = PI * r * r * l;
        // Flat facets inscribed in the wall: a few tenths of a percent short.
        assert!(v < exact && v > 0.98 * exact, "{v} {exact}");
        // Boundary vertices lie on the exact surfaces.
        for f in &mesh.boundary {
            for n in f.nodes {
                let p = mesh.nodes[n as usize];
                let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
                assert!((rho - r).abs() < 1e-9 || p[2].abs() < 1e-12 || (p[2] - l).abs() < 1e-12, "{p:?}");
            }
        }
        let sphere = mesh_of(Shape::Sphere { radius: metres(r) }, 0.003);
        let v: f64 = crate::volumes(&sphere).iter().sum();
        let exact = 4.0 / 3.0 * PI * r.powi(3);
        // Flat facets of about 3 mm inscribed in a 10 mm sphere: 3.7 % short.
        assert!(v < exact && v > 0.95 * exact, "{v} {exact}");
        assert!(sphere.boundary.iter().all(|f| f.tag == Boundary::Pec));
    }

    /// A sphere with a spherical layer, a dielectric inclusion inside it if asked: its mesh
    /// checked to fill the sphere, its shell's tetrahedra a layer of their own.
    fn layered(with_inclusion: bool) {
        use emcraft_model::{Inclusion, Material, Open, OpenShape};
        let (r, d) = (0.02, 0.006);
        let mut m = Model::new();
        m.add_material(Material::new("alumina", 9.8, 1.0).unwrap()).unwrap();
        m.set_solid(Solid { shape: Shape::Sphere { radius: metres(r) }, material: "vacuum".into() }).unwrap();
        m.set_open(Some(Open { thickness: metres(d), reflection: None, shape: OpenShape::Sphere })).unwrap();
        if with_inclusion {
            let ceramic =
                Inclusion { shape: Shape::Sphere { radius: metres(0.005) }, at: Point3::new(0.002, 0.0, 0.0), material: Some("alumina".into()) };
            m.add_inclusion(ceramic).unwrap();
        }
        let mesh = solid(&m, metres(0.004)).unwrap();
        let v = crate::volumes(&mesh);
        assert!(v.iter().all(|x| *x > 0.0));
        let exact = 4.0 / 3.0 * PI * r.powi(3);
        let total: f64 = v.iter().sum();
        assert!(total < exact && total > 0.95 * exact, "{total} {exact}");
        // No tetrahedron crosses the inner sphere: each lies wholly inside or outside it.
        let radius = |n: u32| mesh.nodes[n as usize].to_vec().norm();
        let tol = 1e-9 * r;
        let mut shell = 0.0;
        for (t, tet) in mesh.tets.iter().enumerate() {
            let (lo, hi) = tet.iter().map(|&n| radius(n)).fold((f64::INFINITY, 0.0f64), |(a, b), x| (a.min(x), b.max(x)));
            assert!(hi <= r - d + tol || lo >= r - d - tol, "tetrahedron {t} crosses the layer's inner sphere: {lo} {hi}");
            if lo >= r - d - tol {
                shell += v[t];
            }
        }
        let exact_shell = 4.0 / 3.0 * PI * (r.powi(3) - (r - d).powi(3));
        assert!((shell / exact_shell - 1.0).abs() < 0.05, "{shell} {exact_shell}");
        // The walls are the outer sphere only; the dielectric keeps its region.
        assert!(mesh.boundary.iter().all(|f| f.tag == Boundary::Pec && f.nodes.iter().all(|&n| (radius(n) - r).abs() < 1e-9)));
        assert_eq!(mesh.tet_material.contains(&1), with_inclusion);
    }

    #[test]
    fn a_spherical_layer_is_a_shell_of_its_own() {
        layered(false);
    }

    #[test]
    #[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
    fn a_spherical_layer_holds_inclusions() {
        layered(true);
    }

    #[test]
    fn inclusions_make_regions_that_meet_node_for_node() {
        use emcraft_model::{Inclusion, Material};
        let mut m = Model::new();
        m.add_material(Material::new("alumina", 9.8, 1.0).unwrap()).unwrap();
        m.set_solid(Solid { shape: Shape::Sphere { radius: metres(0.01) }, material: "vacuum".into() }).unwrap();
        m.add_inclusion(Inclusion {
            shape: Shape::Sphere { radius: metres(0.004) },
            at: Point3::new(0.0, 0.0, 0.0),
            material: Some("alumina".into()),
        })
        .unwrap();
        let mesh = solid(&m, metres(0.0025)).unwrap();
        let v = crate::volumes(&mesh);
        assert!(v.iter().all(|x| *x > 0.0));
        let inner: f64 = v.iter().zip(&mesh.tet_material).filter(|(_, m)| **m == 1).map(|(v, _)| v).sum();
        let exact = 4.0 / 3.0 * PI * 0.004f64.powi(3);
        // Flat facets under 0.8 mm long on the 4 mm sphere.
        assert!(inner < exact && inner > 0.97 * exact, "{inner} {exact}");
        // Conforming: every face of the joined mesh is held by one or two tetrahedra, and the
        // boundary is the outer sphere only.
        let topo = crate::Topology::new(&mesh);
        let outer_faces = topo.face_tets.iter().filter(|o| o[1] == crate::NO_TET).count();
        assert_eq!(outer_faces, mesh.boundary.len());
        assert!(mesh.boundary.iter().all(|f| f.nodes.iter().all(|n| {
            let p = mesh.nodes[*n as usize];
            ((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - 0.01).abs() < 1e-9
        })));
        // A conductor instead: a hole, its surface a wall.
        let mut c = m.clone();
        let k = c.mesh_view().inclusion_body[0];
        c.bodies_mut()[k].fill = emcraft_model::Fill::Metal;
        let hole = solid(&c, metres(0.0025)).unwrap();
        assert!(hole.tet_material.iter().all(|m| *m == 0));
        let walls = hole.boundary.len();
        assert!(walls > mesh.boundary.len());
        // Inclusions must lie inside.
        assert!(
            m.add_inclusion(Inclusion { shape: Shape::Sphere { radius: metres(0.004) }, at: Point3::new(0.008, 0.0, 0.0), material: None }).is_err()
        );
    }

    /// A coated sphere: a ball of a second dielectric inside a dielectric ball, and a conductor
    /// in place of the core. Evidence: the volumes of the core and the coat against 4πr³/3 (flat
    /// facets, so a little under), and every face held by one or two tetrahedra.
    #[test]
    fn an_inclusion_inside_another_is_a_region_inside_its_region() {
        use emcraft_model::{Inclusion, Material};
        let mut m = Model::new();
        m.add_material(Material::new("alumina", 9.8, 1.0).unwrap()).unwrap();
        m.add_material(Material::new("teflon", 2.1, 1.0).unwrap()).unwrap();
        m.set_solid(Solid { shape: Shape::Sphere { radius: metres(0.01) }, material: "vacuum".into() }).unwrap();
        let ball = |r: f64, material: Option<&str>| Inclusion {
            shape: Shape::Sphere { radius: metres(r) },
            at: Point3::new(0.001, 0.0, 0.0),
            material: material.map(str::to_owned),
        };
        // The core first: the order the bodies are added in does not matter.
        m.add_inclusion(ball(0.003, Some("teflon"))).unwrap();
        m.add_inclusion(ball(0.006, Some("alumina"))).unwrap();
        let mesh = solid(&m, metres(0.0025)).unwrap();
        let v = crate::volumes(&mesh);
        assert!(v.iter().all(|x| *x > 0.0));
        let of = |k: u32| -> f64 { v.iter().zip(&mesh.tet_material).filter(|(_, m)| **m == k).map(|(v, _)| v).sum() };
        let sphere = |r: f64| 4.0 / 3.0 * PI * r.powi(3);
        let (coat, core) = (of(1), of(2));
        assert!(core < sphere(0.003) && core > 0.97 * sphere(0.003), "{core}");
        let exact = sphere(0.006) - sphere(0.003);
        assert!((coat - exact).abs() < 0.03 * exact, "{coat} {exact}");
        let topo = crate::Topology::new(&mesh);
        assert_eq!(topo.face_tets.iter().filter(|o| o[1] == crate::NO_TET).count(), mesh.boundary.len());
        // The point's filling is the innermost body's.
        assert_eq!(m.material_at(Point3::new(0.001, 0.0, 0.0)), Some(2));
        assert_eq!(m.material_at(Point3::new(0.005, 0.0, 0.0)), Some(1));
        // A conducting core: a hole in the coat, its surface a wall there.
        let mut c = m.clone();
        let k = c.mesh_view().inclusion_body[0];
        c.bodies_mut()[k].fill = emcraft_model::Fill::Metal;
        let hole = solid(&c, metres(0.0025)).unwrap();
        assert!(hole.tet_material.iter().all(|m| *m < 2));
        assert!(hole.boundary.len() > mesh.boundary.len());
        let inner_walls =
            hole.boundary.iter().filter(|f| f.nodes.iter().all(|n| (hole.nodes[*n as usize] - Point3::new(0.001, 0.0, 0.0)).norm() < 0.0031));
        assert!(inner_walls.count() > 20);
        // Refused: a body inside a conductor, and two that cross.
        let mut inside_metal = c.clone();
        inside_metal.add_inclusion(ball(0.001, Some("teflon"))).unwrap();
        let e = solid(&inside_metal, metres(0.0025)).unwrap_err().to_string();
        assert!(e.contains("a conductor"), "{e}");
        let mut crossing = Model::new();
        crossing.add_material(Material::new("alumina", 9.8, 1.0).unwrap()).unwrap();
        crossing.set_solid(Solid { shape: Shape::Sphere { radius: metres(0.01) }, material: "vacuum".into() }).unwrap();
        crossing.add_inclusion(ball(0.004, Some("alumina"))).unwrap();
        let cube = Shape::Box { size: emcraft_model::BoxSize::new(emcraft_model::Vec3::new(0.007, 0.007, 0.007)).unwrap() };
        crossing.add_inclusion(Inclusion { shape: cube, at: Point3::new(-0.0025, -0.0035, -0.0035), material: Some("alumina".into()) }).unwrap();
        let e = solid(&crossing, metres(0.0025)).unwrap_err().to_string();
        assert!(e.contains("cross"), "{e}");
    }

    /// A model of a 30 mm vacuum box with dielectrics `a` and `b` and the inclusions given.
    fn boxed(inclusions: &[emcraft_model::Inclusion]) -> Model {
        use emcraft_model::Material;
        let mut m = Model::new();
        m.add_material(Material::new("a", 4.0, 1.0).unwrap()).unwrap();
        m.add_material(Material::new("b", 9.0, 1.0).unwrap()).unwrap();
        m.set_solid(Solid { shape: Shape::Box { size: BoxSize::new(Vec3::new(0.03, 0.03, 0.03)).unwrap() }, material: "vacuum".into() }).unwrap();
        for inc in inclusions {
            m.add_inclusion(inc.clone()).unwrap();
        }
        m
    }

    fn block(size: [f64; 3], at: [f64; 3], material: Option<&str>) -> emcraft_model::Inclusion {
        emcraft_model::Inclusion {
            shape: Shape::Box { size: BoxSize::new(Vec3::new(size[0], size[1], size[2])).unwrap() },
            at: Point3::new(at[0], at[1], at[2]),
            material: material.map(str::to_owned),
        }
    }

    /// The volume of each material and whether every face is held by one or two tetrahedra,
    /// the boundary's being those held by one.
    fn volumes_and_conformity(mesh: &Mesh, materials: u32) -> Vec<f64> {
        let v = crate::volumes(mesh);
        assert!(v.iter().all(|x| *x > 0.0));
        let topo = crate::Topology::new(mesh);
        assert_eq!(topo.face_tets.iter().filter(|o| o[1] == crate::NO_TET).count(), mesh.boundary.len());
        (0..materials).map(|k| v.iter().zip(&mesh.tet_material).filter(|(_, m)| **m == k).map(|(v, _)| v).sum()).collect()
    }

    /// Dielectrics that touch face to face: a small block on a larger one (the contact inside
    /// the larger one's top face), two blocks of one footprint stacked (their outlines the
    /// same), and one shifted across the other's edge. Evidence: each material's volume is its
    /// block's (flat faces, so exact to rounding) and every interior face has two tetrahedra;
    /// the walls are the box's only.
    #[rstest::rstest]
    #[case::inside_the_face([0.012, 0.012, 0.004], [0.009, 0.009, 0.008], [0.006, 0.005, 0.005], [0.012, 0.0125, 0.012])]
    #[case::the_same_footprint([0.008, 0.008, 0.004], [0.011, 0.011, 0.008], [0.008, 0.008, 0.006], [0.011, 0.011, 0.012])]
    #[case::across_an_edge([0.010, 0.010, 0.004], [0.008, 0.010, 0.008], [0.010, 0.006, 0.005], [0.013, 0.012, 0.012])]
    fn dielectrics_that_touch_share_their_contact(#[case] lower: [f64; 3], #[case] at: [f64; 3], #[case] upper: [f64; 3], #[case] above: [f64; 3]) {
        let m = boxed(&[block(lower, at, Some("a")), block(upper, above, Some("b"))]);
        let mesh = solid(&m, metres(0.006)).unwrap();
        let v = volumes_and_conformity(&mesh, 3);
        let volume = |s: [f64; 3]| s[0] * s[1] * s[2];
        approx::assert_abs_diff_eq!(v[1], volume(lower), epsilon = 1e-9 * volume(lower));
        approx::assert_abs_diff_eq!(v[2], volume(upper), epsilon = 1e-9 * volume(upper));
        approx::assert_abs_diff_eq!(v[0] + v[1] + v[2], 0.03f64.powi(3), epsilon = 1e-9 * 0.03f64.powi(3));
        let on_wall = |n: &u32| mesh.nodes[*n as usize].iter().any(|c| c.abs() < 1e-12 || (c - 0.03).abs() < 1e-12);
        assert!(mesh.boundary.iter().all(|f| f.nodes.iter().all(on_wall)));
    }

    fn plate() -> emcraft_model::Inclusion {
        block([0.016, 0.016, 0.003], [0.007, 0.007, 0.008], Some("a"))
    }

    fn rod(material: Option<&str>) -> emcraft_model::Inclusion {
        emcraft_model::Inclusion {
            shape: Shape::Cylinder { radius: metres(0.003), length: metres(0.006) },
            at: Point3::new(0.015, 0.015, 0.011),
            material: material.map(str::to_owned),
        }
    }

    /// A conducting cylinder standing on a dielectric plate: a hole in the box's region, its
    /// foot (round, inside the plate's top face) a wall of the plate's region. Evidence: the
    /// plate's volume, and the foot's area — the walls in the plate's top plane — a 12-sided
    /// polygon's in the circle (the conductor's surface is divided by π/6), with which the
    /// vacuum's volume is the box's less the plate's and the prism's.
    #[test]
    fn a_conductor_standing_on_a_dielectric_makes_its_foot_a_wall() {
        let mesh = solid(&boxed(&[plate(), rod(None)]), metres(0.006)).unwrap();
        let v = volumes_and_conformity(&mesh, 2);
        approx::assert_abs_diff_eq!(v[1], 0.016 * 0.016 * 0.003, epsilon = 1e-15);
        let foot: f64 = mesh
            .boundary
            .iter()
            .map(|f| f.nodes.map(|n| mesh.nodes[n as usize]))
            .filter(|q| q.iter().all(|p| (p.z - 0.011).abs() < 1e-12))
            .map(|q| 0.5 * (q[1] - q[0]).cross(q[2] - q[0]).norm())
            .sum();
        let circle = PI * 0.003f64.powi(2);
        assert!(foot < circle && foot > 0.9 * circle, "{foot} {circle}");
        approx::assert_abs_diff_eq!(v[0] + v[1], 0.03f64.powi(3) - 0.006 * foot, epsilon = 1e-12);
    }

    /// A dielectric cylinder standing on a plate (its round cap inside the plate's top face),
    /// and three bodies around one edge (a plate under two blocks side by side that touch).
    /// Evidence: the volumes (the cylinder's a prism's inside the circle's).
    #[test]
    #[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
    fn a_cylinder_on_a_plate_and_three_bodies_around_an_edge() {
        let mesh = solid(&boxed(&[plate(), rod(Some("b"))]), metres(0.003)).unwrap();
        let v = volumes_and_conformity(&mesh, 3);
        approx::assert_abs_diff_eq!(v[1], 0.016 * 0.016 * 0.003, epsilon = 1e-15);
        let round = PI * 0.003f64.powi(2) * 0.006;
        assert!(v[2] < round && v[2] > 0.95 * round, "{} {round}", v[2]);
        let left = block([0.006, 0.010, 0.004], [0.009, 0.010, 0.011], Some("b"));
        let right = block([0.006, 0.008, 0.005], [0.015, 0.011, 0.011], Some("a"));
        let mesh = solid(&boxed(&[plate(), left, right]), metres(0.003)).unwrap();
        let v = volumes_and_conformity(&mesh, 3);
        approx::assert_abs_diff_eq!(v[2], 0.006 * 0.010 * 0.004, epsilon = 1e-15);
        approx::assert_abs_diff_eq!(v[1], 0.016 * 0.016 * 0.003 + 0.006 * 0.008 * 0.005, epsilon = 1e-15);
        // A vessel standing on the plate with a liquid inside it: contact and nesting together.
        let liquid = emcraft_model::Inclusion {
            shape: Shape::Cylinder { radius: metres(0.002), length: metres(0.004) },
            at: Point3::new(0.015, 0.015, 0.012),
            material: Some("a".into()),
        };
        let mesh = solid(&boxed(&[plate(), rod(Some("b")), liquid]), metres(0.003)).unwrap();
        let v = volumes_and_conformity(&mesh, 3);
        let (vessel, inside) = (PI * 0.003f64.powi(2) * 0.006, PI * 0.002f64.powi(2) * 0.004);
        assert!(v[2] < vessel - 0.9 * inside && v[2] > 0.93 * (vessel - inside), "{} {vessel} {inside}", v[2]);
        assert!(v[1] > 0.016 * 0.016 * 0.003 + 0.9 * inside && v[1] < 0.016 * 0.016 * 0.003 + inside, "{}", v[1]);
        // A body against its vessel's wall from inside is not made a contact yet: refused.
        let against = emcraft_model::Inclusion {
            shape: Shape::Cylinder { radius: metres(0.002), length: metres(0.004) },
            at: Point3::new(0.015, 0.015, 0.011),
            material: Some("a".into()),
        };
        let e = solid(&boxed(&[plate(), rod(Some("b")), against]), metres(0.003)).unwrap_err().to_string();
        assert!(e.contains("against the wall of inclusion 2"), "{e}");
    }

    #[test]
    fn edges_on_curved_walls_are_bent_onto_them() {
        let (r, l) = (0.01, 0.02);
        let mesh = mesh_of(Shape::Cylinder { radius: metres(r), length: metres(l) }, 0.003);
        assert!(!mesh.curved.is_empty() && mesh.curved.windows(2).all(|w| w[0].0 < w[1].0));
        let boundary_edges: std::collections::HashSet<[u32; 2]> = mesh
            .boundary
            .iter()
            .flat_map(|f| [(0, 1), (1, 2), (0, 2)].map(|(i, j)| [f.nodes[i].min(f.nodes[j]), f.nodes[i].max(f.nodes[j])]))
            .collect();
        for (e, m) in &mesh.curved {
            // On the side wall, between two wall nodes, an edge of the boundary.
            assert!(((m[0] * m[0] + m[1] * m[1]).sqrt() - r).abs() < 1e-12 && boundary_edges.contains(e));
        }
        // Every edge on the side (both ends on the wall, not across a cap) is curved.
        let rho = |n: u32| {
            let p = mesh.nodes[n as usize];
            (p[0] * p[0] + p[1] * p[1]).sqrt()
        };
        let side = mesh
            .boundary
            .iter()
            .filter(|f| {
                f.nodes.iter().all(|n| (rho(*n) - r).abs() < 1e-9) && {
                    let z = f.nodes.map(|n| mesh.nodes[n as usize][2]);
                    !(z.iter().all(|z| z.abs() < 1e-12) || z.iter().all(|z| (z - l).abs() < 1e-12))
                }
            })
            .count();
        assert!(side > 0);
        for f in &mesh.boundary {
            let z = f.nodes.map(|n| mesh.nodes[n as usize][2]);
            let cap = z.iter().all(|z| z.abs() < 1e-12) || z.iter().all(|z| (z - l).abs() < 1e-12);
            if !cap && f.nodes.iter().all(|n| (rho(*n) - r).abs() < 1e-9) {
                for (i, j) in [(0, 1), (1, 2), (0, 2)] {
                    assert!(mesh.midpoint(f.nodes[i], f.nodes[j]).is_some());
                }
            }
        }
        // Straight on request; a box has nothing to bend.
        let mut m = Model::new();
        m.set_solid(Solid { shape: Shape::Cylinder { radius: metres(r), length: metres(l) }, material: "vacuum".into() }).unwrap();
        assert!(solid_with(&m, &SolidOptions { curved: false, ..SolidOptions::new(metres(0.003)) }).unwrap().curved.is_empty());
        assert!(mesh_of(Shape::Box { size: box_size(Vec3::new(0.02, 0.01, 0.03)) }, 0.004).curved.is_empty());
    }

    #[test]
    fn features_combine_with_the_solid() {
        use emcraft_model::Feature;
        let (a, b, l) = (0.02286, 0.01016, 0.03);
        let mut m = Model::new();
        m.set_solid(Solid { shape: Shape::Box { size: box_size(Vec3::new(a, b, l)) }, material: "vacuum".into() }).unwrap();
        // A post across the guide, flush with both broad walls.
        let r = 0.0015;
        m.add_feature(Feature {
            combine: Combine::Subtract,
            shape: Shape::Cylinder { radius: metres(r), length: metres(b) },
            at: Point3::new(a / 2.0, 0.0, l / 2.0),
            axis: Axis::Y,
        })
        .unwrap();
        let mesh = solid(&m, metres(0.003)).unwrap();
        let v: f64 = crate::volumes(&mesh).iter().sum();
        let exact = a * b * l - PI * r * r * b;
        // Flat facets on the post (12 round): the hole a little smaller than the circle.
        assert!(v > exact && v < exact + 0.05 * PI * r * r * b, "{v} {exact}");
        // The post's surface is wall; the ends are still whole ports.
        let port = |k: u32| mesh.boundary.iter().filter(|f| f.tag == Boundary::Port(k)).map(|f| area(&mesh, f)).sum::<f64>();
        assert!((port(0) - a * b).abs() < 1e-12 && (port(1) - a * b).abs() < 1e-12);
        let on_post = |f: &BoundaryFace| {
            f.nodes.iter().all(|n| {
                let p = mesh.nodes[*n as usize];
                ((p[0] - a / 2.0).hypot(p[2] - l / 2.0) - r).abs() < 1e-9
            })
        };
        assert!(mesh.boundary.iter().any(on_post));
        assert!(mesh.curved.iter().any(|(_, p)| ((p[0] - a / 2.0).hypot(p[2] - l / 2.0) - r).abs() < 1e-12));
        // A stub added on the broad wall: a box standing on it (its foot moved into the guide).
        let mut u = Model::new();
        u.set_solid(Solid { shape: Shape::Box { size: box_size(Vec3::new(a, b, l)) }, material: "vacuum".into() }).unwrap();
        u.add_feature(Feature {
            combine: Combine::Add,
            shape: Shape::Box { size: box_size(Vec3::new(0.008, 0.006, 0.01)) },
            at: Point3::new(0.007, b, 0.01),
            axis: Axis::Z,
        })
        .unwrap();
        let v: f64 = crate::volumes(&solid(&u, metres(0.003)).unwrap()).iter().sum();
        // Exact but for the features' shift by a millionth of the element size (1e-9 m here,
        // over the stub's 80 mm² foot).
        assert!((v - (a * b * l + 0.008 * 0.006 * 0.01)).abs() < 1e-7 * v, "{v}");
        // A union flush with both broad walls, reaching out past a side wall: grown past the
        // broad walls' planes, united and cut back at them (`crate::clip`), exactly.
        let mut w = Model::new();
        w.set_solid(Solid { shape: Shape::Box { size: box_size(Vec3::new(a, b, l)) }, material: "vacuum".into() }).unwrap();
        w.add_feature(Feature {
            combine: Combine::Add,
            shape: Shape::Box { size: box_size(Vec3::new(0.03, b, 0.01)) },
            at: Point3::new(0.0, 0.0, 0.01),
            axis: Axis::Z,
        })
        .unwrap();
        let v: f64 = crate::volumes(&solid(&w, metres(0.005)).unwrap()).iter().sum();
        let exact = a * b * l + (0.03 - a) * b * 0.01;
        assert!((v - exact).abs() < 1e-7 * exact, "{v} {exact}");
    }

    /// A union flush with the solid's faces from inside, its faces within theirs: the solid's
    /// faces bound the union, so it draws back from them — a block across the guide changes
    /// nothing, a stub standing on the floor and reaching out through the roof adds only what
    /// stands out.
    #[test]
    fn unions_flush_from_inside_draw_back() {
        use emcraft_model::Feature;
        let (a, b, l) = (0.02286, 0.01016, 0.03);
        for (height, extra) in [(b, 0.0), (b + 0.006, 0.006)] {
            let mut w = Model::new();
            w.set_solid(Solid { shape: Shape::Box { size: box_size(Vec3::new(a, b, l)) }, material: "vacuum".into() }).unwrap();
            w.add_feature(Feature {
                combine: Combine::Add,
                shape: Shape::Box { size: box_size(Vec3::new(0.008, height, 0.01)) },
                at: Point3::new(0.007, 0.0, 0.01),
                axis: Axis::Z,
            })
            .unwrap();
            let v: f64 = crate::volumes(&solid(&w, metres(0.003)).unwrap()).iter().sum();
            let exact = a * b * l + 0.008 * extra * 0.01;
            assert!((v - exact).abs() < 1e-7 * exact, "{v} {exact}");
        }
    }

    /// The area of the faces tagged as port `k`.
    fn port_area(mesh: &Mesh, k: u32) -> f64 {
        mesh.boundary.iter().filter(|f| f.tag == Boundary::Port(k)).map(|f| area(mesh, f)).sum()
    }

    /// Regression: a stub on WR-90's broad wall with its sides in the guide's side walls (an
    /// E-plane tee of two boxes) was refused ("a union whose face is flush with the solid's and
    /// reaches past the solid's face"); so was a box carrying the guide on past its end. Both lie
    /// behind the side walls' planes, so they are grown past them, united and cut back there
    /// (`crate::clip`): the volume is exact and the ports whole.
    #[test]
    fn unions_sharing_the_walls_planes_unite_exactly() {
        use emcraft_model::Feature;
        let (a, b, l) = (0.02286, 0.01016, 0.04);
        let guide = || {
            let mut m = Model::new();
            m.set_solid(Solid { shape: Shape::Box { size: box_size(Vec3::new(a, b, l)) }, material: "vacuum".into() }).unwrap();
            m
        };
        let mut tee = guide();
        tee.add_feature(Feature {
            combine: Combine::Add,
            shape: Shape::Box { size: box_size(Vec3::new(a, 0.01, 0.01)) },
            at: Point3::new(0.0, b, 0.015),
            axis: Axis::Z,
        })
        .unwrap();
        let mesh = solid(&tee, metres(0.004)).unwrap();
        let v: f64 = crate::volumes(&mesh).iter().sum();
        // Exact but for the stub's shift by a millionth of the element size.
        let exact = a * b * l + a * 0.01 * 0.01;
        assert!((v - exact).abs() < 1e-7 * exact, "{v} {exact}");
        assert!((port_area(&mesh, 0) - a * b).abs() < 1e-12 && (port_area(&mesh, 1) - a * b).abs() < 1e-12);
        // Carried on past the end: the end face is inside now, so no port 1 is left there.
        let mut longer = guide();
        longer
            .add_feature(Feature {
                combine: Combine::Add,
                shape: Shape::Box { size: box_size(Vec3::new(a, b, 0.01)) },
                at: Point3::new(0.0, 0.0, l),
                axis: Axis::Z,
            })
            .unwrap();
        let mesh = solid(&longer, metres(0.004)).unwrap();
        let v: f64 = crate::volumes(&mesh).iter().sum();
        assert!((v - a * b * (l + 0.01)).abs() < 1e-7 * v, "{v}");
        assert!((port_area(&mesh, 0) - a * b).abs() < 1e-12 && port_area(&mesh, 1) == 0.0);
        // An L-shaped prism region with a stub on its arm's end against the side wall x = 0:
        // grown past x = 0 and z = 0 and cut back.
        let prism_with = |outline: &[[f64; 2]], at: [f64; 2]| {
            let mut m = Model::new();
            m.set_solid(Solid { shape: Shape::prism(outline, metres(0.01)).unwrap(), material: "vacuum".into() }).unwrap();
            m.add_feature(Feature {
                combine: Combine::Add,
                shape: Shape::Box { size: box_size(Vec3::new(0.01, 0.01, 0.005)) },
                at: Point3::new(at[0], at[1], 0.0),
                axis: Axis::Z,
            })
            .unwrap();
            m
        };
        let ell = [[0.0, 0.0], [0.03, 0.0], [0.03, 0.03], [0.02, 0.03], [0.02, 0.01], [0.0, 0.01]];
        let v: f64 = crate::volumes(&solid(&prism_with(&ell, [0.0, 0.01]), metres(0.004)).unwrap()).iter().sum();
        let exact = 0.03 * 0.01 * 0.01 + 0.01 * 0.02 * 0.01 + 0.01 * 0.01 * 0.005;
        assert!((v - exact).abs() < 1e-6 * exact, "{v} {exact}");
        // A stub whose side lies in a wall's plane where the region reaches past that plane
        // elsewhere (the notch's wall x = 10 mm, the region going on to x = 0 below it): no
        // plane bounds both and no move keeps the union; refused, saying why.
        let notched = [[0.0, 0.0], [0.03, 0.0], [0.03, 0.02], [0.01, 0.02], [0.01, 0.01], [0.0, 0.01]];
        let e = solid(&prism_with(&notched, [0.01, 0.02]), metres(0.004)).unwrap_err().to_string();
        assert!(e.contains("not supported yet") && e.contains("prism"), "{e}");
    }

    /// Regression: an iris plate against three walls of WR-90 meshed at 4 mm, its face at
    /// z = 20 mm on a grid line of the guide's walls (divided every 4 mm), crossed the walls a
    /// millionth of the size from that line; boundary recovery found the slivers' points on
    /// another edge ("the surface is not a valid complex"). The walls' grid line is now moved off
    /// the plate's plane first (`grid`): the volume is exact and the ports whole.
    #[test]
    fn a_face_on_a_wall_grid_line_meshes() {
        use emcraft_model::Feature;
        let (a, b, l) = (0.02286, 0.01016, 0.04);
        for (y, h) in [(0.0, 0.00483), (0.00533, b - 0.00533)] {
            let mut m = Model::new();
            m.set_solid(Solid { shape: Shape::Box { size: box_size(Vec3::new(a, b, l)) }, material: "vacuum".into() }).unwrap();
            m.add_feature(Feature {
                combine: Combine::Subtract,
                shape: Shape::Box { size: box_size(Vec3::new(a, h, 0.001)) },
                at: Point3::new(0.0, y, 0.02),
                axis: Axis::Z,
            })
            .unwrap();
            let mesh = solid(&m, metres(0.004)).unwrap();
            let v: f64 = crate::volumes(&mesh).iter().sum();
            let exact = a * b * l - a * h * 0.001;
            assert!((v - exact).abs() < 1e-7 * exact, "{v} {exact}");
            assert!((port_area(&mesh, 0) - a * b).abs() < 1e-12 && (port_area(&mesh, 1) - a * b).abs() < 1e-12);
        }
    }

    /// Regression: a post from floor to roof of a prism solid (an imported region) was combined
    /// with its caps a millionth of the size from the floor and roof — `feature_body` knows only
    /// a box's and a cylinder's faces — leaving slivers the mesher filled with some 90 000
    /// tetrahedra over a minute and a half. Its caps now move out of the walls as an imported
    /// body's do: a few thousand tetrahedra, the volume that of the post's facets.
    #[test]
    fn a_post_across_a_prism_solid_moves_out_of_its_walls() {
        use emcraft_model::Feature;
        let (a, b, l, r) = (0.02286, 0.01016, 0.04, 0.001);
        let mut m = Model::new();
        m.set_solid(Solid { shape: Shape::prism(&[[0.0, 0.0], [l, 0.0], [l, a], [0.0, a]], metres(b)).unwrap(), material: "vacuum".into() }).unwrap();
        m.add_feature(Feature {
            combine: Combine::Subtract,
            shape: Shape::Cylinder { radius: metres(r), length: metres(b) },
            at: Point3::new(0.02, a / 2.0, 0.0),
            axis: Axis::Z,
        })
        .unwrap();
        let mesh = solid(&m, metres(0.004)).unwrap();
        assert!(mesh.tets.len() < 20_000, "{}", mesh.tets.len());
        let v: f64 = crate::volumes(&mesh).iter().sum();
        // The hole between the post's 12 inscribed facets and its circle (the curved elements'
        // edges bent onto it): the post reaches from wall to wall, no sliver left.
        let (facets, circle) = (0.5 * 12.0 * r * r * (2.0 * PI / 12.0).sin() * b, PI * r * r * b);
        let hole = a * b * l - v;
        assert!(hole > facets * (1.0 - 1e-9) && hole < circle, "{hole} {facets} {circle}");
    }

    /// An imported ramp (a wedge across the guide) standing on the floor and against both side
    /// walls: its foot and ends move out of the walls, its slanted face reaching past the floor
    /// along its own plane, so the air left is exactly the guide less the ramp.
    #[test]
    fn an_imported_ramp_on_the_floor_keeps_its_slope() {
        use emcraft_model::Feature;
        let (a, b, l) = (0.02286, 0.01016, 0.032);
        let (z0, z1, h) = (0.010, 0.020, 0.004);
        let positions = vec![
            Point3::new(0.0, 0.0, z0),
            Point3::new(a, 0.0, z0),
            Point3::new(0.0, 0.0, z1),
            Point3::new(a, 0.0, z1),
            Point3::new(0.0, h, z1),
            Point3::new(a, h, z1),
        ];
        let triangles = vec![[0, 1, 3], [0, 3, 2], [0, 4, 5], [0, 5, 1], [2, 3, 5], [2, 5, 4], [0, 2, 4], [1, 5, 3]];
        let shape = Shape::Surface { name: "Ramp".into(), positions, triangles, tags: vec![0, 0, 1, 1, 2, 2, 3, 4], surfaces: Vec::new() };
        let mut m = Model::new();
        m.set_solid(Solid { shape: imported_box("Guide", [0.0; 3], [a, b, l]), material: "vacuum".into() }).unwrap();
        m.add_feature(Feature { combine: Combine::Subtract, shape, at: Point3::ORIGIN, axis: Axis::Z }).unwrap();
        let mesh = solid(&m, metres(0.003)).unwrap();
        let v: f64 = crate::volumes(&mesh).iter().sum();
        let exact = a * b * l - 0.5 * (z1 - z0) * h * a;
        assert!((v - exact).abs() < 1e-7 * exact, "{v} {exact}");
        let port = |k: u32| mesh.boundary.iter().filter(|f| f.tag == Boundary::Port(k)).map(|f| area(&mesh, f)).sum::<f64>();
        assert!((port(0) - a * b).abs() < 1e-12 && (port(1) - a * b).abs() < 1e-12);
    }

    /// A box [lo, hi] as an imported surface: 12 triangles, a tag per face.
    fn imported_box(name: &str, lo: [f64; 3], hi: [f64; 3]) -> Shape {
        let positions = (0..8).map(|i| Point3::new([lo[0], hi[0]][i & 1], [lo[1], hi[1]][(i >> 1) & 1], [lo[2], hi[2]][(i >> 2) & 1])).collect();
        let quads = [[0, 4, 6, 2], [1, 3, 7, 5], [0, 1, 5, 4], [2, 6, 7, 3], [0, 2, 3, 1], [4, 5, 7, 6]];
        let triangles = quads.iter().flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]).collect();
        Shape::Surface { name: name.into(), positions, triangles, tags: (0..12).map(|i| i / 2).collect(), surfaces: Vec::new() }
    }

    /// Regression: an imported metal body touching the region's walls (an iris plate against a
    /// side wall and both broad walls) was moved by its millionth of the size, leaving slivers of
    /// air a nanometre thick along the walls and flat tetrahedra the solver refused; imported
    /// bodies were also turned about z as if they were round. Now their flush faces move out of
    /// the walls first: the volume is exact, the ports whole, the mesh small and quick.
    #[test]
    fn imported_bodies_touching_the_walls_are_subtracted() {
        use emcraft_model::Feature;
        let (a, b, l) = (0.02286, 0.01016, 0.032);
        for plates in [vec![0.0], vec![0.0, 0.01743]] {
            let mut m = Model::new();
            m.set_solid(Solid { shape: imported_box("Guide", [0.0; 3], [a, b, l]), material: "vacuum".into() }).unwrap();
            for &x in &plates {
                let shape = imported_box("Iris", [x, 0.0, 0.015], [x + 0.00543, b, 0.017]);
                m.add_feature(Feature { combine: Combine::Subtract, shape, at: Point3::ORIGIN, axis: Axis::Z }).unwrap();
            }
            let start = std::time::Instant::now();
            let mesh = solid(&m, metres(0.003)).unwrap();
            assert!(start.elapsed().as_secs() < 20, "{:?}", start.elapsed());
            // The crossings' points on the walls' straight creases collapsed: no tiny segments.
            assert!(mesh.tets.len() < 4000, "{}", mesh.tets.len());
            let v: f64 = crate::volumes(&mesh).iter().sum();
            let exact = a * b * l - plates.len() as f64 * 0.00543 * b * 0.002;
            assert!((v - exact).abs() < 1e-7 * exact, "{v} {exact}");
            let port = |k: u32| mesh.boundary.iter().filter(|f| f.tag == Boundary::Port(k)).map(|f| area(&mesh, f)).sum::<f64>();
            assert!((port(0) - a * b).abs() < 1e-12 && (port(1) - a * b).abs() < 1e-12);
        }
        // A plate a micrometre off the wall leaves a gap no element fits (a sixtieth of the
        // mesher's smallest size): refused at once, naming the gap.
        let start = std::time::Instant::now();
        let e = solid(&plate_off_the_wall(1e-6), metres(0.003)).unwrap_err().to_string();
        assert!(e.contains("gap"), "{e}");
        assert!(start.elapsed().as_secs() < 5, "{:?}", start.elapsed());
    }

    /// WR-90, 32 mm long, less an imported iris plate 2 mm thick against both broad walls,
    /// `gap` (m) off the side wall x = 0.
    fn plate_off_the_wall(gap: f64) -> Model {
        use emcraft_model::Feature;
        let (a, b, l) = (0.02286, 0.01016, 0.032);
        let mut m = Model::new();
        m.set_solid(Solid { shape: imported_box("Guide", [0.0; 3], [a, b, l]), material: "vacuum".into() }).unwrap();
        let shape = imported_box("Iris", [gap, 0.0, 0.015], [0.00543, b, 0.017]);
        m.add_feature(Feature { combine: Combine::Subtract, shape, at: Point3::ORIGIN, axis: Axis::Z }).unwrap();
        m
    }

    /// Regression: a face closer than a tenth of the element size to a parallel one was refused
    /// ("a gap narrower than … is not meshed"). Now the gap is filled with elements about as
    /// wide as it is (`narrow`): the plate 0.2 mm off the side wall at 3 mm meshes, its volume
    /// exact and the gap's elements small.
    #[test]
    fn a_narrow_gap_is_filled_with_small_elements() {
        let (b, gap) = (0.01016, 0.0002);
        let mesh = solid(&plate_off_the_wall(gap), metres(0.003)).unwrap();
        let v: f64 = crate::volumes(&mesh).iter().sum();
        let exact = 0.02286 * b * 0.032 - (0.00543 - gap) * b * 0.002;
        assert!((v - exact).abs() < 1e-7 * exact, "{v} {exact}");
        // Every tetrahedron in the gap (its centroid there, away from its ends) no longer than a
        // few gap widths; at the uniform size they would be ten times that.
        let longest = mesh
            .tets
            .iter()
            .map(|t| t.map(|n| mesh.nodes[n as usize]))
            .filter(|p| {
                let c = (0..3).map(|k| p.iter().map(|q| q[k]).sum::<f64>() / 4.0).collect::<Vec<_>>();
                c[0] < gap && c[2] > 0.0152 && c[2] < 0.0168 && c[1] > 0.0005 && c[1] < b - 0.0005
            })
            .map(|p| (0..4).flat_map(|i| (i + 1..4).map(move |j| (i, j))).map(|(i, j)| p[i].distance(p[j])).fold(0.0, f64::max))
            .fold(0.0, f64::max);
        assert!(longest > 0.0 && longest < 6.0 * gap, "{longest}");
    }

    /// A coaxial line's inner conductor in two pieces meeting end to end: the solid (radius
    /// 3.5 mm, 20 mm long) less two cylinders of radius 1 mm and `r2`, at z 0..10 and 10..20 mm.
    fn stepped_coax(r2: f64) -> Model {
        use emcraft_model::Feature;
        let mut m = Model::new();
        m.set_solid(Solid { shape: Shape::Cylinder { radius: metres(0.0035), length: metres(0.02) }, material: "vacuum".into() }).unwrap();
        for (radius, z) in [(0.001, 0.0), (r2, 0.01)] {
            m.add_feature(Feature {
                combine: Combine::Subtract,
                shape: Shape::Cylinder { radius: metres(radius), length: metres(0.01) },
                at: Point3::new(0.0, 0.0, z),
                axis: Axis::Z,
            })
            .unwrap();
        }
        m
    }

    /// Regression: the two cylinders' facing caps, each moved by its own millionth of the size,
    /// left two boundary parts a nanometre apart, and boundary edge recovery split the same few
    /// edges a hair at a time for many minutes. Combined as they are (not joined first), they
    /// now stop at the pass cap with an error, in seconds.
    #[test]
    fn nearly_touching_boundaries_stop_quickly() {
        let (m, size) = (stepped_coax(0.001), 0.0015);
        let s = m.domain.solid.as_ref().unwrap();
        let main = combine_all(tessellate(&s.shape, size), m.features(), &s.shape, size).unwrap();
        let start = std::time::Instant::now();
        let e = region(&main, size, Vec::new(), &[]).err().expect("nearly touching boundaries are refused");
        assert!(e.to_string().contains("nearly touch"), "{e}");
        assert!(start.elapsed().as_secs() < 30, "{:?}", start.elapsed());
    }

    /// The stepped coaxial line meshes within seconds, its volume the exact one but for the
    /// flat facets on the round walls, both ends whole ports.
    fn stepped_coax_meshes(r2: f64, size: f64) {
        let start = std::time::Instant::now();
        let mesh = solid(&stepped_coax(r2), metres(size)).unwrap_or_else(|e| panic!("r2 {r2}: {e}"));
        // A bound against a runaway, not a benchmark: CI's shared runners take about 60 s for the
        // finer hole where a desktop takes a fraction of it.
        assert!(start.elapsed().as_secs() < 240, "r2 {r2}: {:?}", start.elapsed());
        let v: f64 = crate::volumes(&mesh).iter().sum();
        let hole = |r: f64| PI * r * r * 0.01;
        let exact = PI * 0.0035f64.powi(2) * 0.02 - hole(0.001) - hole(r2);
        assert!((v - exact).abs() < 0.02 * exact, "r2 {r2}: {v} {exact}");
        let port = |k: u32| mesh.boundary.iter().filter(|f| f.tag == Boundary::Port(k)).map(|f| area(&mesh, f)).sum::<f64>();
        assert!(port(0) > 0.0 && port(1) > 0.0, "r2 {r2}");
    }

    /// Regression: features meeting end to end (a stepped inner conductor) used to hang the
    /// mesher. Now with the same cross-section they are one feature; with one inside the other,
    /// the smaller reaches into the larger; overlapping otherwise, they are refused.
    #[test]
    fn features_meeting_end_to_end_mesh() {
        stepped_coax_meshes(0.0015, 0.0015);
        // Boxes face to face: the same rectangle becomes one box.
        use emcraft_model::Feature;
        let mut m = Model::new();
        m.set_solid(Solid { shape: Shape::Box { size: box_size(Vec3::new(0.02, 0.01, 0.03)) }, material: "vacuum".into() }).unwrap();
        for z in [0.005, 0.01] {
            let f = Feature {
                combine: Combine::Subtract,
                shape: Shape::Box { size: box_size(Vec3::new(0.004, 0.004, 0.005)) },
                at: Point3::new(0.008, 0.003, z),
                axis: Axis::Z,
            };
            m.add_feature(f).unwrap();
        }
        let v: f64 = crate::volumes(&solid(&m, metres(0.003)).unwrap()).iter().sum();
        let exact = 0.02 * 0.01 * 0.03 - 0.004 * 0.004 * 0.01;
        assert!((v - exact).abs() < 1e-7 * exact, "{v} {exact}");
        // Cross-sections that overlap without one holding the other are refused at once.
        let mut w = Model::new();
        w.set_solid(Solid { shape: Shape::Box { size: box_size(Vec3::new(0.02, 0.01, 0.03)) }, material: "vacuum".into() }).unwrap();
        for (x, z) in [(0.008, 0.005), (0.010, 0.01)] {
            let f = Feature {
                combine: Combine::Subtract,
                shape: Shape::Box { size: box_size(Vec3::new(0.004, 0.004, 0.005)) },
                at: Point3::new(x, 0.003, z),
                axis: Axis::Z,
            };
            w.add_feature(f).unwrap();
        }
        let e = solid(&w, metres(0.003)).expect_err("overlapping sections are refused");
        assert!(e.to_string().contains("reach into"), "{e}");
    }

    #[test]
    #[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
    fn features_meeting_end_to_end_mesh_heavy() {
        let size = 0.0015;
        for r2 in [0.001, 0.0005] {
            stepped_coax_meshes(r2, size);
        }
        // One radius: the same mesh as one hole 20 mm long.
        let mut one = stepped_coax(0.001);
        let first = one.mesh_view().feature_body[0];
        let mut bodies = one.bodies_mut();
        let kept: Vec<emcraft_model::Body> =
            bodies.iter().enumerate().filter(|(k, b)| *k == first || b.feature().is_none()).map(|(_, b)| b.clone()).collect();
        *bodies = kept;
        if let Some(Shape::Cylinder { length, .. }) = bodies.iter_mut().find(|b| b.feature().is_some()).map(|b| &mut b.shape) {
            *length = metres(0.02);
        }
        drop(bodies);
        let digest = |m: &Model| {
            let mesh = solid(m, metres(size)).unwrap();
            (mesh.nodes.len(), mesh.tets.len(), crate::volumes(&mesh).iter().sum::<f64>())
        };
        assert_eq!(digest(&stepped_coax(0.001)), digest(&one));
    }

    #[test]
    fn points_are_located_in_unstructured_meshes() {
        let mesh = mesh_of(Shape::Cylinder { radius: metres(0.01), length: metres(0.02) }, 0.003);
        for p in [[0.0, 0.0, 0.01], [0.007, 0.0, 0.001], [-0.003, 0.004, 0.019]].map(Point3::from) {
            let (t, l) = crate::locate(&mesh, p).unwrap();
            assert!(l.iter().all(|x| *x >= -1e-9), "{l:?}");
            let c = mesh.corners(t).unwrap();
            for d in 0..3 {
                let q: f64 = (0..4).map(|i| l[i] * c[i][d]).sum();
                approx::assert_abs_diff_eq!(q, p[d], epsilon = 1e-12);
            }
        }
        assert!(crate::locate(&mesh, Point3::new(0.02, 0.0, 0.01)).is_none());
    }

    /// The post through WR-90 of the driven gate (`a_post_across_the_guide_conserves_power`).
    fn post_model() -> Model {
        use emcraft_model::Feature;
        let (a, b, l) = (0.02286, 0.01016, 0.03);
        let mut m = Model::new();
        m.set_solid(Solid { shape: Shape::Box { size: box_size(Vec3::new(a, b, l)) }, material: "vacuum".into() }).unwrap();
        m.add_feature(Feature {
            combine: Combine::Subtract,
            shape: Shape::Cylinder { radius: metres(0.0015), length: metres(b) },
            at: Point3::new(a / 2.0, 0.0, l / 2.0),
            axis: Axis::Y,
        })
        .unwrap();
        m
    }

    /// A digest of a mesh's nodes and tetrahedra, bit for bit (FNV-1a).
    fn digest(mesh: &Mesh) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |x: u64| {
            for b in x.to_le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for p in &mesh.nodes {
            for c in [p.x, p.y, p.z] {
                eat(c.to_bits());
            }
        }
        for t in &mesh.tets {
            for v in t {
                eat(u64::from(*v));
            }
        }
        h
    }

    /// Times the post at 4 mm (the ROADMAP's quality-stage case): `cargo test --release -p
    /// emcraft-mesh post_mesh_timing -- --ignored --nocapture`.
    #[test]
    #[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
    fn post_mesh_timing() {
        // The process's CPU time (user + system, Linux), steadier than the wall clock on a busy
        // machine; 0 where /proc is missing.
        let cpu = || -> f64 {
            let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
            let fields: Vec<&str> = stat.rsplit(')').next().unwrap_or("").split_whitespace().collect();
            let tick = |i: usize| fields.get(i).and_then(|f| f.parse::<f64>().ok()).unwrap_or(0.0);
            (tick(11) + tick(12)) / 100.0
        };
        let m = post_model();
        for h in [0.004, 0.003] {
            let (t, c) = (std::time::Instant::now(), cpu());
            let mesh = solid(&m, metres(h)).unwrap();
            let (s, c) = (t.elapsed().as_secs_f64(), cpu() - c);
            eprintln!(
                "post at {} mm: {s:.2} s ({c:.2} s CPU), {} tetrahedra, {} nodes, digest {:016x}",
                h * 1e3,
                mesh.tets.len(),
                mesh.nodes.len(),
                digest(&mesh)
            );
        }
    }

    /// A unit cube whose four sides are split at mid-height, as a design's tall sides may be:
    /// stopped at that height, or at a third, it is the part below, closed (what is above brought
    /// down onto the plane, the flattened rows gone); it is no plain extrusion when a vertex on
    /// the plane lies on no line along z through the top's.
    #[test]
    fn an_extrusion_split_at_the_plane_is_stopped_there() {
        // Corners k (x = k & 1, y = k >> 1 & 1) at z = 0, ½ and 1: index k + 4 level.
        let positions: Vec<Point3> = (0..12).map(|i| Point3::new((i & 1) as f64, ((i >> 1) & 1) as f64, (i / 4) as f64 * 0.5)).collect();
        let mut triangles: Vec<[u32; 3]> = vec![[0, 2, 3], [0, 3, 1], [8, 9, 11], [8, 11, 10]];
        // Each side, counter-clockwise from outside, its two rows.
        for (a, b) in [(0, 1), (1, 3), (3, 2), (2, 0)] {
            for level in 0..2 {
                let (p, q, r, s) = (a + 4 * level, b + 4 * level, b + 4 * (level + 1), a + 4 * (level + 1));
                triangles.extend([[p, q, r], [p, r, s]]);
            }
        }
        let tags = vec![0; triangles.len()];
        let cube = Shape::Surface { name: "cube".into(), positions, triangles, tags, surfaces: Vec::new() };
        let Some(Shape::Surface { positions, triangles, .. }) = shortened(&cube, 2, 0.5, 1e-9) else { panic!("not stopped") };
        assert_eq!((positions.len(), triangles.len()), (8, 12));
        assert!(crate::shared::closed(&triangles) && positions.iter().all(|p| p.z <= 0.5));
        let Some(Shape::Surface { positions, triangles, .. }) = shortened(&cube, 2, 1.0 / 3.0, 1e-9) else { panic!("not stopped") };
        assert!(crate::shared::closed(&triangles) && positions.iter().all(|p| p.z <= 1.0 / 3.0) && positions.len() == 8);
        let Shape::Surface { mut positions, triangles, tags, .. } = cube else { panic!("a surface") };
        positions[5].x = 0.9;
        let moved = Shape::Surface { name: "cube".into(), positions, triangles, tags, surfaces: Vec::new() };
        assert!(shortened(&moved, 2, 0.5, 1e-9).is_none());
        // A cut of no tolerance (a box of no size) stops nothing rather than welding every vertex.
        assert!(shortened(&moved, 2, 0.5, 0.0).is_none());
    }
}
