//! The Cartesian particle-in-cell run's kernels (`emcraft_fdtd::pic`, whose docs give the Yee
//! lattices, the shapes and the scheme): the particles pushed and their current deposited by
//! Esirkepov's scheme, one unit a particle; the current turned back from fixed point and
//! smoothed; B's half steps and E's step a component at a time, one unit a point; the probes;
//! the particles' kinetic energy summed in a fixed tree. The current is summed in fixed point
//! (`kernels::particles`).
//!
//! Lattices (`emcraft_fdtd::pic::device::Plan`'s order), each x fastest: E_a at the cells'
//! centres along a and the nodes across, B_a at the nodes along a and the centres across; an axis
//! between conducting walls has cells + 1 nodes, a periodic one as many as cells.

use cubecl::prelude::*;

use crate::buffer::Real;
use crate::kernels::particles::{Deaths, U, add, finite, gamma, kinetic as kinetic_of, leave, push, shape, take};

/// The box: cells along each axis, whether each is periodic (1) or closed by conducting walls,
/// the cells' sizes (m), the step (s) and the speed of light (m/s).
#[derive(CubeLaunch, CubeType)]
pub(crate) struct Grid<F: Float> {
    pub(crate) nx: u32,
    pub(crate) ny: u32,
    pub(crate) nz: u32,
    pub(crate) px: u32,
    pub(crate) py: u32,
    pub(crate) pz: u32,
    pub(crate) hx: F,
    pub(crate) hy: F,
    pub(crate) hz: F,
    pub(crate) dt: F,
    pub(crate) c: F,
}

/// The six components.
#[derive(CubeLaunch, CubeType)]
pub(crate) struct Fields<F: Float> {
    pub(crate) ex: Box<[F]>,
    pub(crate) ey: Box<[F]>,
    pub(crate) ez: Box<[F]>,
    pub(crate) bx: Box<[F]>,
    pub(crate) by: Box<[F]>,
    pub(crate) bz: Box<[F]>,
}

/// The steady B the run started with (on B's lattices; T), held apart from B's changes, which
/// the device's fields hold: a strong field held with them would leave single precision's last
/// digits to its changes.
#[derive(CubeLaunch, CubeType)]
pub(crate) struct Steady<F: Float> {
    pub(crate) bx: Box<[F]>,
    pub(crate) by: Box<[F]>,
    pub(crate) bz: Box<[F]>,
}

/// The current's fixed-point sums on E's lattices, and their scales.
#[derive(CubeLaunch, CubeType)]
pub(crate) struct Current<F: Float> {
    pub(crate) x: Box<[Atomic<u64>]>,
    pub(crate) y: Box<[Atomic<u64>]>,
    pub(crate) z: Box<[Atomic<u64>]>,
    pub(crate) scale_x: F,
    pub(crate) scale_y: F,
    pub(crate) scale_z: F,
}

/// The particles: position (m), momentum γv (m/s), weight, the step each is born at, whether it
/// is still in the run (1 or 0) and its kind.
#[derive(CubeLaunch, CubeType)]
pub(crate) struct Particles<F: Float> {
    pub(crate) x: Box<[F]>,
    pub(crate) y: Box<[F]>,
    pub(crate) z: Box<[F]>,
    pub(crate) ux: Box<[F]>,
    pub(crate) uy: Box<[F]>,
    pub(crate) uz: Box<[F]>,
    pub(crate) w: Box<[F]>,
    pub(crate) born: Box<[u32]>,
    pub(crate) alive: Box<[u32]>,
    pub(crate) kind: Box<[u32]>,
}

/// Each kind's q Δt/m, charge (C) and rest energy (J).
#[derive(CubeLaunch, CubeType)]
pub(crate) struct Kinds<F: Float> {
    pub(crate) kick: Box<[F]>,
    pub(crate) charge: Box<[F]>,
    pub(crate) rest: Box<[F]>,
}

/// The absorbing layers' (b, a) pairs, each axis's at its nodes and at its cells' centres laid
/// end to end: those of axis d at `start[2d]` (nodes) and `start[2d + 1]` (centres).
#[derive(CubeLaunch, CubeType)]
pub(crate) struct Grading<F: Float> {
    pub(crate) ba: Box<[F]>,
    pub(crate) start: Box<[u32]>,
}

/// A point of a lattice, or a lattice's points along each axis.
#[derive(CubeType)]
pub(crate) struct P {
    i: u32,
    j: u32,
    k: u32,
}

/// Cells along axis `d`.
#[cube]
fn cells<F: Real>(g: &Grid<F>, #[comptime] d: u32) -> u32 {
    let mut n = g.nz;
    if comptime![d == 0] {
        n = g.nx;
    } else if comptime![d == 1] {
        n = g.ny;
    }
    n
}

/// Whether axis `d` is periodic.
#[cube]
fn periodic<F: Real>(g: &Grid<F>, #[comptime] d: u32) -> bool {
    let mut p = g.pz;
    if comptime![d == 0] {
        p = g.px;
    } else if comptime![d == 1] {
        p = g.py;
    }
    p == 1
}

/// The cells' size along axis `d` (m).
#[cube]
fn size<F: Real>(g: &Grid<F>, #[comptime] d: u32) -> F {
    let mut h = g.hz;
    if comptime![d == 0] {
        h = g.hx;
    } else if comptime![d == 1] {
        h = g.hy;
    }
    h
}

/// A lattice's points along axis `d`: its nodes (`node`) or its cells' centres.
#[cube]
fn points<F: Real>(g: &Grid<F>, #[comptime] d: u32, #[comptime] node: bool) -> u32 {
    let n = cells::<F>(g, d);
    let mut out = n;
    if node && !periodic::<F>(g, d) {
        out = n + 1;
    }
    out
}

/// Point `i` along axis `d` wrapped round a periodic axis; −1 past a wall.
#[cube]
fn wrap<F: Real>(g: &Grid<F>, #[comptime] d: u32, i: i32, #[comptime] node: bool) -> i32 {
    let n = points::<F>(g, d, node) as i32;
    let mut out = i32::new(-1);
    if periodic::<F>(g, d) {
        out = ((i % n) + n) % n;
    } else if i >= 0 && i < n {
        out = i;
    }
    out
}

/// Point `i` along axis `d` wrapped round a periodic axis, the nearest past a wall.
#[cube]
fn nearest<F: Real>(g: &Grid<F>, #[comptime] d: u32, i: i32, #[comptime] node: bool) -> u32 {
    let n = points::<F>(g, d, node) as i32;
    let mut out = clamp(i, 0, n - 1);
    if periodic::<F>(g, d) {
        out = ((i % n) + n) % n;
    }
    out as u32
}

/// The points along each axis of the lattice on the nodes of the axes `n0`, `n1`, `n2`.
#[cube]
fn dims<F: Real>(g: &Grid<F>, #[comptime] n0: bool, #[comptime] n1: bool, #[comptime] n2: bool) -> P {
    P { i: points::<F>(g, 0u32, n0), j: points::<F>(g, 1u32, n1), k: points::<F>(g, 2u32, n2) }
}

/// The index of point `p` of a lattice of `d` points.
#[cube]
fn index(p: &P, d: &P) -> usize {
    (p.i + d.i * (p.j + d.j * p.k)) as usize
}

/// Point `n` of a lattice of `d` points.
#[cube]
fn point_of(n: u32, d: &P) -> P {
    P { i: n % d.i, j: (n / d.i) % d.j, k: n / (d.i * d.j) }
}

/// Component `d` of point `p`.
#[cube]
fn get(p: &P, #[comptime] d: u32) -> u32 {
    let mut v = p.k;
    if comptime![d == 0] {
        v = p.i;
    } else if comptime![d == 1] {
        v = p.j;
    }
    v
}

/// Point `p` with its component `d` set to `v`.
#[cube]
fn with(p: &P, #[comptime] d: u32, v: u32) -> P {
    let mut out = P { i: p.i, j: p.j, k: p.k };
    if comptime![d == 0] {
        out.i = v;
    } else if comptime![d == 1] {
        out.j = v;
    } else {
        out.k = v;
    }
    out
}

/// The point whose components along `a`, `b` and `c` (a permutation of the axes) are `va`, `vb`
/// and `vc`.
#[cube]
fn arrange(#[comptime] a: u32, #[comptime] b: u32, va: u32, vb: u32, vc: u32) -> P {
    let p = P { i: vc, j: vc, k: vc };
    let p = with(&p, a, va);
    with(&p, b, vb)
}

/// Component `d` of `u`.
#[cube]
fn of<F: Real>(u: &U<F>, #[comptime] d: u32) -> F {
    let mut v = u.z;
    if comptime![d == 0] {
        v = u.x;
    } else if comptime![d == 1] {
        v = u.y;
    }
    v
}

/// `values` on the lattice on the nodes of `n0`, `n1`, `n2` at `x` (m), linearly along each axis
/// (the particles' first-order shape); past a wall, the nearest point's.
#[cube]
fn gather<F: Real>(values: &[F], g: &Grid<F>, x: &U<F>, #[comptime] n0: bool, #[comptime] n1: bool, #[comptime] n2: bool) -> F {
    let one = F::new(1.0_f32);
    let half = F::new(0.5_f32);
    let d = dims::<F>(g, n0, n1, n2);
    let mut sx = x.x / g.hx;
    let mut sy = x.y / g.hy;
    let mut sz = x.z / g.hz;
    if !n0 {
        sx -= half;
    }
    if !n1 {
        sy -= half;
    }
    if !n2 {
        sz -= half;
    }
    let (fx, fy, fz) = (sx.floor(), sy.floor(), sz.floor());
    let (wx, wy, wz) = (sx - fx, sy - fy, sz - fz);
    let (ix, iy, iz) = (i32::cast_from(fx), i32::cast_from(fy), i32::cast_from(fz));
    let mut sum = F::new(0.0_f32);
    #[unroll]
    for k in 0..2 {
        let pk = nearest::<F>(g, 2u32, iz + k, n2);
        let w2 = if k == 0 { one - wz } else { wz };
        #[unroll]
        for j in 0..2 {
            let pj = nearest::<F>(g, 1u32, iy + j, n1);
            let w1 = if j == 0 { one - wy } else { wy };
            #[unroll]
            for i in 0..2 {
                let pi = nearest::<F>(g, 0u32, ix + i, n0);
                let w0 = if i == 0 { one - wx } else { wx };
                sum += w0 * w1 * w2 * values[index(&P { i: pi, j: pj, k: pk }, &d)];
            }
        }
    }
    sum
}

/// E at `x` (m).
#[cube]
fn e_at<F: Real>(f: &Fields<F>, g: &Grid<F>, x: &U<F>) -> U<F> {
    U::<F> {
        x: gather::<F>(&f.ex, g, x, false, true, true),
        y: gather::<F>(&f.ey, g, x, true, false, true),
        z: gather::<F>(&f.ez, g, x, true, true, false),
    }
}

/// The current of charge `q` (C) along axis `a` (`b`, `c` the two across) moving from `s0` to
/// `s1` (in cells, each window starting at `first`): Esirkepov's running sums along a, each
/// added to J_a (`emcraft_fdtd::pic::deposit::current`).
#[cube]
fn deposit_along<F: Real>(
    sum: &[Atomic<u64>],
    scale: F,
    g: &Grid<F>,
    s0: &U<F>,
    s1: &U<F>,
    first: &P,
    q: F,
    #[comptime] a: u32,
    #[comptime] b: u32,
    #[comptime] c: u32,
) {
    let half = F::new(0.5_f32);
    let third = F::new(3.0_f32);
    let zero = F::new(0.0_f32);
    let factor = -q / (g.dt * size::<F>(g, b) * size::<F>(g, c));
    let fa = get(first, a) as i32 - 1;
    let fb = get(first, b) as i32 - 1;
    let fc = get(first, c) as i32 - 1;
    let d = dims::<F>(g, comptime![a != 0], comptime![a != 1], comptime![a != 2]);
    #[unroll]
    for mb in 0..4 {
        let at_b = F::cast_from(fb + mb);
        let s0b = shape::<F>(of::<F>(s0, b) - at_b);
        let dsb = shape::<F>(of::<F>(s1, b) - at_b) - s0b;
        let pb = wrap::<F>(g, b, fb + mb, true);
        #[unroll]
        for mc in 0..4 {
            let at_c = F::cast_from(fc + mc);
            let s0c = shape::<F>(of::<F>(s0, c) - at_c);
            let dsc = shape::<F>(of::<F>(s1, c) - at_c) - s0c;
            let across = s0b * s0c + half * dsb * s0c + half * s0b * dsc + dsb * dsc / third;
            let pc = wrap::<F>(g, c, fc + mc, true);
            if across != zero && pb >= 0 && pc >= 0 {
                let mut run = zero;
                // The edge above node m along a; the last one's sum is zero (Σ ΔS = 0).
                #[unroll]
                for ma in 0..3 {
                    let at_a = F::cast_from(fa + ma);
                    let s0a = shape::<F>(of::<F>(s0, a) - at_a);
                    let dsa = shape::<F>(of::<F>(s1, a) - at_a) - s0a;
                    run += factor * dsa * across;
                    let pa = wrap::<F>(g, a, fa + ma, false);
                    if pa >= 0 {
                        let p = arrange(a, b, pa as u32, pb as u32, pc as u32);
                        add::<F>(sum, index(&p, &d) as u32, run, scale);
                    }
                }
            }
        }
    }
}

/// Whether the cell holding `x` (m) is metal.
#[cube]
fn in_metal<F: Real>(metal: &[u32], g: &Grid<F>, x: &U<F>, #[comptime] any: bool) -> bool {
    let mut out = false;
    if any {
        let p = P {
            i: nearest::<F>(g, 0u32, i32::cast_from((x.x / g.hx).floor()), false),
            j: nearest::<F>(g, 1u32, i32::cast_from((x.y / g.hy).floor()), false),
            k: nearest::<F>(g, 2u32, i32::cast_from((x.z / g.hz).floor()), false),
        };
        out = metal[index(&p, &dims::<F>(g, false, false, false))] == 1;
    }
    out
}

/// A move along axis `d` from `x0` to `x1` (m): the fraction of it before it meets a conducting
/// wall, 2 when it does not.
#[cube]
fn meets<F: Real>(g: &Grid<F>, x0: F, x1: F, #[comptime] d: u32) -> F {
    let zero = F::new(0.0_f32);
    let length = F::cast_from(cells::<F>(g, d)) * size::<F>(g, d);
    let mut f = F::new(2.0_f32);
    if !periodic::<F>(g, d) {
        if x1 < zero {
            f = x0 / (x0 - x1);
        } else if x1 > length {
            f = (length - x0) / (x1 - x0);
        }
    }
    f
}

/// Whether a move along axis `d` ending at `x1` (m) goes past its high end.
#[cube]
fn past_high<F: Real>(g: &Grid<F>, x1: F, #[comptime] d: u32) -> bool {
    x1 > F::cast_from(cells::<F>(g, d)) * size::<F>(g, d)
}

/// Where a move along axis `d` ends: `x0` + f (`x1` − `x0`), inside conducting walls, on the
/// wall it hit (`hit`, `high`).
#[cube]
fn ends<F: Real>(g: &Grid<F>, x0: F, x1: F, f: F, hit: bool, high: bool, #[comptime] d: u32) -> F {
    let zero = F::new(0.0_f32);
    let length = F::cast_from(cells::<F>(g, d)) * size::<F>(g, d);
    let mut to = x0 + f * (x1 - x0);
    if !periodic::<F>(g, d) {
        to = F::clamp(to, zero, length);
    }
    if hit {
        to = if high { length } else { zero };
    }
    to
}

/// A position along axis `d` back inside a periodic box.
#[cube]
fn inside<F: Real>(g: &Grid<F>, x: F, #[comptime] d: u32) -> F {
    let mut out = x;
    if periodic::<F>(g, d) {
        let length = F::cast_from(cells::<F>(g, d)) * size::<F>(g, d);
        out = x - length * (x / length).floor();
    }
    out
}

/// Particle `ABSOLUTE_POS` (of `len`) pushed and moved a step (`step`), its current added, its
/// leaving written to `deaths` (`emcraft_fdtd::pic`'s `advance`): how 1 + 2 × axis + (1 at the
/// high end) at a wall, 7 in the metal; its charge (C) as its weight.
#[cube(launch)]
pub(crate) fn push_particles<F: Real>(
    mut p: Particles<F>,
    fields: &Fields<F>,
    steady: &Steady<F>,
    g: &Grid<F>,
    kinds: &Kinds<F>,
    metal: &[u32],
    j: &Current<F>,
    mut deaths: Deaths<F>,
    step: u32,
    len: u32,
    #[comptime] vay: bool,
    #[comptime] any_metal: bool,
    #[comptime] any_steady: bool,
) {
    let n = ABSOLUTE_POS as u32;
    if n < len && p.alive[n as usize] == 1 && p.born[n as usize] <= step {
        let id = n as usize;
        let zero = F::new(0.0_f32);
        let one = F::new(1.0_f32);
        let kind = p.kind[id] as usize;
        let w = p.w[id];
        let x0 = U::<F> { x: p.x[id], y: p.y[id], z: p.z[id] };
        let e = e_at::<F>(fields, g, &x0);
        let mut b = U::<F> {
            x: gather::<F>(&fields.bx, g, &x0, true, false, false),
            y: gather::<F>(&fields.by, g, &x0, false, true, false),
            z: gather::<F>(&fields.bz, g, &x0, false, false, true),
        };
        if any_steady {
            b.x += gather::<F>(&steady.bx, g, &x0, true, false, false);
            b.y += gather::<F>(&steady.by, g, &x0, false, true, false);
            b.z += gather::<F>(&steady.bz, g, &x0, false, false, true);
        }
        let u = push::<F>(U::<F> { x: p.ux[id], y: p.uy[id], z: p.uz[id] }, e, b, kinds.kick[kind], g.c, vay);
        if !finite::<F>(&u) {
            leave::<F>(&mut deaths, step, n, 0u32, zero, zero);
            p.alive[id] = 0u32;
        } else {
            let gu = gamma::<F>(u.x, u.y, u.z, g.c);
            let x1 = U::<F> { x: x0.x + u.x / gu * g.dt, y: x0.y + u.y / gu * g.dt, z: x0.z + u.z / gu * g.dt };
            // The first conducting wall the move meets, in the axes' order.
            let f0 = meets::<F>(g, x0.x, x1.x, 0u32);
            let f1 = meets::<F>(g, x0.y, x1.y, 1u32);
            let f2 = meets::<F>(g, x0.z, x1.z, 2u32);
            let mut f = one;
            let mut hit = 0u32;
            let mut high = false;
            if f0 < f {
                f = F::max(f0, zero);
                hit = 1u32;
                high = past_high::<F>(g, x1.x, 0u32);
            }
            if f1 < f {
                f = F::max(f1, zero);
                hit = 2u32;
                high = past_high::<F>(g, x1.y, 1u32);
            }
            if f2 < f {
                f = F::max(f2, zero);
                hit = 3u32;
                high = past_high::<F>(g, x1.z, 2u32);
            }
            let to = U::<F> {
                x: ends::<F>(g, x0.x, x1.x, f, hit == 1, high, 0u32),
                y: ends::<F>(g, x0.y, x1.y, f, hit == 2, high, 1u32),
                z: ends::<F>(g, x0.z, x1.z, f, hit == 3, high, 2u32),
            };
            let q = kinds.charge[kind] * w;
            // Esirkepov's windows, in cells: from the node below the old cell (the cell's own
            // node here, a position never below 0).
            let s0 = U::<F> { x: x0.x / g.hx, y: x0.y / g.hy, z: x0.z / g.hz };
            let s1 = U::<F> { x: to.x / g.hx, y: to.y / g.hy, z: to.z / g.hz };
            let first = P { i: u32::cast_from(s0.x.floor()), j: u32::cast_from(s0.y.floor()), k: u32::cast_from(s0.z.floor()) };
            deposit_along::<F>(&j.x, j.scale_x, g, &s0, &s1, &first, q, 0u32, 1u32, 2u32);
            deposit_along::<F>(&j.y, j.scale_y, g, &s0, &s1, &first, q, 1u32, 2u32, 0u32);
            deposit_along::<F>(&j.z, j.scale_z, g, &s0, &s1, &first, q, 2u32, 0u32, 1u32);
            let mut how = 0u32;
            if hit != 0 {
                how = 2 * hit - 1;
                if high {
                    how += 1;
                }
            } else if in_metal::<F>(metal, g, &to, any_metal) {
                how = 7u32;
            }
            if how != 0 {
                let kinetic = kinetic_of::<F>(&u, w, kinds.rest[kind], g.c);
                leave::<F>(&mut deaths, step, n, how, q, kinetic);
                p.alive[id] = 0u32;
            } else {
                p.x[id] = inside::<F>(g, to.x, 0u32);
                p.y[id] = inside::<F>(g, to.y, 1u32);
                p.z[id] = inside::<F>(g, to.z, 2u32);
                p.ux[id] = u.x;
                p.uy[id] = u.y;
                p.uz[id] = u.z;
            }
        }
    }
}

/// J_a's fixed-point sums (`sum`, `scale`) turned into J (A/m²) in `j`, the sums cleared; one
/// unit per point of E_a's lattice.
#[cube(launch)]
pub(crate) fn settle<F: Real>(sum: &[Atomic<u64>], j: &mut [F], scale: F, count: u32) {
    let n = ABSOLUTE_POS as u32;
    if n < count {
        j[n as usize] = take::<F>(sum, n as usize, scale);
    }
}

/// One axis's binomial passes over J_a (`emcraft_fdtd::pic`'s `filter::smooth`): each point of
/// E_a's lattice the stencil `weights` (2k + 1 of them) over its neighbours along `d`, zero past
/// a wall; from `from` into `to`.
#[cube(launch)]
pub(crate) fn smooth<F: Real>(from: &[F], to: &mut [F], weights: &[F], width: u32, g: &Grid<F>, #[comptime] a: u32, #[comptime] d: u32) {
    let n = ABSOLUTE_POS as u32;
    let lat = dims::<F>(g, comptime![a != 0], comptime![a != 1], comptime![a != 2]);
    if n < lat.i * lat.j * lat.k {
        let p = point_of(n, &lat);
        let k = (width / 2) as i32;
        let at = get(&p, d) as i32;
        let mut sum = F::new(0.0_f32);
        let mut t = 0u32;
        while t < width {
            let m = wrap::<F>(g, d, at + t as i32 - k, comptime![a != d]);
            if m >= 0 {
                sum += weights[t as usize] * from[index(&with(&p, d, m as u32), &lat)];
            }
            t += 1;
        }
        to[n as usize] = sum;
    }
}

/// A difference along axis `d` through the layers at point `i` of its nodes (`node`) or
/// centres: d + ψ, ψ advanced by the (b, a) there (`emcraft_fdtd::pic::fields`' `stretch`;
/// outside the layers b = 1, a = 0 and ψ stays zero).
#[cube]
fn stretch<F: Real>(
    grading: &Grading<F>,
    psi: &mut [F],
    id: usize,
    i: u32,
    diff: F,
    #[comptime] d: u32,
    #[comptime] node: bool,
    #[comptime] absorbing: bool,
) -> F {
    let mut out = diff;
    if absorbing {
        let mut at = grading.start[(2 * d + 1) as usize] + i;
        if node {
            at = grading.start[(2 * d) as usize] + i;
        }
        let s = grading.ba[(2 * at) as usize] * psi[id] + grading.ba[(2 * at + 1) as usize] * diff;
        psi[id] = s;
        out = diff + s;
    }
    out
}

/// B_a advanced by `dt` (s): ∂B_a/∂t = −(∂_q E_r − ∂_r E_q), (a, q, r) cyclic, one unit per point
/// of B_a's lattice; those whose differences reach past a wall keep theirs.
#[cube(launch)]
pub(crate) fn faraday<F: Real>(
    b: &mut [F],
    eq: &[F],
    er: &[F],
    psi_q: &mut [F],
    psi_r: &mut [F],
    grading: &Grading<F>,
    g: &Grid<F>,
    dt: F,
    #[comptime] a: u32,
    #[comptime] q: u32,
    #[comptime] r: u32,
    #[comptime] absorbing: bool,
) {
    let n = ABSOLUTE_POS as u32;
    let lat = dims::<F>(g, comptime![a == 0], comptime![a == 1], comptime![a == 2]);
    if n < lat.i * lat.j * lat.k {
        let p = point_of(n, &lat);
        let nq = wrap::<F>(g, q, get(&p, q) as i32 + 1, true);
        let nr = wrap::<F>(g, r, get(&p, r) as i32 + 1, true);
        if nq >= 0 && nr >= 0 {
            let lr = dims::<F>(g, comptime![r != 0], comptime![r != 1], comptime![r != 2]);
            let lq = dims::<F>(g, comptime![q != 0], comptime![q != 1], comptime![q != 2]);
            let id = n as usize;
            let dq = (er[index(&with(&p, q, nq as u32), &lr)] - er[index(&p, &lr)]) / size::<F>(g, q);
            let dr = (eq[index(&with(&p, r, nr as u32), &lq)] - eq[index(&p, &lq)]) / size::<F>(g, r);
            let dq = stretch::<F>(grading, psi_q, id, get(&p, q), dq, q, false, absorbing);
            let dr = stretch::<F>(grading, psi_r, id, get(&p, r), dr, r, false, absorbing);
            b[id] -= dt * (dq - dr);
        }
    }
}

/// E_a advanced a step: ε₀εr ∂E_a/∂t = (∂_q B_r − ∂_r B_q)/μ₀ − J_a, (a, q, r) cyclic, off the
/// walls it is tangential to and the edges metal holds; one unit per point of E_a's lattice.
/// With a steady field its curl's part (`curl`, c²∇ × B₀) is added.
#[cube(launch)]
pub(crate) fn ampere<F: Real>(
    e: &mut [F],
    bq: &[F],
    br: &[F],
    j: &[F],
    curl: &[F],
    held: &[u32],
    eps: &[F],
    psi_q: &mut [F],
    psi_r: &mut [F],
    grading: &Grading<F>,
    g: &Grid<F>,
    c2: F,
    eps0: F,
    #[comptime] a: u32,
    #[comptime] q: u32,
    #[comptime] r: u32,
    #[comptime] absorbing: bool,
    #[comptime] any_media: bool,
    #[comptime] any_steady: bool,
) {
    let n = ABSOLUTE_POS as u32;
    let lat = dims::<F>(g, comptime![a != 0], comptime![a != 1], comptime![a != 2]);
    if n < lat.i * lat.j * lat.k {
        let p = point_of(n, &lat);
        let on_q = !periodic::<F>(g, q) && (get(&p, q) == 0 || get(&p, q) == cells::<F>(g, q));
        let on_r = !periodic::<F>(g, r) && (get(&p, r) == 0 || get(&p, r) == cells::<F>(g, r));
        let mq = wrap::<F>(g, q, get(&p, q) as i32 - 1, false);
        let mr = wrap::<F>(g, r, get(&p, r) as i32 - 1, false);
        if !on_q && !on_r && mq >= 0 && mr >= 0 {
            let lr = dims::<F>(g, comptime![r == 0], comptime![r == 1], comptime![r == 2]);
            let lq = dims::<F>(g, comptime![q == 0], comptime![q == 1], comptime![q == 2]);
            let id = n as usize;
            let dq = (br[index(&p, &lr)] - br[index(&with(&p, q, mq as u32), &lr)]) / size::<F>(g, q);
            let dr = (bq[index(&p, &lq)] - bq[index(&with(&p, r, mr as u32), &lq)]) / size::<F>(g, r);
            let dq = stretch::<F>(grading, psi_q, id, get(&p, q), dq, q, true, absorbing);
            let dr = stretch::<F>(grading, psi_r, id, get(&p, r), dr, r, true, absorbing);
            let mut free = true;
            let mut er = F::new(1.0_f32);
            if any_media {
                free = held[id] == 0;
                er = eps[id];
            }
            if free {
                let mut drive = c2 * (dq - dr);
                if any_steady {
                    drive += curl[id];
                }
                e[id] += g.dt / er * (drive - j[id] / eps0);
            }
        }
    }
}

/// E (V/m) at each of `count` probes (x, y, z triples, m) into `out` at slot `slot`.
#[cube(launch)]
pub(crate) fn probe<F: Real>(f: &Fields<F>, g: &Grid<F>, points: &[F], out: &mut [F], slot: u32, count: u32) {
    let p = ABSOLUTE_POS as u32;
    if p < count {
        let x = U::<F> { x: points[(3 * p) as usize], y: points[(3 * p + 1) as usize], z: points[(3 * p + 2) as usize] };
        let e = e_at::<F>(f, g, &x);
        let base = (3 * (slot * count + p)) as usize;
        out[base] = e.x;
        out[base + 1] = e.y;
        out[base + 2] = e.z;
    }
}

/// The particles' kinetic energy (J) in `out[0]`: one cube, each unit summing every `width`-th
/// particle from its own, then pairs folded in a fixed tree.
#[cube(launch)]
pub(crate) fn kinetic<F: Real>(p: &Particles<F>, kinds: &Kinds<F>, g: &Grid<F>, out: &mut [F], len: u32, step: u32, #[comptime] width: usize) {
    let u = UNIT_POS as usize;
    let units = CUBE_DIM as usize;
    let mut acc = F::new(0.0_f32);
    let mut m = u;
    while m < len as usize {
        if p.alive[m] == 1 && p.born[m] <= step {
            let v = U::<F> { x: p.ux[m], y: p.uy[m], z: p.uz[m] };
            acc += kinetic_of::<F>(&v, p.w[m], kinds.rest[p.kind[m] as usize], g.c);
        }
        m += units;
    }
    let total = crate::kernels::fold::<F>(acc, width);
    if u == 0 {
        out[0] = total;
    }
}

/// Σ wᵢ (vᵢ + sᵢ)² over `count` values, a partial sum a cube into `out[cube]`: each unit summing
/// every (cubes × units)-th value from its own, then the cube's units folded in a fixed tree
/// (the weights `w` with `weighted`, the steady part `s` with `shifted`).
#[cube(launch)]
pub(crate) fn squares<F: Real>(
    values: &[F],
    w: &[F],
    s: &[F],
    out: &mut [F],
    count: u32,
    #[comptime] weighted: bool,
    #[comptime] shifted: bool,
    #[comptime] width: usize,
) {
    let u = UNIT_POS as usize;
    let stride = CUBE_COUNT * CUBE_DIM as usize;
    let mut acc = F::new(0.0_f32);
    let mut m = CUBE_POS * CUBE_DIM as usize + u;
    while m < count as usize {
        let mut v = values[m];
        if shifted {
            v += s[m];
        }
        let mut square = v * v;
        if weighted {
            square *= w[m];
        }
        acc += square;
        m += stride;
    }
    let total = crate::kernels::fold::<F>(acc, width);
    if u == 0 {
        out[CUBE_POS] = total;
    }
}
