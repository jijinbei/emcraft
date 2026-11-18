//! The axisymmetric particle-in-cell run's kernels (`emcraft_fdtd::axi`, whose docs give the
//! r–z Yee lattice, the rings and the scheme): the rings pushed and their current deposited,
//! one unit a ring; B's half steps and E's step, one unit a point; the probes; the rings'
//! kinetic energy summed in a fixed tree. The current is summed in fixed point
//! (`kernels::particles`).
//!
//! Lattices (`emcraft_fdtd::axi::device::Plan`'s order): E_r nr × Nz, E_z (nr + 1) × Cz,
//! E_θ (nr + 1) × Nz, B_r (nr + 1) × Cz, B_z nr × Nz, B_θ nr × Cz; Nz z's nodes (cells + 1
//! between conducting ends, cells when periodic), Cz z's centres (cells); r runs fastest.

use cubecl::prelude::*;

use crate::buffer::Real;
use crate::kernels::particles::{Deaths, U, add, finite, gamma, kinetic as kinetic_of, leave, push, shape, take};

/// The r–z grid: cells along r and z, whether z is periodic, the cells' sizes (m) and the step
/// (s); with the speed of light (m/s) and π in the run's precision (literals in a kernel are
/// single).
#[derive(CubeLaunch, CubeType)]
pub(crate) struct Grid<F: Float> {
    pub(crate) nr: u32,
    pub(crate) nz: u32,
    pub(crate) periodic: u32,
    pub(crate) dr: F,
    pub(crate) dz: F,
    pub(crate) dt: F,
    pub(crate) c: F,
    pub(crate) pi: F,
}

/// The six components.
#[derive(CubeLaunch, CubeType)]
pub(crate) struct Fields<F: Float> {
    pub(crate) er: Box<[F]>,
    pub(crate) ez: Box<[F]>,
    pub(crate) et: Box<[F]>,
    pub(crate) br: Box<[F]>,
    pub(crate) bz: Box<[F]>,
    pub(crate) bt: Box<[F]>,
}

/// The steady magnetic field the run starts with, held apart from B's changes (which the
/// device's fields hold): B₀ itself, for the rings, and the parts of E's curl it makes, as the
/// host computed them in double precision (∂z B_θ₀ at E_r, ∂z B_r₀ and ∂r B_z₀ at E_θ, the curl
/// of B_θ₀ at E_z). A strong field held as it is would leave single precision's last digits to
/// its changes, and the curl that E_θ takes of B_z would lose them.
#[derive(CubeLaunch, CubeType)]
pub(crate) struct Steady<F: Float> {
    pub(crate) br: Box<[F]>,
    pub(crate) bz: Box<[F]>,
    pub(crate) bt: Box<[F]>,
    pub(crate) er_dz: Box<[F]>,
    pub(crate) et_dz: Box<[F]>,
    pub(crate) et_dr: Box<[F]>,
    pub(crate) ez_curl: Box<[F]>,
}

/// The current's fixed-point sums on E_r's, E_z's and E_θ's lattices, and their scales.
#[derive(CubeLaunch, CubeType)]
pub(crate) struct Current<F: Float> {
    pub(crate) r: Box<[Atomic<u64>]>,
    pub(crate) z: Box<[Atomic<u64>]>,
    pub(crate) t: Box<[Atomic<u64>]>,
    pub(crate) scale_r: F,
    pub(crate) scale_z: F,
    pub(crate) scale_t: F,
}

/// The absorbing layers along z: ψ of E_r's, E_θ's, B_θ's and B_r's z differences.
#[derive(CubeLaunch, CubeType)]
pub(crate) struct Psi<F: Float> {
    pub(crate) er: Box<[F]>,
    pub(crate) et: Box<[F]>,
    pub(crate) bt: Box<[F]>,
    pub(crate) br: Box<[F]>,
}

/// The layers' (b, a): of E's steps at z's nodes and of B's half steps at its centres.
#[derive(CubeLaunch, CubeType)]
pub(crate) struct Grading<F: Float> {
    pub(crate) e: Box<[F]>,
    pub(crate) b: Box<[F]>,
}

/// The metal: whether each E_r, E_z, E_θ point touches it, and whether each cell is it (1 or 0).
#[derive(CubeLaunch, CubeType)]
pub(crate) struct Metal {
    pub(crate) er: Box<[u32]>,
    pub(crate) ez: Box<[u32]>,
    pub(crate) et: Box<[u32]>,
    pub(crate) cells: Box<[u32]>,
}

/// The rings: radius and z (m), momentum γv (m/s), weight, the step each is born at and whether
/// it is still in the run (1 or 0).
#[derive(CubeLaunch, CubeType)]
pub(crate) struct Rings<F: Float> {
    pub(crate) r: Box<[F]>,
    pub(crate) z: Box<[F]>,
    pub(crate) ur: Box<[F]>,
    pub(crate) ut: Box<[F]>,
    pub(crate) uz: Box<[F]>,
    pub(crate) w: Box<[F]>,
    pub(crate) born: Box<[u32]>,
    pub(crate) alive: Box<[u32]>,
}

/// The species: q Δt/m, the charge (C) and the rest energy (J).
#[derive(CubeLaunch, CubeType)]
pub(crate) struct Species<F: Float> {
    pub(crate) kick: F,
    pub(crate) charge: F,
    pub(crate) rest: F,
}

/// z's points along a lattice: its nodes (`node`) or centres.
#[cube]
fn z_points<F: Real>(g: &Grid<F>, #[comptime] node: bool) -> i32 {
    let cells = g.nz as i32;
    if node && g.periodic == 0 { cells + 1 } else { cells }
}

/// Point `k` along z: wrapped round a periodic axis, the nearest past a conducting end.
#[cube]
fn z_clamp<F: Real>(g: &Grid<F>, k: i32, #[comptime] node: bool) -> u32 {
    let n = z_points::<F>(g, node);
    let mut out = clamp(k, 0, n - 1);
    if g.periodic == 1 {
        out = ((k % n) + n) % n;
    }
    out as u32
}

/// Point `k` along z wrapped round a periodic axis; −1 past a conducting end.
#[cube]
fn z_wrap<F: Real>(g: &Grid<F>, k: i32, #[comptime] node: bool) -> i32 {
    let n = z_points::<F>(g, node);
    let mut out = i32::new(-1);
    if g.periodic == 1 {
        out = ((k % n) + n) % n;
    } else if k >= 0 && k < n {
        out = k;
    }
    out
}

/// Whether z node `k` lies on a conducting end.
#[cube]
fn z_on_wall<F: Real>(g: &Grid<F>, k: u32) -> bool {
    g.periodic == 0 && (k == 0 || k == g.nz)
}

/// `values` on a lattice of `nr` points along r at (r, z), linearly; across the axis odd
/// (`odd`) or even, past the wall and the ends the nearest point's.
#[cube]
fn gather<F: Real>(values: &[F], g: &Grid<F>, nr: u32, r: F, z: F, #[comptime] r_node: bool, #[comptime] z_node: bool, #[comptime] odd: bool) -> F {
    let half = F::new(0.5_f32);
    let one = F::new(1.0_f32);
    let mut sr = r / g.dr;
    if !r_node {
        sr -= half;
    }
    let mut sz = z / g.dz;
    if !z_node {
        sz -= half;
    }
    let i0 = sr.floor();
    let k0 = sz.floor();
    let fr = sr - i0;
    let fz = sz - k0;
    let i0 = i32::cast_from(i0);
    let k0 = i32::cast_from(k0);
    let mut sum = F::new(0.0_f32);
    #[unroll]
    for di in 0..2 {
        #[unroll]
        for dk in 0..2 {
            let mut i = i0 + di;
            let mut sign = one;
            if i < 0 {
                i = if r_node { -i } else { -i - 1 };
                if odd {
                    sign = -one;
                }
            }
            let i = min(i as u32, nr - 1);
            let k = z_clamp::<F>(g, k0 + dk, z_node);
            let wr = if di == 0 { one - fr } else { fr };
            let wz = if dk == 0 { one - fz } else { fz };
            sum += wr * wz * sign * values[(i + nr * k) as usize];
        }
    }
    sum
}

/// Whether cell (i, k) is metal (none outside the grid).
#[cube]
fn metal_cell<F: Real>(metal: &Metal, g: &Grid<F>, i: i32, k: i32, #[comptime] any: bool) -> bool {
    let mut out = false;
    if any && i >= 0 && i < g.nr as i32 {
        let kk = z_wrap::<F>(g, k, false);
        if kk >= 0 {
            out = metal.cells[(i as u32 + g.nr * kk as u32) as usize] == 1;
        }
    }
    out
}

/// The area of node ring `i`'s faces normal to z (m²).
#[cube]
fn area<F: Real>(g: &Grid<F>, i: u32) -> F {
    let pi = g.pi;
    let mut a = F::new(2.0_f32) * pi * F::cast_from(i) * g.dr * g.dr;
    if i == 0 {
        a = pi * F::new(0.25_f32) * g.dr * g.dr;
    }
    a
}

/// Ring `ABSOLUTE_POS` (of `len`) pushed and moved a step (`step` of the batch), its current
/// added, its leaving written to `deaths` (`emcraft_fdtd::axi::AxiPic`'s `advance`).
#[cube(launch)]
pub(crate) fn push_rings<F: Real>(
    mut rings: Rings<F>,
    fields: &Fields<F>,
    steady: &Steady<F>,
    g: &Grid<F>,
    species: &Species<F>,
    metal: &Metal,
    j: &Current<F>,
    mut deaths: Deaths<F>,
    step: u32,
    len: u32,
    #[comptime] vay: bool,
    #[comptime] any_metal: bool,
    #[comptime] any_steady: bool,
) {
    let n = ABSOLUTE_POS as u32;
    if n < len && rings.alive[n as usize] == 1 && rings.born[n as usize] <= step {
        let id = n as usize;
        let zero = F::new(0.0_f32);
        let one = F::new(1.0_f32);
        let half = F::new(0.5_f32);
        let nr = g.nr;
        let (r, z, w) = (rings.r[id], rings.z[id], rings.w[id]);
        let e = U::<F> {
            x: gather::<F>(&fields.er, g, nr, r, z, false, true, true),
            y: gather::<F>(&fields.et, g, nr + 1, r, z, true, true, true),
            z: gather::<F>(&fields.ez, g, nr + 1, r, z, true, false, false),
        };
        let mut b = U::<F> {
            x: gather::<F>(&fields.br, g, nr + 1, r, z, true, false, true),
            y: gather::<F>(&fields.bt, g, nr, r, z, false, false, true),
            z: gather::<F>(&fields.bz, g, nr, r, z, false, true, false),
        };
        if any_steady {
            b.x += gather::<F>(&steady.br, g, nr + 1, r, z, true, false, true);
            b.y += gather::<F>(&steady.bt, g, nr, r, z, false, false, true);
            b.z += gather::<F>(&steady.bz, g, nr, r, z, false, true, false);
        }
        let u = push::<F>(U::<F> { x: rings.ur[id], y: rings.ut[id], z: rings.uz[id] }, e, b, species.kick, g.c, vay);
        if !finite::<F>(&u) {
            leave::<F>(&mut deaths, step, n, 0u32, w, zero);
            rings.alive[id] = 0u32;
        } else {
            let g_u = gamma::<F>(u.x, u.y, u.z, g.c);
            let (vr, vt, vz) = (u.x / g_u, u.y / g_u, u.z / g_u);
            // Straight in the plane θ = 0, then turned back onto it.
            let x = r + vr * g.dt;
            let y = vt * g.dt;
            let mut r1 = (x * x + y * y).sqrt();
            let mut z1 = z + vz * g.dt;
            let mut c = one;
            let mut s = zero;
            if r1 > zero {
                c = x / r1;
                s = y / r1;
            }
            let u1 = U::<F> { x: c * u.x + s * u.y, y: -s * u.x + c * u.y, z: u.z };
            // The walls: the conductor at R and conducting ends.
            let radius = F::cast_from(nr) * g.dr;
            let length = F::cast_from(g.nz) * g.dz;
            let mut how = 0u32;
            let mut f = one;
            if r1 > radius {
                f = F::clamp((radius - r) / (r1 - r), zero, one);
                how = 1u32;
            }
            if g.periodic == 0 {
                if z1 < zero {
                    let fe = F::clamp((zero - z) / (z1 - z), zero, one);
                    if fe < f {
                        f = fe;
                        how = 2u32;
                    }
                }
                if z1 > length {
                    let fe = F::clamp((length - z) / (z1 - z), zero, one);
                    if fe < f {
                        f = fe;
                        how = 3u32;
                    }
                }
            }
            if how != 0 {
                r1 = F::clamp(r + f * (r1 - r), zero, radius);
                z1 = F::clamp(z + f * (z1 - z), zero, length);
            }
            let q = species.charge * w;
            deposit::<F>(j, g, r, z, r1, z1, q);
            deposit_theta::<F>(j, g, half * (r + r1), half * (z + z1), q * vt);
            if how == 0 && metal_cell::<F>(metal, g, i32::cast_from((r1 / g.dr).floor()), i32::cast_from((z1 / g.dz).floor()), any_metal) {
                how = 4u32;
            }
            if how != 0 {
                let kinetic = kinetic_of::<F>(&u1, w, species.rest, g.c);
                leave::<F>(&mut deaths, step, n, how, w, kinetic);
                rings.alive[id] = 0u32;
            } else {
                rings.r[id] = r1;
                let mut zn = z1;
                if g.periodic == 1 {
                    zn = z1 - length * (z1 / length).floor();
                }
                rings.z[id] = zn;
                rings.ur[id] = u1.x;
                rings.ut[id] = u1.y;
                rings.uz[id] = u1.z;
            }
        }
    }
}

/// The current of charge `q` (C) moving from (r0, z0) to (r1, z1) in a step: Esirkepov's flux
/// through each node ring's faces, over their areas.
#[cube]
fn deposit<F: Real>(j: &Current<F>, g: &Grid<F>, r0: F, z0: F, r1: F, z1: F, q: F) {
    let half = F::new(0.5_f32);
    let zero = F::new(0.0_f32);
    let pi = g.pi;
    let (sr0, sr1, sz0, sz1) = (r0 / g.dr, r1 / g.dr, z0 / g.dz, z1 / g.dz);
    let fr = i32::cast_from(sr0.floor()) - 1;
    let fz = i32::cast_from(sz0.floor()) - 1;
    let nr = g.nr as i32;
    // I_r through r_{i+½}: the running sum over i.
    #[unroll]
    for mz in 0..4 {
        let k = z_wrap::<F>(g, fz + mz, true);
        let at = F::cast_from(fz + mz);
        let b0 = shape::<F>(sz0 - at);
        let across = b0 + half * (shape::<F>(sz1 - at) - b0);
        if k >= 0 && across != zero {
            let mut flux = zero;
            #[unroll]
            for mr in 0..3 {
                let ar = F::cast_from(fr + mr);
                let ds = shape::<F>(sr1 - ar) - shape::<F>(sr0 - ar);
                flux -= q * ds * across / g.dt;
                let i = fr + mr;
                if i >= 0 && i < nr {
                    let face = F::new(2.0_f32) * pi * (F::cast_from(i) + half) * g.dr * g.dz;
                    add::<F>(&j.r, i as u32 + g.nr * k as u32, flux / face, j.scale_r);
                }
            }
        }
    }
    // I_z through the node rings' areas: the running sum over k.
    #[unroll]
    for mr in 0..4 {
        let i = fr + mr;
        let ar = F::cast_from(i);
        let b0 = shape::<F>(sr0 - ar);
        let across = b0 + half * (shape::<F>(sr1 - ar) - b0);
        if i >= 0 && i <= nr && across != zero {
            let a = area::<F>(g, i as u32);
            let mut flux = zero;
            #[unroll]
            for mz in 0..3 {
                let at = F::cast_from(fz + mz);
                let ds = shape::<F>(sz1 - at) - shape::<F>(sz0 - at);
                flux -= q * ds * across / g.dt;
                let k = z_wrap::<F>(g, fz + mz, false);
                if k >= 0 {
                    add::<F>(&j.z, i as u32 + (g.nr + 1) * k as u32, flux / a, j.scale_z);
                }
            }
        }
    }
}

/// The azimuthal current `qv` (A·m) at (r, z) on E_θ's nodes, over their rings' volumes.
#[cube]
fn deposit_theta<F: Real>(j: &Current<F>, g: &Grid<F>, r: F, z: F, qv: F) {
    let one = F::new(1.0_f32);
    let sr = r / g.dr;
    let sz = z / g.dz;
    let i0 = sr.floor();
    let k0 = sz.floor();
    let (fr, fz) = (sr - i0, sz - k0);
    let (i0, k0) = (i32::cast_from(i0), i32::cast_from(k0));
    #[unroll]
    for di in 0..2 {
        let i = i0 + di;
        if i >= 1 && i <= g.nr as i32 {
            let wr = if di == 0 { one - fr } else { fr };
            #[unroll]
            for dk in 0..2 {
                let k = z_wrap::<F>(g, k0 + dk, true);
                if k >= 0 {
                    let wz = if dk == 0 { one - fz } else { fz };
                    let volume = area::<F>(g, i as u32) * g.dz;
                    add::<F>(&j.t, i as u32 + (g.nr + 1) * k as u32, qv * wr * wz / volume, j.scale_t);
                }
            }
        }
    }
}

/// A z difference through the layers: d + ψ, ψ advanced by the (b, a) at `k` of `ba`.
#[cube]
fn stretch<F: Real>(ba: &[F], psi: &mut [F], id: usize, k: u32, d: F, #[comptime] absorbing: bool) -> F {
    let mut out = d;
    if absorbing {
        let p = ba[(2 * k) as usize] * psi[id] + ba[(2 * k + 1) as usize] * d;
        psi[id] = p;
        out = d + p;
    }
    out
}

/// B advanced by `dt` (s): ∂B/∂t = −∇ × E, one unit per point (i, k), i < nr + 1, k < Nz.
#[cube(launch)]
pub(crate) fn faraday<F: Real>(mut f: Fields<F>, mut psi: Psi<F>, grading: &Grading<F>, g: &Grid<F>, dt: F, #[comptime] absorbing: bool) {
    let nr = g.nr;
    let pos = ABSOLUTE_POS as u32;
    let i = pos % (nr + 1);
    let k = pos / (nr + 1);
    let cz = g.nz;
    let nz = z_points::<F>(g, true) as u32;
    if k < nz {
        let kw = z_wrap::<F>(g, k as i32 + 1, true);
        let kp = kw as u32;
        // B_θ: −(∂z E_r − ∂r E_z).
        if i < nr && k < cz && kw >= 0 {
            let id = (i + nr * k) as usize;
            let dz = (f.er[(i + nr * kp) as usize] - f.er[(i + nr * k) as usize]) / g.dz;
            let dz = stretch::<F>(&grading.b, &mut psi.bt, id, k, dz, absorbing);
            let dr = (f.ez[(i + 1 + (nr + 1) * k) as usize] - f.ez[(i + (nr + 1) * k) as usize]) / g.dr;
            f.bt[id] -= dt * (dz - dr);
        }
        // B_r: +∂z E_θ.
        if k < cz && kw >= 0 {
            let id = (i + (nr + 1) * k) as usize;
            let dz = (f.et[(i + (nr + 1) * kp) as usize] - f.et[(i + (nr + 1) * k) as usize]) / g.dz;
            f.br[id] += dt * stretch::<F>(&grading.b, &mut psi.br, id, k, dz, absorbing);
        }
        // B_z: −(1/r) ∂r (r E_θ).
        if i < nr {
            let id = (i + nr * k) as usize;
            let r0 = F::cast_from(i) * g.dr;
            let r1 = F::cast_from(i + 1) * g.dr;
            let rm = (F::cast_from(i) + F::new(0.5_f32)) * g.dr;
            f.bz[id] -= dt * (r1 * f.et[(i + 1 + (nr + 1) * k) as usize] - r0 * f.et[(i + (nr + 1) * k) as usize]) / (rm * g.dr);
        }
    }
}

/// E advanced a step: ε₀ ∂E/∂t = ∇ × B/μ₀ − J, off the walls and the metal it is tangential to;
/// one unit per point (i, k), i < nr + 1, k < Nz. Every point's current is taken (and cleared),
/// those of E held too.
#[cube(launch)]
pub(crate) fn ampere<F: Real>(
    mut f: Fields<F>,
    mut psi: Psi<F>,
    grading: &Grading<F>,
    j: &Current<F>,
    metal: &Metal,
    steady: &Steady<F>,
    g: &Grid<F>,
    c2: F,
    eps0: F,
    #[comptime] absorbing: bool,
    #[comptime] any_metal: bool,
    #[comptime] any_steady: bool,
) {
    let nr = g.nr;
    let pos = ABSOLUTE_POS as u32;
    let i = pos % (nr + 1);
    let k = pos / (nr + 1);
    let cz = g.nz;
    let nz = z_points::<F>(g, true) as u32;
    let dt = g.dt;
    if k < nz {
        let wall = z_on_wall::<F>(g, k);
        let kw = z_wrap::<F>(g, k as i32 - 1, false);
        let km = kw as u32;
        // E_r: −c² ∂z B_θ − J_r/ε₀, off the conducting ends.
        if i < nr {
            let id = (i + nr * k) as usize;
            let jr = take::<F>(&j.r, id, j.scale_r);
            if !wall && kw >= 0 {
                let kc = min(k, cz - 1);
                let mut dz = (f.bt[(i + nr * kc) as usize] - f.bt[(i + nr * km) as usize]) / g.dz;
                if any_steady {
                    dz += steady.er_dz[id];
                }
                let dz = stretch::<F>(&grading.e, &mut psi.er, id, k, dz, absorbing);
                if !(any_metal && metal.er[id] == 1) {
                    f.er[id] += dt * (-c2 * dz - jr / eps0);
                }
            }
        }
        // E_θ: c²(∂z B_r − ∂r B_z) − J_θ/ε₀, off the axis, the wall and the ends.
        {
            let id = (i + (nr + 1) * k) as usize;
            let jt = take::<F>(&j.t, id, j.scale_t);
            if !wall && kw >= 0 && i >= 1 && i < nr {
                let kc = min(k, cz - 1);
                let mut dz = (f.br[(i + (nr + 1) * kc) as usize] - f.br[(i + (nr + 1) * km) as usize]) / g.dz;
                let mut dr = (f.bz[(i + nr * k) as usize] - f.bz[(i - 1 + nr * k) as usize]) / g.dr;
                if any_steady {
                    dz += steady.et_dz[id];
                    dr += steady.et_dr[id];
                }
                let dz = stretch::<F>(&grading.e, &mut psi.et, id, k, dz, absorbing);
                if !(any_metal && metal.et[id] == 1) {
                    f.et[id] += dt * (c2 * (dz - dr) - jt / eps0);
                }
            }
        }
        // E_z: c² (1/r) ∂r (r B_θ) − J_z/ε₀, off the wall; on the axis 4B_θ(r_½)/Δr.
        if k < cz {
            let id = (i + (nr + 1) * k) as usize;
            let jz = take::<F>(&j.z, id, j.scale_z);
            if i < nr {
                let mut curl = F::new(4.0_f32) * f.bt[(nr * k) as usize] / g.dr;
                if i > 0 {
                    let half = F::new(0.5_f32);
                    let rp = (F::cast_from(i) + half) * g.dr;
                    let rm = (F::cast_from(i) - half) * g.dr;
                    let r = F::cast_from(i) * g.dr;
                    curl = (rp * f.bt[(i + nr * k) as usize] - rm * f.bt[(i - 1 + nr * k) as usize]) / (r * g.dr);
                }
                if any_steady {
                    curl += steady.ez_curl[id];
                }
                if !(any_metal && metal.ez[id] == 1) {
                    f.ez[id] += dt * (c2 * curl - jz / eps0);
                }
            }
        }
    }
}

/// The values at `count` taps of the fields into `out` at slot `slot`: each tap its component
/// (`which`, 0–5 in the order of [`Fields`]) and its point there (`at`).
#[cube(launch)]
pub(crate) fn tap<F: Real>(f: &Fields<F>, which: &[u32], at: &[u32], out: &mut [F], slot: u32, count: u32) {
    let p = ABSOLUTE_POS as u32;
    if p < count {
        let w = which[p as usize];
        let i = at[p as usize] as usize;
        let o = (slot * count + p) as usize;
        if w == 0 {
            out[o] = f.er[i];
        } else if w == 1 {
            out[o] = f.ez[i];
        } else if w == 2 {
            out[o] = f.et[i];
        } else if w == 3 {
            out[o] = f.br[i];
        } else if w == 4 {
            out[o] = f.bz[i];
        } else {
            out[o] = f.bt[i];
        }
    }
}

/// E (E_r, E_θ, E_z; V/m) at each of `count` probes (r, z pairs, m) into `out` at slot `slot`.
#[cube(launch)]
pub(crate) fn probe<F: Real>(f: &Fields<F>, g: &Grid<F>, points: &[F], out: &mut [F], slot: u32, count: u32) {
    let p = ABSOLUTE_POS as u32;
    if p < count {
        let r = points[(2 * p) as usize];
        let z = points[(2 * p + 1) as usize];
        let nr = g.nr;
        let base = (3 * (slot * count + p)) as usize;
        out[base] = gather::<F>(&f.er, g, nr, r, z, false, true, true);
        out[base + 1] = gather::<F>(&f.et, g, nr + 1, r, z, true, true, true);
        out[base + 2] = gather::<F>(&f.ez, g, nr + 1, r, z, true, false, false);
    }
}

/// The rings' kinetic energy (J) in `out[0]`: one cube, each unit summing every `width`-th
/// ring from its own, then pairs folded in a fixed tree.
#[cube(launch)]
pub(crate) fn kinetic<F: Real>(rings: &Rings<F>, species: &Species<F>, g: &Grid<F>, out: &mut [F], len: u32, step: u32, #[comptime] width: usize) {
    let u = UNIT_POS as usize;
    let units = CUBE_DIM as usize;
    let c = g.c;
    let mut acc = F::new(0.0_f32);
    let mut m = u;
    while m < len as usize {
        if rings.alive[m] == 1 && rings.born[m] <= step {
            let u = U::<F> { x: rings.ur[m], y: rings.ut[m], z: rings.uz[m] };
            acc += kinetic_of::<F>(&u, rings.w[m], species.rest, c);
        }
        m += units;
    }
    let total = crate::kernels::fold::<F>(acc, width);
    if u == 0 {
        out[0] = total;
    }
}
