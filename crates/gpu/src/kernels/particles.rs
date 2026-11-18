//! What the particle runs' kernels share, whatever their grid (`axi`'s rings about an axis,
//! `pic`'s particles in a box): the pushers, the fixed-point current, the list of what left the
//! run, and the copies that place newborn particles into their slots.
//!
//! **The current, deterministically.** Many particles add to the same point in one step, in an
//! order the device does not fix; floating-point sums in that order would differ run to run. So
//! each addition is made an integer, J × `scale` rounded toward zero to 64 bits, and the
//! integers are added atomically: integer addition is associative, so the sum is the same in any
//! order, bit for bit, and E's step turns it back into J (the fixed-point accumulation of
//! S. Le Grand, A. W. Götz, R. C. Walker, *SPFP: Speed without compromise — a mixed precision
//! model for GPU accelerated molecular dynamics simulations*, Comput. Phys. Commun.
//! 184:374–380, 2013). The host picks `scale` so that no point's sum can overflow.

use cubecl::prelude::*;

use crate::buffer::Real;

/// The particles that left the run: each one's step and slot, how (0 dropped, its momentum not
/// a number; the others the grid's walls and metal, its own numbering), its charge or weight (as
/// the grid's host reads it) and its kinetic energy (J); `count` how many so far.
#[derive(CubeLaunch, CubeType)]
pub(crate) struct Deaths<F: Float> {
    pub(crate) step: Box<[u32]>,
    pub(crate) slot: Box<[u32]>,
    pub(crate) how: Box<[u32]>,
    pub(crate) w: Box<[F]>,
    pub(crate) kinetic: Box<[F]>,
    pub(crate) count: Box<[Atomic<u32>]>,
}

/// Particle `slot` written to the list of those that left at `step`.
#[cube]
pub(crate) fn leave<F: Real>(deaths: &mut Deaths<F>, step: u32, slot: u32, how: u32, w: F, kinetic: F) {
    let at = deaths.count[0].fetch_add(1u32) as usize;
    deaths.step[at] = step;
    deaths.slot[at] = slot;
    deaths.how[at] = how;
    deaths.w[at] = w;
    deaths.kinetic[at] = kinetic;
}

/// Adds `v` to point `id` of a fixed-point sum.
#[cube]
pub(crate) fn add<F: Real>(sum: &[Atomic<u64>], id: u32, v: F, scale: F) {
    sum[id as usize].fetch_add(u64::cast_from(i64::cast_from(v * scale)));
}

/// The fixed-point sum at `id` as the value it sums, the sum cleared for the next step.
#[cube]
pub(crate) fn take<F: Real>(sum: &[Atomic<u64>], id: usize, scale: F) -> F {
    let v = sum[id].load();
    sum[id].store(0u64);
    F::cast_from(i64::cast_from(v)) / scale
}

/// The first-order shape: the weight on a node `s` cells away.
#[cube]
pub(crate) fn shape<F: Real>(s: F) -> F {
    F::max(F::new(1.0_f32) - s.abs(), F::new(0.0_f32))
}

/// γ of u (m/s), `c` the speed of light (m/s).
#[cube]
pub(crate) fn gamma<F: Real>(u0: F, u1: F, u2: F, c: F) -> F {
    (F::new(1.0_f32) + (u0 * u0 + u1 * u1 + u2 * u2) / (c * c)).sqrt()
}

/// The kinetic energy (J) of `w` particles of rest energy `rest` (J) at momentum u (m/s):
/// w mc² u²/c² / (γ + 1), (γ − 1) without its cancellation.
#[cube]
pub(crate) fn kinetic<F: Real>(u: &U<F>, w: F, rest: F, c: F) -> F {
    let u2 = (u.x * u.x + u.y * u.y + u.z * u.z) / (c * c);
    w * rest * u2 / (gamma::<F>(u.x, u.y, u.z, c) + F::new(1.0_f32))
}

/// A vector of three components in a right-handed frame: x, y, z in a box, r, θ, z about an
/// axis.
#[derive(CubeType)]
pub(crate) struct U<F: Float> {
    pub(crate) x: F,
    pub(crate) y: F,
    pub(crate) z: F,
}

/// Whether each component of `u` is a finite number (a comparison with NaN fails).
#[cube]
pub(crate) fn finite<F: Real>(u: &U<F>) -> bool {
    let most = F::new(f32::MAX);
    u.x.abs() <= most && u.y.abs() <= most && u.z.abs() <= most
}

/// u pushed by E and B a step (`emcraft_particles::push`): Boris's or Vay's (`vay`), `kick`
/// q Δt/m.
#[cube]
pub(crate) fn push<F: Real>(u: U<F>, e: U<F>, b: U<F>, kick: F, c: F, #[comptime] vay: bool) -> U<F> {
    let one = F::new(1.0_f32);
    let half = F::new(0.5_f32);
    let two = F::new(2.0_f32);
    let mut out = U::<F> { x: u.x, y: u.y, z: u.z };
    if vay {
        let g = gamma::<F>(u.x, u.y, u.z, c);
        let (vx, vy, vz) = (u.x / g, u.y / g, u.z / g);
        // u′ = u + k (E + ½ v × B).
        let px = u.x + kick * (e.x + half * (vy * b.z - vz * b.y));
        let py = u.y + kick * (e.y + half * (vz * b.x - vx * b.z));
        let pz = u.z + kick * (e.z + half * (vx * b.y - vy * b.x));
        let (tx, ty, tz) = (half * kick * b.x, half * kick * b.y, half * kick * b.z);
        let tau2 = tx * tx + ty * ty + tz * tz;
        let ustar = (px * tx + py * ty + pz * tz) / c;
        let gp = gamma::<F>(px, py, pz, c);
        let sigma = gp * gp - tau2;
        let gn = (half * (sigma + (sigma * sigma + F::new(4.0_f32) * (tau2 + ustar * ustar)).sqrt())).sqrt();
        let (sx, sy, sz) = (tx / gn, ty / gn, tz / gn);
        let s = one / (one + sx * sx + sy * sy + sz * sz);
        let pt = px * sx + py * sy + pz * sz;
        out.x = s * (px + sx * pt + (py * sz - pz * sy));
        out.y = s * (py + sy * pt + (pz * sx - px * sz));
        out.z = s * (pz + sz * pt + (px * sy - py * sx));
    } else {
        let k = half * kick;
        let mx = u.x + e.x * k;
        let my = u.y + e.y * k;
        let mz = u.z + e.z * k;
        let g = gamma::<F>(mx, my, mz, c);
        let (tx, ty, tz) = (b.x * (k / g), b.y * (k / g), b.z * (k / g));
        let qx = mx + (my * tz - mz * ty);
        let qy = my + (mz * tx - mx * tz);
        let qz = mz + (mx * ty - my * tx);
        let f = two / (one + tx * tx + ty * ty + tz * tz);
        let (sx, sy, sz) = (tx * f, ty * f, tz * f);
        out.x = mx + (qy * sz - qz * sy) + e.x * k;
        out.y = my + (qz * sx - qx * sz) + e.y * k;
        out.z = mz + (qx * sy - qy * sx) + e.z * k;
    }
    out
}

/// `count` values of `from` copied into `to` from `start` on.
#[cube(launch)]
pub(crate) fn copy_values<F: Real>(to: &mut [F], from: &[F], start: u32, count: u32) {
    let m = ABSOLUTE_POS as u32;
    if m < count {
        to[(start + m) as usize] = from[m as usize];
    }
}

/// `count` indices of `from` copied into `to` from `start` on.
#[cube(launch)]
pub(crate) fn copy_ints(to: &mut [u32], from: &[u32], start: u32, count: u32) {
    let m = ABSOLUTE_POS as u32;
    if m < count {
        to[(start + m) as usize] = from[m as usize];
    }
}
