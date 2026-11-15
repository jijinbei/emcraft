//! Real roots of a real function: its sign changes found along a grid, each bracket narrowed by
//! bisection (W. H. Press et al., *Numerical Recipes*, 3rd ed., Cambridge 2007, §9.1). One
//! bisection for every closed form here, each caller saying how fine it needs its roots.
//!
//! ```
//! use emcraft_theory::roots::{bisect, roots_on_grid};
//! // √2 to rounding, and to a millionth.
//! let r = bisect(|x| x * x - 2.0, 0.0, 2.0, 0.0).unwrap();
//! assert!((r - 2f64.sqrt()).abs() < 1e-15);
//! assert!((bisect(|x| x * x - 2.0, 0.0, 2.0, 1e-6).unwrap() - 2f64.sqrt()).abs() < 2e-6);
//! // No sign change, no root.
//! assert!(bisect(|x| x * x + 1.0, -1.0, 1.0, 0.0).is_none());
//! // sin's zeros in (1, 10): π, 2π, 3π.
//! let zeros = roots_on_grid(f64::sin, (0..=900).map(|i| 1.0 + 0.01 * i as f64), 1e-12);
//! assert_eq!(zeros.len(), 3);
//! assert!((zeros[2] - 3.0 * std::f64::consts::PI).abs() < 1e-10);
//! ```

/// The most halvings of a bracket: past f64's 2⁻¹⁰⁷⁴, more than any bracket of finite ends needs
/// to reach rounding.
pub const MAX_BISECTIONS: usize = 200;

/// The x in [`lo`, `hi`] where `g` changes sign (`g(lo)` and `g(hi)` of opposite signs, or one
/// of them zero), by bisection until the bracket is within `tolerance` of the larger end's size
/// (0: until its midpoint is one of its ends, rounding); a midpoint where `g` is zero is the
/// root. None without a sign change, or for ends that are not finite.
pub fn bisect(g: impl Fn(f64) -> f64, lo: f64, hi: f64, tolerance: f64) -> Option<f64> {
    if !(lo.is_finite() && hi.is_finite()) {
        return None;
    }
    let (g_lo, g_hi) = (g(lo), g(hi));
    if g_lo == 0.0 {
        return Some(lo);
    }
    if g_hi == 0.0 {
        return Some(hi);
    }
    (g_lo.signum() != g_hi.signum()).then(|| narrowed(&g, (lo, g_lo), hi, tolerance))
}

/// The bracket [`lo`, `hi`] (`g` of `lo`'s sign at `lo`, of the other at `hi`) halved down to
/// `tolerance` ([`bisect`]): its midpoint.
fn narrowed(g: &impl Fn(f64) -> f64, (mut lo, g_lo): (f64, f64), mut hi: f64, tolerance: f64) -> f64 {
    let side = g_lo.signum();
    for _ in 0..MAX_BISECTIONS {
        let mid = 0.5 * (lo + hi);
        if mid <= lo.min(hi) || mid >= lo.max(hi) || (hi - lo).abs() <= tolerance * lo.abs().max(hi.abs()) {
            break;
        }
        let g_mid = g(mid);
        if g_mid == 0.0 {
            return mid;
        }
        if g_mid.signum() == side {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The roots of `g` along `grid` (its points in order, either way): a point where `g` is zero,
/// and each sign change between neighbours narrowed by [`bisect`] to `tolerance`, in the grid's
/// order.
pub fn roots_on_grid(g: impl Fn(f64) -> f64, grid: impl IntoIterator<Item = f64>, tolerance: f64) -> Vec<f64> {
    let mut points = grid.into_iter().map(|x| (x, g(x)));
    let Some(mut before) = points.next() else { return Vec::new() };
    let mut out = Vec::new();
    for now in points {
        if before.1 == 0.0 {
            out.push(before.0);
        } else if now.1 != 0.0 && before.1.signum() != now.1.signum() {
            out.push(narrowed(&g, before, now.0, tolerance));
        }
        before = now;
    }
    out
}
