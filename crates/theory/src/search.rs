//! Golden-section search (J. Kiefer, Proc. Amer. Math. Soc. 4:502, 1953; W. H. Press et al.,
//! *Numerical Recipes*, 3rd ed., Cambridge 2007, §10.2): the best of a function with one
//! extremum over an interval, the bracket narrowed by 1/φ each step, one new value a step.
//!
//! ```
//! use emcraft_theory::search::golden;
//! // (x − 1)² is least at 1; −(x − 1)² most there.
//! let least = golden(0.0, 3.0, 60, |x| (x - 1.0).powi(2), |a, b| a <= b);
//! assert!((least.at - 1.0).abs() < 1e-9);
//! let most = golden(0.0, 3.0, 60, |x| -(x - 1.0).powi(2), |a, b| a > b);
//! assert!((most.at - 1.0).abs() < 1e-9 && most.value <= 0.0);
//! ```

/// Where a search ended and the function's value there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Golden {
    pub at: f64,
    pub value: f64,
}

/// The best of `f` over [`lo`, `hi`] after `steps` narrowings: `prefer(a, b)` says whether a
/// value `a` at the lower interior point is better than `b` at the upper (`a <= b` for a
/// minimum, ties to the lower; `a > b` for a maximum, ties to the upper), and the better of the
/// last two points is the answer.
pub fn golden(mut lo: f64, mut hi: f64, steps: usize, f: impl Fn(f64) -> f64, prefer: impl Fn(f64, f64) -> bool) -> Golden {
    // 1/φ, the share of the bracket each interior point lies from its far end.
    let r = 0.5 * (5f64.sqrt() - 1.0);
    let (mut x1, mut x2) = (hi - r * (hi - lo), lo + r * (hi - lo));
    let (mut f1, mut f2) = (f(x1), f(x2));
    for _ in 0..steps {
        if prefer(f1, f2) {
            hi = x2;
            (x2, f2) = (x1, f1);
            x1 = hi - r * (hi - lo);
            f1 = f(x1);
        } else {
            lo = x1;
            (x1, f1) = (x2, f2);
            x2 = lo + r * (hi - lo);
            f2 = f(x2);
        }
    }
    if prefer(f1, f2) { Golden { at: x1, value: f1 } } else { Golden { at: x2, value: f2 } }
}
