//! What the grids' tests share.

/// A deterministic sequence uniform in [0, 1): Knuth's MMIX linear congruential generator, its
/// top 53 bits (the same numbers on every run, so the tests are reproducible).
pub(crate) struct Lcg(pub(crate) u64);

impl Lcg {
    pub(crate) fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}
