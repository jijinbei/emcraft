//! S-parameters of two 3 mm full-height posts 20 mm apart across WR-90 (60 mm of guide, the posts
//! 20 mm from either port plane) at the frequencies given (GHz), from the closed-form reference:
//! the oracle's `design-two-posts`. Prints f, |S11|, ∠S11 (°), |S21|, ∠S21 (°).
fn main() {
    use emcraft_theory::post::{Post, Posts};
    let post = |z| Post { x: 11.43e-3, z, radius: 1.5e-3 };
    let pair = Posts { width: 22.86e-3, length: 60e-3, posts: vec![post(20e-3), post(40e-3)] };
    for f in std::env::args().skip(1).filter_map(|a| a.parse::<f64>().ok()) {
        match pair.solve(f * 1e9, 128) {
            Ok(s) => println!("{f} {} {} {} {}", s.s11.norm(), s.s11.arg().to_degrees(), s.s21.norm(), s.s21.arg().to_degrees()),
            Err(e) => println!("{f} error {e}"),
        }
    }
}
