//! Small dense matrices: the determinant of a closed form's matching matrix.

/// The determinant of a square matrix (row-major, `n` × `n`) by elimination with partial
/// pivoting.
pub fn det(mut a: Vec<f64>, n: usize) -> f64 {
    let mut d = 1.0;
    for c in 0..n {
        let Some(p) = (c..n).max_by(|x, y| a[x * n + c].abs().total_cmp(&a[y * n + c].abs())) else { return 0.0 };
        if a[p * n + c] == 0.0 {
            return 0.0;
        }
        if p != c {
            for k in 0..n {
                a.swap(p * n + k, c * n + k);
            }
            d = -d;
        }
        let pivot = a[c * n + c];
        d *= pivot;
        for r in c + 1..n {
            let f = a[r * n + c] / pivot;
            for k in c..n {
                a[r * n + k] -= f * a[c * n + k];
            }
        }
    }
    d
}
