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

#[cfg(test)]
mod tests {
    use super::*;

    /// Determinants known in closed form: a Vandermonde matrix's ∏(xⱼ − xᵢ), a triangular
    /// matrix's diagonal, a zero pivot that needs rows exchanged, and matrices of lower rank.
    #[test]
    fn determinants_are_their_closed_forms() {
        let x: [f64; 5] = [0.5, -1.0, 2.0, 3.0, -2.5];
        let vandermonde: Vec<f64> = x.iter().flat_map(|x| (0..5).map(move |k| x.powi(k))).collect();
        let want: f64 = (0..5).flat_map(|i| (i + 1..5).map(move |j| (i, j))).map(|(i, j)| x[j] - x[i]).product();
        assert!((det(vandermonde, 5) / want - 1.0).abs() < 1e-13, "{want}");
        // 0·(2 − 0) − 2·(2 − 0) + 1·(0 − 3) = −7, by the first row.
        assert!((det(vec![0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 3.0, 0.0, 2.0], 3) + 7.0).abs() < 1e-14);
        // A permutation of two rows: −1; of three in a cycle: +1.
        assert_eq!(det(vec![0.0, 1.0, 1.0, 0.0], 2), -1.0);
        assert_eq!(det(vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], 3), 1.0);
        assert_eq!(det(vec![2.0, 7.0, -3.0, 0.0, -4.0, 5.0, 0.0, 0.0, 0.5], 3), -4.0);
        assert_eq!(det(vec![1.0, 2.0, 3.0, 4.0], 2), -2.0);
        assert_eq!(det(vec![1.0, 2.0, 2.0, 4.0], 2), 0.0);
        assert_eq!(det(vec![0.0, 0.0, 3.0, 1.0], 2), 0.0);
        assert_eq!(det(vec![], 0), 1.0);
        assert_eq!(det(vec![-3.0], 1), -3.0);
    }
}
