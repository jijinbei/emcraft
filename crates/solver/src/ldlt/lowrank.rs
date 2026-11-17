//! Low-rank tiles of the factor (block low-rank, BLR): a block of L below a supernode's diagonal
//! is replaced by X Yᵀ (X rows × r, Y columns × r) when that is smaller, to a given accuracy.
//! The blocks of a front that couple two separated groups of unknowns far apart carry
//! interactions that are smooth in space, so their numerical rank is small (P. Amestoy, C.
//! Ashcraft, O. Boiteau, A. Buttari, J.-Y. L'Excellent, C. Weisbecker, *Improving multifrontal
//! methods by means of block low-rank representations*, SIAM J. Sci. Comput. 37(3):A1451–A1474,
//! 2015; for the curl–curl equation, the same holds of its discretization's fronts).
//!
//! **Compression** by QR with column pivoting truncated where what is left is small (P.
//! Businger, G. H. Golub, *Linear least squares solutions by Householder transformations*,
//! Numer. Math. 7:269–276, 1965): Householder reflections, each step taking the column of
//! largest norm left, stopping when the Frobenius norm of the trailing block, which is exactly the
//! error ‖T − X Yᵀ‖_F, is at most the tolerance — or keeping the tile dense when the rank
//! reached would store no fewer entries. Every step runs in one fixed order on one thread, so a
//! tile compresses the same bit for bit wherever it is.

use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use num_complex::Complex64 as C;

/// A tile as kept: dense (rows × columns by column) or X Yᵀ.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Tile {
    Dense(Vec<C>),
    /// X (rows × rank, by column) then Y (columns × rank, by column), one after the other.
    LowRank {
        rank: usize,
        xy: Vec<C>,
    },
}

/// Tile `t` (rows × columns, by column) as X Yᵀ to within `tol` in the Frobenius norm, or dense
/// when no rank below rows · columns / (rows + columns) reaches it.
pub(super) fn compress(t: &[C], rows: usize, cols: usize, tol: f64) -> Tile {
    let dense = || Tile::Dense(t.to_vec());
    if rows == 0 || cols == 0 || t.len() != rows * cols || !(tol >= 0.0) {
        return dense();
    }
    // The most rank worth keeping: below it X Yᵀ stores fewer entries than the tile.
    let worth = (rows * cols) / (rows + cols);
    let mut a = t.to_vec();
    let mut perm: Vec<usize> = (0..cols).collect();
    // Householder vectors (rows i.. of each) and their β.
    let mut reflectors: Vec<(Vec<C>, f64)> = Vec::new();
    let col_norm2 = |a: &[C], j: usize, from: usize| a[j * rows + from..(j + 1) * rows].iter().map(C::norm_sqr).sum::<f64>();
    let mut rank = 0;
    loop {
        let i = rank;
        let norms: Vec<f64> = (i..cols).map(|j| if i < rows { col_norm2(&a, j, i) } else { 0.0 }).collect();
        let left: f64 = norms.iter().sum();
        if left.sqrt() <= tol || i >= rows.min(cols) {
            break;
        }
        if i >= worth {
            return dense();
        }
        // The column of largest norm left (ties to the first), brought to i.
        let (best, _) = norms.iter().enumerate().fold((0, -1.0), |b, (o, &n)| if n > b.1 { (o, n) } else { b });
        let p = i + best;
        if p != i {
            for r in 0..rows {
                a.swap(i * rows + r, p * rows + r);
            }
            perm.swap(i, p);
        }
        // The reflection H = I − β v vᴴ taking column i's rows i.. to α e₁.
        let x = &a[i * rows + i..(i + 1) * rows];
        let xnorm = norms[best].sqrt();
        let phase = if x[0].norm() > 0.0 { x[0] / x[0].norm() } else { C::new(1.0, 0.0) };
        let alpha = -phase * xnorm;
        let mut v: Vec<C> = x.to_vec();
        v[0] -= alpha;
        let vv: f64 = v.iter().map(C::norm_sqr).sum();
        let beta = if vv > 0.0 { 2.0 / vv } else { 0.0 };
        for j in i..cols {
            let c = &mut a[j * rows + i..(j + 1) * rows];
            let s: C = v.iter().zip(c.iter()).map(|(vi, ci)| vi.conj() * ci).sum::<C>() * beta;
            for (ci, vi) in c.iter_mut().zip(&v) {
                *ci -= vi * s;
            }
        }
        reflectors.push((v, beta));
        rank += 1;
    }
    // X = Q's first `rank` columns: H₁ ⋯ H_r applied to e_t.
    let mut xy = vec![C::new(0.0, 0.0); rank * (rows + cols)];
    for t in 0..rank {
        let col = &mut xy[t * rows..(t + 1) * rows];
        col[t] = C::new(1.0, 0.0);
        for (i, (v, beta)) in reflectors.iter().enumerate().rev() {
            let c = &mut col[i..];
            let s: C = v.iter().zip(c.iter()).map(|(vi, ci)| vi.conj() * ci).sum::<C>() * *beta;
            for (ci, vi) in c.iter_mut().zip(v) {
                *ci -= vi * s;
            }
        }
    }
    // Y = (R Pᵀ)ᵀ: original column perm[j] holds R's column j (its rows 0..min(j + 1, rank)).
    let y = &mut xy[rank * rows..];
    for (j, &orig) in perm.iter().enumerate() {
        for t in 0..rank.min(j + 1) {
            y[t * cols + orig] = a[j * rows + t];
        }
    }
    Tile::LowRank { rank, xy }
}

impl Tile {
    /// Its entries: the dense tile's, or X's then Y's.
    pub(super) fn into_entries(self) -> Vec<C> {
        match self {
            Tile::Dense(v) => v,
            Tile::LowRank { xy, .. } => xy,
        }
    }

    /// Its rank (`None`: dense) and its entries, as [`sub_mul`] takes them.
    pub(super) fn parts(&self) -> (Option<usize>, &[C]) {
        match self {
            Tile::Dense(v) => (None, v),
            Tile::LowRank { rank, xy } => (Some(*rank), xy),
        }
    }
}

/// A kept tile as the products read it: its rank (`None`: dense), entries and shape.
#[derive(Clone, Copy)]
pub(super) struct TileRef<'a> {
    pub rank: Option<usize>,
    pub data: &'a [C],
    pub rows: usize,
    pub cols: usize,
}

/// The entries a tile of `rows` × `cols` keeps at `rank` (`None`: dense).
pub(super) fn entries(rank: Option<usize>, rows: usize, cols: usize) -> usize {
    rank.map_or(rows * cols, |r| r * (rows + cols))
}

/// dst (rows × n, by row) −= T src (columns × n, by row): the forward solve's product.
pub(super) fn sub_mul(t: TileRef<'_>, src: &[C], dst: &mut [C], n: usize) {
    let TileRef { rank, data, rows, cols } = t;
    let minus = C::new(-1.0, 0.0);
    let s = MatRef::from_row_major_slice(src, cols, n);
    let d = MatMut::from_row_major_slice_mut(dst, rows, n);
    match rank {
        None => matmul(d, Accum::Add, MatRef::from_column_major_slice(data, rows, cols), s, minus, Par::Seq),
        Some(rank) => {
            let (rank, xy) = (&rank, data);
            let (x, y) = xy.split_at(rows * rank);
            let x = MatRef::from_column_major_slice(x, rows, *rank);
            let y = MatRef::from_column_major_slice(y, cols, *rank);
            let mut w = vec![C::new(0.0, 0.0); rank * n];
            matmul(MatMut::from_row_major_slice_mut(&mut w, *rank, n), Accum::Replace, y.transpose(), s, C::new(1.0, 0.0), Par::Seq);
            matmul(d, Accum::Add, x, MatRef::from_row_major_slice(&w, *rank, n), minus, Par::Seq);
        }
    }
}

/// dst (columns × n, by row) −= Tᵀ src (rows × n, by row): the backward solve's product.
pub(super) fn sub_mul_transposed(t: TileRef<'_>, src: &[C], dst: &mut [C], n: usize) {
    let TileRef { rank, data, rows, cols } = t;
    let minus = C::new(-1.0, 0.0);
    let s = MatRef::from_row_major_slice(src, rows, n);
    let d = MatMut::from_row_major_slice_mut(dst, cols, n);
    match rank {
        None => matmul(d, Accum::Add, MatRef::from_column_major_slice(data, rows, cols).transpose(), s, minus, Par::Seq),
        Some(rank) => {
            let (rank, xy) = (&rank, data);
            let (x, y) = xy.split_at(rows * rank);
            let x = MatRef::from_column_major_slice(x, rows, *rank);
            let y = MatRef::from_column_major_slice(y, cols, *rank);
            let mut w = vec![C::new(0.0, 0.0); rank * n];
            matmul(MatMut::from_row_major_slice_mut(&mut w, *rank, n), Accum::Replace, x.transpose(), s, C::new(1.0, 0.0), Par::Seq);
            matmul(d, Accum::Add, y, MatRef::from_row_major_slice(&w, *rank, n), minus, Par::Seq);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multifrontal::testing::Rng;

    fn dense_of(tile: &Tile, rows: usize, cols: usize) -> Vec<C> {
        // T = T I: the backward product with the identity's rows gives Tᵀ, read back by column.
        let mut out = vec![C::new(0.0, 0.0); rows * cols];
        let eye: Vec<C> = (0..cols * cols).map(|i| if i / cols == i % cols { C::new(1.0, 0.0) } else { C::new(0.0, 0.0) }).collect();
        let mut by_row = vec![C::new(0.0, 0.0); rows * cols];
        let (rank, data) = tile.parts();
        sub_mul(TileRef { rank, data, rows, cols }, &eye, &mut by_row, cols);
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = -by_row[r * cols + c];
            }
        }
        out
    }

    /// A rows × cols tile of rank `rank` plus noise of size `noise`.
    fn tile_of_rank(rows: usize, cols: usize, rank: usize, noise: f64, rng: &mut Rng) -> Vec<C> {
        let u: Vec<C> = (0..rows * rank).map(|_| rng.c()).collect();
        let v: Vec<C> = (0..cols * rank).map(|_| rng.c()).collect();
        (0..rows * cols)
            .map(|i| {
                let (r, c) = (i % rows, i / rows);
                (0..rank).map(|t| u[t * rows + r] * v[t * cols + c]).sum::<C>() + rng.c() * noise
            })
            .collect()
    }

    fn frobenius(a: &[C], b: &[C]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).norm_sqr()).sum::<f64>().sqrt()
    }

    #[test]
    fn a_tile_of_low_rank_is_kept_as_its_rank_within_the_tolerance() {
        // Rank 5 exactly: compressed to rank 5, reproduced to round-off; with noise 1e-9 and a
        // tolerance above the noise's norm, still rank 5 and within the tolerance.
        let mut rng = Rng(7);
        for (noise, tol) in [(0.0, 1e-10), (1e-9, 1e-6)] {
            let t = tile_of_rank(60, 40, 5, noise, &mut rng);
            let c = compress(&t, 60, 40, tol);
            let Tile::LowRank { rank, .. } = c else { panic!("not compressed") };
            assert_eq!(rank, 5);
            assert!(frobenius(&dense_of(&c, 60, 40), &t) <= tol * 1.0001, "{}", frobenius(&dense_of(&c, 60, 40), &t));
        }
    }

    #[test]
    fn a_tile_of_full_rank_stays_dense() {
        let mut rng = Rng(11);
        let t: Vec<C> = (0..30 * 20).map(|_| rng.c()).collect();
        assert_eq!(compress(&t, 30, 20, 1e-8), Tile::Dense(t));
    }

    #[test]
    fn the_products_match_the_dense_tile() {
        // Forward and backward products of a compressed tile against the dense ones, three
        // right-hand sides.
        let mut rng = Rng(3);
        let t = tile_of_rank(50, 30, 4, 0.0, &mut rng);
        let c = compress(&t, 50, 30, 1e-12);
        assert!(matches!(c, Tile::LowRank { rank: 4, .. }));
        let d = Tile::Dense(t);
        let src: Vec<C> = (0..30 * 3).map(|_| rng.c()).collect();
        let (mut a, mut b) = (vec![C::new(0.0, 0.0); 50 * 3], vec![C::new(0.0, 0.0); 50 * 3]);
        sub_mul(TileRef { rank: c.parts().0, data: c.parts().1, rows: 50, cols: 30 }, &src, &mut a, 3);
        sub_mul(TileRef { rank: d.parts().0, data: d.parts().1, rows: 50, cols: 30 }, &src, &mut b, 3);
        assert!(frobenius(&a, &b) < 1e-10, "{}", frobenius(&a, &b));
        let src: Vec<C> = (0..50 * 3).map(|_| rng.c()).collect();
        let (mut a, mut b) = (vec![C::new(0.0, 0.0); 30 * 3], vec![C::new(0.0, 0.0); 30 * 3]);
        sub_mul_transposed(TileRef { rank: c.parts().0, data: c.parts().1, rows: 50, cols: 30 }, &src, &mut a, 3);
        sub_mul_transposed(TileRef { rank: d.parts().0, data: d.parts().1, rows: 50, cols: 30 }, &src, &mut b, 3);
        assert!(frobenius(&a, &b) < 1e-10, "{}", frobenius(&a, &b));
    }

    #[test]
    fn hostile_tiles_are_kept_dense() {
        // Wrong lengths, an empty tile, a tolerance that is not a number: kept as given.
        let t = vec![C::new(1.0, 0.0); 6];
        assert_eq!(compress(&t, 4, 2, 1e-8), Tile::Dense(t.clone()));
        assert_eq!(compress(&[], 0, 3, 1e-8), Tile::Dense(Vec::new()));
        assert_eq!(compress(&t, 3, 2, f64::NAN), Tile::Dense(t));
    }
}
