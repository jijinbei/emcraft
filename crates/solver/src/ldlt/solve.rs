//! The solves with the factor: L z = b, D w = z, Lᵀ x = w, supernode by supernode.

use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use num_complex::Complex64 as C;

use super::front::Pivot;
use super::{Ldlt, Supernode, lowrank};
use crate::multifrontal::{FactorError, refine, refine_many};
use crate::sparse::Csr;

/// The fewest refinement steps a compressed factor's solve takes: each step gains about the
/// digits the factor keeps.
const COMPRESSED_STEPS: usize = 30;

impl Ldlt {
    /// Solves A x = b in place.
    pub fn solve(&self, b: &mut [C]) -> Result<(), FactorError> {
        let mut one = [b.to_vec()];
        self.solve_many(&mut one)?;
        b.copy_from_slice(&one[0]);
        Ok(())
    }

    /// Solves A X = B in place for several right-hand sides at once: each pass over the factor
    /// serves them all (the solve reads the whole factor and does little with each entry): each
    /// supernode's rows below its diagonal block as one matrix product with all of them. The
    /// products run on one thread, so the result does not depend on the machine's.
    pub fn solve_many(&self, bs: &mut [Vec<C>]) -> Result<(), FactorError> {
        let r = bs.len();
        if r == 0 {
            return Ok(());
        }
        if bs.iter().any(|b| b.len() != self.n) {
            return Err(FactorError::Dimension(format!("right-hand sides for {} unknowns", self.n)));
        }
        // y[position · r + j].
        let mut y = crate::memory::zeros(self.n * r, "the right-hand sides")?;
        for (pos, &u) in self.perm.iter().enumerate() {
            for (j, b) in bs.iter().enumerate() {
                y[pos * r + j] = b[u] * self.scale[u];
            }
        }
        let mut local: Vec<C> = Vec::new();
        for s in &self.supernodes {
            self.forward(s, &mut y, r, &mut local)?;
        }
        for s in &self.supernodes {
            diagonal(s, &mut y, r);
        }
        let mut acc = Vec::new();
        for s in self.supernodes.iter().rev() {
            self.backward(s, &mut y, r, &mut local, &mut acc)?;
        }
        for (pos, &u) in self.perm.iter().enumerate() {
            for (j, b) in bs.iter_mut().enumerate() {
                b[u] = y[pos * r + j] * self.scale[u];
            }
        }
        Ok(())
    }

    /// Supernode `s`'s part of L z = b on the `r` interleaved right-hand sides `y`, through the
    /// buffer `local`: its diagonal block by columns, then the rows below it at once,
    /// local[k..] −= L₂₁ local[..k] (a matrix product: the factor is read once per pass however
    /// many right-hand sides share it). Kept as tiles: each diagonal tile in turn, then the
    /// tiles below it in its columns.
    fn forward(&self, s: &Supernode, y: &mut [C], r: usize, local: &mut Vec<C>) -> Result<(), FactorError> {
        let (m, k) = (s.rows.len(), s.k);
        checked(s)?;
        gather(local, y, &s.rows, r);
        if s.tiles.is_empty() {
            let cols = self.piece(s, 0, m * k)?;
            diagonal_forward(Diagonal { block: &cols, n: m, from: 0, to: k, pivots: &s.pivots }, local, r);
            if m > k && r > 0 {
                let (top, below) = local.split_at_mut(k * r);
                let l21 = MatRef::from_column_major_slice(&cols, m, k).subrows(k, m - k);
                let x1 = MatRef::from_row_major_slice(top, k, r);
                let dst = MatMut::from_row_major_slice_mut(below, m - k, r);
                matmul(dst, Accum::Add, l21, x1, C::new(-1.0, 0.0), Par::Seq);
            }
        } else {
            for (d, below) in panels(&s.tiles) {
                let block = self.piece(s, d.0, d.1.len())?;
                diagonal_forward(Diagonal { block: &block, n: d.1.rows, from: d.1.col0, to: d.1.col0 + d.1.cols, pivots: &s.pivots }, local, r);
                for (i, t) in below {
                    let data = self.piece(s, i, t.len())?;
                    // Rows below the columns: split the buffer between them.
                    let (head, tail) = local.split_at_mut(t.row0 * r);
                    let src = &head[t.col0 * r..(t.col0 + t.cols) * r];
                    lowrank::sub_mul(lowrank::TileRef { rank: t.rank, data: &data, rows: t.rows, cols: t.cols }, src, &mut tail[..t.rows * r], r);
                }
            }
        }
        scatter(local, y, &s.rows, r);
        Ok(())
    }

    /// Supernode `s`'s part of Lᵀ x = w, with the buffers `local` and `acc`: the rows below the
    /// diagonal block at once, local[..k] −= L₂₁ᵀ local[k..], then the block by columns. Kept as
    /// tiles: the diagonal tiles last to first, each after the tiles below it.
    fn backward(&self, s: &Supernode, y: &mut [C], r: usize, local: &mut Vec<C>, acc: &mut Vec<C>) -> Result<(), FactorError> {
        let (m, k) = (s.rows.len(), s.k);
        checked(s)?;
        gather(local, y, &s.rows, r);
        if s.tiles.is_empty() {
            let cols = self.piece(s, 0, m * k)?;
            if m > k && r > 0 {
                let (top, below) = local.split_at_mut(k * r);
                let l21 = MatRef::from_column_major_slice(&cols, m, k).subrows(k, m - k);
                let x2 = MatRef::from_row_major_slice(&*below, m - k, r);
                let dst = MatMut::from_row_major_slice_mut(top, k, r);
                matmul(dst, Accum::Add, l21.transpose(), x2, C::new(-1.0, 0.0), Par::Seq);
            }
            diagonal_backward(Diagonal { block: &cols, n: m, from: 0, to: k, pivots: &s.pivots }, local, r, acc);
        } else {
            for (d, below) in panels(&s.tiles).into_iter().rev() {
                for (i, t) in below {
                    let data = self.piece(s, i, t.len())?;
                    let (head, tail) = local.split_at_mut(t.row0 * r);
                    let dst = &mut head[t.col0 * r..(t.col0 + t.cols) * r];
                    lowrank::sub_mul_transposed(
                        lowrank::TileRef { rank: t.rank, data: &data, rows: t.rows, cols: t.cols },
                        &tail[..t.rows * r],
                        dst,
                        r,
                    );
                }
                let block = self.piece(s, d.0, d.1.len())?;
                diagonal_backward(Diagonal { block: &block, n: d.1.rows, from: d.1.col0, to: d.1.col0 + d.1.cols, pivots: &s.pivots }, local, r, acc);
            }
        }
        scatter(local, y, &s.rows[..k], r);
        Ok(())
    }

    /// Piece `i` of supernode `s`'s columns (from memory or disk), refused unless it holds
    /// `len` entries.
    fn piece<'a>(&'a self, s: &'a Supernode, i: usize, len: usize) -> Result<std::borrow::Cow<'a, [C]>, FactorError> {
        let piece = s.l.get(i).ok_or_else(|| FactorError::Dimension(format!("a supernode without its piece {i}")))?.get(self.scratch.as_deref())?;
        if piece.len() != len {
            return Err(FactorError::Dimension(format!("a supernode's piece {i} holds {} entries, not {len}", piece.len())));
        }
        Ok(piece)
    }

    /// Solves A x = b with iterative refinement until the backward error
    /// ‖b − A x‖∞ / (‖A‖ ‖x‖∞ + ‖b‖∞) stops falling (at most `steps` corrections); returns x and
    /// that error.
    pub fn solve_refined(&self, a: &Csr, b: &[C], steps: usize) -> Result<(Vec<C>, f64), FactorError> {
        refine(a, b, self.steps(steps), |x| self.solve(x))
    }

    /// The refinement steps a solve takes: `steps`, or with a compressed factor (a few digits
    /// short of exact) at least [`COMPRESSED_STEPS`].
    fn steps(&self, steps: usize) -> usize {
        if self.compressed.is_some() { steps.max(COMPRESSED_STEPS) } else { steps }
    }

    /// [`Ldlt::solve_refined`] for several right-hand sides, each pass over the factor serving
    /// them all ([`Ldlt::solve_many`]).
    pub fn solve_refined_many(&self, a: &Csr, bs: &[Vec<C>], steps: usize) -> Result<Vec<(Vec<C>, f64)>, FactorError> {
        refine_many(a, bs, self.steps(steps), |xs| self.solve_many(xs))
    }
}

/// Checks supernode `s`'s tiles lie within its front, each diagonal one square and dense, each
/// other below one, a piece for each and the diagonal ones covering its k columns.
fn checked(s: &Supernode) -> Result<(), FactorError> {
    let (m, k) = (s.rows.len(), s.k);
    let tile_ok = |t: &super::Placed| {
        t.row0 + t.rows <= m
            && t.col0 + t.cols <= k
            && if t.row0 == t.col0 { t.rows == t.cols && t.rank.is_none() } else { t.row0 >= t.col0 + t.cols }
    };
    let diagonal = panels(&s.tiles).iter().map(|(d, _)| d.1.cols).sum::<usize>();
    let pieces = if s.tiles.is_empty() { 1 } else { s.tiles.len() };
    if s.l.len() != pieces || !s.tiles.iter().all(tile_ok) || (!s.tiles.is_empty() && diagonal != k) {
        return Err(FactorError::Dimension(format!("a supernode's {} tiles do not make its {m} × {k} columns", s.tiles.len())));
    }
    Ok(())
}

/// The tiles as panels: each diagonal tile (in the order kept) with the tiles below it in its
/// columns, each with its index (its piece).
type Indexed = (usize, super::Placed);
fn panels(tiles: &[super::Placed]) -> Vec<(Indexed, Vec<Indexed>)> {
    let mut out: Vec<(Indexed, Vec<Indexed>)> = Vec::new();
    for (i, t) in tiles.iter().enumerate() {
        if t.row0 == t.col0 {
            out.push(((i, *t), Vec::new()));
        } else if let Some((_, below)) = out.iter_mut().rev().find(|((_, d), _)| t.col0 >= d.col0 && t.col0 < d.col0 + d.cols) {
            below.push((i, *t));
        }
    }
    out
}

/// A unit lower diagonal block: its entries (by column, `n` apart) for the columns `from..to`
/// (front rows `from..` likewise), and the supernode's pivots.
struct Diagonal<'a> {
    block: &'a [C],
    n: usize,
    from: usize,
    to: usize,
    pivots: &'a [Pivot],
}

/// Forward substitution on `local` with the diagonal block `d`, 2 × 2 pivots' columns together.
fn diagonal_forward(d: Diagonal<'_>, local: &mut [C], r: usize) {
    let Diagonal { block, n, from, to, pivots } = d;
    let zero = C::new(0.0, 0.0);
    let mut p = from;
    while p < to {
        let width = if matches!(pivots.get(p), Some(Pivot::Two(..))) { 2 } else { 1 };
        for c in p..(p + width).min(to) {
            let xc: Vec<C> = local[c * r..c * r + r].to_vec();
            if xc.iter().all(|v| *v == zero) {
                continue;
            }
            for i in p + width..to {
                let l = block[(c - from) * n + (i - from)];
                if l == zero {
                    continue;
                }
                for (t, x) in local[i * r..i * r + r].iter_mut().zip(&xc) {
                    *t -= l * x;
                }
            }
        }
        p += width;
    }
}

/// [`diagonal_forward`]'s transpose: back substitution on `local`, through the buffer `acc`.
fn diagonal_backward(d: Diagonal<'_>, local: &mut [C], r: usize, acc: &mut Vec<C>) {
    let Diagonal { block, n, from, to, pivots } = d;
    let zero = C::new(0.0, 0.0);
    acc.clear();
    acc.resize(r, zero);
    let mut p = to;
    while p > from {
        let width = if p >= from + 2 && matches!(pivots.get(p - 2), Some(Pivot::Two(..))) { 2 } else { 1 };
        let start = p - width;
        for c in start..p {
            acc.iter_mut().for_each(|a| *a = zero);
            for i in p..to {
                let l = block[(c - from) * n + (i - from)];
                for (a, x) in acc.iter_mut().zip(&local[i * r..i * r + r]) {
                    *a += l * x;
                }
            }
            for (t, a) in local[c * r..c * r + r].iter_mut().zip(acc.iter()) {
                *t -= a;
            }
        }
        p = start;
    }
}

/// Supernode `s`'s part of D w = z.
fn diagonal(s: &Supernode, y: &mut [C], r: usize) {
    let mut p = 0;
    while p < s.k {
        match s.pivots[p] {
            Pivot::One(d) => {
                for v in &mut y[s.rows[p] * r..s.rows[p] * r + r] {
                    *v /= d;
                }
                p += 1;
            }
            Pivot::Two(a11, a21, a22) => {
                let (r0, r1) = (s.rows[p], s.rows[p + 1]);
                let det = a11 * a22 - a21 * a21;
                for j in 0..r {
                    let (b0, b1) = (y[r0 * r + j], y[r1 * r + j]);
                    y[r0 * r + j] = (a22 * b0 - a21 * b1) / det;
                    y[r1 * r + j] = (a11 * b1 - a21 * b0) / det;
                }
                p += 2;
            }
            Pivot::Second => p += 1,
        }
    }
}

/// `local` = the `r` values of each of `rows` in `y`.
fn gather(local: &mut Vec<C>, y: &[C], rows: &[usize], r: usize) {
    local.clear();
    for &row in rows {
        local.extend_from_slice(&y[row * r..row * r + r]);
    }
}

/// The values of `local` back to `rows` of `y`.
fn scatter(local: &[C], y: &mut [C], rows: &[usize], r: usize) {
    for (k, &row) in rows.iter().enumerate() {
        y[row * r..row * r + r].copy_from_slice(&local[k * r..k * r + r]);
    }
}
