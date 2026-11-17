//! A large front factored by panels, never formed whole: its columns in panels of [`PANEL`]
//! (the supernode's own columns first, then the update's), each assembled from A and the
//! children's updates on its own, brought up to date from the panels factored before it
//! (left-looking: the update of panel J is Σ_K L_{·K} D_K L_{JK}ᵀ over the earlier panels K),
//! then — one of the supernode's — eliminated with the same block pivoting as a whole front
//! ([`super::front::factor_panel`]) and kept as tiles, each low-rank where that is smaller
//! ([`super::lowrank`]); or — one of the update's — handed to the parent as it is. The front
//! then takes a panel of m rows and the factor's tiles, not m² entries (the left-looking form of
//! the block low-rank factorization: P. Amestoy, C. Ashcraft, O. Boiteau, A. Buttari, J.-Y.
//! L'Excellent, C. Weisbecker, SIAM J. Sci. Comput. 37(3):A1451–A1474, 2015, §3).
//!
//! Pivots are chosen among a panel's own columns; an interchange of two of its variables swaps
//! their rows in the tiles of the earlier panels too (the factor's rows follow the front's), and
//! touches no later panel (those hold only rows below their own first column). Each tile's
//! update sums the earlier panels in order, so the result is the same on any number of threads.

use std::ops::Range;

use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use num_complex::Complex64 as C;
use rayon::prelude::*;

use super::front::{Pivot, PivotCounts, factor_panel};
use super::{Placed, lowrank};
use crate::multifrontal::FactorError;

/// A front factored by panels: its tiles (each diagonal one followed by those below it) and
/// their entries, its pivots, what the pivoting counted, and its update for the parent (lower,
/// packed by column).
pub(super) struct ByPanels {
    pub pieces: Vec<Vec<C>>,
    pub tiles: Vec<Placed>,
    pub pivots: Vec<Pivot>,
    pub counts: PivotCounts,
    pub update: Vec<C>,
}

/// `0..k` then `k..m` in pieces of `width` at most.
fn ranges(m: usize, k: usize, width: usize) -> Vec<Range<usize>> {
    let split = |r: Range<usize>| (r.start..r.end).step_by(width.max(1)).map(move |a| a..(a + width).min(r.end)).collect::<Vec<_>>();
    let mut out = split(0..k);
    out.extend(split(k..m));
    out
}

/// A kept tile as a dense matrix (rows × cols, by column).
fn dense(t: &Placed, data: &[C]) -> Vec<C> {
    match t.rank {
        None => data.to_vec(),
        Some(r) => {
            let (x, y) = data.split_at(t.rows * r);
            let mut out = vec![C::new(0.0, 0.0); t.rows * t.cols];
            matmul(
                MatMut::from_column_major_slice_mut(&mut out, t.rows, t.cols),
                Accum::Replace,
                MatRef::from_column_major_slice(x, t.rows, r),
                MatRef::from_column_major_slice(y, t.cols, r).transpose(),
                C::new(1.0, 0.0),
                Par::Seq,
            );
            out
        }
    }
}

/// D Lᵀ for a panel's `pivots` and a tile `l` (rows × w, by column) of its columns: w × rows,
/// by column.
fn d_lt(pivots: &[Pivot], l: &[C], rows: usize, w: usize) -> Vec<C> {
    let mut out = vec![C::new(0.0, 0.0); w * rows];
    let at = |i: usize, t: usize| l[t * rows + i];
    let mut t = 0;
    while t < w {
        match pivots.get(t) {
            Some(Pivot::Two(a11, a21, a22)) if t + 1 < w => {
                for i in 0..rows {
                    let (x0, x1) = (at(i, t), at(i, t + 1));
                    out[i * w + t] = a11 * x0 + a21 * x1;
                    out[i * w + t + 1] = a21 * x0 + a22 * x1;
                }
                t += 2;
            }
            Some(Pivot::One(d)) => {
                for i in 0..rows {
                    out[i * w + t] = d * at(i, t);
                }
                t += 1;
            }
            _ => t += 1,
        }
    }
    out
}

/// out (rows × cols, by column) = T W for a kept tile T (rows × w) and W (w × cols, by column).
fn product(t: &Placed, data: &[C], w_mat: &[C], cols: usize, out: &mut [C]) {
    let wm = MatRef::from_column_major_slice(w_mat, t.cols, cols);
    let dst = MatMut::from_column_major_slice_mut(out, t.rows, cols);
    match t.rank {
        None => matmul(dst, Accum::Replace, MatRef::from_column_major_slice(data, t.rows, t.cols), wm, C::new(1.0, 0.0), Par::Seq),
        Some(r) => {
            let (x, y) = data.split_at(t.rows * r);
            let mut yw = vec![C::new(0.0, 0.0); r * cols];
            matmul(
                MatMut::from_column_major_slice_mut(&mut yw, r, cols),
                Accum::Replace,
                MatRef::from_column_major_slice(y, t.cols, r).transpose(),
                wm,
                C::new(1.0, 0.0),
                Par::Seq,
            );
            matmul(
                dst,
                Accum::Replace,
                MatRef::from_column_major_slice(x, t.rows, r),
                MatRef::from_column_major_slice(&yw, r, cols),
                C::new(1.0, 0.0),
                Par::Seq,
            );
        }
    }
}

/// Swaps rows p and q of a kept tile.
fn swap_rows(t: &Placed, data: &mut [C], p: usize, q: usize) {
    let (rows, cols) = match t.rank {
        None => (t.rows, t.cols),
        Some(r) => (t.rows, r),
    };
    for c in 0..cols {
        data.swap(c * rows + p, c * rows + q);
    }
}

/// How a front is factored by panels: its first `k` rows the supernode's, the static-pivot
/// threshold, the tiles' tolerance per entry (`compress`, [`super::with_compression`]) and the
/// panels' columns.
pub(super) struct Panels {
    pub k: usize,
    pub tau: f64,
    pub compress: Option<f64>,
    pub width: usize,
}

/// Factors a front of `rows.len()` rows (`rows`, its positions, follow the interchanges) by
/// panels as `how` says: `assemble(columns, panel)` adds A's and the children's entries of
/// `columns` (rows from the first of them down, the front's order before any interchange) into
/// the panel, column-major with m − columns.start rows.
pub(super) fn factor_by_panels(
    how: &Panels,
    rows: &mut [usize],
    mut assemble: impl FnMut(Range<usize>, &mut [C]) -> Result<(), FactorError>,
) -> Result<ByPanels, FactorError> {
    let &Panels { k, tau, compress, width } = how;
    let m = rows.len();
    let panels = ranges(m, k, width);
    let mu = m - k;
    let mut out = ByPanels {
        pieces: Vec::new(),
        tiles: Vec::new(),
        pivots: vec![Pivot::Second; k],
        counts: PivotCounts::default(),
        update: crate::memory::with_capacity(mu * (mu + 1) / 2, "an update matrix")?,
    };
    // first[K][I − K]: the piece of tile (I, K), for the supernode's panels K.
    let mut first: Vec<usize> = Vec::new();
    for (j, cols) in panels.iter().enumerate() {
        let (c0, w) = (cols.start, cols.len());
        let mj = m - c0;
        let mut panel = crate::memory::zeros(mj * w, "a panel")?;
        assemble(cols.clone(), &mut panel)?;
        // The earlier panels of the supernode, in order: panel −= Σ L_{·K} D_K L_{JK}ᵀ.
        for (kk, &base) in first.iter().enumerate() {
            let tile_j = base + (j - kk);
            let (tj, kcols) = (out.tiles[tile_j], &panels[kk]);
            let w_mat = d_lt(&out.pivots[kcols.clone()], &dense(&tj, &out.pieces[tile_j]), tj.rows, kcols.len());
            let below: Vec<usize> = (tile_j..base + (panels.len() - kk)).collect();
            let products: Vec<Vec<C>> = below
                .par_iter()
                .map(|&i| {
                    let t = &out.tiles[i];
                    let mut prod = vec![C::new(0.0, 0.0); t.rows * w];
                    product(t, &out.pieces[i], &w_mat, w, &mut prod);
                    prod
                })
                .collect();
            for (&i, prod) in below.iter().zip(&products) {
                let t = &out.tiles[i];
                for c in 0..w {
                    let dst = &mut panel[c * mj + (t.row0 - c0)..c * mj + (t.row0 - c0) + t.rows];
                    for (d, p) in dst.iter_mut().zip(&prod[c * t.rows..(c + 1) * t.rows]) {
                        *d -= p;
                    }
                }
            }
        }
        if c0 >= k {
            // One of the update's panels: its columns, lower, for the parent.
            for c in 0..w {
                out.update.extend_from_slice(&panel[c * mj + c..(c + 1) * mj]);
            }
            continue;
        }
        // The diagonal block whole (its upper triangle the lower's mirror), then eliminated.
        for c in 0..w {
            for r in 0..c {
                panel[c * mj + r] = panel[r * mj + c];
            }
        }
        let (pivots, counts, swaps) = factor_panel(&mut panel, mj, w, &mut rows[c0..], tau)?;
        out.counts.perturbed += counts.perturbed;
        out.counts.two_by_two += counts.two_by_two;
        out.pivots[c0..c0 + w].copy_from_slice(&pivots);
        // The interchanges in the earlier panels' tiles of these rows.
        for (kk, &base) in first.iter().enumerate() {
            let i = base + (j - kk);
            let t = out.tiles[i];
            for &(p, q) in &swaps {
                swap_rows(&t, &mut out.pieces[i], p, q);
            }
        }
        // Kept: the diagonal tile, then each tile below, compressed where asked.
        first.push(out.tiles.len());
        for rr in &panels[j..] {
            let r0 = rr.start - c0;
            let entries: Vec<C> = (0..w).flat_map(|c| panel[c * mj + r0..c * mj + r0 + rr.len()].iter().copied()).collect();
            let diagonal = rr.start == c0;
            let tile = match compress {
                Some(tol) if !diagonal => lowrank::compress(&entries, rr.len(), w, tol * ((rr.len() * w) as f64).sqrt()),
                _ => lowrank::Tile::Dense(entries),
            };
            out.tiles.push(Placed { row0: rr.start, rows: rr.len(), col0: c0, cols: w, rank: tile.parts().0 });
            out.pieces.push(tile.into_entries());
        }
    }
    Ok(out)
}
