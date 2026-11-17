//! Sparse complex symmetric factorization P A Pᵀ = L D Lᵀ (A = Aᵀ, not Hermitian) by the
//! multifrontal method: L unit lower triangular, D block diagonal with 1 × 1 and 2 × 2 pivots.
//!
//! **Structure.** The method's ordering, supernodes and traversal are
//! [`crate::multifrontal`]'s: each supernode is factored in a dense frontal matrix holding its
//! columns of A plus the update matrices its children left behind (extend-add), the supernode's
//! own columns eliminated, and the Schur complement passed to the parent (I. S. Duff, J. K. Reid,
//! *The multifrontal solution of indefinite sparse symmetric linear equations*, ACM TOMS
//! 9(3):302–325, 1983; J. W. H. Liu, *The multifrontal method for sparse matrix solution: theory
//! and practice*, SIAM Review 34(1):82–109, 1992). A front is symmetric: its lower triangle is
//! assembled into both, and the update matrices are passed packed (lower, by column).
//!
//! **Pivoting** ([`front`]). Within a front, the supernode's columns are fully summed and may be
//! interchanged freely without changing any structure. The pivot is chosen among them by the
//! Bunch–Kaufman test (J. R. Bunch, L. Kaufman, *Some stable methods for calculating inertia
//! and solving symmetric linear systems*, Math. Comp. 31(137):163–179, 1977) on the moduli of the
//! complex entries, as LAPACK's `zsytrf` does for complex symmetric matrices: with
//! α = (1 + √17)/8 and ω_j the largest off-diagonal modulus of column j, column p is a 1 × 1
//! pivot when |a_pp| ≥ α ω_p; otherwise its largest entry's row r gives a 1 × 1 pivot at r or the
//! 2 × 2 pivot (p, r). When r lies outside the supernode (not yet fully summed), another column
//! of the supernode that passes the test is taken, else the best 2 × 2 or 1 × 1 inside it, and a
//! pivot below √ε ‖A‖ is replaced by that value (static pivoting, as in O. Schenk, K. Gärtner,
//! *Solving unsymmetric sparse systems of linear equations with PARDISO*, FGCS 20(3):475–487,
//! 2004); [`Ldlt::solve_refined`] then recovers the accuracy by iterative refinement and reports
//! the backward error.
//!
//! **Storage** (`storage`). The factored columns are kept in memory up to a budget, the rest
//! in a scratch file on disk. Within [`with_compression`], the rows of a large front's columns
//! below its diagonal block are kept as tiles, each low-rank where that is smaller
//! (`lowrank`): the factor is then exact to that tolerance, and [`Ldlt::solve_refined`] (or a
//! Krylov method it preconditions) recovers the rest.

use std::cell::Cell;
use std::ops::Range;
use std::sync::Arc;

use emcraft_model::Point3;
use num_complex::Complex64 as C;

use crate::multifrontal::{
    Child, Counts, Eliminated, Front, FrontKernel, PANEL, Plan, Shape, Triangle, Update, Waiting, available_memory, check, factor_tree, fill, threads,
};
use crate::sparse::Csr;

mod front;
mod lowrank;
mod panel;
mod solve;
mod storage;

pub use crate::multifrontal::{Analysis, FactorError, MAX_FRONT};
pub(crate) use storage::memory_budget;

use front::{Pivot, factor_front, factor_front_on};
use storage::{Columns, Scratch, Store};

#[derive(Clone, Debug)]
struct Supernode {
    /// Positions of the front's rows; the first `k` are the supernode's columns in pivot order.
    rows: Vec<usize>,
    k: usize,
    /// The front's first k columns after elimination: L below the diagonal (m × k, by column).
    l: Vec<Columns>,
    pivots: Vec<Pivot>,
    /// Empty: `l` is one piece, the m × k columns. Otherwise `l` holds a piece per tile, in this
    /// order: each diagonal tile followed by those below it in its columns.
    tiles: Vec<Placed>,
}

/// A tile of a supernode's columns: its rows and columns in the front, its rank (`None`:
/// dense). A diagonal tile (rows from its first
/// column) is square, dense and unit lower; the others lie below one, in its columns.
#[derive(Clone, Copy, Debug)]
struct Placed {
    row0: usize,
    rows: usize,
    col0: usize,
    cols: usize,
    rank: Option<usize>,
}

impl Placed {
    fn len(&self) -> usize {
        lowrank::entries(self.rank, self.rows, self.cols)
    }
}

/// The side of a compressed tile (rows and columns). Fronts are compressed from twice as many
/// rows below their diagonal block and half as many columns: smaller ones would hold tiles too
/// small to fall in rank.
const TILE: usize = 256;

thread_local! {
    static COMPRESSION: Cell<Option<f64>> = const { Cell::new(None) };
}

/// Runs `f` with the factorizations it starts on this thread compressed to `tol` (`None`: kept
/// exact): each tile of a large front below its diagonal block within `tol` per entry, root mean
/// square, of the equilibrated matrix (whose largest entry in each row is one). The previous
/// setting is restored after, even if `f` unwinds.
///
/// ```
/// use emcraft_solver::ldlt::{compression, with_compression};
/// assert_eq!(with_compression(Some(1e-8), compression), Some(1e-8));
/// assert_eq!(compression(), None);
/// // A tolerance that is not a positive number compresses nothing.
/// assert_eq!(with_compression(Some(-1.0), compression), None);
/// ```
pub fn with_compression<T>(tol: Option<f64>, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<f64>);
    impl Drop for Restore {
        fn drop(&mut self) {
            COMPRESSION.with(|c| c.set(self.0));
        }
    }
    let tol = tol.filter(|t| t.is_finite() && *t > 0.0);
    let _restore = Restore(COMPRESSION.with(|c| c.replace(tol)));
    f()
}

/// The calling thread's compression tolerance ([`with_compression`]).
pub fn compression() -> Option<f64> {
    COMPRESSION.with(Cell::get)
}

/// The factored columns of a front (column-major m × k, the first k of `f`) as the k × k
/// diagonal block and the tiles (`tile` rows and columns at most) of the rows below it, each
/// compressed to `tol` per entry (tiles on threads; each compresses the same wherever it runs).
fn compressed(f: &[C], m: usize, k: usize, tol: f64, tile: usize) -> Result<(Vec<Vec<C>>, Vec<Placed>), FactorError> {
    use rayon::prelude::*;
    let blocks: Vec<(usize, usize, usize, usize)> =
        (k..m).step_by(tile).flat_map(|r0| (0..k).step_by(tile).map(move |c0| (r0, (m - r0).min(tile), c0, (k - c0).min(tile)))).collect();
    let tiles: Vec<lowrank::Tile> = blocks
        .par_iter()
        .map(|&(r0, rows, c0, cols)| {
            let t: Vec<C> = (0..cols).flat_map(|c| f[(c0 + c) * m + r0..(c0 + c) * m + r0 + rows].iter().copied()).collect();
            lowrank::compress(&t, rows, cols, tol * ((rows * cols) as f64).sqrt())
        })
        .collect::<Result<_, _>>()?;
    let mut diagonal = crate::memory::with_capacity(k * k, "the factor's diagonal block")?;
    for c in 0..k {
        diagonal.extend_from_slice(&f[c * m..c * m + k]);
    }
    let mut pieces = vec![diagonal];
    let mut placed = vec![Placed { row0: 0, rows: k, col0: 0, cols: k, rank: None }];
    for (&(row0, rows, col0, cols), t) in blocks.iter().zip(tiles) {
        placed.push(Placed { row0, rows, col0, cols, rank: t.parts().0 });
        pieces.push(t.into_entries());
    }
    Ok((pieces, placed))
}

/// What a test asks of a factorization beyond its plan: the updates on disk, fronts by panels
/// from a size.
#[derive(Clone, Copy, Debug, Default)]
struct Forced {
    spill: bool,
    panel_from: Option<usize>,
}

/// A factorization.
#[derive(Clone, Debug)]
pub struct Ldlt {
    n: usize,
    /// `perm[position] = unknown`.
    perm: Vec<usize>,
    supernodes: Vec<Supernode>,
    /// The columns kept on disk, when the factor outgrew the memory budget.
    scratch: Option<Arc<Scratch>>,
    /// Bytes of the factor on disk.
    pub bytes_on_disk: u64,
    /// The symmetric scaling the factor is of: S A S, s_i = 1/√(max_j |a_ij|).
    scale: Vec<f64>,
    /// Pivots replaced by √ε ‖S A S‖.
    pub perturbed: usize,
    pub two_by_two: usize,
    /// Entries of L (with the diagonal blocks).
    pub entries: usize,
    /// The largest front.
    pub largest_front: usize,
    /// The tolerance its large fronts were compressed to ([`with_compression`]), if any.
    pub compressed: Option<f64>,
    /// Seconds of the symbolic analysis (when made with it) and of the numeric factorization.
    pub analysis_seconds: f64,
    pub numeric_seconds: f64,
}

/// The start of column b of a packed lower triangle of order m (columns stored from their
/// diagonal down).
fn packed(b: usize, m: usize) -> usize {
    fill(b, m)
}

/// The LDLᵀ's fronts: A's lower columns by position (S A S), the static-pivot threshold, and
/// where the factored columns go.
struct Kernel {
    /// Column j's entries on and below the diagonal: (row position, value).
    a_cols: Vec<Vec<(usize, C)>>,
    tau: f64,
    store: Store,
    /// The tiles' tolerance and side ([`with_compression`]).
    compress: Option<(f64, usize)>,
    /// Fronts of this many rows or more factored by panels ([`panel`]).
    panel_from: Option<usize>,
}

/// The LDLᵀ's fronts: each update the lower triangle of m − k rows (by column), the factor the
/// front's first k columns, which may go to disk.
const SHAPE: Shape = Shape { update: Triangle::Lower, kept: |k, m| k * m, out_of_core: true, panels: true };

impl FrontKernel for Kernel {
    type Node = Supernode;

    fn assemble(&self, columns: Range<usize>, front: &mut Front) -> Result<(), FactorError> {
        let m = front.m();
        for j in columns.clone() {
            let cj = j - columns.start;
            for &(i, v) in self.a_cols.get(j).map_or(&[][..], Vec::as_slice) {
                let r = front.place_entry(i)?;
                front.f[cj * m + r] += v;
                if r != cj {
                    front.f[r * m + cj] += v;
                }
            }
        }
        Ok(())
    }

    /// The update is packed lower by columns (i ≥ j); positions in this front ascend with its
    /// rows.
    fn extend_add(&self, f: &mut [C], m: usize, place: &[usize], u: &[C]) {
        let mu = place.len();
        for b in 0..mu {
            let cb = place[b];
            let start = packed(b, mu);
            for a in b..mu {
                let v = u[start + a - b];
                let ra = place[a];
                f[cb * m + ra] += v;
                if ra != cb {
                    f[ra * m + cb] += v;
                }
            }
        }
    }

    /// Factors the front's columns (on the calling thread's device when it takes the front) and
    /// keeps them (in memory, or on disk beyond the budget).
    fn eliminate(&self, front: Front) -> Result<Eliminated<Supernode>, FactorError> {
        let Front { mut rows, k, mut f } = front;
        let m = rows.len();
        let (pivots, pc) = match crate::accel::current().filter(|a| a.takes(m, k)) {
            Some(device) => factor_front_on(&*device, &mut f, m, k, &mut rows, self.tau)?,
            None => factor_front(&mut f, m, k, &mut rows, self.tau)?,
        };
        // The update matrix U = F₂₂ − L₂₁ D L₂₁ᵀ (lower) for the parent: the trailing block.
        let update = Update::trailing(&f, &rows, k, Triangle::Lower)?;
        // Keep the k factored columns only, in an allocation of their own (the front's m × m is
        // freed), as tiles when the front is large and compression is asked for.
        let (l, tiles) = match self.compress.filter(|&(_, tile)| m - k >= 2 * tile && 2 * k >= tile) {
            Some((tol, tile)) => {
                let kept = compressed(&f, m, k, tol, tile)?;
                drop(f);
                kept
            }
            None => {
                // The first k columns are the front's first k · m entries: kept in place, the
                // rest given back (a large allocation shrinks where it lies).
                f.truncate(k * m);
                f.shrink_to_fit();
                (vec![f], Vec::new())
            }
        };
        let entries = if tiles.is_empty() {
            fill(k, m)
        } else {
            tiles.iter().map(|t| if t.row0 == t.col0 { fill(t.cols, t.rows) } else { t.len() }).sum::<usize>()
        };
        let l = l.into_iter().map(|piece| self.store.keep(piece)).collect::<Result<Vec<_>, _>>()?;
        let counts = Counts { perturbed: pc.perturbed, two_by_two: pc.two_by_two, entries, largest_front: 0 };
        Ok(Eliminated { node: Supernode { rows, k, l, pivots, tiles }, update, counts })
    }

    fn by_panels(&self, m: usize) -> bool {
        self.panel_from.is_some_and(|p| m >= p)
    }

    /// The front by panels ([`panel`]): each panel assembled from A's columns among its own and
    /// from the children's update columns that land in it (read a panel's worth at a time).
    fn eliminate_panels(
        &self,
        columns: Range<usize>,
        rows: Vec<usize>,
        children: Vec<Child>,
        waiting: &Waiting,
    ) -> Result<Eliminated<Supernode>, FactorError> {
        let (m, k) = (rows.len(), columns.len());
        let original = rows.clone();
        let mut rows = rows;
        let place = |p: usize| original.binary_search(&p).map_err(|_| FactorError::Dimension("an entry outside its front".into()));
        let assemble = |cols: Range<usize>, panel: &mut [C]| -> Result<(), FactorError> {
            let (c0, mj) = (cols.start, m - cols.start);
            for c in cols.start..cols.end.min(k) {
                for &(i, v) in self.a_cols.get(columns.start + c).map_or(&[][..], Vec::as_slice) {
                    let r = place(i)?;
                    if r >= c {
                        panel[(c - c0) * mj + (r - c0)] += v;
                    }
                }
            }
            for child in &children {
                let mu = child.place.len();
                let (b0, b1) = (child.place.partition_point(|&x| x < cols.start), child.place.partition_point(|&x| x < cols.end));
                if b0 == b1 {
                    continue;
                }
                let values = waiting.columns(&child.held, mu, b0..b1)?;
                let mut at = 0;
                for b in b0..b1 {
                    let cb = child.place[b] - c0;
                    for a in b..mu {
                        panel[cb * mj + (child.place[a] - c0)] += values[at];
                        at += 1;
                    }
                }
            }
            Ok(())
        };
        let width = self.compress.map_or(PANEL, |c| c.1);
        let how = panel::Panels { k, tau: self.tau, compress: self.compress.map(|c| c.0), width };
        let by = panel::factor_by_panels(&how, &mut rows, assemble)?;
        drop(children);
        let update = (m > k).then(|| Update { rows: rows[k..].to_vec(), values: by.update });
        let entries = by.tiles.iter().map(|t| if t.row0 == t.col0 { fill(t.cols, t.rows) } else { t.len() }).sum::<usize>();
        let l = by.pieces.into_iter().map(|piece| self.store.keep(piece)).collect::<Result<Vec<_>, _>>()?;
        let counts = Counts { perturbed: by.counts.perturbed, two_by_two: by.counts.two_by_two, entries, largest_front: 0 };
        Ok(Eliminated { node: Supernode { rows, k, l, pivots: by.pivots, tiles: by.tiles }, update, counts })
    }
}

/// The equilibration S A S with each row's largest entry about one: s_i = 1/√(max_j |a_ij|).
fn symmetric_scaling(a: &Csr) -> Vec<f64> {
    (0..a.n)
        .map(|i| {
            let m = a.val[a.row(i)].iter().fold(0.0f64, |m, v| m.max(v.norm()));
            if m > 0.0 { 1.0 / m.sqrt() } else { 1.0 }
        })
        .collect()
}

/// The columns of `a` by position (`inv[unknown] = position`), lower part with the diagonal:
/// (row position, value).
fn lower_columns(a: &Csr, inv: &[usize]) -> Vec<Vec<(usize, C)>> {
    let mut a_cols: Vec<Vec<(usize, C)>> = vec![Vec::new(); a.n];
    for u in 0..a.n {
        let pu = inv[u];
        for (&v, &x) in a.col[a.row(u)].iter().zip(&a.val[a.row(u)]) {
            let pv = inv[v];
            if pu >= pv {
                a_cols[pv].push((pu, x));
            }
        }
    }
    a_cols
}

impl Ldlt {
    /// Factors `a` (complex symmetric, both triangles stored) with the unknowns ordered by
    /// nested dissection on `xyz` (a point per unknown).
    pub fn factor(a: &Csr, xyz: &[Point3]) -> Result<Ldlt, FactorError> {
        Ldlt::factor_within(a, xyz, memory_budget())
    }

    /// [`Ldlt::factor`] keeping at most `budget` bytes of the factor in memory, the rest in a
    /// scratch file on disk.
    pub fn factor_within(a: &Csr, xyz: &[Point3], budget: u64) -> Result<Ldlt, FactorError> {
        Ldlt::factor_bordered(a, xyz, budget, 0)
    }

    /// [`Ldlt::factor_within`] with the last `last` unknowns eliminated after all the others
    /// (a bordered matrix's border: unknowns each coupled to many).
    pub fn factor_bordered(a: &Csr, xyz: &[Point3], budget: u64, last: usize) -> Result<Ldlt, FactorError> {
        let analysis = Analysis::new(a, xyz, last)?;
        Ldlt::factor_analysed(a, &analysis, budget)
    }

    /// Factors `a` with an [`Analysis`] of its pattern made before (the same pattern: a sweep's
    /// frequencies share one), skipping the ordering and the symbolic factorization.
    pub fn factor_analysed(a: &Csr, an: &Analysis, budget: u64) -> Result<Ldlt, FactorError> {
        Ldlt::factor_on(a, an, budget, threads())
    }

    /// [`Ldlt::factor_analysed`] on at most `threads` threads.
    fn factor_on(a: &Csr, an: &Analysis, budget: u64, threads: usize) -> Result<Ldlt, FactorError> {
        Ldlt::factor_tiled(a, an, budget, threads, compression().map(|tol| (tol, TILE)), Forced::default())
    }

    /// [`Ldlt::factor_on`] with the large fronts' rows below their diagonal blocks kept as
    /// tiles of a side, compressed to a tolerance (`compress`); the top's updates on disk when
    /// the plan says so or `spill`.
    fn factor_tiled(
        a: &Csr,
        an: &Analysis,
        budget: u64,
        threads: usize,
        compress: Option<(f64, usize)>,
        forced: Forced,
    ) -> Result<Ldlt, FactorError> {
        let t0 = std::time::Instant::now();
        check(a, an)?;
        // Equilibrate, so the static-pivot threshold below means the same for every unknown.
        // (Graded meshes mix elements whose matrices differ by many orders; against the largest
        // entry of the whole matrix, the threshold would replace the pivots of every small one.)
        let scale = symmetric_scaling(a);
        let mut scaled = a.clone();
        for i in 0..scaled.n {
            for k in scaled.row(i) {
                scaled.val[k] *= scale[i] * scale[scaled.col[k]];
            }
        }
        let tau = f64::EPSILON.sqrt() * scaled.max_abs();
        let a_cols = lower_columns(&scaled, &an.inv);
        drop(scaled);
        let mut plan = Plan::new(an, SHAPE, budget, available_memory(), threads)?;
        plan.spill |= forced.spill;
        plan.panel_from = forced.panel_from.or(plan.panel_from);
        let kernel = Kernel { a_cols, tau, store: Store::new(plan.in_memory), compress, panel_from: plan.panel_from };
        let (supernodes, counts) = factor_tree(&kernel, an, &plan)?;
        let scratch = kernel.store.into_scratch()?;
        let bytes_on_disk = scratch.as_ref().map_or(0, |s| s.end);
        Ok(Ldlt {
            compressed: compress.map(|(tol, _)| tol),
            numeric_seconds: t0.elapsed().as_secs_f64(),
            analysis_seconds: an.seconds,
            n: a.n,
            perm: an.perm.clone(),
            supernodes,
            scratch: scratch.map(Arc::new),
            bytes_on_disk,
            scale,
            perturbed: counts.perturbed,
            two_by_two: counts.two_by_two,
            entries: counts.entries,
            largest_front: counts.largest_front,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accel::Accelerator;
    use crate::multifrontal::testing::{Rng, grid_pattern};

    /// A random complex symmetric matrix on a 3D grid graph (7-point stencil), points per node.
    fn grid_matrix(nx: usize, ny: usize, nz: usize, diag: C, rng: &mut Rng) -> (Csr, Vec<Point3>) {
        let (mut a, edges, xyz) = grid_pattern(nx, ny, nz);
        for g in &edges {
            let v = rng.c();
            a.add(g[0], g[1], v);
            a.add(g[1], g[0], v);
        }
        for i in 0..a.n {
            let v = diag + rng.c() * 0.1;
            a.add(i, i, v);
        }
        (a, xyz)
    }

    fn residual(a: &Csr, x: &[C], b: &[C]) -> f64 {
        let ax = a.mul(x);
        let r = b.iter().zip(&ax).fold(0.0f64, |m, (b, a)| m.max((b - a).norm()));
        let xn = x.iter().fold(0.0f64, |m, v| m.max(v.norm()));
        r / (a.max_abs() * xn)
    }

    /// A damped Helmholtz operator on a 3D grid (the 7-point Laplacian, −1 to each neighbour,
    /// 6 − k² + jσ on the diagonal): its fronts' blocks below the diagonal are smooth in space.
    fn helmholtz(n: usize) -> (Csr, Vec<Point3>) {
        let (mut a, edges, xyz) = grid_pattern(n, n, n);
        for g in &edges {
            a.add(g[0], g[1], C::new(-1.0, 0.0));
            a.add(g[1], g[0], C::new(-1.0, 0.0));
        }
        for i in 0..a.n {
            a.add(i, i, C::new(5.5, 0.05));
        }
        (a, xyz)
    }

    #[test]
    fn a_compressed_factor_solves_to_its_tolerance_and_refinement_recovers_the_rest() {
        // Tiles of 12 on a 14³ grid: the large fronts' rows below their diagonal blocks become
        // low-rank tiles. At 1e-10 the solve alone is accurate to about that; at 1e-5 the factor
        // keeps fewer entries, the solve alone is off by about that much, and iterative
        // refinement brings the backward error down to round-off all the same.
        let (a, xyz) = helmholtz(14);
        let an = Analysis::new(&a, &xyz, 0).unwrap();
        let mut rng = Rng(5);
        let x_true: Vec<C> = (0..a.n).map(|_| rng.c()).collect();
        let b = a.mul(&x_true);
        let exact = Ldlt::factor_tiled(&a, &an, u64::MAX, threads(), None, Forced::default()).unwrap();
        let mut previous = exact.entries;
        for (tol, alone) in [(1e-10, 1e-7), (1e-5, 1e-2)] {
            let f = Ldlt::factor_tiled(&a, &an, u64::MAX, threads(), Some((tol, 12)), Forced::default()).unwrap();
            assert!(f.supernodes.iter().any(|s| s.tiles.iter().any(|t| t.rank.is_some())), "no tile compressed at {tol}");
            assert!(f.entries < previous, "{tol}: {} entries, {previous} before", f.entries);
            previous = f.entries;
            let mut x = b.clone();
            f.solve(&mut x).unwrap();
            let err = x.iter().zip(&x_true).fold(0.0f64, |m, (p, q)| m.max((p - q).norm()));
            assert!(err < alone, "{tol}: {err}");
            let (xr, berr) = f.solve_refined(&a, &b, 30).unwrap();
            assert!(berr < 1e-14, "{tol}: {berr}");
            let err = xr.iter().zip(&x_true).fold(0.0f64, |m, (p, q)| m.max((p - q).norm()));
            assert!(err < 1e-9, "{tol}: {err}");
        }
    }

    #[test]
    fn a_compressed_factor_is_the_same_on_any_number_of_threads() {
        let (a, xyz) = helmholtz(12);
        let an = Analysis::new(&a, &xyz, 0).unwrap();
        let b: Vec<C> = (0..a.n).map(|i| C::new((i % 7) as f64, (i % 3) as f64)).collect();
        let solve = |threads: usize| {
            let f = Ldlt::factor_tiled(&a, &an, u64::MAX, threads, Some((1e-6, 10)), Forced::default()).unwrap();
            let mut x = b.clone();
            f.solve(&mut x).unwrap();
            x.iter().flat_map(|z| [z.re.to_bits(), z.im.to_bits()]).collect::<Vec<u64>>()
        };
        assert_eq!(solve(1), solve(threads()));
    }

    #[test]
    fn fronts_by_panels_solve_as_whole_fronts_do() {
        // Every front of 60 rows or more factored by panels (of 256 columns, or of 16 with
        // compression): an indefinite random matrix (2 × 2 pivots) and the Helmholtz operator,
        // exact panels solve as the whole fronts do; compressed ones refine to round-off; with
        // the updates on disk too, the same.
        let mut rng = Rng(23);
        let (indefinite, ixyz) = grid_matrix(13, 12, 11, C::new(0.0, 0.0), &mut rng);
        let (helm, hxyz) = helmholtz(14);
        for (a, xyz) in [(&indefinite, &ixyz), (&helm, &hxyz)] {
            let an = Analysis::new(a, xyz, 0).unwrap();
            assert!(an.super_rows.iter().any(|r| r.len() >= 60), "no front of 60 rows");
            let x_true: Vec<C> = (0..a.n).map(|_| rng.c()).collect();
            let b = a.mul(&x_true);
            let whole = Ldlt::factor_tiled(a, &an, u64::MAX, threads(), None, Forced::default()).unwrap();
            let by_panels = Forced { spill: false, panel_from: Some(60) };
            let exact = Ldlt::factor_tiled(a, &an, u64::MAX, threads(), None, by_panels).unwrap();
            assert!(exact.supernodes.iter().any(|s| s.tiles.len() > 1), "no front by panels");
            let (mut x, mut y) = (b.clone(), b.clone());
            whole.solve(&mut x).unwrap();
            exact.solve(&mut y).unwrap();
            let d = x.iter().zip(&y).fold(0.0f64, |m, (p, q)| m.max((p - q).norm()));
            assert!(d < 1e-8, "exact panels: {d}");
            for spill in [false, true] {
                let f = Ldlt::factor_tiled(a, &an, u64::MAX, threads(), Some((1e-8, 16)), Forced { spill, panel_from: Some(60) }).unwrap();
                assert!(f.supernodes.iter().any(|s| s.tiles.iter().any(|t| t.rank.is_some())), "no tile compressed");
                let (xr, berr) = f.solve_refined(a, &b, 30).unwrap();
                assert!(berr < 1e-14, "compressed panels: {berr}");
                let err = xr.iter().zip(&x_true).fold(0.0f64, |m, (p, q)| m.max((p - q).norm()));
                assert!(err < 1e-8, "compressed panels: {err}");
            }
        }
        // The same on any number of threads.
        let an = Analysis::new(&helm, &hxyz, 0).unwrap();
        let bits = |threads: usize| {
            let f = Ldlt::factor_tiled(&helm, &an, u64::MAX, threads, Some((1e-6, 16)), Forced { spill: true, panel_from: Some(60) }).unwrap();
            let mut x: Vec<C> = (0..helm.n).map(|i| C::new(1.0, (i % 3) as f64)).collect();
            f.solve(&mut x).unwrap();
            x.iter().flat_map(|z| [z.re.to_bits(), z.im.to_bits()]).collect::<Vec<u64>>()
        };
        assert_eq!(bits(1), bits(threads()));
    }

    #[test]
    fn updates_on_disk_give_the_same_factor() {
        // The top's updates written to the scratch file and read back as their parents are
        // assembled: the same solution, bit for bit.
        let (a, xyz) = helmholtz(12);
        let an = Analysis::new(&a, &xyz, 0).unwrap();
        let b: Vec<C> = (0..a.n).map(|i| C::new((i % 5) as f64, 1.0)).collect();
        let solve = |spill: bool| {
            let f = Ldlt::factor_tiled(&a, &an, u64::MAX, threads(), None, Forced { spill, panel_from: None }).unwrap();
            let mut x = b.clone();
            f.solve(&mut x).unwrap();
            x.iter().flat_map(|z| [z.re.to_bits(), z.im.to_bits()]).collect::<Vec<u64>>()
        };
        assert_eq!(solve(true), solve(false));
    }

    #[test]
    fn spilled_subtree_updates_are_on_disk_as_the_subtrees_finish() {
        // The plan counts no subtree's update held in memory when it spills them: each goes to
        // disk on its thread as its subtree finishes, not after all of them (an oven of 2.3
        // million unknowns held them all and was killed for memory).
        let (a, xyz) = helmholtz(12);
        let an = Analysis::new(&a, &xyz, 0).unwrap();
        let kernel = Kernel { a_cols: lower_columns(&a, &an.inv), tau: 1e-12, store: Store::new(u64::MAX), compress: None, panel_from: None };
        let on_disk = crate::multifrontal::subtree_updates_on_disk(&kernel, &an, 4).unwrap();
        assert!(!on_disk.is_empty());
        assert!(on_disk.iter().all(|&d| d));
    }

    #[test]
    fn compression_is_set_for_the_calling_thread_only() {
        // Set on this thread, not seen on another; a tolerance that is not a positive number
        // keeps the factor exact.
        with_compression(Some(1e-6), || {
            assert_eq!(compression(), Some(1e-6));
            assert_eq!(std::thread::spawn(compression).join().unwrap(), None);
        });
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(with_compression(Some(bad), compression), None);
        }
    }

    #[test]
    fn the_factor_is_the_same_on_any_number_of_threads() {
        // Large enough for subtrees on threads and threaded products in the top: one thread,
        // fewer (a pool of their own) and the machine's give the same solution, bit for bit.
        let mut rng = Rng(17);
        let (a, xyz) = grid_matrix(20, 18, 16, C::new(-1.0, 0.3), &mut rng);
        let an = Analysis::new(&a, &xyz, 0).unwrap();
        let b: Vec<C> = (0..a.n).map(|_| rng.c()).collect();
        let solve = |threads: usize| {
            let f = Ldlt::factor_on(&a, &an, u64::MAX, threads).unwrap();
            let mut x = b.clone();
            f.solve(&mut x).unwrap();
            x.iter().flat_map(|z| [z.re.to_bits(), z.im.to_bits()]).collect::<Vec<u64>>()
        };
        let all = solve(threads());
        assert_eq!(solve(1), all);
        assert_eq!(solve(3), all);
    }

    #[test]
    fn solves_random_complex_symmetric_systems() {
        let mut rng = Rng(0x9e3779b97f4a7c15);
        for (diag, label) in [(C::new(4.0, 0.0), "dominant"), (C::new(0.0, 0.0), "indefinite"), (C::new(-1.0, 0.3), "shifted")] {
            let (a, xyz) = grid_matrix(7, 5, 6, diag, &mut rng);
            let f = Ldlt::factor(&a, &xyz).unwrap();
            let x_true: Vec<C> = (0..a.n).map(|_| rng.c()).collect();
            let b = a.mul(&x_true);
            let mut x = b.clone();
            f.solve(&mut x).unwrap();
            assert!(residual(&a, &x, &b) < 1e-11, "{label}: {}", residual(&a, &x, &b));
            let (xr, berr) = f.solve_refined(&a, &b, 5).unwrap();
            assert!(berr < 1e-14, "{label}: {berr}");
            let err = xr.iter().zip(&x_true).fold(0.0f64, |m, (p, q)| m.max((p - q).norm()));
            assert!(err < 1e-8, "{label}: {err}");
        }
    }

    #[test]
    fn several_right_hand_sides_solve_together() {
        // Three right-hand sides in one pass over the factor (each supernode's rows below its
        // block as one product) and with refinement: each as accurate as alone.
        let mut rng = Rng(0x2545f4914f6cdd1d);
        let (a, xyz) = grid_matrix(7, 6, 5, C::new(0.0, 0.0), &mut rng);
        let f = Ldlt::factor(&a, &xyz).unwrap();
        let truths: Vec<Vec<C>> = (0..3).map(|_| (0..a.n).map(|_| rng.c()).collect()).collect();
        let bs: Vec<Vec<C>> = truths.iter().map(|x| a.mul(x)).collect();
        let mut xs = bs.clone();
        f.solve_many(&mut xs).unwrap();
        for (x, b) in xs.iter().zip(&bs) {
            assert!(residual(&a, x, b) < 1e-11, "{}", residual(&a, x, b));
        }
        for ((xr, berr), truth) in f.solve_refined_many(&a, &bs, 5).unwrap().iter().zip(&truths) {
            assert!(*berr < 1e-14, "{berr}");
            let err = xr.iter().zip(truth).fold(0.0f64, |m, (p, q)| m.max((p - q).norm()));
            assert!(err < 1e-8, "{err}");
        }
    }

    #[test]
    fn needs_two_by_two_pivots() {
        // A zero diagonal with the largest entries inside one supernode: [[0, 1], [1, 0]] blocks
        // coupled densely, all in one front.
        let n = 6;
        let groups = [(0..n).collect::<Vec<_>>()];
        let mut a = Csr::from_groups(n, &groups).unwrap();
        let mut rng = Rng(7);
        for i in 0..n {
            for j in i + 1..n {
                let v = if j == i + 1 && i % 2 == 0 { C::new(5.0, 1.0) } else { rng.c() * 0.2 };
                a.add(i, j, v);
                a.add(j, i, v);
            }
        }
        let xyz: Vec<Point3> = (0..n).map(|i| Point3::new(i as f64, 0.0, 0.0)).collect();
        let f = Ldlt::factor(&a, &xyz).unwrap();
        assert!(f.two_by_two > 0);
        assert_eq!(f.perturbed, 0);
        let b: Vec<C> = (0..n).map(|i| C::new(i as f64, 1.0)).collect();
        let mut x = b.clone();
        f.solve(&mut x).unwrap();
        assert!(residual(&a, &x, &b) < 1e-13);
    }

    /// A device held on the host, doing what `DeviceFront` documents in the plainest way.
    struct HostDevice(std::sync::atomic::AtomicUsize);

    struct HostFront {
        f: Vec<C>,
        m: usize,
    }

    impl Accelerator for HostDevice {
        fn name(&self) -> String {
            "host".into()
        }
        fn takes(&self, m: usize, _: usize) -> bool {
            m >= 40
        }
        fn front(&self, f: &[C], m: usize) -> Result<Box<dyn crate::accel::DeviceFront>, crate::accel::DeviceError> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(Box::new(HostFront { f: f.to_vec(), m }))
        }
    }

    impl crate::accel::DeviceFront for HostFront {
        fn read(&mut self, cols: Range<usize>, out: &mut [C]) -> Result<(), crate::accel::DeviceError> {
            out.copy_from_slice(&self.f[cols.start * self.m..cols.end * self.m]);
            Ok(())
        }
        fn swap_rows(&mut self, cols: Range<usize>, swaps: &[(usize, usize)]) -> Result<(), crate::accel::DeviceError> {
            for c in cols {
                for &(p, q) in swaps {
                    self.f.swap(c * self.m + p, c * self.m + q);
                }
            }
            Ok(())
        }
        fn update(&mut self, be: usize, k: usize, w: &[C], l: &[C]) -> Result<(), crate::accel::DeviceError> {
            let (m, below) = (self.m, self.m - be);
            let nb = w.len() / below;
            for j in be..m {
                for i in (if j < k { be } else { j })..m {
                    let sum = (0..nb).fold(C::new(0.0, 0.0), |s, t| s + w[t * below + i - be] * l[t * below + j - be]);
                    self.f[j * m + i] -= sum;
                }
            }
            Ok(())
        }
    }

    #[test]
    fn a_device_holding_the_large_fronts_factors_alike() {
        let mut rng = Rng(7);
        for diag in [C::new(4.0, 0.0), C::new(0.0, 0.0)] {
            let (a, xyz) = grid_matrix(9, 8, 7, diag, &mut rng);
            let x_true: Vec<C> = (0..a.n).map(|_| rng.c()).collect();
            let b = a.mul(&x_true);
            let host = Ldlt::factor(&a, &xyz).unwrap();
            let device = std::sync::Arc::new(HostDevice(Default::default()));
            let on = crate::accel::with(Some(device.clone()), || Ldlt::factor(&a, &xyz)).unwrap();
            assert!(device.0.load(std::sync::atomic::Ordering::Relaxed) > 0);
            assert_eq!((on.perturbed, on.two_by_two), (host.perturbed, host.two_by_two));
            let (mut x, mut y) = (b.clone(), b.clone());
            host.solve(&mut x).unwrap();
            on.solve(&mut y).unwrap();
            let gap = x.iter().zip(&y).fold(0.0f64, |m, (p, q)| m.max((p - q).norm()));
            assert!(residual(&a, &y, &b) < 1e-11 && gap < 1e-9, "{gap:e}");
        }
        // Outside `with`, no device.
        assert!(crate::accel::current().is_none());
    }

    #[test]
    fn deterministic_bit_for_bit() {
        let mut r1 = Rng(42);
        let (a, xyz) = grid_matrix(6, 6, 6, C::new(0.5, 0.0), &mut r1);
        let b: Vec<C> = (0..a.n).map(|i| C::new((i % 7) as f64, 1.0)).collect();
        let run = || {
            let mut x = b.clone();
            Ldlt::factor(&a, &xyz).unwrap().solve(&mut x).unwrap();
            x
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn badly_scaled_matrices_keep_their_pivots() {
        // Two blocks of a graded mesh's sizes, 1e12 apart: against the whole matrix's largest
        // entry, √ε‖A‖ would replace every pivot of the small one; equilibrated, none.
        let n = 40;
        let groups: Vec<Vec<usize>> = (0..n - 1).map(|i| vec![i, i + 1]).collect();
        let mut a = Csr::from_groups(n, &groups).unwrap();
        for i in 0..n {
            let s = if i < n / 2 { 1e12 } else { 1.0 };
            a.add(i, i, C::new(4.0 * s, 0.3));
            if i + 1 < n {
                let s = if i + 1 < n / 2 { 1e12 } else { 1.0 };
                a.add(i, i + 1, C::new(-s, 0.0));
                a.add(i + 1, i, C::new(-s, 0.0));
            }
        }
        let xyz: Vec<Point3> = (0..n).map(|i| Point3::new(i as f64, 0.0, 0.0)).collect();
        let f = Ldlt::factor(&a, &xyz).unwrap();
        assert_eq!(f.perturbed, 0);
        let x: Vec<C> = (0..n).map(|i| C::new(1.0 + i as f64, -0.5)).collect();
        let mut b = a.mul(&x);
        f.solve(&mut b).unwrap();
        let err = b.iter().zip(&x).map(|(p, q)| (p - q).norm()).fold(0.0, f64::max);
        assert!(err < 1e-8, "{err:e}");
    }

    #[test]
    fn hostile_matrices_are_refused() {
        let groups = [vec![0, 1]];
        let mut a = Csr::from_groups(2, &groups).unwrap();
        let xyz = [Point3::ORIGIN, Point3::new(1.0, 0.0, 0.0)];
        assert_eq!(Ldlt::factor(&a, &xyz).unwrap_err(), FactorError::Zero);
        a.add(0, 0, C::new(f64::NAN, 0.0));
        assert_eq!(Ldlt::factor(&a, &xyz).unwrap_err(), FactorError::NotFinite);
        assert!(matches!(Ldlt::factor(&a, &xyz[..1]), Err(FactorError::Dimension(_))));
    }

    #[test]
    fn a_factor_on_disk_solves_the_same() {
        // Out of core from the first column: the same solution bit for bit.
        let mut rng = Rng(7);
        let (a, xyz) = grid_matrix(6, 5, 4, C::new(0.3, 0.1), &mut rng);
        let b: Vec<C> = (0..a.n).map(|i| C::new((i % 5) as f64, -1.0)).collect();
        let mem = Ldlt::factor_within(&a, &xyz, u64::MAX).unwrap();
        let disk = Ldlt::factor_within(&a, &xyz, 0).unwrap();
        assert_eq!(mem.bytes_on_disk, 0);
        let (mut x, mut y) = (b.clone(), b.clone());
        mem.solve(&mut x).unwrap();
        disk.solve(&mut y).unwrap();
        assert_eq!(x, y);
        // Clones share the file.
        let again = disk;
        let mut z = b;
        again.solve(&mut z).unwrap();
        assert_eq!(x, z);
    }

    #[test]
    #[ignore = "benchmark: cargo test --release -p emcraft-solver --lib dense_front_rate -- --ignored --nocapture"]
    fn dense_front_rate() {
        // A dense complex symmetric matrix, one front: the factorization's arithmetic rate.
        for n in [800usize, 1600] {
            let all: Vec<usize> = (0..n).collect();
            let mut a = Csr::from_groups(n, std::iter::once(all.as_slice())).unwrap();
            let mut rng = Rng(0x1234_5678);
            for i in 0..n {
                for j in 0..=i {
                    let v = C::new(rng.next(), rng.next());
                    let v = if i == j { v + C::new(n as f64, 0.0) } else { v };
                    a.add(i, j, v);
                    if i != j {
                        a.add(j, i, v);
                    }
                }
            }
            let xyz: Vec<Point3> = (0..n).map(|i| Point3::new(i as f64, 0.0, 0.0)).collect();
            let t = std::time::Instant::now();
            let f = Ldlt::factor(&a, &xyz).unwrap();
            let s = t.elapsed().as_secs_f64();
            // n³/3 complex multiply–adds of 8 real flops each.
            let gflops = 8.0 * (n as f64).powi(3) / 3.0 / s / 1e9;
            eprintln!("n {n}: {s:.3} s, {gflops:.1} GFLOP/s, front {}", f.largest_front);
        }
    }
}
