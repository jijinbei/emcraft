//! The multifrontal method shared by the symmetric [`crate::ldlt`] and the unsymmetric
//! [`crate::lu`] factorizations of matrices with a symmetric pattern.
//!
//! **Structure.** The unknowns are ordered by nested dissection ([`crate::ordering`]), then by a
//! postorder of the elimination tree (J. W. H. Liu, *The role of elimination trees in sparse
//! factorization*, SIAM J. Matrix Anal. Appl. 11(1):134–172, 1990), and consecutive columns
//! with nested structures are merged into supernodes ([`Analysis`]). Each supernode is factored
//! in a dense frontal matrix: its rows and columns of A plus the update matrices its children
//! left behind (extend-add), the supernode's own variables eliminated, and the Schur complement
//! passed to the parent (I. S. Duff, J. K. Reid, *The multifrontal solution of indefinite sparse
//! symmetric linear equations*, ACM TOMS 9(3):302–325, 1983; J. W. H. Liu, *The multifrontal
//! method for sparse matrix solution: theory and practice*, SIAM Review 34(1):82–109, 1992).
//!
//! **The driver** ([`factor_tree`]) is the same for every factorization; what a front holds,
//! how A and the updates enter it and how its variables are eliminated is a [`FrontKernel`]'s.
//! Subtrees light enough go to threads whole, each in postorder as alone (so each keeps the
//! memory of a sequential factorization); the heavy top, whose fronts are large, follows on the
//! calling thread in order, its fronts' products threaded inside. Every front receives its
//! children's updates in the children's order whichever thread made them, so the factor is the
//! same bit for bit on any number of threads.

use std::collections::BTreeMap;
use std::ops::Range;

use num_complex::Complex64 as C;
use rayon::prelude::*;

use crate::sparse::Csr;

mod analysis;
mod memory;
mod scratch;

pub(crate) use crate::memory::available_memory;
pub use analysis::Analysis;
pub(crate) use analysis::fill;
pub(crate) use memory::{PANEL, Plan, Shape, products};
pub(crate) use scratch::Scratch;

/// The largest frontal matrix (rows) a factorization may form: m² complex entries (40 GB at the
/// limit; the [`Plan`] refuses before that what the memory cannot hold).
pub const MAX_FRONT: usize = 50_000;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum FactorError {
    #[error("a frontal matrix of {rows} rows exceeds the limit of {limit}: the problem is too large for the direct solver")]
    FrontTooLarge { rows: usize, limit: usize },
    #[error("the matrix has a non-finite entry")]
    NotFinite,
    #[error("the matrix is zero")]
    Zero,
    #[error("dimension mismatch: {0}")]
    Dimension(String),
    #[error("the factor's scratch file: {0}")]
    Scratch(String),
    #[error("the device failed: {0}")]
    Device(String),
    #[error("the factor needs {:.1} GB, beyond the {:.1} GB budget (an unsymmetric factor is kept in memory)", *needs as f64 / 1e9, *budget as f64 / 1e9)]
    FactorTooLarge { needs: u64, budget: u64 },
    #[error("the factorization's threads: {0}")]
    Threads(String),
    #[error(transparent)]
    Memory(#[from] crate::memory::OutOfMemory),
}

/// A supernode's dense frontal matrix, column-major m × m.
pub(crate) struct Front {
    /// Positions of the front's rows (and columns), ascending; the first `k` are the
    /// supernode's own variables.
    pub rows: Vec<usize>,
    pub k: usize,
    pub f: Vec<C>,
}

impl Front {
    /// The order m.
    pub fn m(&self) -> usize {
        self.rows.len()
    }

    /// The front row of position `p`.
    pub fn place(&self, p: usize) -> Option<usize> {
        self.rows.binary_search(&p).ok()
    }

    /// [`Front::place`] of an entry of A, refused when outside the front.
    pub fn place_entry(&self, p: usize) -> Result<usize, FactorError> {
        self.place(p).ok_or_else(|| FactorError::Dimension("an entry outside its front".into()))
    }
}

/// A Schur complement passed to the parent: its rows (positions) and its values in the kernel's
/// layout.
pub(crate) struct Update {
    pub rows: Vec<usize>,
    pub values: Vec<C>,
}

/// What of a front's trailing block its parent is handed: all of it (an LU's), or its lower
/// triangle with the diagonal (a symmetric factorization's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Triangle {
    Full,
    Lower,
}

impl Triangle {
    /// The entries of an update of `mu` rows.
    pub(crate) const fn entries(self, mu: usize) -> usize {
        match self {
            Triangle::Full => mu * mu,
            Triangle::Lower => mu * (mu + 1) / 2,
        }
    }
}

impl Update {
    /// The update the eliminated front `f` (m × m column-major over `rows`, its first `k`
    /// factored) hands its parent: its trailing block, column by column as `triangle` keeps it;
    /// none when all its rows were factored.
    pub(crate) fn trailing(f: &[C], rows: &[usize], k: usize, triangle: Triangle) -> Result<Option<Update>, crate::memory::OutOfMemory> {
        let m = rows.len();
        let mu = m - k;
        if mu == 0 {
            return Ok(None);
        }
        let mut u = crate::memory::with_capacity(triangle.entries(mu), "an update matrix")?;
        for j in 0..mu {
            let first = match triangle {
                Triangle::Full => k,
                Triangle::Lower => k + j,
            };
            u.extend_from_slice(&f[(k + j) * m + first..(k + j + 1) * m]);
        }
        Ok(Some(Update { rows: rows[k..].to_vec(), values: u }))
    }
}

/// An update waiting for its parent: in memory, or in the factorization's scratch file for
/// updates (its rows kept here).
pub(crate) enum Held {
    Memory(Update),
    Disk { rows: Vec<usize>, offset: u64, len: usize },
}

impl Held {
    /// Its rows (positions).
    pub(crate) fn rows(&self) -> &[usize] {
        match self {
            Held::Memory(u) => &u.rows,
            Held::Disk { rows, .. } => rows,
        }
    }
}

/// A child's update as a front factored by panels reads it: where each of its rows goes in the
/// front, and the update itself.
pub(crate) struct Child {
    pub place: Vec<usize>,
    pub held: Held,
}

/// Where the top's updates wait for their parents: in memory, or (`spill`) each written to a
/// scratch file of their own as it is made and read back as its parent is assembled, so the top
/// holds one front and one child's update at a time (Liu 1986's out-of-core stack). The file is
/// made when first needed and goes with the factorization.
pub(crate) struct Waiting {
    spill: bool,
    file: std::sync::Mutex<Option<Scratch>>,
}

impl Waiting {
    fn new(spill: bool) -> Waiting {
        Waiting { spill, file: std::sync::Mutex::new(None) }
    }

    fn hold(&self, u: Update) -> Result<Held, FactorError> {
        if !self.spill {
            return Ok(Held::Memory(u));
        }
        let mut guard = self.file.lock().map_err(|_| FactorError::Scratch("the updates' file's lock was poisoned".into()))?;
        if guard.is_none() {
            *guard = Some(Scratch::new()?);
        }
        match guard.as_mut() {
            Some(file) => {
                let (offset, len) = file.write(&u.values, "an update matrix on its way to disk")?;
                Ok(Held::Disk { rows: u.rows, offset, len })
            }
            None => Ok(Held::Memory(u)),
        }
    }

    /// The columns `cols` of a lower-packed update of `mu` rows (each from its diagonal down,
    /// one after another), from memory or read back from the file.
    pub(crate) fn columns<'a>(&self, h: &'a Held, mu: usize, cols: Range<usize>) -> Result<std::borrow::Cow<'a, [C]>, FactorError> {
        let (from, to) = (fill(cols.start, mu), fill(cols.end, mu));
        match h {
            Held::Memory(u) => {
                u.values.get(from..to).map(std::borrow::Cow::Borrowed).ok_or_else(|| FactorError::Dimension("an update shorter than its rows".into()))
            }
            Held::Disk { offset, len, .. } => {
                if to > *len {
                    return Err(FactorError::Dimension("an update on disk shorter than its rows".into()));
                }
                let guard = self.file.lock().map_err(|_| FactorError::Scratch("the updates' file's lock was poisoned".into()))?;
                let file = guard.as_ref().ok_or_else(|| FactorError::Scratch("an update on disk without a file".into()))?;
                Ok(std::borrow::Cow::Owned(file.read(offset + (from as u64) * 16, to - from, "an update's columns read back")?))
            }
        }
    }

    fn take(&self, h: Held) -> Result<Update, FactorError> {
        match h {
            Held::Memory(u) => Ok(u),
            Held::Disk { rows, offset, len } => {
                let guard = self.file.lock().map_err(|_| FactorError::Scratch("the updates' file's lock was poisoned".into()))?;
                let file = guard.as_ref().ok_or_else(|| FactorError::Scratch("an update on disk without a file".into()))?;
                Ok(Update { rows, values: file.read(offset, len, "an update matrix read back")? })
            }
        }
    }
}

/// What a factorization counts.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Counts {
    /// Pivots replaced by the static-pivot threshold.
    pub perturbed: usize,
    /// 2 × 2 pivots.
    pub two_by_two: usize,
    /// Entries of the factor.
    pub entries: usize,
    /// The largest front (rows).
    pub largest_front: usize,
}

impl Counts {
    fn add(&mut self, o: Counts) {
        self.perturbed += o.perturbed;
        self.two_by_two += o.two_by_two;
        self.entries += o.entries;
        self.largest_front = self.largest_front.max(o.largest_front);
    }
}

/// A front eliminated: what the factor keeps of it, the update for the parent and its counts.
pub(crate) struct Eliminated<N> {
    pub node: N,
    pub update: Option<Update>,
    pub counts: Counts,
}

/// One factorization's part of the multifrontal method: how A's entries and the children's
/// updates enter a front, and how the front's own variables are eliminated.
pub(crate) trait FrontKernel: Sync {
    /// What the factor keeps of a supernode.
    type Node: Send;

    /// Adds A's entries of the supernode's own `columns` (positions) into `front`.
    fn assemble(&self, columns: Range<usize>, front: &mut Front) -> Result<(), FactorError>;

    /// Adds a child's update `values` (of [`FrontKernel::eliminate`]'s layout), its row a at
    /// front row `place[a]`, into the front `f` of order m.
    fn extend_add(&self, f: &mut [C], m: usize, place: &[usize], values: &[C]);

    /// Eliminates the front's first k variables.
    fn eliminate(&self, front: Front) -> Result<Eliminated<Self::Node>, FactorError>;

    /// Whether a front of `m` rows is factored by panels, never formed whole
    /// ([`FrontKernel::eliminate_panels`]).
    fn by_panels(&self, _m: usize) -> bool {
        false
    }

    /// Assembles and eliminates supernode `columns`' front of `rows` panel by panel, reading its
    /// children's updates from `waiting` a panel's columns at a time.
    fn eliminate_panels(
        &self,
        _columns: Range<usize>,
        _rows: Vec<usize>,
        _children: Vec<Child>,
        _waiting: &Waiting,
    ) -> Result<Eliminated<Self::Node>, FactorError> {
        Err(FactorError::Dimension("not supported yet: this factorization by panels".into()))
    }
}

/// Refuses a matrix that is not of `an`'s pattern, has a non-finite entry or is zero.
pub(crate) fn check(a: &Csr, an: &Analysis) -> Result<(), FactorError> {
    if !an.fits(a) {
        return Err(FactorError::Dimension("the analysis is of another pattern".into()));
    }
    if a.val.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
        return Err(FactorError::NotFinite);
    }
    if a.n > 0 && !(a.max_abs() > 0.0) {
        return Err(FactorError::Zero);
    }
    Ok(())
}

/// The supernodal tree: each supernode's parent and children (ascending), and the first
/// supernode of its subtree (a subtree is the contiguous range from it to the root).
struct Tree {
    parent: Vec<Option<usize>>,
    children: Vec<Vec<usize>>,
    first: Vec<usize>,
    /// Multiply–adds of each subtree, about k m² per front.
    work: Vec<f64>,
}

impl Tree {
    fn new(an: &Analysis) -> Tree {
        let ns = an.supernodes();
        let parent = an.supernode_parents();
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); ns];
        for (s, p) in parent.iter().enumerate() {
            if let Some(p) = p {
                children[*p].push(s);
            }
        }
        let mut first: Vec<usize> = (0..ns).collect();
        let mut work = vec![0.0f64; ns];
        for s in 0..ns {
            let (k, m) = (an.columns(s).len() as f64, an.super_rows[s].len() as f64);
            work[s] += k * m * m;
            if let Some(p) = parent[s] {
                first[p] = first[p].min(first[s]);
                work[p] += work[s];
            }
        }
        Tree { parent, children, first, work }
    }

    fn roots(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.parent.len()).filter(|&s| self.parent[s].is_none())
    }

    /// The roots of the subtrees that go to threads whole (each with under 1/(8 · threads) of
    /// the work, or a leaf), and the supernodes above them, both ascending.
    fn split(&self, threads: usize) -> (Vec<usize>, Vec<usize>) {
        let total: f64 = self.roots().map(|s| self.work[s]).sum();
        let light = total / (8 * threads) as f64;
        let (mut tasks, mut top): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        let mut stack: Vec<usize> = self.roots().collect();
        while let Some(s) = stack.pop() {
            if self.work[s] <= light || self.children[s].is_empty() {
                tasks.push(s);
            } else {
                top.push(s);
                stack.extend(self.children[s].iter().copied());
            }
        }
        tasks.sort_unstable();
        top.sort_unstable();
        (tasks, top)
    }
}

/// A subtree factored on a thread.
struct Subtree<N> {
    root: usize,
    nodes: Vec<(usize, N)>,
    update: Option<Update>,
    counts: Counts,
}

/// Assembles supernode `s`'s front from A and its children's `updates` (in the children's
/// order, each taken from `waiting` as it is added) and eliminates it.
fn node<K: FrontKernel>(
    kernel: &K,
    an: &Analysis,
    s: usize,
    rows: Vec<usize>,
    updates: Vec<Held>,
    waiting: &Waiting,
) -> Result<Eliminated<K::Node>, FactorError> {
    let columns = an.columns(s);
    let m = rows.len();
    if m > MAX_FRONT {
        return Err(FactorError::FrontTooLarge { rows: m, limit: MAX_FRONT });
    }
    if kernel.by_panels(m) {
        let children = updates
            .into_iter()
            .map(|held| {
                let place = held.rows().iter().map(|p| rows.binary_search(p).ok()).collect::<Option<Vec<usize>>>();
                let place = place.ok_or_else(|| FactorError::Dimension("an update outside its parent's front".into()))?;
                Ok(Child { place, held })
            })
            .collect::<Result<Vec<_>, FactorError>>()?;
        let mut done = kernel.eliminate_panels(columns, rows, children, waiting)?;
        done.counts.largest_front = m;
        return Ok(done);
    }
    let mut front = Front { rows, k: columns.len(), f: crate::memory::zeros(m * m, "a frontal matrix")? };
    kernel.assemble(columns, &mut front)?;
    for u in updates {
        let u = waiting.take(u)?;
        let place = u.rows.iter().map(|&p| front.place(p)).collect::<Option<Vec<usize>>>();
        let place = place.ok_or_else(|| FactorError::Dimension("an update outside its parent's front".into()))?;
        kernel.extend_add(&mut front.f, m, &place, &u.values);
    }
    let mut done = kernel.eliminate(front)?;
    done.counts.largest_front = m;
    Ok(done)
}

/// Factors the subtree of `root` (its supernodes in postorder with their rows), as the
/// sequential factorization would.
fn subtree<K: FrontKernel>(
    kernel: &K,
    an: &Analysis,
    tree: &Tree,
    root: usize,
    list: Vec<(usize, Vec<usize>)>,
) -> Result<Subtree<K::Node>, FactorError> {
    let _alone = memory::SubtreeGuard::enter();
    let mut pending: BTreeMap<usize, Vec<Held>> = BTreeMap::new();
    let mut out = Subtree { root, nodes: Vec::with_capacity(list.len()), update: None, counts: Counts::default() };
    let in_memory = Waiting::new(false);
    for (s, rows) in list {
        let updates = pending.remove(&s).unwrap_or_default();
        let f = node(kernel, an, s, rows, updates, &in_memory)?;
        out.counts.add(f.counts);
        if s == root {
            out.update = f.update;
        } else if let (Some(u), Some(p)) = (f.update, tree.parent.get(s).copied().flatten()) {
            pending.entry(p).or_default().push(Held::Memory(u));
        }
        out.nodes.push((s, f.node));
    }
    Ok(out)
}

/// The threads a factorization may use: the machine's.
pub(crate) fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
}

/// A light subtree for a thread: its root, and each of its supernodes with its rows.
type Job = (usize, Vec<(usize, Vec<usize>)>);

/// The jobs of the subtrees `tasks`, their rows taken out of `super_rows`.
fn jobs(tree: &Tree, tasks: &[usize], super_rows: &mut [Option<Vec<usize>>]) -> Vec<Job> {
    tasks.iter().map(|&s| (s, (tree.first[s]..=s).map(|t| (t, super_rows[t].take().unwrap_or_default())).collect())).collect()
}

/// The light subtrees `jobs` factored on `threads` threads, each root's update for its parent
/// handed to `waiting` on the subtree's own thread as it finishes: when the plan spills, it is on
/// disk from then on, not held in memory until every subtree is done (the plan counts none of
/// them held).
fn subtrees<K: FrontKernel>(
    kernel: &K,
    an: &Analysis,
    tree: &Tree,
    jobs: Vec<Job>,
    threads: usize,
    waiting: &Waiting,
) -> Result<Vec<(Subtree<K::Node>, Option<Held>)>, FactorError> {
    let factor_jobs = || -> Vec<Result<(Subtree<K::Node>, Option<Held>), FactorError>> {
        jobs.into_par_iter()
            .map(|(root, list)| {
                let mut d = subtree(kernel, an, tree, root, list)?;
                let held = match (d.update.take(), tree.parent[d.root]) {
                    (Some(u), Some(_)) => Some(waiting.hold(u)?),
                    _ => None,
                };
                Ok((d, held))
            })
            .collect()
    };
    // Fewer threads than the machine's: a pool of that many.
    let done = if threads >= self::threads().min(rayon::current_num_threads()) {
        factor_jobs()
    } else {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e| FactorError::Threads(e.to_string()))?.install(factor_jobs)
    };
    done.into_iter().collect()
}

/// For tests: whether each light subtree's update for its parent is on disk when the subtrees
/// on `threads` threads are done, the updates spilled.
#[cfg(test)]
pub(crate) fn subtree_updates_on_disk<K: FrontKernel>(kernel: &K, an: &Analysis, threads: usize) -> Result<Vec<bool>, FactorError> {
    let tree = Tree::new(an);
    let (tasks, _) = tree.split(threads);
    let mut super_rows: Vec<Option<Vec<usize>>> = an.super_rows.iter().cloned().map(Some).collect();
    let waiting = Waiting::new(true);
    let done = subtrees(kernel, an, &tree, jobs(&tree, &tasks, &mut super_rows), threads, &waiting)?;
    Ok(done.iter().filter_map(|(_, held)| held.as_ref().map(|h| matches!(h, Held::Disk { .. }))).collect())
}

/// Factors every supernode of `an` with `kernel` as `plan` says: the light subtrees on its
/// threads, the top after them on the calling thread (where [`crate::accel`]'s device is), the
/// updates waiting for a top front on disk when it spills them. Returns the nodes by supernode
/// and the counts.
pub(crate) fn factor_tree<K: FrontKernel>(kernel: &K, an: &Analysis, plan: &Plan) -> Result<(Vec<K::Node>, Counts), FactorError> {
    let threads = plan.threads;
    let waiting = Waiting::new(plan.spill);
    let ns = an.supernodes();
    let tree = Tree::new(an);
    let (tasks, top) = tree.split(threads);

    let mut nodes: Vec<Option<K::Node>> = (0..ns).map(|_| None).collect();
    let mut super_rows: Vec<Option<Vec<usize>>> = an.super_rows.iter().cloned().map(Some).collect();
    let done = subtrees(kernel, an, &tree, jobs(&tree, &tasks, &mut super_rows), threads, &waiting)?;
    // Updates waiting for a top supernode: (child, update).
    let mut pending: Vec<Vec<(usize, Held)>> = (0..ns).map(|_| Vec::new()).collect();
    let mut counts = Counts::default();
    for (d, held) in done {
        counts.add(d.counts);
        for (s, node) in d.nodes {
            nodes[s] = Some(node);
        }
        if let (Some(u), Some(p)) = (held, tree.parent[d.root]) {
            pending[p].push((d.root, u));
        }
    }
    for s in top {
        let rows = super_rows[s].take().unwrap_or_default();
        let mut from_children = std::mem::take(&mut pending[s]);
        from_children.sort_by_key(|c| c.0);
        let updates = from_children.into_iter().map(|(_, u)| u).collect();
        let f = node(kernel, an, s, rows, updates, &waiting)?;
        counts.add(f.counts);
        if let (Some(u), Some(p)) = (f.update, tree.parent[s]) {
            pending[p].push((s, waiting.hold(u)?));
        }
        nodes[s] = Some(f.node);
    }
    let nodes = nodes
        .into_iter()
        .map(|n| n.ok_or_else(|| FactorError::Dimension("a supernode was left unfactored".into())))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((nodes, counts))
}

/// Solves A x = b by `solve` with iterative refinement until the backward error
/// ‖b − A x‖∞ / (‖A‖ ‖x‖∞ + ‖b‖∞) stops falling (at most `steps` corrections); returns x and
/// that error.
pub(crate) fn refine(a: &Csr, b: &[C], steps: usize, solve: impl Fn(&mut [C]) -> Result<(), FactorError>) -> Result<(Vec<C>, f64), FactorError> {
    let mut one = refine_many(a, std::slice::from_ref(&b.to_vec()), steps, |bs| bs.iter_mut().try_for_each(|b| solve(b)))?;
    one.pop().ok_or_else(|| FactorError::Dimension("a refined solve returned nothing".into()))
}

/// [`refine`] for several right-hand sides at once: each pass of `solve_many` serves every one
/// whose refinement goes on (each stopping as the single refinement would), so a pass over the
/// factor serves them all rather than each in turn.
pub(crate) fn refine_many(
    a: &Csr,
    bs: &[Vec<C>],
    steps: usize,
    solve_many: impl Fn(&mut [Vec<C>]) -> Result<(), FactorError>,
) -> Result<Vec<(Vec<C>, f64)>, FactorError> {
    let norm_a = a.max_abs();
    let inf = |v: &[C]| v.iter().fold(0.0f64, |m, z| m.max(z.norm()));
    let berr = |b: &[C], x: &[C]| -> (Vec<C>, f64) {
        let ax = a.mul(x);
        let r: Vec<C> = b.iter().zip(&ax).map(|(b, a)| b - a).collect();
        let e = inf(&r) / (norm_a * inf(x) + inf(b)).max(f64::MIN_POSITIVE);
        (r, e)
    };
    let mut xs: Vec<Vec<C>> = bs.to_vec();
    solve_many(&mut xs)?;
    // Per right-hand side: its solution, residual, error and whether it is still refined.
    let mut state: Vec<(Vec<C>, Vec<C>, f64, bool)> = xs
        .into_iter()
        .zip(bs)
        .map(|(x, b)| {
            let (r, e) = berr(b, &x);
            (x, r, e, true)
        })
        .collect();
    for _ in 0..steps.min(20) {
        for s in &mut state {
            s.3 &= s.2 >= 4.0 * f64::EPSILON;
        }
        let active: Vec<usize> = (0..state.len()).filter(|&j| state[j].3).collect();
        if active.is_empty() {
            break;
        }
        let mut corrections: Vec<Vec<C>> = active.iter().map(|&j| state[j].1.clone()).collect();
        solve_many(&mut corrections)?;
        for (&j, d) in active.iter().zip(corrections) {
            let (x, r, err, going) = &mut state[j];
            let trial: Vec<C> = x.iter().zip(&d).map(|(x, d)| x + d).collect();
            let (r2, e2) = berr(&bs[j], &trial);
            if !(e2 < *err) {
                *going = false;
                continue;
            }
            *x = trial;
            *r = r2;
            *err = e2;
        }
    }
    Ok(state.into_iter().map(|(x, _, e, _)| (x, e)).collect())
}

/// Shared by the factorizations' tests: a deterministic pseudo-random sequence and matrices on
/// a 3D grid graph.
#[cfg(test)]
pub(crate) mod testing {
    use emcraft_model::Point3;
    use num_complex::Complex64 as C;

    use crate::sparse::Csr;

    /// A deterministic pseudo-random sequence (xorshift).
    pub struct Rng(pub u64);

    impl Rng {
        pub fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        }

        pub fn c(&mut self) -> C {
            C::new(self.next(), self.next())
        }
    }

    /// The pattern of a 3D grid graph (7-point stencil), its edges and a point per node.
    pub fn grid_pattern(nx: usize, ny: usize, nz: usize) -> (Csr, Vec<[usize; 2]>, Vec<Point3>) {
        let id = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);
        let n = nx * ny * nz;
        let mut edges: Vec<[usize; 2]> = Vec::new();
        let mut xyz = vec![Point3::ORIGIN; n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    xyz[id(i, j, k)] = Point3::new(i as f64, j as f64, k as f64);
                    if i + 1 < nx {
                        edges.push([id(i, j, k), id(i + 1, j, k)]);
                    }
                    if j + 1 < ny {
                        edges.push([id(i, j, k), id(i, j + 1, k)]);
                    }
                    if k + 1 < nz {
                        edges.push([id(i, j, k), id(i, j, k + 1)]);
                    }
                }
            }
        }
        let a = Csr::from_groups(n, &edges).unwrap();
        (a, edges, xyz)
    }
}
