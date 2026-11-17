//! Where the factor's columns are kept: in memory up to a budget, beyond it in a scratch file on
//! disk (out-of-core).

use std::borrow::Cow;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use num_complex::Complex64 as C;

use crate::multifrontal::FactorError;
pub(crate) use crate::multifrontal::Scratch;

/// The most bytes of factor kept in memory before the rest goes to the scratch file: the
/// environment's `EMCRAFT_FACTOR_MEMORY_GB`, else no cap of its own — the factorization's
/// [`crate::multifrontal::Plan`] keeps in memory what the system's available memory leaves
/// beside its fronts.
pub(crate) fn memory_budget() -> u64 {
    std::env::var("EMCRAFT_FACTOR_MEMORY_GB")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|g| g.is_finite() && *g >= 0.0)
        .map_or(u64::MAX, |gb| (gb * 1e9) as u64)
}

/// Where a supernode's factored columns are kept.
#[derive(Clone, Debug)]
pub(super) enum Columns {
    Memory(Vec<C>),
    /// In the scratch file: `len` complex numbers from byte `offset`.
    Disk {
        offset: u64,
        len: usize,
    },
}

impl Columns {
    /// The columns, from memory or read back from `scratch`.
    pub fn get<'a>(&'a self, scratch: Option<&Scratch>) -> Result<Cow<'a, [C]>, FactorError> {
        match self {
            Columns::Memory(v) => Ok(Cow::Borrowed(v)),
            Columns::Disk { offset, len } => {
                let sc = scratch.ok_or_else(|| FactorError::Scratch("columns on disk without a file".into()))?;
                Ok(Cow::Owned(sc.read(*offset, *len, "a factor's columns read back")?))
            }
        }
    }
}

/// Keeps factored columns, shared by the threads of a factorization: in memory while they fit
/// the budget, then in the scratch file (made when first needed).
pub(super) struct Store {
    budget: u64,
    in_memory: AtomicU64,
    scratch: Mutex<Option<Scratch>>,
}

impl Store {
    pub fn new(budget: u64) -> Store {
        Store { budget, in_memory: AtomicU64::new(0), scratch: Mutex::new(None) }
    }

    /// Keeps `l`.
    pub fn keep(&self, l: Vec<C>) -> Result<Columns, FactorError> {
        let bytes = (l.len() * 16) as u64;
        let fits =
            self.in_memory.try_update(Ordering::SeqCst, Ordering::SeqCst, |used| (used + bytes <= self.budget).then_some(used + bytes)).is_ok();
        if fits {
            return Ok(Columns::Memory(l));
        }
        let mut guard = self.scratch.lock().map_err(|_| FactorError::Dimension("the scratch file's lock was poisoned".into()))?;
        if guard.is_none() {
            *guard = Some(Scratch::new()?);
        }
        match guard.as_mut() {
            Some(sc) => sc.write(&l, "a factor's columns on their way to disk").map(|(offset, len)| Columns::Disk { offset, len }),
            None => Ok(Columns::Memory(l)),
        }
    }

    /// The scratch file, when any columns went to it.
    pub fn into_scratch(self) -> Result<Option<Scratch>, FactorError> {
        self.scratch.into_inner().map_err(|_| FactorError::Dimension("the scratch file's lock was poisoned".into()))
    }
}
