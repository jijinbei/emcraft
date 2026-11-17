//! The model's error type and the checks its constructors and setters share.

/// Why a model change was refused.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ModelError {
    #[error("invalid {what}: {why}")]
    Invalid { what: String, why: String },
    #[error("{kind} `{name}` already exists")]
    Duplicate { kind: &'static str, name: String },
    #[error("unknown {kind} `{name}`")]
    Unknown { kind: &'static str, name: String },
}

pub type Result<T> = std::result::Result<T, ModelError>;

use crate::quantity::{Positive, Quantity};

/// An [`ModelError::Invalid`] naming `what` and `why`.
pub(crate) fn invalid(what: impl Into<String>, why: impl Into<String>) -> ModelError {
    ModelError::Invalid { what: what.into(), why: why.into() }
}

/// A finite number > 0: [`Positive`]'s rule, for a field kept as an `f64`.
pub fn positive(what: &str, v: f64) -> Result<f64> {
    Positive::named(what, v).map(Positive::get)
}

/// Every value finite, else "`what` must be finite".
pub(crate) fn all_finite(what: impl Into<String>, values: &[f64]) -> Result<()> {
    if values.iter().all(|v| v.is_finite()) { Ok(()) } else { Err(invalid(what, "must be finite")) }
}

/// A name with something besides white space in it.
pub(crate) fn named(what: &str, name: &str) -> Result<()> {
    if name.trim().is_empty() { Err(invalid(what, "must not be empty")) } else { Ok(()) }
}

/// Whether `x` and `y` agree component by component within `tol`.
pub(crate) fn near<const N: usize>(x: [f64; N], y: [f64; N], tol: f64) -> bool {
    x.iter().zip(y).all(|(a, b)| (a - b).abs() <= tol)
}
