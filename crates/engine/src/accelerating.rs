//! An eigenmode as an accelerating cavity along a straight line, what `results.cavity` reports
//! and `results.track` drives its electrons with: the field's own voltage V₀ = |∫ E·dl| along
//! the line, the energy U the mode stores (its images through symmetry planes counted) and its
//! R/Q, V₀²/(ωU) (the linac convention: T. P. Wangler, *RF Linear Accelerators*, 2nd ed., Wiley
//! 2008, §2.2–2.3).

use emcraft_model::{Model, Point3};
use emcraft_solver::Complex64 as C;
use emcraft_solver::driven::Driven;
use emcraft_solver::eigen::{Eigen, Mode};

use crate::{EngineError, Params, Result};

/// Below this V₀²/(ωU) (Ω) a mode has no field along the line to speak of (a TE mode on the
/// axis, its V₀ the solve's round-off).
pub(crate) const MIN_FIELD_R_OVER_Q: f64 = 1e-3;

/// A line's two ends (m).
pub(crate) type Ends = (Point3, Point3);

/// How many times the region solved its images make the whole: each symmetry plane halves it.
pub(crate) fn images(model: &Model) -> f64 {
    2f64.powi(i32::try_from(model.domain.symmetry.len()).unwrap_or(0))
}

/// Refused: a rotational sector or a periodic cell, whose mode is not the whole cavity's.
pub(crate) fn whole_cavity(model: &Model) -> Result<()> {
    if model.domain.rotation.is_some() || model.domain.periodic.is_some() {
        return Err(EngineError::failed("not supported yet: a cavity's voltage on a rotational sector or a periodic cell"));
    }
    Ok(())
}

/// Mode `index` (from 1) of `e`, or why not under parameter `mode`.
pub(crate) fn mode_at<'e>(p: &Params, e: &'e Eigen, index: usize) -> Result<&'e Mode> {
    e.modes.get(index.wrapping_sub(1)).ok_or_else(|| p.error("mode", format!("must be 1 to {}", e.modes.len())))
}

/// A mode's own field along a line: V₀ (V), the energy it stores (J) and its R/Q (Ω, 0 when it
/// stores none), all at the mode's arbitrary amplitude.
pub(crate) struct OwnField {
    pub(crate) voltage: C,
    pub(crate) energy: f64,
    pub(crate) r_over_q: f64,
}

impl OwnField {
    /// Mode `m` (solved by `driven`) along `line`, integrated in `intervals`, its stored energy
    /// counted `images` times.
    pub(crate) fn of(driven: &Driven, m: &Mode, line: Ends, intervals: usize, images: f64) -> Result<OwnField> {
        let voltage = driven.line_integral(&m.vector, m.frequency, line, intervals, |_| C::new(1.0, 0.0))?;
        let energy = driven.stored_energy(&m.vector, m.frequency)? * images;
        let v0 = voltage.norm();
        let r_over_q = if energy > 0.0 { v0 * v0 / (2.0 * std::f64::consts::PI * m.frequency * energy) } else { 0.0 };
        Ok(OwnField { voltage, energy, r_over_q })
    }

    /// Whether it has a field along the line to accelerate with.
    pub(crate) fn accelerates(&self) -> bool {
        self.r_over_q >= MIN_FIELD_R_OVER_Q
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::Session;

    /// A sector's or a periodic cell's mode is not the whole cavity's: refused by both
    /// results.cavity and results.track (the tracking once went ahead on a quarter's mode).
    #[test]
    fn a_sector_is_no_whole_cavity() {
        let mut s = Session::new();
        s.execute("geometry.solid", &json!({"shape": "cylinder", "radius": 10, "length": 13.06})).unwrap();
        assert!(super::whole_cavity(&s.model).is_ok());
        s.execute("model.rotation", &json!({"n": 4})).unwrap();
        assert!(super::whole_cavity(&s.model).is_err());
    }
}
