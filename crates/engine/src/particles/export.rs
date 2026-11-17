//! `results.particles.export`: a particle run's records as CSV for other programs — its history
//! (time, the field's and the particles' energy, the particles held, the current emitted, the
//! power through each output plane), its probes' fields over time, or its particles at the end
//! (position, momentum, energy: the phase space). Times in ns, lengths in the display's unit,
//! fields in V/m, momenta γv in m/s, energies in J or eV as each column's header says.

use std::fmt::Write as _;
use std::io::Write as _;

use serde_json::{Value, json};

use super::PicResult;
use crate::{EngineError, Params, Result, Session};

emcraft_macros::named_enum! {
    /// What of a run is written.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(crate) enum What {
        History = "history",
        Probes = "probes",
        Particles = "particles",
    }
}

emcraft_macros::named_serde!(What);

/// `results.particles.export`.
pub fn export(s: &mut Session, p: &Params) -> Result<Value> {
    let path = p.string("path").required()?;
    let what = p.choice("what").or(What::History)?;
    let run = s.results.particles.clone().ok_or_else(|| EngineError::failed("no particle run: solve.particles first"))?;
    let (text, rows) = match what {
        What::History => history(s, &run),
        What::Probes => probes(s, &run),
        What::Particles => particles(s, &run),
    };
    let mut file = std::fs::File::create(path).map_err(|e| EngineError::cannot_write(path, e))?;
    file.write_all(text.as_bytes()).map_err(|e| EngineError::cannot_write(path, e))?;
    Ok(json!({"path": path, "what": what.id(), "rows": rows}))
}

/// A length for a header: to a nanometre's precision in millimetres, its trailing zeros dropped
/// (7, not 7.000000000000001).
fn short(x: f64) -> String {
    let rounded = format!("{x:.6}");
    rounded.trim_end_matches('0').trim_end_matches('.').to_owned()
}

/// The history's records: a header, then a row each.
fn history(s: &Session, run: &PicResult) -> (String, usize) {
    let l = s.display.length.symbol();
    let mut text = String::from("t (ns),field (J),kinetic (J),particles,emitted (A)");
    for o in &run.outputs {
        let _ = write!(text, ",power at z = {} {l} (W)", short(s.show_length(o.z)));
    }
    text.push('\n');
    for r in &run.history {
        let _ = write!(text, "{},{},{},{},{}", r.t * 1e9, r.field, r.kinetic, r.particles, r.emitted);
        for p in &r.power {
            let _ = write!(text, ",{p}");
        }
        text.push('\n');
    }
    (text, run.history.len())
}

/// The probes' E over time: a header, then a row each sample (every probe sampled together).
fn probes(s: &Session, run: &PicResult) -> (String, usize) {
    let l = s.display.length.symbol();
    let mut text = String::from("t (ns)");
    for (k, p) in run.probes.iter().enumerate() {
        let [x, y, z] = p.at.map(|c| short(s.show_length(c)));
        for c in ["Ex", "Ey", "Ez"] {
            let _ = write!(text, ",probe {} ({x} {y} {z} {l}) {c} (V/m)", k + 1);
        }
    }
    text.push('\n');
    let samples = run.probes.iter().map(|p| p.e.len()).min().unwrap_or(0);
    let every = run.probes.first().map_or(1, |p| p.every);
    for i in 0..samples {
        let _ = write!(text, "{}", (i + 1) as f64 * every as f64 * run.dt * 1e9);
        for p in &run.probes {
            for c in p.e[i] {
                let _ = write!(text, ",{c}");
            }
        }
        text.push('\n');
    }
    (text, samples)
}

/// The particles at the end (the phase space): a header, then a row each.
fn particles(s: &Session, run: &PicResult) -> (String, usize) {
    let l = s.display.length.symbol();
    let mut text = format!("x ({l}),y ({l}),z ({l}),ux (m/s),uy (m/s),uz (m/s),energy (eV),population\n");
    for p in &run.snapshot {
        let [x, y, z] = p.x.map(|c| s.show_length(c));
        let _ = writeln!(text, "{x},{y},{z},{},{},{},{},{}", p.u[0], p.u[1], p.u[2], p.energy, p.population);
    }
    (text, run.snapshot.len())
}
