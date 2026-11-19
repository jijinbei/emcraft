//! `pcb.crosstalk`: what the routed nets beside a net couple into it — each of its segments with
//! another net's parallel segment on its layer near it solved as two coupled lines (the
//! cross-section `pcb.impedance` builds, the neighbour's strip beside it), their near- and far-end
//! crosstalk from the per-unit-length matrices.
//!
//! **Weakly coupled lines** (C. R. Paul, *Analysis of Multiconductor Transmission Lines*, 2nd ed.,
//! 2008, §§7.3–7.4; the backward and forward coupling of a matched pair). With the capacitance
//! coupling k_C = −C₁₂/√(C₁₁C₂₂) and the inductive k_L = L₁₂/√(L₁₁L₂₂) — L = μ₀ε₀ (C⁰)⁻¹ makes it
//! −C⁰₁₂/√(C⁰₁₁C⁰₂₂) — a step of rise time t_r on the aggressor, both lines matched, gives at the
//! victim's near end k_b = (k_L + k_C)/4 once the coupled run's round trip 2T_d outlasts the
//! rise (2T_d/t_r of it before), and at its far end (k_C − k_L)/2 · T_d/t_r: none in one
//! dielectric (a stripline's k_L = k_C), negative on a microstrip (more of the field in the air
//! couples less capacitively). T_d is the run's length over the victim's speed.
//!
//! **A route.** Each segment's run beside the neighbour is the length the two overlap along it.
//! A net's far-end crosstalk from one aggressor is the sum of its segments' (they add along the
//! way); its near-end the sum of the segments' unsaturated ones, at most the largest saturated
//! one — an estimate, not a waveform: lines matched at both ends, the coupling weak.

use std::collections::BTreeMap;

use emcraft_solver::portmode::strips::Line;
use serde_json::{Value, json};

use crate::pcb_lines::{Beside, Outcome, solve_net, thousandth};
use crate::{EngineError, Params, Result, Session};

/// How far from a segment's edge another net's parallel segment is taken as a neighbour (m),
/// when `within` is not given.
const WITHIN: f64 = 1e-3;
/// The aggressor's rise time (s), when `rise_time` is not given.
const RISE: f64 = 100e-12;

/// Two lines' coupling, the `own`th the victim: (k_C, k_L, the victim's delay per unit length).
fn coupling(line: &Line, own: usize) -> Option<(f64, f64, f64)> {
    let other = 1 - own.min(1);
    let k = |m: &[Vec<f64>]| {
        let (vv, aa, va) = (m.get(own)?.get(own)?, m.get(other)?.get(other)?, m.get(own)?.get(other)?);
        Some(-va / (vv * aa).sqrt())
    };
    let (c, c0) = (line.capacitance.get(own)?.get(own)?, line.vacuum.get(own)?.get(own)?);
    let delay = (c / c0).sqrt() / emcraft_model::units::C0;
    Some((k(&line.capacitance)?, k(&line.vacuum)?, delay)).filter(|(a, b, d)| a.is_finite() && b.is_finite() && d.is_finite())
}

/// What one aggressor does to the route: its segments beside it, the near- and far-end sums and
/// the largest saturated near-end coefficient.
#[derive(Default)]
struct Aggressor {
    segments: usize,
    run: f64,
    near: f64,
    near_max: f64,
    far: f64,
}

/// `pcb.crosstalk`: the crosstalk into `net` from the nets beside it (module docs).
pub fn crosstalk(s: &mut Session, p: &Params) -> Result<Value> {
    let board = s.model.pcb().ok_or_else(|| EngineError::failed("load a PCB with pcb.import, pcb.load or pcb.sample"))?;
    let net = p.string("net").required()?;
    let has = |n: &str| board.segments.iter().any(|g| g.net == n);
    if !has(net) {
        return Err(p.error("net", "names no routed net of this board"));
    }
    let aggressor = match p.string("aggressor").optional()? {
        Some(a) if a == net => return Err(p.error("aggressor", "must be another net")),
        Some(a) if !has(a) => return Err(p.error("aggressor", "names no routed net of this board")),
        a => a,
    };
    let within = p.positive_length("within").check(|w| *w <= 0.1, "is at most 100 mm").optional()?.unwrap_or(WITHIN);
    let rise = p.positive("rise_time").check(|t| *t <= 1e6, "is in picoseconds, at most 1 µs").optional()?.map_or(RISE, |t| t * 1e-12);
    let rows = solve_net(board, net, Beside::Neighbour { net: aggressor, within }, false).map_err(|e| p.error("net", &e))?;
    let length = s.display.length;
    let percent = |x: f64| thousandth(100.0 * x);
    let (mut shown, mut by): (Vec<Value>, BTreeMap<String, Aggressor>) = (Vec::new(), BTreeMap::new());
    let mut unsolved = Vec::new();
    for r in &rows {
        let g = r.segment;
        match &r.outcome {
            Outcome::Coupled { neighbour, gap, overlap, line, own } => {
                let Some((kc, kl, delay)) = coupling(line, *own) else {
                    unsolved.push(json!({"segment": g.id, "note": "its coupling could not be found"}));
                    continue;
                };
                let td = overlap * delay;
                let (kb, kf) = (0.25 * (kl + kc), 0.5 * (kc - kl));
                let (next, fext) = (kb * (2.0 * td / rise).min(1.0), kf * td / rise);
                shown.push(json!({
                    "segment": g.id,
                    "layer": g.layer,
                    "aggressor": neighbour,
                    "gap": length.from_si(*gap),
                    "run": length.from_si(*overlap),
                    "k_capacitive": kc,
                    "k_inductive": kl,
                    "k_next": kb,
                    "next_percent": percent(next),
                    "fext_percent": percent(fext),
                }));
                let a = by.entry(neighbour.clone()).or_default();
                a.segments += 1;
                a.run += overlap;
                a.near += kb * 2.0 * td / rise;
                a.near_max = a.near_max.max(kb.abs());
                a.far += fext;
            }
            Outcome::None(why) => unsolved.push(json!({"segment": g.id, "note": why})),
            Outcome::Single { .. } | Outcome::Pair { .. } => {}
        }
    }
    let aggressors: Vec<Value> = by
        .iter()
        .map(|(name, a)| {
            json!({
                "net": name,
                "segments": a.segments,
                "run": length.from_si(a.run),
                "next_percent": percent(a.near.signum() * a.near.abs().min(a.near_max)),
                "fext_percent": percent(a.far),
            })
        })
        .collect();
    Ok(json!({
        "net": net,
        "rise_time": thousandth(rise * 1e12),
        "within": length.from_si(within),
        "length_unit": length.symbol(),
        "aggressors": aggressors,
        "segments": shown,
        "unsolved": unsolved,
        "model": "each segment beside another net's parallel segment on its layer (its edge within `within`; the nearest) solved as two coupled quasi-TEM lines (pcb.impedance's cross-section); weak coupling, both lines matched (Paul 2008, §7.3): near end (k_L + k_C)/4 saturating once 2 T_d ≥ t_r, far end (k_C − k_L)/2 · T_d/t_r, T_d the run's delay; a route's far end the sum over its segments, its near end the sum up to the largest saturated one",
    }))
}
