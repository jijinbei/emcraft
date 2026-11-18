//! A board's net classes and the impedances their lines should have (`pcb.class`), checked
//! segment by segment (`pcb.check`): each class's routed nets solved as `pcb.impedance` solves
//! them — alone against a `z0` target, beside their differential partner against a `z_diff`
//! one — every segment passing within the class's tolerance or found out: off its target, with
//! no reference plane, or (a pair's) not beside its partner. A finding names its segment by the
//! id the board's source gave it (a PCBCraft board's tracks keep their PCBCraft ids), its net,
//! layer and middle, so a layout tool can point at it.

use emcraft_model::pcb::{Board, NetClass};
use serde_json::{Value, json};

use crate::pcb_lines::{Outcome, Solved, ohms, partner_name, solve_net};
use crate::{EngineError, Params, Result, Session};

fn failed(why: impl std::fmt::Display) -> EngineError {
    EngineError::failed(why)
}

/// The most findings a reply lists (the counts cover them all).
const MAX_FINDINGS: usize = 500;
/// A class's tolerance when `pcb.class` leaves it out: ±10 %.
const TOLERANCE: f64 = 0.1;

/// Applies `change` to a copy of the board's classes, checks the board and keeps it.
fn edit(s: &mut Session, change: impl FnOnce(&mut Vec<NetClass>) -> Result<Value>) -> Result<Value> {
    let mut board = s.model.pcb().cloned().ok_or_else(|| failed("load a PCB first: pcb.import, pcb.load or pcb.sample"))?;
    let reply = change(&mut board.classes)?;
    board.check()?;
    s.model.set_pcb(Some(board));
    Ok(reply)
}

/// `pcb.class`: a net class made, or the one named `name` changed (only what is given).
pub fn class(s: &mut Session, p: &Params) -> Result<Value> {
    let name = p.string("name").required()?.to_owned();
    let nets = match p.list("nets", 256).optional()? {
        Some(list) => Some(
            list.iter()
                .map(|v| v.as_str().map(str::to_owned).ok_or_else(|| p.error("nets", "lists net names or patterns (`*` any characters)")))
                .collect::<Result<Vec<_>>>()?,
        ),
        None => None,
    };
    let z0 = p.positive("z0").optional()?;
    let z_diff = p.positive("z_diff").optional()?;
    let tolerance = p.positive("tolerance").check(|t| *t <= 100.0, "is a percentage, at most 100").optional()?.map(|t| 0.01 * t);
    edit(s, |classes| {
        let made = match classes.iter_mut().find(|c| c.name == name) {
            Some(c) => {
                c.nets = nets.unwrap_or_else(|| std::mem::take(&mut c.nets));
                c.z0 = z0.or(c.z0);
                c.z_diff = z_diff.or(c.z_diff);
                c.tolerance = tolerance.unwrap_or(c.tolerance);
                false
            }
            None => {
                let nets = nets.ok_or_else(|| p.error("nets", "a new class needs its nets"))?;
                if z0.is_none() && z_diff.is_none() {
                    return Err(p.error("z0", "a new class needs a target: z0, z_diff or both"));
                }
                classes.push(NetClass { name: name.clone(), nets, z0, z_diff, tolerance: tolerance.unwrap_or(TOLERANCE) });
                true
            }
        };
        Ok(json!({"class": name, "made": made, "classes": classes.len()}))
    })
}

/// `pcb.class.delete`: the class named `name` removed.
pub fn class_delete(s: &mut Session, p: &Params) -> Result<Value> {
    let name = p.string("name").required()?.to_owned();
    edit(s, |classes| {
        let before = classes.len();
        classes.retain(|c| c.name != name);
        if classes.len() == before {
            return Err(p.error("name", "names no class of this board"));
        }
        Ok(json!({"classes": classes.len()}))
    })
}

/// What one class asks of one segment, and what it found: the quantity (`z0` or `z_diff`), the
/// value (Ω) if the segment has one.
struct Check<'a> {
    net: String,
    solved: Solved<'a>,
    quantity: &'static str,
    target: f64,
}

/// A segment's value for the quantity asked: a single line's z0, a pair's 2 z_odd; or why it has
/// none (`no reference plane`, `not beside its partner`, the solver's word).
fn value(c: &Check) -> std::result::Result<f64, String> {
    match (&c.solved.outcome, c.quantity) {
        (Outcome::Single { z, .. }, "z0") => Ok(*z),
        (Outcome::Pair { odd, .. }, "z_diff") => Ok(2.0 * odd.0),
        (Outcome::Single { .. }, _) => Err("not beside its differential partner".into()),
        (Outcome::Pair { .. }, _) => Err("solved as a pair".into()),
        (Outcome::None(why), _) if why.starts_with("no reference plane") => Err("no reference plane".into()),
        (Outcome::None(why), _) => Err(why.clone()),
    }
}

/// The checks `class` asks for on `board`: each of its routed nets alone against `z0`, each pair
/// (the net and the one its name pairs with, both routed, once) against `z_diff`.
fn checks<'a>(board: &'a Board, class: &NetClass) -> std::result::Result<Vec<Check<'a>>, String> {
    let mut nets: Vec<&str> = board.segments.iter().map(|g| g.net.as_str()).filter(|n| class.matches(n)).collect();
    nets.sort_unstable();
    nets.dedup();
    let has = |n: &str| board.segments.iter().any(|g| g.net == n);
    let mut out = Vec::new();
    if let Some(target) = class.z0 {
        for net in &nets {
            out.extend(solve_net(board, net, None)?.into_iter().map(|solved| Check { net: (*net).to_owned(), solved, quantity: "z0", target }));
        }
    }
    if let Some(target) = class.z_diff {
        let mut done: Vec<String> = Vec::new();
        for net in &nets {
            if done.iter().any(|d| d == net) {
                continue;
            }
            let partner = partner_name(net, has);
            done.extend(partner.clone());
            let solved = solve_net(board, net, partner.as_deref())?;
            out.extend(solved.into_iter().map(|solved| Check { net: (*net).to_owned(), solved, quantity: "z_diff", target }));
        }
    }
    Ok(out)
}

/// `pcb.check`: every class's (or the one named `class`'s) segments against its targets.
pub fn check(s: &mut Session, p: &Params) -> Result<Value> {
    let board = s.model.pcb().ok_or_else(|| failed("load a PCB with pcb.import, pcb.load or pcb.sample"))?;
    let only = p.string("class").optional()?;
    let classes: Vec<&NetClass> = board.classes.iter().filter(|c| only.is_none_or(|n| n == c.name)).collect();
    if classes.is_empty() {
        return Err(match only {
            Some(_) => p.error("class", "names no class of this board"),
            None => failed("the board has no net classes: make one with pcb.class"),
        });
    }
    let length = s.display.length;
    let (mut findings, mut found, mut rows) = (Vec::new(), 0usize, Vec::new());
    for class in classes {
        let checks = checks(board, class).map_err(failed)?;
        let nets: std::collections::BTreeSet<&str> = checks.iter().map(|c| c.net.as_str()).collect();
        let mut failing = 0usize;
        let mut range = (f64::INFINITY, f64::NEG_INFINITY);
        for c in &checks {
            let g = c.solved.segment;
            let problem = match value(c) {
                Ok(z) => {
                    range = (range.0.min(z), range.1.max(z));
                    let off = z / c.target - 1.0;
                    (off.abs() > class.tolerance).then(|| (json!(ohms(z)), json!((off * 1e3).round() / 10.0), "off target".to_owned()))
                }
                Err(why) => Some((Value::Null, Value::Null, why)),
            };
            let Some((value, deviation, problem)) = problem else { continue };
            failing += 1;
            found += 1;
            if findings.len() < MAX_FINDINGS {
                let at = [0.5 * (g.start[0] + g.end[0]), 0.5 * (g.start[1] + g.end[1])];
                findings.push(json!({
                    "class": class.name,
                    "net": c.net,
                    "segment": g.id,
                    "layer": g.layer,
                    "at": [length.from_si(at[0]), length.from_si(at[1])],
                    "length": length.from_si(emcraft_model::pcb::distance(g.start, g.end)),
                    "quantity": c.quantity,
                    "value": value,
                    "target": c.target,
                    "deviation_percent": deviation,
                    "problem": problem,
                }));
            }
        }
        let verdict = match (checks.len(), failing) {
            (0, _) => "no routed nets",
            (_, 0) => "pass",
            _ => "fail",
        };
        rows.push(json!({
            "class": class.name,
            "nets": nets,
            "z0": class.z0,
            "z_diff": class.z_diff,
            "tolerance_percent": 100.0 * class.tolerance,
            "segments": checks.len(),
            "failing": failing,
            "range": range.0.is_finite().then(|| json!([ohms(range.0), ohms(range.1)])),
            "verdict": verdict,
        }));
    }
    let verdict = if found == 0 { "pass" } else { "fail" };
    Ok(json!({
        "verdict": verdict,
        "classes": rows,
        "findings": findings,
        "findings_total": found,
        "length_unit": length.symbol(),
        "ids": "each finding's segment is the id its board's source gave it: a PCBCraft board's track id, a KiCad board's track uuid (an arc's chords `uuid:k`)",
        "model": "pcb.impedance's: each segment's cross-section quasi-statically, references the nearest planes of other nets; z_diff = 2 z_odd beside the partner its name pairs with",
    }))
}
