//! A board's net classes and the impedances their lines should have (`pcb.class`), checked
//! segment by segment (`pcb.check`): each class's routed nets solved as `pcb.impedance` solves
//! them — alone against a `z0` target, beside their differential partner against a `z_diff`
//! one — every segment passing within the class's tolerance or found out: off its target, with
//! no reference plane, or (a pair's) not beside its partner. A finding names its segment by the
//! id the board's source gave it (a PCBCraft board's tracks keep their PCBCraft ids), its net,
//! layer and middle, so a layout tool can point at it.
//!
//! A class's limits are on each net's route as a whole: its routed length, its delay (the
//! segments' lengths at their modes' speeds — a pair's odd mode's beside its partner), a pair's
//! skew (the difference of its two nets' delays) and its loss at a frequency (`pcb.impedance`'s,
//! copper's conductivity). A route with a segment of no impedance has no delay or loss, and is
//! found out for it.

use emcraft_model::pcb::{Board, Limits, NetClass};
use serde_json::{Value, json};

use crate::pcb_lines::{Beside, COPPER, Lossy, Outcome, Solved, db, ohms, partner_name, solve_net, thousandth};
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
    let z0 = p.edit("z0", || p.positive("z0").optional())?;
    let z_diff = p.edit("z_diff", || p.positive("z_diff").optional())?;
    let tolerance = p.positive("tolerance").check(|t| *t <= 100.0, "is a percentage, at most 100").optional()?.map(|t| 0.01 * t);
    let picoseconds = |name: &'static str| p.edit(name, || Ok(p.positive(name).optional()?.map(|t| t * 1e-12)));
    let (length, delay, skew) =
        (p.edit("max_length", || p.positive_length("max_length").optional())?, picoseconds("max_delay")?, picoseconds("max_skew")?);
    let (loss, frequency) = (p.edit("max_loss", || p.positive("max_loss").optional())?, p.edit("frequency", || p.frequency("frequency").optional())?);
    let limited = |old: Limits| Limits {
        length: length.apply(old.length),
        delay: delay.apply(old.delay),
        skew: skew.apply(old.skew),
        loss: loss.apply(old.loss),
        frequency: frequency.apply(old.frequency),
    };
    edit(s, |classes| {
        let made = match classes.iter_mut().find(|c| c.name == name) {
            Some(c) => {
                c.nets = nets.unwrap_or_else(|| std::mem::take(&mut c.nets));
                c.z0 = z0.apply(c.z0);
                c.z_diff = z_diff.apply(c.z_diff);
                c.tolerance = tolerance.unwrap_or(c.tolerance);
                c.limits = limited(c.limits);
                false
            }
            None => {
                let nets = nets.ok_or_else(|| p.error("nets", "a new class needs its nets"))?;
                let (z0, z_diff, limits) = (z0.apply(None), z_diff.apply(None), limited(Limits::default()));
                if z0.is_none() && z_diff.is_none() && limits.is_empty() {
                    return Err(p.error("z0", "a new class needs a target or a limit: z0, z_diff, max_length, max_delay, max_skew or max_loss"));
                }
                classes.push(NetClass { name: name.clone(), nets, z0, z_diff, tolerance: tolerance.unwrap_or(TOLERANCE), limits });
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
        (Outcome::Single { mode, .. }, "z0") => Ok(mode.impedance),
        (Outcome::Pair { odd, .. }, "z_diff") => Ok(2.0 * odd.impedance),
        (Outcome::Single { .. }, _) => Err("not beside its differential partner".into()),
        (Outcome::Pair { .. }, _) => Err("solved as a pair".into()),
        (Outcome::Coupled { .. }, _) => Err("solved beside another net".into()),
        (Outcome::None(why), _) if why.starts_with("no reference plane") => Err("no reference plane".into()),
        (Outcome::None(why), _) => Err(why.clone()),
    }
}

/// The class's routed nets on `board`, sorted.
fn nets_of<'a>(board: &'a Board, class: &NetClass) -> Vec<&'a str> {
    let mut nets: Vec<&str> = board.segments.iter().map(|g| g.net.as_str()).filter(|n| class.matches(n)).collect();
    nets.sort_unstable();
    nets.dedup();
    nets
}

/// The checks `class` asks for on `board`: each of its routed nets alone against `z0`, each pair
/// (the net and the one its name pairs with, both routed, once) against `z_diff`.
fn checks<'a>(board: &'a Board, class: &NetClass) -> std::result::Result<Vec<Check<'a>>, String> {
    let nets = nets_of(board, class);
    let has = |n: &str| board.segments.iter().any(|g| g.net == n);
    let mut out = Vec::new();
    if let Some(target) = class.z0 {
        for net in &nets {
            let solved = solve_net(board, net, Beside::Alone, false)?;
            out.extend(solved.into_iter().map(|solved| Check { net: (*net).to_owned(), solved, quantity: "z0", target }));
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
            let solved = solve_net(board, net, partner.as_deref().map_or(Beside::Alone, Beside::Partner), false)?;
            out.extend(solved.into_iter().map(|solved| Check { net: (*net).to_owned(), solved, quantity: "z_diff", target }));
        }
    }
    Ok(out)
}

/// A net's route as a class's limits weigh it: its routed length (m), and over its segments with
/// an impedance its delay (s) and its conductor and dielectric loss (dB; the conductor's none
/// when a segment's copper is a sheet); how many segments have no impedance.
struct Route {
    length: f64,
    delay: f64,
    loss: Option<(Option<f64>, f64)>,
    unsolved: usize,
}

/// `net`'s route (beside `partner`'s segments when given), its loss at `lossy`.
fn route(board: &Board, net: &str, partner: Option<&str>, lossy: Option<Lossy>) -> std::result::Result<Route, String> {
    let solved = solve_net(board, net, partner.map_or(Beside::Alone, Beside::Partner), lossy.is_some())?;
    let mut r = Route { length: 0.0, delay: 0.0, loss: lossy.map(|_| (Some(0.0), 0.0)), unsolved: 0 };
    for s in &solved {
        let len = emcraft_model::pcb::distance(s.segment.start, s.segment.end);
        r.length += len;
        let Some(mode) = s.outcome.mode() else {
            r.unsolved += 1;
            continue;
        };
        r.delay += mode.delay() * len;
        if let (Some(l), Some((conductor, dielectric))) = (lossy, r.loss.as_mut()) {
            let (c, d) = l.loss(mode, len);
            *conductor = conductor.zip(c).map(|(a, b)| a + b);
            *dielectric += d;
        }
    }
    Ok(r)
}

/// A route's finding: `(net, pair, quantity, value, limit, problem)`, values in display units
/// (length), ps and dB.
type RouteFinding = (String, Option<String>, &'static str, Option<f64>, f64, String);

/// What a class's limits find on its routes — each net's length, delay and loss (and each pair's
/// skew) over its limit or unknown — and each net's route as the reply shows it.
fn routes(board: &Board, class: &NetClass, unit: emcraft_model::units::Length) -> std::result::Result<(Vec<RouteFinding>, Vec<Value>), String> {
    let l = class.limits;
    if l.is_empty() {
        return Ok((vec![], vec![]));
    }
    let lossy = l.frequency.map(|frequency| Lossy { frequency, conductivity: COPPER });
    let has = |n: &str| board.segments.iter().any(|g| g.net == n);
    let paired = class.z_diff.is_some() || l.skew.is_some();
    let (mut found, mut shown, mut delays) = (Vec::new(), Vec::new(), std::collections::BTreeMap::new());
    let ps = |t: f64| thousandth(t * 1e12);
    for net in nets_of(board, class) {
        let partner = paired.then(|| partner_name(net, has)).flatten();
        let r = route(board, net, partner.as_deref(), lossy)?;
        let known = r.unsolved == 0;
        let loss = r.loss.and_then(|(c, d)| c.map(|c| c + d)).filter(|_| known);
        shown.push(json!({
            "net": net,
            "length": unit.from_si(r.length),
            "delay": known.then(|| ps(r.delay)),
            "loss": loss.map(db),
            "unsolved": r.unsolved,
        }));
        let mut over = |quantity: &'static str, value: Option<f64>, limit: Option<f64>, show: fn(f64) -> f64, why: &str| {
            if let Some(limit) = limit {
                match value {
                    Some(v) if v > limit => found.push((net.to_owned(), None, quantity, Some(show(v)), show(limit), "over its limit".to_owned())),
                    Some(_) => {}
                    None => found.push((net.to_owned(), None, quantity, None, show(limit), why.to_owned())),
                }
            }
        };
        let unknown = format!("{} segment(s) without an impedance", r.unsolved);
        over("length", Some(r.length), l.length, |x| x, "");
        over("delay", known.then_some(r.delay), l.delay, |t| thousandth(t * 1e12), &unknown);
        let lossless = if known { "a segment's copper has no thickness: no conductor loss" } else { unknown.as_str() };
        over("loss", loss, l.loss, db, lossless);
        delays.insert(net.to_owned(), (known.then_some(r.delay), partner));
    }
    // Each length found is shown in display units.
    for f in found.iter_mut().filter(|f| f.2 == "length") {
        f.3 = f.3.map(|v| unit.from_si(v));
        f.4 = unit.from_si(f.4);
    }
    if let Some(limit) = l.skew {
        for (net, (delay, partner)) in &delays {
            // Each pair once: from its lesser name, or from the one net of it in the class.
            let Some(partner) = partner.as_ref().filter(|p| net < *p || !delays.contains_key(*p)) else { continue };
            // The partner is routed, but may not be in the class: its delay solved beside the net.
            let theirs = match delays.get(partner) {
                Some((d, _)) => *d,
                None => {
                    let r = route(board, partner, Some(net), None)?;
                    (r.unsolved == 0).then_some(r.delay)
                }
            };
            match delay.zip(theirs).map(|(a, b)| (a - b).abs()) {
                Some(skew) if skew > limit => {
                    found.push((net.clone(), Some(partner.clone()), "skew", Some(ps(skew)), ps(limit), "over its limit".into()))
                }
                Some(_) => {}
                None => found.push((net.clone(), Some(partner.clone()), "skew", None, ps(limit), "a segment of the pair has no impedance".into())),
            }
        }
        if delays.values().all(|(_, p)| p.is_none()) {
            found.push((String::new(), None, "skew", None, ps(limit), "the class has no differential pairs".into()));
        }
    }
    Ok((found, shown))
}

/// A class's limits as replies show them: display units of length and frequency, ps and dB.
fn limits(l: &Limits, display: emcraft_model::units::Display) -> Value {
    json!({
        "max_length": l.length.map(|x| display.length.from_si(x)),
        "max_delay": l.delay.map(|t| thousandth(t * 1e12)),
        "max_skew": l.skew.map(|t| thousandth(t * 1e12)),
        "max_loss": l.loss,
        "frequency": l.frequency.map(|f| display.frequency.from_si(f)),
    })
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
        let nets = nets_of(board, class);
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
        let (limited, routes) = routes(board, class, length).map_err(failed)?;
        for (net, pair, quantity, value, limit, problem) in limited {
            failing += 1;
            found += 1;
            if findings.len() < MAX_FINDINGS {
                findings.push(json!({
                    "class": class.name,
                    "net": net,
                    "pair": pair,
                    "quantity": quantity,
                    "value": value,
                    "target": limit,
                    "problem": problem,
                }));
            }
        }
        let verdict = match (nets.len(), failing) {
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
            "limits": limits(&class.limits, s.display),
            "routes": routes,
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
        "delay_unit": "ps",
        "loss_unit": "dB",
        "ids": "each finding's segment is the id its board's source gave it: a PCBCraft board's track id, a KiCad board's track uuid (an arc's chords `uuid:k`)",
        "model": "pcb.impedance's: each segment's cross-section quasi-statically, references the nearest planes of other nets; z_diff = 2 z_odd beside the partner its name pairs with; a route's length, delay (a pair's odd mode's) and loss (copper's conductivity, smooth) over its segments, vias left out; a pair's skew the difference of its nets' delays",
    }))
}
