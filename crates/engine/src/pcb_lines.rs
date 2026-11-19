//! `pcb.impedance`: every routed segment of a net (and of its differential partner) as the line
//! it is, its cross-section solved quasi-statically by the finite elements
//! (`emcraft_solver::portmode::strips`: capacitances with and without the dielectrics):
//! the strip at its layer's height, the board's dielectrics as slabs, its reference planes the
//! box's floor and ceiling, an open side's lid far away; the partner's parallel segment beside it
//! for a pair. Outer and inner layers alike — microstrips, embedded microstrips, striplines.
//!
//! **Copper and mask.** A layer's copper thickness makes its strips (and its planes) that thick:
//! on the top layer standing up into the air, on the bottom one hanging down, on an inner layer
//! centred on the layer (half into each dielectric beside it). A solder mask on an outer layer is
//! a coat its thickness deep over the board's face and over the strips on that layer, their top
//! and sides.
//!
//! **References.** A segment's reference planes are the nearest layers below and above it with
//! copper of another net (a pour, a plane, another net's wide trace) under its middle — the rule
//! `pcb.port` follows. A segment with none has no defined impedance and is reported so.
//!
//! **Pairs.** The partner net is the one given, else the one the net's name pairs with (`+`/`-`,
//! `_P`/`_N`, `P`/`N` at its end). A segment of the net paired is one with a parallel segment of the
//! partner on its layer beside its middle: their odd and even modes, Z_diff = 2 Z_odd and
//! Z_common = Z_even / 2. A segment without one is solved alone.
//!
//! Each distinct cross-section is solved once (on the threads), segments sharing it share its
//! result. The model is quasi-TEM and lossless, the box's sides conducting walls far from the
//! strips.

use std::collections::BTreeMap;

use emcraft_model::pcb::{Board, Segment};
use emcraft_solver::portmode::strips::{Block, Line, Section, Slab, Strip};
use rayon::prelude::*;
use serde_json::{Value, json};

use crate::{EngineError, Params, Result, Session};

fn failed(s: impl ToString) -> EngineError {
    EngineError::failed(s.to_string())
}

/// A segment's direction (unit) and length (m).
fn direction(s: &Segment) -> ([f64; 2], f64) {
    let d = [s.end[0] - s.start[0], s.end[1] - s.start[1]];
    let len = d[0].hypot(d[1]);
    ([d[0] / len, d[1] / len], len)
}

/// The segment's middle (m).
fn middle(s: &Segment) -> [f64; 2] {
    [0.5 * (s.start[0] + s.end[0]), 0.5 * (s.start[1] + s.end[1])]
}

/// Whether `p` lies inside `polygon` (even–odd).
fn inside(polygon: &[[f64; 2]], p: [f64; 2]) -> bool {
    let mut odd = false;
    for (i, a) in polygon.iter().enumerate() {
        let b = polygon[(i + 1) % polygon.len()];
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]) {
            odd = !odd;
        }
    }
    odd
}

/// The distance (m) from `p` to segment `a`–`b`.
fn to_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let len2 = d[0] * d[0] + d[1] * d[1];
    let t = if len2 > 0.0 { (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / len2).clamp(0.0, 1.0) } else { 0.0 };
    (p[0] - a[0] - t * d[0]).hypot(p[1] - a[1] - t * d[1])
}

/// The net `net`'s name pairs with on `board`: `+`/`-`, `_P`/`_N` or `P`/`N` at its end, either
/// case, when the board has a net of that name.
///
/// ```
/// use emcraft_engine::pcb_lines::partner_name;
/// let nets = ["USB_D+", "USB_D-", "CLK_P", "CLK_N", "SDA"];
/// let has = |n: &str| nets.contains(&n);
/// assert_eq!(partner_name("USB_D+", has).as_deref(), Some("USB_D-"));
/// assert_eq!(partner_name("CLK_N", has).as_deref(), Some("CLK_P"));
/// assert_eq!(partner_name("SDA", has), None);
/// ```
pub fn partner_name(net: &str, has: impl Fn(&str) -> bool) -> Option<String> {
    let swaps = [("+", "-"), ("-", "+"), ("_P", "_N"), ("_N", "_P"), ("_p", "_n"), ("_n", "_p"), ("P", "N"), ("N", "P"), ("p", "n"), ("n", "p")];
    swaps.iter().find_map(|(a, b)| net.strip_suffix(a).map(|stem| format!("{stem}{b}")).filter(|other| other != net && has(other)))
}

/// A segment's cross-section as the solver takes it: its strips, slabs, box; its references'
/// layers (below, above) and the closed form a single line's kind has, if any.
struct Built {
    section: Section,
    below: Option<usize>,
    above: Option<usize>,
    partner: Option<(String, f64)>,
    closed: Option<f64>,
}

/// The cross-section of segment `g` (its partner `partner`'s segments beside it, if any).
fn build(board: &Board, g: &Segment, net: &str, partner: Option<&str>) -> Option<Built> {
    let heights = board.heights();
    let li = board.layers.iter().position(|l| l.name == g.layer)?;
    let h0 = heights[li];
    let at = middle(g);
    let foreign = |n: &str| n != net && Some(n) != partner;
    let covered = |l: usize| {
        let name = &board.layers[l].name;
        board.copper.iter().any(|c| &c.layer == name && foreign(&c.net) && inside(&c.polygon, at))
            || board.segments.iter().any(|x| &x.layer == name && foreign(&x.net) && to_segment(at, x.start, x.end) <= 0.5 * x.width)
    };
    let below = (0..board.layers.len()).filter(|&l| heights[l] < h0 && covered(l)).max_by(|&a, &b| heights[a].total_cmp(&heights[b]));
    let above = (0..board.layers.len()).filter(|&l| heights[l] > h0 && covered(l)).min_by(|&a, &b| heights[a].total_cmp(&heights[b]));
    if below.is_none() && above.is_none() {
        return Some(Built {
            section: Section { half_width: 0.0, floor: 0.0, ceiling: 0.0, slabs: vec![], blocks: vec![], strips: vec![] },
            below,
            above,
            partner: None,
            closed: None,
        });
    }
    // The partner's parallel segment on the layer beside the middle: its offset across (signed,
    // to the left of the segment's direction) and width.
    let (dir, _) = direction(g);
    let beside = partner.and_then(|p| {
        board
            .segments
            .iter()
            .filter(|x| x.net == p && x.layer == g.layer)
            .filter_map(|x| {
                let (d, len) = direction(x);
                let parallel = (dir[0] * d[1] - dir[1] * d[0]).abs() < 0.02;
                let along = (at[0] - x.start[0]) * d[0] + (at[1] - x.start[1]) * d[1];
                // The foot of the middle on the partner's centre line, seen across the segment
                // (its left normal positive).
                let foot = [x.start[0] + along * d[0], x.start[1] + along * d[1]];
                let offset = (foot[0] - at[0]) * -dir[1] + (foot[1] - at[1]) * dir[0];
                (parallel && (0.0..=len).contains(&along) && offset.abs() > 0.5 * (g.width + x.width)).then_some((offset, x.width))
            })
            .min_by(|a, b| a.0.abs().total_cmp(&b.0.abs()))
    });
    let reach = [below.map(|l| h0 - heights[l]), above.map(|l| heights[l] - h0)].into_iter().flatten().fold(f64::INFINITY, f64::min);
    let span = g.width + beside.map_or(0.0, |(o, w)| o.abs() + 0.5 * w);
    let open = 20.0 * reach.max(span);
    // Each layer's copper: its thickness and how far it reaches below and above the layer — up
    // into the air on top, down on the bottom, half each way inside.
    let last = board.layers.len() - 1;
    let reaches = |l: usize| {
        let t = board.layers[l].copper;
        match l {
            0 => (0.0, t),
            l if l == last => (t, 0.0),
            _ => (0.5 * t, 0.5 * t),
        }
    };
    // The planes' faces towards the strip.
    let (floor, ceiling) = (below.map_or(h0 - open, |l| heights[l] + reaches(l).1), above.map_or(h0 + open, |l| heights[l] - reaches(l).0));
    let half_width = 12.0 * reach.max(span) + span;
    // The dielectrics between the layers (top down: each layer's gap below it), clipped to the box.
    let slabs = (0..board.layers.len().saturating_sub(1))
        .filter_map(|i| {
            let (top, bottom) = (heights[i].min(ceiling), heights[i + 1].max(floor));
            (bottom < top && board.layers[i].eps_below > 1.0).then_some(Slab { bottom, top, eps: board.layers[i].eps_below })
        })
        .collect();
    let (down, thickness) = (reaches(li).0, board.layers[li].copper);
    let strip = |left: f64, right: f64| Strip { left, right, height: h0 - down, thickness };
    let mut strips = vec![strip(-0.5 * g.width, 0.5 * g.width)];
    if let Some((o, w)) = beside {
        strips.push(strip(o - 0.5 * w, o + 0.5 * w));
        strips.sort_by(|a, b| a.left.total_cmp(&b.left));
    }
    // The masks over the outer faces in the box: a coat over the face, and over the strips when
    // they are on that face's layer.
    let mut blocks = Vec::new();
    for m in &board.masks {
        let Some(l) = [0, last].into_iter().find(|&l| board.layers[l].name == m.layer) else { continue };
        let (face, sign) = if l == 0 { (heights[0], 1.0) } else { (heights[last], -1.0) };
        let (lo, hi) = if sign > 0.0 { (face, face + m.thickness) } else { (face - m.thickness, face) };
        if hi <= floor || lo >= ceiling {
            continue;
        }
        blocks.push(Block { left: -half_width, right: half_width, bottom: lo, top: hi, eps: m.eps });
        if l == li {
            for s in &strips {
                let (bottom, top) = if sign > 0.0 { (face, s.top() + m.thickness) } else { (s.height - m.thickness, face) };
                blocks.push(Block { left: s.left - m.thickness, right: s.right + m.thickness, bottom, top, eps: m.eps });
            }
        }
    }
    let masked = !blocks.is_empty();
    // A closed form for a single line of one dielectric: Hammerstad and Jensen's outer microstrip,
    // Cohn's centred stripline.
    let eps_between = |a: f64, b: f64| {
        let ls: Vec<f64> =
            (0..board.layers.len().saturating_sub(1)).filter(|&i| heights[i + 1] < b && heights[i] > a).map(|i| board.layers[i].eps_below).collect();
        (ls.windows(2).all(|w| w[0] == w[1])).then(|| ls.first().copied()).flatten()
    };
    // Hammerstad and Jensen's microstrip (with its thickness) has no mask; Cohn's stripline no
    // thickness either.
    let microstrip = |h: f64, e: f64| emcraft_theory::microstrip::thick(g.width / h, thickness / h, e).0;
    let closed = match (beside, below, above) {
        (None, Some(l), None) if li == 0 && !masked => eps_between(heights[l], h0).map(|e| microstrip(h0 - (heights[l] + reaches(l).1), e)),
        (None, None, Some(l)) if li == last && !masked => eps_between(h0, heights[l]).map(|e| microstrip(heights[l] - reaches(l).0 - h0, e)),
        (None, Some(lb), Some(la)) if thickness == 0.0 && floor == heights[lb] && ceiling == heights[la] => {
            let (b, centred) = (heights[la] - heights[lb], ((h0 - heights[lb]) - (heights[la] - h0)).abs());
            eps_between(heights[lb], heights[la]).filter(|_| centred <= 0.01 * b).map(|e| emcraft_theory::stripline::z0(g.width, b, e))
        }
        _ => None,
    };
    let partner = beside.and(partner).map(|p| (p.to_owned(), beside.map_or(0.0, |(o, w)| o.abs() - 0.5 * (g.width + w))));
    Some(Built { section: Section { half_width, floor, ceiling, slabs, blocks, strips }, below, above, partner, closed })
}

/// A key equal for segments whose cross-sections are the same to a nanometre.
fn key(b: &Built) -> String {
    let nm = |x: f64| (x * 1e9).round() as i64;
    let s = &b.section;
    let strips: Vec<_> = s.strips.iter().map(|x| [nm(x.left), nm(x.right), nm(x.height), nm(x.thickness)]).collect();
    let slabs: Vec<_> = s.slabs.iter().map(|x| [nm(x.bottom), nm(x.top), (x.eps * 1e9).round() as i64]).collect();
    let blocks: Vec<_> = s.blocks.iter().map(|x| [nm(x.left), nm(x.right), nm(x.bottom), nm(x.top), (x.eps * 1e9).round() as i64]).collect();
    format!("{:?}{:?}{:?}{:?}{:?}", [nm(s.half_width), nm(s.floor), nm(s.ceiling)], strips, slabs, blocks, (b.below, b.above))
}

/// What a segment's cross-section gives: nothing (and why), a single line's impedance (with its
/// closed form, if its kind has one) or a pair's odd and even modes beside its partner.
pub(crate) enum Outcome {
    None(String),
    Single {
        z: f64,
        eps_eff: f64,
        closed: Option<f64>,
    },
    /// The partner, the gap between the strips' edges (m), the odd and even modes' (Z, ε_eff).
    Pair {
        partner: String,
        gap: f64,
        odd: (f64, f64),
        even: (f64, f64),
    },
}

/// A segment of a net and what its cross-section gives: its references' layers (below, above)
/// and its kind (`microstrip`, `embedded microstrip`, `stripline`) when it has a reference.
pub(crate) struct Solved<'a> {
    pub segment: &'a Segment,
    pub below: Option<usize>,
    pub above: Option<usize>,
    pub kind: Option<&'static str>,
    pub outcome: Outcome,
}

/// The most segments of a net solved at once.
const MAX_ROUTE: usize = 512;

/// Every segment of `net` solved (its pair's segments beside it with `partner`), each distinct
/// cross-section once, on the threads. Refused: more than [`MAX_ROUTE`] segments.
pub(crate) fn solve_net<'a>(board: &'a Board, net: &str, partner: Option<&str>) -> std::result::Result<Vec<Solved<'a>>, String> {
    let segments: Vec<&Segment> = board.segments.iter().filter(|g| g.net == net).collect();
    if segments.len() > MAX_ROUTE {
        return Err(format!("`{net}` has more than {MAX_ROUTE} segments"));
    }
    let built: Vec<Option<Built>> = segments.iter().map(|g| build(board, g, net, partner)).collect();
    let mut distinct: BTreeMap<String, &Section> = BTreeMap::new();
    for b in built.iter().flatten().filter(|b| !b.section.strips.is_empty()) {
        distinct.entry(key(b)).or_insert(&b.section);
    }
    let solved: BTreeMap<String, std::result::Result<Line, String>> =
        distinct.into_par_iter().map(|(k, section)| (k, section.solve().map_err(|e| e.to_string()))).collect();
    let last = board.layers.len().saturating_sub(1);
    let rows = segments
        .into_iter()
        .zip(built)
        .map(|(g, b)| {
            let Some(b) = b else {
                return Solved { segment: g, below: None, above: None, kind: None, outcome: Outcome::None("its layer is not in the stackup".into()) };
            };
            let (below, above) = (b.below, b.above);
            if b.section.strips.is_empty() {
                let why = "no reference plane above or below: no impedance is defined".into();
                return Solved { segment: g, below, above, kind: None, outcome: Outcome::None(why) };
            }
            let li = board.layers.iter().position(|l| l.name == g.layer).unwrap_or(0);
            let kind = match (below.is_some() && above.is_some(), li == 0 || li == last) {
                (true, _) => "stripline",
                (false, true) => "microstrip",
                (false, false) => "embedded microstrip",
            };
            let outcome = match solved.get(&key(&b)) {
                Some(Ok(line)) => match (line.single(), line.pair(), b.partner) {
                    (Some(z), _, _) => Outcome::Single { z, eps_eff: line.modes[0].eps_eff, closed: b.closed },
                    (None, Some((odd, even)), Some((partner, gap))) => {
                        Outcome::Pair { partner, gap, odd: (odd.impedance, odd.eps_eff), even: (even.impedance, even.eps_eff) }
                    }
                    _ => Outcome::None("the pair's modes could not be told apart".into()),
                },
                Some(Err(e)) => Outcome::None(e.clone()),
                None => Outcome::None("not solved".into()),
            };
            Solved { segment: g, below, above, kind: Some(kind), outcome }
        })
        .collect();
    Ok(rows)
}

/// The differential partner of `net`: `pair` when given (another routed net), else the one its
/// name pairs with. `has` tells a routed net of the board.
pub(crate) fn partner(p: &Params, net: &str, has: impl Fn(&str) -> bool) -> Result<Option<String>> {
    match p.string("pair").optional()? {
        Some(q) if q == net => Err(p.error("pair", "must be another net")),
        Some(q) if has(q) => Ok(Some(q.to_owned())),
        Some(_) => Err(p.error("pair", "names no routed net of this board")),
        None => Ok(partner_name(net, has)),
    }
}

/// Rounded to a milliohm, as replies show impedances.
pub(crate) fn ohms(z: f64) -> f64 {
    (z * 1e3).round() / 1e3
}

/// `pcb.impedance`: the impedance of every segment of `net` (and of its pair), its cross-section
/// solved (module docs).
pub fn impedance(s: &mut Session, p: &Params) -> Result<Value> {
    let board = s.model.pcb().ok_or_else(|| failed("load a PCB with pcb.import, pcb.load or pcb.sample"))?;
    let net = p.string("net").required()?;
    let has = |n: &str| board.segments.iter().any(|g| g.net == n);
    if !has(net) {
        return Err(p.error("net", "names no routed net of this board"));
    }
    let partner = partner(p, net, has)?;
    let rows = solve_net(board, net, partner.as_deref()).map_err(|e| p.error("net", &e))?;
    let length = s.display.length;
    let name = |l: Option<usize>| l.and_then(|l| board.layers.get(l)).map(|l| l.name.clone());
    let mut shown = Vec::new();
    let mut weighted: Vec<(f64, f64)> = Vec::new();
    for r in &rows {
        let g = r.segment;
        let len = direction(g).1;
        let mut row = json!({"segment": g.id, "layer": g.layer, "width": length.from_si(g.width), "length": length.from_si(len)});
        if r.kind.is_some() || r.below.is_some() || r.above.is_some() {
            row["below"] = json!(name(r.below));
            row["above"] = json!(name(r.above));
        }
        if let Some(kind) = r.kind {
            row["kind"] = json!(kind);
        }
        match &r.outcome {
            Outcome::None(why) => row["note"] = json!(why),
            Outcome::Single { z, eps_eff, closed } => {
                row["z0"] = json!(ohms(*z));
                row["eps_eff"] = json!(eps_eff);
                if let Some(c) = closed {
                    row["closed_form"] = json!(ohms(*c));
                }
                weighted.push((*z, len));
            }
            Outcome::Pair { partner, gap, odd, even } => {
                row["pair"] = json!(partner);
                row["gap"] = json!(length.from_si(*gap));
                row["z_odd"] = json!(ohms(odd.0));
                row["z_even"] = json!(ohms(even.0));
                row["z_diff"] = json!(ohms(2.0 * odd.0));
                row["z_common"] = json!(ohms(0.5 * even.0));
                row["eps_eff_odd"] = json!(odd.1);
                row["eps_eff_even"] = json!(even.1);
                weighted.push((2.0 * odd.0, len));
            }
        }
        shown.push(row);
    }
    let total: f64 = weighted.iter().map(|w| w.1).sum();
    let summary = (!weighted.is_empty()).then(|| {
        let (lo, hi) = weighted.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), (z, _)| (a.min(*z), b.max(*z)));
        json!({"min": ohms(lo), "max": ohms(hi), "mean": ohms(weighted.iter().map(|(z, l)| z * l).sum::<f64>() / total)})
    });
    Ok(json!({
        "net": net,
        "pair": partner,
        "length_unit": length.symbol(),
        "sections": shown,
        "impedance": summary,
        "model": "each segment's cross-section by the finite elements, quasi-statically (its capacitances with and without the dielectrics): quasi-TEM, lossless, each layer's copper its thickness (outer layers' standing into the air, inner ones' centred), the solder masks a coat over the outer faces and their strips, references the nearest planes of other nets above and below; a pair's segments with the partner's parallel segment beside (z_diff = 2 z_odd); the summary is length-weighted, of z0 or of z_diff",
    }))
}
