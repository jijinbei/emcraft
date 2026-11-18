//! `pcb.impedance`: every routed segment of a net (and of its differential partner) as the line
//! it is, its cross-section solved quasi-statically by the finite elements
//! (`emcraft_solver::portmode::strips`: capacitances with and without the dielectrics):
//! the strip at its layer's height, the board's dielectrics as slabs, its reference planes the
//! box's floor and ceiling, an open side's lid far away; the partner's parallel segment beside it
//! for a pair. Outer and inner layers alike — microstrips, embedded microstrips, striplines.
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
//! result. The model is quasi-TEM, lossless, the copper of no thickness, the box's sides
//! conducting walls far from the strips.

use std::collections::BTreeMap;

use emcraft_model::pcb::{Board, Segment};
use emcraft_solver::portmode::strips::{Line, Section, Slab, Strip};
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
            section: Section { half_width: 0.0, floor: 0.0, ceiling: 0.0, slabs: vec![], strips: vec![] },
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
    let (floor, ceiling) = (below.map_or(h0 - open, |l| heights[l]), above.map_or(h0 + open, |l| heights[l]));
    let half_width = 12.0 * reach.max(span) + span;
    // The dielectrics between the layers (top down: each layer's gap below it), clipped to the box.
    let slabs = (0..board.layers.len().saturating_sub(1))
        .filter_map(|i| {
            let (top, bottom) = (heights[i].min(ceiling), heights[i + 1].max(floor));
            (bottom < top && board.layers[i].eps_below > 1.0).then_some(Slab { bottom, top, eps: board.layers[i].eps_below })
        })
        .collect();
    let mut strips = vec![Strip { left: -0.5 * g.width, right: 0.5 * g.width, height: h0 }];
    if let Some((o, w)) = beside {
        strips.push(Strip { left: o - 0.5 * w, right: o + 0.5 * w, height: h0 });
        strips.sort_by(|a, b| a.left.total_cmp(&b.left));
    }
    // A closed form for a single line of one dielectric: Hammerstad and Jensen's outer microstrip,
    // Cohn's centred stripline.
    let eps_between = |a: f64, b: f64| {
        let ls: Vec<f64> =
            (0..board.layers.len().saturating_sub(1)).filter(|&i| heights[i + 1] < b && heights[i] > a).map(|i| board.layers[i].eps_below).collect();
        (ls.windows(2).all(|w| w[0] == w[1])).then(|| ls.first().copied()).flatten()
    };
    let outer = li == 0 || li + 1 == board.layers.len();
    let closed = match (beside, below, above) {
        (None, Some(l), None) if outer && li == 0 => {
            let h = h0 - heights[l];
            eps_between(heights[l], h0).map(|e| emcraft_theory::microstrip::z0(g.width / h, e))
        }
        (None, None, Some(l)) if outer && li + 1 == board.layers.len() => {
            let h = heights[l] - h0;
            eps_between(h0, heights[l]).map(|e| emcraft_theory::microstrip::z0(g.width / h, e))
        }
        (None, Some(lb), Some(la)) => {
            let (b, centred) = (heights[la] - heights[lb], ((h0 - heights[lb]) - (heights[la] - h0)).abs());
            eps_between(heights[lb], heights[la]).filter(|_| centred <= 0.01 * b).map(|e| emcraft_theory::stripline::z0(g.width, b, e))
        }
        _ => None,
    };
    let partner = beside.and(partner).map(|p| (p.to_owned(), beside.map_or(0.0, |(o, w)| o.abs() - 0.5 * (g.width + w))));
    Some(Built { section: Section { half_width, floor, ceiling, slabs, strips }, below, above, partner, closed })
}

/// A key equal for segments whose cross-sections are the same to a nanometre.
fn key(b: &Built) -> String {
    let nm = |x: f64| (x * 1e9).round() as i64;
    let s = &b.section;
    let strips: Vec<_> = s.strips.iter().map(|x| [nm(x.left), nm(x.right), nm(x.height)]).collect();
    let slabs: Vec<_> = s.slabs.iter().map(|x| [nm(x.bottom), nm(x.top), (x.eps * 1e9).round() as i64]).collect();
    format!("{:?}{:?}{:?}{:?}", [nm(s.half_width), nm(s.floor), nm(s.ceiling)], strips, slabs, (b.below, b.above))
}

/// `pcb.impedance`: the impedance of every segment of `net` (and of its pair), its cross-section
/// solved (module docs).
pub fn impedance(s: &mut Session, p: &Params) -> Result<Value> {
    let board = s.model.pcb().ok_or_else(|| failed("load a PCB with pcb.import, pcb.load or pcb.sample"))?;
    let net = p.string("net").required()?;
    if !board.segments.iter().any(|g| g.net == net) {
        return Err(p.error("net", "names no routed net of this board"));
    }
    let has = |n: &str| board.segments.iter().any(|g| g.net == n);
    let partner = match p.string("pair").optional()? {
        Some(q) if q == net => return Err(p.error("pair", "must be another net")),
        Some(q) if has(q) => Some(q.to_owned()),
        Some(_) => return Err(p.error("pair", "names no routed net of this board")),
        None => partner_name(net, has),
    };
    let segments: Vec<&Segment> = board.segments.iter().filter(|g| g.net == net).collect();
    if segments.len() > 512 {
        return Err(p.error("net", "has more than 512 segments"));
    }
    let built: Vec<Option<Built>> = segments.iter().map(|g| build(board, g, net, partner.as_deref())).collect();
    // Each distinct cross-section solved once, on the threads.
    let mut distinct: BTreeMap<String, &Section> = BTreeMap::new();
    for b in built.iter().flatten().filter(|b| !b.section.strips.is_empty()) {
        distinct.entry(key(b)).or_insert(&b.section);
    }
    let solved: BTreeMap<String, std::result::Result<Line, String>> =
        distinct.into_par_iter().map(|(k, section)| (k, section.solve().map_err(|e| e.to_string()))).collect();
    let (length, ohms) = (s.display.length, |z: f64| (z * 1e3).round() / 1e3);
    let name = |l: Option<usize>| l.and_then(|l| board.layers.get(l)).map(|l| l.name.clone());
    let mut rows = Vec::new();
    let mut weighted: Vec<(f64, f64)> = Vec::new();
    for (g, b) in segments.iter().zip(&built) {
        let len = direction(g).1;
        let mut row = json!({"segment": g.id, "layer": g.layer, "width": length.from_si(g.width), "length": length.from_si(len)});
        let Some(b) = b else {
            row["note"] = json!("its layer is not in the stackup");
            rows.push(row);
            continue;
        };
        row["below"] = json!(name(b.below));
        row["above"] = json!(name(b.above));
        if b.section.strips.is_empty() {
            row["note"] = json!("no reference plane above or below: no impedance is defined");
            rows.push(row);
            continue;
        }
        let li = board.layers.iter().position(|l| l.name == g.layer).unwrap_or(0);
        let outer = li == 0 || li + 1 == board.layers.len();
        row["kind"] = json!(match (b.below.is_some() && b.above.is_some(), outer) {
            (true, _) => "stripline",
            (false, true) => "microstrip",
            (false, false) => "embedded microstrip",
        });
        match solved.get(&key(b)) {
            Some(Ok(line)) => match (line.single(), line.pair(), &b.partner) {
                (Some(z), _, _) => {
                    row["z0"] = json!(ohms(z));
                    row["eps_eff"] = json!(line.modes[0].eps_eff);
                    if let Some(c) = b.closed {
                        row["closed_form"] = json!(ohms(c));
                    }
                    weighted.push((z, len));
                }
                (None, Some((odd, even)), Some((pname, gap))) => {
                    row["pair"] = json!(pname);
                    row["gap"] = json!(length.from_si(*gap));
                    row["z_odd"] = json!(ohms(odd.impedance));
                    row["z_even"] = json!(ohms(even.impedance));
                    row["z_diff"] = json!(ohms(2.0 * odd.impedance));
                    row["z_common"] = json!(ohms(0.5 * even.impedance));
                    row["eps_eff_odd"] = json!(odd.eps_eff);
                    row["eps_eff_even"] = json!(even.eps_eff);
                    weighted.push((2.0 * odd.impedance, len));
                }
                _ => row["note"] = json!("the pair's modes could not be told apart"),
            },
            Some(Err(e)) => row["note"] = json!(e),
            None => row["note"] = json!("not solved"),
        }
        rows.push(row);
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
        "sections": rows,
        "impedance": summary,
        "model": "each segment's cross-section by the finite elements, quasi-statically (its capacitances with and without the dielectrics): quasi-TEM, lossless, copper of no thickness, references the nearest planes of other nets above and below; a pair's segments with the partner's parallel segment beside (z_diff = 2 z_odd); the summary is length-weighted, of z0 or of z_diff",
    }))
}
