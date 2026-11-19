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
//! **Delay and losses.** A segment's delay is its length over the mode's speed, c/√ε_eff (a
//! pair's odd mode's). At a `frequency`, its losses: the dielectrics' by their loss tangents (each
//! layer's `loss_below`, a mask's `loss`) weighed by the electric energy in each, the copper's by
//! Wheeler's incremental inductance rule at the skin effect's surface resistance √(π f μ₀/σ) —
//! smooth copper many skin depths thick (a sheet has no conductor loss, copper under two skin
//! depths thick is said to be).
//!
//! Each distinct cross-section is solved once (on the threads), segments sharing it share its
//! result. The model is quasi-TEM, the box's sides conducting walls far from the strips.

use std::collections::BTreeMap;

use emcraft_model::pcb::{Board, Segment};
use emcraft_model::units::MU0;
use emcraft_solver::portmode::strips::{Block, Line, Mode, Section, Slab, Strip};
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

/// What a cross-section takes beside its segment: nothing, its differential partner's parallel
/// segment (however far across), or the nearest parallel segment of another net (`net`, else
/// any) whose edge is within `within` (m) of the segment's.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Beside<'a> {
    Alone,
    Partner(&'a str),
    Neighbour { net: Option<&'a str>, within: f64 },
}

/// The segment beside one: its net, the gap between the two strips' edges and the length the
/// two run side by side (m).
struct Near {
    net: String,
    gap: f64,
    overlap: f64,
}

/// A segment's cross-section as the solver takes it: its strips (the segment's the `own`th),
/// slabs, box; its references' layers (below, above), what lies beside it and the closed form a
/// single line's kind has, if any.
struct Built {
    section: Section,
    below: Option<usize>,
    above: Option<usize>,
    near: Option<Near>,
    own: usize,
    closed: Option<f64>,
}

/// The cross-section of segment `g` (with what `beside` takes beside it).
fn build(board: &Board, g: &Segment, net: &str, beside: Beside) -> Option<Built> {
    let heights = board.heights();
    let li = board.layers.iter().position(|l| l.name == g.layer)?;
    let h0 = heights[li];
    let at = middle(g);
    let named = |n: &str| match beside {
        Beside::Alone | Beside::Neighbour { net: None, .. } => false,
        Beside::Partner(p) | Beside::Neighbour { net: Some(p), .. } => n == p,
    };
    let foreign = |n: &str| n != net && !named(n);
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
            near: None,
            own: 0,
            closed: None,
        });
    }
    // The parallel segment beside the middle on the layer: its offset across (signed, to the
    // left of the segment's direction), width, net, gap and the length it runs beside.
    let (dir, length) = direction(g);
    let (wanted, within): (&dyn Fn(&str) -> bool, f64) = match beside {
        Beside::Alone => (&|_| false, 0.0),
        Beside::Partner(p) => (&move |n: &str| n == p, f64::INFINITY),
        Beside::Neighbour { net: a, within } => (&move |n: &str| n != net && a.is_none_or(|a| n == a), within),
    };
    let beside = board
        .segments
        .iter()
        .filter(|x| x.layer == g.layer && wanted(&x.net))
        .filter_map(|x| {
            let (d, len) = direction(x);
            let parallel = (dir[0] * d[1] - dir[1] * d[0]).abs() < 0.02;
            let along = (at[0] - x.start[0]) * d[0] + (at[1] - x.start[1]) * d[1];
            // The foot of the middle on the other's centre line, seen across the segment (its
            // left normal positive).
            let foot = [x.start[0] + along * d[0], x.start[1] + along * d[1]];
            let offset = (foot[0] - at[0]) * -dir[1] + (foot[1] - at[1]) * dir[0];
            let gap = offset.abs() - 0.5 * (g.width + x.width);
            // The other's ends along the segment, and how much of the segment lies between them.
            let ends = [x.start, x.end].map(|e| (e[0] - g.start[0]) * dir[0] + (e[1] - g.start[1]) * dir[1]);
            let overlap = (ends[0].max(ends[1]).min(length) - ends[0].min(ends[1]).max(0.0)).max(0.0);
            (parallel && (0.0..=len).contains(&along) && gap > 0.0 && gap <= within).then_some((offset, x.width, x.net.as_str(), gap, overlap))
        })
        .min_by(|a, b| a.0.abs().total_cmp(&b.0.abs()));
    let reach = [below.map(|l| h0 - heights[l]), above.map(|l| heights[l] - h0)].into_iter().flatten().fold(f64::INFINITY, f64::min);
    let span = g.width + beside.map_or(0.0, |(o, w, ..)| o.abs() + 0.5 * w);
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
            let l = &board.layers[i];
            (bottom < top && (l.eps_below > 1.0 || l.loss_below > 0.0)).then_some(Slab { bottom, top, eps: l.eps_below, loss: l.loss_below })
        })
        .collect();
    let (down, thickness) = (reaches(li).0, board.layers[li].copper);
    let strip = |left: f64, right: f64| Strip { left, right, height: h0 - down, thickness };
    let mut strips = vec![strip(-0.5 * g.width, 0.5 * g.width)];
    if let Some((o, w, ..)) = beside {
        strips.push(strip(o - 0.5 * w, o + 0.5 * w));
        strips.sort_by(|a, b| a.left.total_cmp(&b.left));
    }
    // The segment's strip among the two sorted across: the second when the other's offset is
    // negative (it sorts first).
    let own = usize::from(beside.is_some_and(|(o, ..)| o < 0.0));
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
        blocks.push(Block { left: -half_width, right: half_width, bottom: lo, top: hi, eps: m.eps, loss: m.loss });
        if l == li {
            for s in &strips {
                let (bottom, top) = if sign > 0.0 { (face, s.top() + m.thickness) } else { (s.height - m.thickness, face) };
                blocks.push(Block { left: s.left - m.thickness, right: s.right + m.thickness, bottom, top, eps: m.eps, loss: m.loss });
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
    let near = beside.map(|(_, _, net, gap, overlap)| Near { net: net.to_owned(), gap, overlap });
    Some(Built { section: Section { half_width, floor, ceiling, slabs, blocks, strips }, below, above, near, own, closed })
}

/// A key equal for segments whose cross-sections are the same to a nanometre.
fn key(b: &Built) -> String {
    let nm = |x: f64| (x * 1e9).round() as i64;
    let s = &b.section;
    let strips: Vec<_> = s.strips.iter().map(|x| [nm(x.left), nm(x.right), nm(x.height), nm(x.thickness)]).collect();
    let n9 = |x: f64| (x * 1e9).round() as i64;
    let slabs: Vec<_> = s.slabs.iter().map(|x| [nm(x.bottom), nm(x.top), n9(x.eps), n9(x.loss)]).collect();
    let blocks: Vec<_> = s.blocks.iter().map(|x| [nm(x.left), nm(x.right), nm(x.bottom), nm(x.top), n9(x.eps), n9(x.loss)]).collect();
    format!("{:?}{:?}{:?}{:?}{:?}", [nm(s.half_width), nm(s.floor), nm(s.ceiling)], strips, slabs, blocks, (b.below, b.above))
}

/// What a segment's cross-section gives: nothing (and why), a single line's mode (with its
/// impedance's closed form, if its kind has one), a pair's odd and even modes beside its partner,
/// or two coupled lines beside a neighbour.
pub(crate) enum Outcome {
    None(String),
    Single {
        mode: Mode,
        closed: Option<f64>,
    },
    /// The partner, the gap between the strips' edges (m), the odd and even modes.
    Pair {
        partner: String,
        gap: f64,
        odd: Mode,
        even: Mode,
    },
    /// The neighbour, the gap between the strips' edges and the length they run side by side
    /// (m), the two lines (the segment's the `own`th strip).
    Coupled {
        neighbour: String,
        gap: f64,
        overlap: f64,
        line: Line,
        own: usize,
    },
}

impl Outcome {
    /// The mode a signal on the segment travels in: a single line's, a pair's odd.
    pub(crate) fn mode(&self) -> Option<&Mode> {
        match self {
            Outcome::Single { mode, .. } | Outcome::Pair { odd: mode, .. } => Some(mode),
            Outcome::None(_) | Outcome::Coupled { .. } => None,
        }
    }
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

/// Every segment of `net` solved (with what `beside` takes beside it; with the modes' resistance
/// when `lossy`), each distinct cross-section once, on the threads. Refused: more than
/// [`MAX_ROUTE`] segments.
pub(crate) fn solve_net<'a>(board: &'a Board, net: &str, beside: Beside, lossy: bool) -> std::result::Result<Vec<Solved<'a>>, String> {
    let segments: Vec<&Segment> = board.segments.iter().filter(|g| g.net == net).collect();
    if segments.len() > MAX_ROUTE {
        return Err(format!("`{net}` has more than {MAX_ROUTE} segments"));
    }
    let built: Vec<Option<Built>> = segments.iter().map(|g| build(board, g, net, beside)).collect();
    let mut distinct: BTreeMap<String, &Section> = BTreeMap::new();
    for b in built.iter().flatten().filter(|b| !b.section.strips.is_empty()) {
        distinct.entry(key(b)).or_insert(&b.section);
    }
    let solved: BTreeMap<String, std::result::Result<Line, String>> = distinct
        .into_par_iter()
        .map(|(k, section)| (k, if lossy { section.solve_lossy() } else { section.solve() }.map_err(|e| e.to_string())))
        .collect();
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
                Some(Ok(line)) => match (b.near, beside, &line.modes[..]) {
                    (None, _, [mode]) => Outcome::Single { mode: *mode, closed: b.closed },
                    (Some(n), Beside::Partner(_), _) => match line.pair() {
                        Some((odd, even)) => Outcome::Pair { partner: n.net, gap: n.gap, odd, even },
                        None => Outcome::None("the pair's modes could not be told apart".into()),
                    },
                    (Some(n), _, _) => Outcome::Coupled { neighbour: n.net, gap: n.gap, overlap: n.overlap, line: line.clone(), own: b.own },
                    (None, ..) => Outcome::None("not one line".into()),
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

/// Rounded to a thousandth, as replies show picoseconds and percentages.
pub(crate) fn thousandth(x: f64) -> f64 {
    (x * 1e3).round() / 1e3
}

/// Rounded to a micro-decibel, as replies show losses.
pub(crate) fn db(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

/// Copper's conductivity (S/m, annealed: IEC 60028), when `conductivity` is not given.
pub(crate) const COPPER: f64 = 5.8e7;

/// Decibels in a neper, 20/ln 10.
const NEPER: f64 = 20.0 / std::f64::consts::LN_10;

/// A line's losses at `frequency` (Hz) with conductors of `conductivity` (S/m).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Lossy {
    pub frequency: f64,
    pub conductivity: f64,
}

impl Lossy {
    /// The parameters `frequency` (display units) and `conductivity` (S/m, default copper's):
    /// none without a frequency.
    pub(crate) fn read(p: &Params) -> Result<Option<Lossy>> {
        let conductivity = p.positive("conductivity").check(|s| (1e5..=1e9).contains(s), "is in S/m, 1e5–1e9").optional()?.unwrap_or(COPPER);
        Ok(p.frequency("frequency").optional()?.map(|frequency| Lossy { frequency, conductivity }))
    }

    /// The skin depth (m), 1/√(π f μ₀ σ).
    pub(crate) fn depth(&self) -> f64 {
        1.0 / (std::f64::consts::PI * self.frequency * MU0 * self.conductivity).sqrt()
    }

    /// A mode's conductor loss (none for a sheet) and dielectric loss over `length` (m), dB: at
    /// the surface resistance 1/(σ δ).
    pub(crate) fn loss(&self, mode: &Mode, length: f64) -> (Option<f64>, f64) {
        let (conductor, dielectric) = mode.attenuation(self.frequency, 1.0 / (self.conductivity * self.depth()));
        (conductor.map(|a| NEPER * a * length), NEPER * dielectric * length)
    }

    /// What its model leaves out on `board`: a sheet's conductor loss, copper under two skin
    /// depths thick.
    pub(crate) fn notes(&self, board: &Board) -> Vec<String> {
        let depth = self.depth();
        let sheets = board.layers.iter().filter(|l| l.copper <= 0.0).map(|l| l.name.as_str()).collect::<Vec<_>>();
        let thin = board.layers.iter().filter(|l| l.copper > 0.0 && l.copper < 2.0 * depth).map(|l| l.name.as_str()).collect::<Vec<_>>();
        let mut notes = Vec::new();
        if !sheets.is_empty() {
            notes.push(format!("{} have no copper thickness: their lines' conductor loss is not given", sheets.join(", ")));
        }
        if !thin.is_empty() {
            notes.push(format!(
                "{}'s copper is under two skin depths ({:.2} µm) thick at this frequency: its conductor loss is underestimated",
                thin.join(", "),
                depth * 1e6
            ));
        }
        notes.push("smooth copper: a rough foil's extra loss is not included".into());
        notes
    }
}

/// `pcb.impedance`: the impedance of every segment of `net` (and of its pair), its cross-section
/// solved (module docs); its delay, and at a `frequency` its losses.
pub fn impedance(s: &mut Session, p: &Params) -> Result<Value> {
    let board = s.model.pcb().ok_or_else(|| failed("load a PCB with pcb.import, pcb.load or pcb.sample"))?;
    let net = p.string("net").required()?;
    let has = |n: &str| board.segments.iter().any(|g| g.net == n);
    if !has(net) {
        return Err(p.error("net", "names no routed net of this board"));
    }
    let partner = partner(p, net, has)?;
    let lossy = Lossy::read(p)?;
    let beside = partner.as_deref().map_or(Beside::Alone, Beside::Partner);
    let rows = solve_net(board, net, beside, lossy.is_some()).map_err(|e| p.error("net", &e))?;
    let length = s.display.length;
    let name = |l: Option<usize>| l.and_then(|l| board.layers.get(l)).map(|l| l.name.clone());
    let mut shown = Vec::new();
    let mut weighted: Vec<(f64, f64)> = Vec::new();
    let (mut delay, mut conductor, mut dielectric, mut unsolved) = (0.0, Some(0.0), 0.0, 0usize);
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
            Outcome::Single { mode, closed } => {
                row["z0"] = json!(ohms(mode.impedance));
                row["eps_eff"] = json!(mode.eps_eff);
                if let Some(c) = closed {
                    row["closed_form"] = json!(ohms(*c));
                }
                weighted.push((mode.impedance, len));
            }
            Outcome::Pair { partner, gap, odd, even } => {
                row["pair"] = json!(partner);
                row["gap"] = json!(length.from_si(*gap));
                row["z_odd"] = json!(ohms(odd.impedance));
                row["z_even"] = json!(ohms(even.impedance));
                row["z_diff"] = json!(ohms(2.0 * odd.impedance));
                row["z_common"] = json!(ohms(0.5 * even.impedance));
                row["eps_eff_odd"] = json!(odd.eps_eff);
                row["eps_eff_even"] = json!(even.eps_eff);
                weighted.push((2.0 * odd.impedance, len));
            }
            Outcome::Coupled { .. } => row["note"] = json!("solved beside another net"),
        }
        match r.outcome.mode() {
            Some(mode) => {
                row["delay"] = json!(thousandth(mode.delay() * len * 1e12));
                delay += mode.delay() * len;
                if let Some(l) = lossy {
                    let (c, d) = l.loss(mode, len);
                    row["loss"] = json!({"conductor": c.map(db), "dielectric": db(d), "total": c.map(|c| db(c + d))});
                    conductor = conductor.zip(c).map(|(a, b)| a + b);
                    dielectric += d;
                }
            }
            None => unsolved += 1,
        }
        shown.push(row);
    }
    let total: f64 = weighted.iter().map(|w| w.1).sum();
    let summary = (!weighted.is_empty()).then(|| {
        let (lo, hi) = weighted.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), (z, _)| (a.min(*z), b.max(*z)));
        json!({"min": ohms(lo), "max": ohms(hi), "mean": ohms(weighted.iter().map(|(z, l)| z * l).sum::<f64>() / total)})
    });
    let routed: f64 = rows.iter().map(|r| direction(r.segment).1).sum();
    let mut route = json!({"length": length.from_si(routed), "delay": thousandth(delay * 1e12), "unsolved": unsolved});
    let mut notes = Vec::new();
    if let Some(l) = lossy {
        route["loss"] = json!({"conductor": conductor.map(db), "dielectric": db(dielectric), "total": conductor.map(|c| db(c + dielectric))});
        route["frequency"] = json!(s.display.frequency.from_si(l.frequency));
        notes = l.notes(board);
    }
    if unsolved > 0 {
        notes.push(format!("{unsolved} segment(s) without an impedance are left out of the route's delay and loss"));
    }
    Ok(json!({
        "net": net,
        "pair": partner,
        "length_unit": length.symbol(),
        "delay_unit": "ps",
        "loss_unit": "dB",
        "sections": shown,
        "impedance": summary,
        "route": route,
        "notes": notes,
        "model": "each segment's cross-section by the finite elements, quasi-statically (its capacitances with and without the dielectrics): quasi-TEM, each layer's copper its thickness (outer layers' standing into the air, inner ones' centred), the solder masks a coat over the outer faces and their strips, references the nearest planes of other nets above and below; a pair's segments with the partner's parallel segment beside (z_diff = 2 z_odd); the summary is length-weighted, of z0 or of z_diff. A segment's delay is its length at the speed c/√ε_eff (a pair's odd mode's); its loss at `frequency` the dielectrics' (their loss tangents weighed by the electric energy in each) and the copper's (Wheeler's incremental inductance rule at the skin effect's surface resistance, smooth copper); the route's are the sums over its segments (vias left out)",
    }))
}
