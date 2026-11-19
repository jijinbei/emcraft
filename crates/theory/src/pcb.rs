//! PCB route screening plus an explicitly limited, lossless quasi-TEM line model.
//! Impedance: Hammerstad–Jensen (1980), existing `microstrip` implementation. Cascading:
//! Pozar, Microwave Engineering, 4th ed., §4.4 (`coupled::line`/`s_params`). Geometry findings
//! are not a field solution or an EMC/SI compliance verdict. No guessed ground across gaps.
use crate::{coupled, microstrip};
use emcraft_model::pcb::{Board, Segment, distance};
use emcraft_model::units::C0;
use num_complex::Complex64 as C;
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone, Debug)]
pub struct Finding {
    pub code: Code,
    pub detail: String,
    pub segment: Option<String>,
}

emcraft_macros::named_enum! {
    /// What a geometric finding is about.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Code {
        /// The route branches, breaks or closes: no two-terminal line.
        RouteTopology = "route_topology",
        /// The route changes layers through a via.
        ViaTransition = "via_transition",
        /// No reference copper on another layer.
        NoReference = "no_reference",
        /// Reference copper missing under part of the route.
        ReturnGap = "return_gap",
        /// The return corridor reaches the reference's edge or a slot.
        ReferenceEdge = "reference_edge",
        /// The cross-section is outside the line model.
        UnsupportedCrossSection = "unsupported_cross_section",
        /// Another signal runs close and parallel.
        NearbySignal = "nearby_signal",
        /// No return via beside a layer transition.
        MissingReturnVia = "missing_return_via",
    }
}

emcraft_macros::named_serde!(Code);

impl Code {
    /// Whether it is about the reference copper under the route: none, a gap, an edge (the
    /// return path the illustrations must not invent).
    ///
    /// ```
    /// use emcraft_theory::pcb::Code;
    /// assert!(Code::ReturnGap.reference() && !Code::NearbySignal.reference());
    /// ```
    pub fn reference(self) -> bool {
        matches!(self, Code::ReturnGap | Code::ReferenceEdge | Code::NoReference)
    }
}

#[derive(Clone, Debug)]
pub struct Section {
    pub id: String,
    pub reference: String,
    pub length: f64,
    pub h: f64,
    pub impedance: f64,
    pub delay: f64,
    pub uncovered: Vec<[f64; 2]>,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub findings: Vec<Finding>,
    pub sections: Vec<Section>,
    pub path: Vec<usize>,
    pub model_available: bool,
    pub length: f64,
}

fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

/// The distance (m) from `p` to the segment `a`–`b` and where along it (0 at `a`, 1 at `b`) the
/// nearest point lies: for picking and the geometric screening.
///
/// ```
/// let (d, t) = emcraft_theory::pcb::near([0.5, 0.2], [0.0, 0.0], [1.0, 0.0]);
/// assert!((d - 0.2).abs() < 1e-15 && (t - 0.5).abs() < 1e-15);
/// // Beyond an end, the end is nearest.
/// assert_eq!(emcraft_theory::pcb::near([2.0, 0.0], [0.0, 0.0], [1.0, 0.0]), (1.0, 1.0));
/// ```
pub fn near(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> (f64, f64) {
    let d = sub(b, a);
    let q = sub(p, a);
    let t = ((q[0] * d[0] + q[1] * d[1]) / (d[0] * d[0] + d[1] * d[1]).max(1e-30)).clamp(0.0, 1.0);
    (distance(p, [a[0] + t * d[0], a[1] + t * d[1]]), t)
}

fn inside(p: [f64; 2], poly: &[[f64; 2]]) -> bool {
    let mut on = false;
    for (a, b) in poly.iter().zip(poly.iter().cycle().skip(1)).take(poly.len()) {
        if near(p, *a, *b).0 < 1e-12 {
            return true;
        }
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
            on = !on;
        }
    }
    on
}
// Split at every polygon intersection; midpoint membership is exact between intersections,
// unlike fixed sampling which can miss narrow slots. Collinear boundary endpoints count too.
fn gaps(a: [f64; 2], b: [f64; 2], polys: &[&[[f64; 2]]], budget: &mut usize) -> Result<Vec<[f64; 2]>, String> {
    let mut cuts = vec![0.0, 1.0];
    let d = sub(b, a);
    for poly in polys {
        for (c, e) in poly.iter().zip(poly.iter().cycle().skip(1)).take(poly.len()) {
            if *budget == 0 {
                return Err("return geometry work limit exceeded; select a smaller net".into());
            }
            *budget -= 1;
            let edge = sub(*e, *c);
            let den = cross(d, edge);
            if den.abs() > 1e-24 {
                let t = cross(sub(*c, a), edge) / den;
                let u = cross(sub(*c, a), d) / den;
                if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
                    cuts.push(t);
                }
            } else if cross(sub(*c, a), d).abs() < 1e-20 {
                for p in [*c, *e] {
                    let (r, t) = near(p, a, b);
                    if r < 1e-12 {
                        cuts.push(t);
                    }
                }
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    let mut out = Vec::new();
    for v in cuts.windows(2) {
        let t = (v[0] + v[1]) / 2.0;
        let p = [a[0] + t * d[0], a[1] + t * d[1]];
        let mut covered = false;
        for poly in polys {
            if *budget < poly.len() {
                return Err("return geometry work limit exceeded".into());
            }
            *budget -= poly.len();
            if inside(p, poly) {
                covered = true;
                break;
            }
        }
        if !covered {
            out.push([v[0], v[1]]);
        }
    }
    Ok(out)
}

fn key(layer: &str, p: [f64; 2]) -> (String, i64, i64) {
    (layer.into(), (p[0] / 1e-9).round() as i64, (p[1] / 1e-9).round() as i64)
}

fn path(segments: &[&Segment]) -> Option<Vec<usize>> {
    let mut graph: BTreeMap<_, Vec<usize>> = BTreeMap::new();
    for (i, s) in segments.iter().enumerate() {
        for p in [s.start, s.end] {
            graph.entry(key(&s.layer, p)).or_default().push(i);
        }
    }
    let ends: Vec<_> = graph.iter().filter(|(_, v)| v.len() == 1).map(|(k, _)| k.clone()).collect();
    if ends.len() != 2 || graph.values().any(|v| v.len() > 2) {
        return None;
    }
    // Non-adjacent same-layer copper crossings/overlaps create electrical shortcuts even
    // when their centreline endpoints alone form a simple chain.
    for (i, a) in segments.iter().enumerate() {
        for b in segments.iter().skip(i + 1) {
            if a.layer != b.layer || [a.start, a.end].iter().any(|p| [b.start, b.end].iter().any(|q| key(&a.layer, *p) == key(&b.layer, *q))) {
                continue;
            }
            let d = sub(a.end, a.start);
            let e = sub(b.end, b.start);
            let den = cross(d, e);
            let intersects = den.abs() > 1e-24
                && (0.0..=1.0).contains(&(cross(sub(b.start, a.start), e) / den))
                && (0.0..=1.0).contains(&(cross(sub(b.start, a.start), d) / den));
            let gap =
                [near(a.start, b.start, b.end).0, near(a.end, b.start, b.end).0, near(b.start, a.start, a.end).0, near(b.end, a.start, a.end).0]
                    .into_iter()
                    .fold(f64::INFINITY, f64::min);
            if intersects || gap < (a.width + b.width) / 2.0 {
                return None;
            }
        }
    }
    let mut at = ends[0].clone();
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for _ in 0..segments.len() {
        let i = *graph.get(&at)?.iter().find(|i| !seen.contains(*i))?;
        seen.insert(i);
        out.push(i);
        let s = segments[i];
        at = if key(&s.layer, s.start) == at { key(&s.layer, s.end) } else { key(&s.layer, s.start) };
    }
    (seen.len() == segments.len() && at == ends[1]).then_some(out)
}

/// Screens a routed net (lengths in m): its gaps in the reference copper, nearby signals, its
/// topology and vias. Its line model (and so [`s_parameters`]) is given only for an unbranched
/// outer microstrip on one layer over continuous ground.
///
/// ```
/// let empty = emcraft_model::pcb::Board { name: "empty".into(), layers: vec![], segments: vec![], vias: vec![], copper: vec![], notes: vec![], region: None, classes: vec![], masks: vec![] };
/// assert!(emcraft_theory::pcb::inspect(&empty, "CLK", "GND").is_err());
/// ```
pub fn inspect(board: &Board, net: &str, ground: &str) -> Result<Report, String> {
    board.check().map_err(|e| e.to_string())?;
    if net == ground {
        return Err("select a signal net different from the reference net".into());
    }
    let traces: Vec<_> = board.segments.iter().filter(|s| s.net == net).collect();
    if traces.is_empty() || traces.len() > 512 {
        return Err("net must contain 1..512 routed segments".into());
    }
    let mut report = Report {
        findings: vec![],
        sections: vec![],
        path: vec![],
        model_available: true,
        length: traces.iter().map(|s| distance(s.start, s.end)).sum(),
    };
    match path(&traces) {
        Some(p) => report.path = p,
        None => report.unmodelled(
            Code::RouteTopology,
            "Branched, disconnected or closed route: a two-terminal line model cannot represent it.".into(),
            None,
        ),
    }
    let signal_vias: Vec<_> = board.vias.iter().filter(|v| v.net == net).collect();
    if !signal_vias.is_empty() {
        report.unmodelled(Code::ViaTransition, "Signal vias require a via/transition model; no two-port prediction is made.".into(), None);
    }
    let mut budget = 1_000_000;
    for s in &traces {
        screen(board, s, ground, &mut report, &mut budget)?;
    }
    if report.sections.len() != traces.len() {
        report.model_available = false;
    }
    let refs: BTreeSet<_> = report.sections.iter().map(|s| s.reference.clone()).collect();
    if refs.len() >= 2 {
        for via in signal_vias {
            if !stitched(board, via.at, &refs, ground) {
                report.note(
                    Code::MissingReturnVia,
                    "No reference-net via within 2 mm spanning the selected reference layers. Verify the return transition.".into(),
                    None,
                );
            }
        }
    }
    Ok(report)
}

impl Report {
    fn note(&mut self, code: Code, detail: String, segment: Option<&Segment>) {
        self.findings.push(Finding { code, detail, segment: segment.map(|s| s.id.clone()) });
    }

    /// A finding that leaves the net without its line model.
    fn unmodelled(&mut self, code: Code, detail: String, segment: Option<&Segment>) {
        self.model_available = false;
        self.note(code, detail, segment);
    }
}

/// Segment `s` of the net screened against its `ground` reference: the nearest reference
/// layer, the gaps under and beside it, its cross-section and line estimate (a [`Section`]),
/// and a nearby signal. `budget` bounds the work shared by every segment.
fn screen(board: &Board, s: &Segment, ground: &str, report: &mut Report, budget: &mut usize) -> Result<(), String> {
    let Some(index) = board.layers.iter().position(|l| l.name == s.layer) else {
        return Err("unknown layer".into());
    };
    let reference = board
        .layers
        .iter()
        .enumerate()
        .filter(|(i, l)| *i != index && board.copper.iter().any(|c| c.net == ground && c.layer == l.name))
        .min_by(|(_, a), (_, b)| (a.z - board.layers[index].z).abs().total_cmp(&(b.z - board.layers[index].z).abs()));
    let Some((ri, plane)) = reference else {
        report.unmodelled(Code::NoReference, format!("No filled {ground} reference copper on another layer."), Some(s));
        return Ok(());
    };
    let h = (plane.z - board.layers[index].z).abs();
    let polys: Vec<&[[f64; 2]]> = board.copper.iter().filter(|c| c.net == ground && c.layer == plane.name).map(|c| c.polygon.as_slice()).collect();
    let uncovered = gaps(s.start, s.end, &polys, budget)?;
    if !uncovered.is_empty() {
        report.unmodelled(
            Code::ReturnGap,
            format!("The route crosses missing {ground} copper on {}. Restore a continuous reference or reroute.", plane.name),
            Some(s),
        );
    }
    let d = sub(s.end, s.start);
    let len = distance(s.start, s.end);
    let offset = [-d[1] / len * (h + s.width / 2.0), d[0] / len * (h + s.width / 2.0)];
    for sign in [-1.0, 1.0] {
        let a = [s.start[0] + sign * offset[0], s.start[1] + sign * offset[1]];
        let b = [s.end[0] + sign * offset[0], s.end[1] + sign * offset[1]];
        if !gaps(a, b, &polys, budget)?.is_empty() {
            report.unmodelled(
                Code::ReferenceEdge,
                "Reference copper is missing within h + W/2 of this route; infinite-plane impedance is not applicable.".into(),
                Some(s),
            );
            break;
        }
    }
    let dielectric = index.min(ri);
    let er = board.layers[dielectric].eps_below;
    let outer = index == 0 || index + 1 == board.layers.len();
    if !outer || index.abs_diff(ri) != 1 || !(0.01..=100.0).contains(&(s.width / h)) {
        report.unmodelled(
            Code::UnsupportedCrossSection,
            "Only adjacent-plane outer microstrip within 0.01 ≤ W/h ≤ 100 is supported by this model.".into(),
            Some(s),
        );
    }
    let ee = microstrip::eps_eff(s.width / h, er);
    let impedance = microstrip::z0(s.width / h, er);
    if !impedance.is_finite() || !ee.is_finite() {
        return Err("non-finite line estimate".into());
    }
    report.sections.push(Section {
        id: s.id.clone(),
        reference: plane.name.clone(),
        length: len,
        h,
        impedance,
        delay: len * ee.sqrt() / C0,
        uncovered,
    });
    if let Some((other, gap)) = nearby_signal(board, s, ground, h, budget)? {
        report.note(
            Code::NearbySignal,
            format!("{other} runs close and parallel (gap {:0.3e} m). This is geometric screening, not calculated crosstalk.", gap.max(0.0)),
            Some(s),
        );
    }
    Ok(())
}

/// The first other signal on `s`'s layer that runs parallel within 3h of it, and the gap
/// between their edges (m).
fn nearby_signal<'b>(board: &'b Board, s: &Segment, ground: &str, h: f64, budget: &mut usize) -> Result<Option<(&'b str, f64)>, String> {
    let d = sub(s.end, s.start);
    let len = distance(s.start, s.end);
    for other in board.segments.iter().filter(|o| o.net != s.net && o.net != ground && o.layer == s.layer) {
        if *budget == 0 {
            return Err("coupling screening work limit exceeded".into());
        }
        *budget -= 1;
        let gap = [
            near(other.start, s.start, s.end).0,
            near(other.end, s.start, s.end).0,
            near(s.start, other.start, other.end).0,
            near(s.end, other.start, other.end).0,
        ]
        .into_iter()
        .fold(f64::INFINITY, f64::min)
            - (s.width + other.width) / 2.0;
        let od = sub(other.end, other.start);
        let parallel = cross(d, od).abs() / (len * distance(other.start, other.end));
        if gap < 3.0 * h && parallel < 0.1 {
            return Ok(Some((other.net.as_str(), gap)));
        }
    }
    Ok(None)
}

/// Whether a `ground` via within 2 mm of `at` spans every reference layer of `refs`.
fn stitched(board: &Board, at: [f64; 2], refs: &BTreeSet<String>, ground: &str) -> bool {
    let layer = |name: &str| board.layers.iter().position(|l| l.name == name);
    board.vias.iter().any(|v| {
        v.net == ground
            && distance(v.at, at) <= 0.002
            && refs.iter().all(|r| match (layer(r), layer(&v.layers[0]), layer(&v.layers[1])) {
                (Some(i), Some(a), Some(b)) => i >= a.min(b) && i <= a.max(b),
                _ => false,
            })
    })
}

/// S11 and S21 at `frequency` (Hz) of the route's sections cascaded as lossless lines, referred
/// to `reference` ohms at both ends (Pozar §4.4). Bends, vias, pads, losses and coupling are
/// left out on purpose; a report with no model (not a simple outer microstrip) is refused.
///
/// ```
/// use emcraft_model::pcb::{Board, Copper, Layer, Segment};
/// // 30 mm of 0.3 mm trace over 0.18 mm of FR-4 with a whole ground plane below.
/// let ground = Copper { net: "GND".into(), layer: "B".into(), polygon: vec![[0.0, 0.0], [0.04, 0.0], [0.04, 0.02], [0.0, 0.02]] };
/// let trace = Segment { id: "0".into(), net: "S".into(), layer: "F".into(), start: [0.005, 0.01], end: [0.035, 0.01], width: 3e-4 };
/// let layers = vec![Layer { name: "F".into(), z: 0.0, eps_below: 4.2, loss_below: 0.0, copper: 0.0 }, Layer { name: "B".into(), z: 1.8e-4, eps_below: 1.0, loss_below: 0.0, copper: 0.0 }];
/// let board = Board { name: "line".into(), layers, segments: vec![trace], vias: vec![], copper: vec![ground], notes: vec![], region: None, classes: vec![], masks: vec![] };
/// let report = emcraft_theory::pcb::inspect(&board, "S", "GND").unwrap();
/// let (s11, s21) = emcraft_theory::pcb::s_parameters(&report, 1e9, 50.0).unwrap();
/// // A lossless line: no power lost.
/// assert!((s11.norm_sqr() + s21.norm_sqr() - 1.0).abs() < 1e-12);
/// assert!(emcraft_theory::pcb::s_parameters(&report, -1.0, 50.0).is_err());
/// ```
pub fn s_parameters(report: &Report, frequency: f64, reference: f64) -> Result<(C, C), String> {
    if !report.model_available
        || !frequency.is_finite()
        || frequency < 0.0
        || frequency > 1e12
        || !reference.is_finite()
        || !(0.1..=10000.0).contains(&reference)
    {
        return Err("transmission-line model unavailable or invalid frequency/impedance".into());
    }
    let mut matrix = coupled::line(reference, 0.0);
    for i in &report.path {
        let s = report.sections.get(*i).ok_or("incomplete route")?;
        let q = coupled::line(s.impedance, std::f64::consts::TAU * frequency * s.delay);
        matrix = coupled::product(&matrix, &q);
    }
    let out = coupled::s_params(&matrix, reference);
    if ![out.0.re, out.0.im, out.1.re, out.1.im].iter().all(|v| v.is_finite()) {
        return Err("non-finite cascade".into());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use emcraft_model::pcb::{Copper, Layer};
    fn board() -> Board {
        Board {
            name: "test".into(),
            layers: vec![
                Layer { name: "top".into(), z: 0.0, eps_below: 4.2, loss_below: 0.0, copper: 0.0 },
                Layer { name: "ground".into(), z: 0.00018, eps_below: 1.0, loss_below: 0.0, copper: 0.0 },
            ],
            segments: vec![Segment {
                id: "route".into(),
                net: "DATA".into(),
                layer: "top".into(),
                start: [0.005, 0.01],
                end: [0.045, 0.01],
                width: 0.0003,
            }],
            vias: vec![],
            copper: vec![Copper { net: "GND".into(), layer: "ground".into(), polygon: vec![[0.0, 0.0], [0.05, 0.0], [0.05, 0.02], [0.0, 0.02]] }],
            notes: vec![],
            region: None,
            classes: vec![],
            masks: vec![],
        }
    }

    #[test]
    fn narrow_ground_slot_is_found_and_disables_si() {
        let mut b = board();
        let valid = inspect(&b, "DATA", "GND").unwrap();
        assert!(valid.model_available);
        let mut left = b.copper[0].clone();
        left.polygon[1][0] = 0.020001;
        left.polygon[2][0] = 0.020001;
        let mut right = b.copper[0].clone();
        right.polygon[0][0] = 0.020002;
        right.polygon[3][0] = 0.020002;
        b.copper = vec![left, right];
        let r = inspect(&b, "DATA", "GND").unwrap();
        assert!(!r.model_available);
        assert!(r.findings.iter().any(|f| f.code == Code::ReturnGap));
        assert!(s_parameters(&r, 1e9, 50.0).is_err());
    }

    #[test]
    fn a_reference_edge_invalidates_the_infinite_plane_model_even_under_covered_centreline() {
        let mut b = board();
        b.copper[0].polygon[2][1] = 0.0102;
        b.copper[0].polygon[3][1] = 0.0102;
        let r = inspect(&b, "DATA", "GND").unwrap();
        assert!(!r.model_available);
        assert!(r.findings.iter().any(|f| f.code == Code::ReferenceEdge));
        assert!(!r.findings.iter().any(|f| f.code == Code::ReturnGap));
    }

    #[test]
    fn a_return_via_screen_checks_distance_and_layer_span_without_claiming_a_via_model() {
        let mut b = board();
        b.layers.extend([
            Layer { name: "g2".into(), z: 0.0014, eps_below: 4.2, loss_below: 0.0, copper: 0.0 },
            Layer { name: "bottom".into(), z: 0.00158, eps_below: 1.0, loss_below: 0.0, copper: 0.0 },
        ]);
        let mut plane = b.copper[0].clone();
        plane.layer = "g2".into();
        b.copper.push(plane);
        b.segments[0].end = [0.025, 0.01];
        let mut end = b.segments[0].clone();
        end.id = "last".into();
        end.layer = "bottom".into();
        end.start = [0.025, 0.01];
        end.end = [0.045, 0.01];
        b.segments.push(end);
        b.vias.push(emcraft_model::pcb::Via {
            net: "DATA".into(),
            at: [0.025, 0.01],
            layers: ["top".into(), "bottom".into()],
            diameter: 0.0006,
            drill: None,
        });
        let r = inspect(&b, "DATA", "GND").unwrap();
        assert!(!r.model_available);
        assert!(r.findings.iter().any(|f| f.code == Code::MissingReturnVia));
        let mut via = b.vias[0].clone();
        via.net = "GND".into();
        via.at[0] += 0.0005;
        b.vias.push(via);
        let r = inspect(&b, "DATA", "GND").unwrap();
        assert!(!r.model_available);
        assert!(r.findings.iter().any(|f| f.code == Code::ViaTransition));
        assert!(!r.findings.iter().any(|f| f.code == Code::MissingReturnVia));
        b.vias[1].layers[1] = "ground".into();
        assert!(inspect(&b, "DATA", "GND").unwrap().findings.iter().any(|f| f.code == Code::MissingReturnVia));
        b.vias[1].layers[1] = "bottom".into();
        b.vias[1].at[0] += 0.01;
        assert!(inspect(&b, "DATA", "GND").unwrap().findings.iter().any(|f| f.code == Code::MissingReturnVia));
    }

    #[test]
    fn matched_quarter_wave_has_independent_closed_form_and_no_reflection() {
        let r = Report {
            findings: vec![],
            sections: vec![Section {
                id: "line".into(),
                reference: "gnd".into(),
                length: 0.1,
                h: 0.001,
                impedance: 50.0,
                delay: 1e-9,
                uncovered: vec![],
            }],
            path: vec![0],
            model_available: true,
            length: 0.1,
        };
        let (s11, s21) = s_parameters(&r, 0.25e9, 50.0).unwrap();
        assert!(s11.norm() < 1e-14);
        assert!((s21 - C::new(0.0, -1.0)).norm() < 1e-14);
        approx::assert_abs_diff_eq!(s11.norm_sqr() + s21.norm_sqr(), 1.0, epsilon = 1e-14);
        assert!(s_parameters(&r, f64::NAN, 50.0).is_err());
    }

    #[test]
    fn topology_layer_and_reference_failures_do_not_get_plausible_si() {
        let mut b = board();
        let mut branch = b.segments[0].clone();
        branch.id = "branch".into();
        branch.end = [0.005, 0.015];
        b.segments.push(branch.clone());
        branch.id = "third".into();
        branch.end = [0.01, 0.015];
        b.segments.push(branch);
        assert!(!inspect(&b, "DATA", "GND").unwrap().model_available);
        b.segments.pop();
        b.segments.pop();
        b.copper.clear();
        let r = inspect(&b, "DATA", "GND").unwrap();
        assert!(!r.model_available);
        assert!(r.findings.iter().any(|f| f.code == Code::NoReference));
        assert!(inspect(&b, "missing", "GND").is_err());
        assert!(inspect(&b, "DATA", "DATA").is_err());
        b.segments[0].width = f64::NAN;
        assert!(inspect(&b, "DATA", "GND").is_err());
    }

    #[test]
    fn a_route_crossing_itself_is_not_a_serial_line() {
        let mut b = board();
        let points = [[0.005, 0.005], [0.04, 0.015], [0.04, 0.005], [0.005, 0.015]];
        b.segments = points
            .windows(2)
            .enumerate()
            .map(|(i, p)| Segment { id: format!("track-{i}"), net: "DATA".into(), layer: "top".into(), start: p[0], end: p[1], width: 0.0003 })
            .collect();
        let r = inspect(&b, "DATA", "GND").unwrap();
        assert!(r.findings.iter().any(|f| f.code == Code::RouteTopology));
        assert!(s_parameters(&r, 1e9, 50.0).is_err());
    }

    #[test]
    fn serial_width_steps_do_not_depend_on_input_order() {
        let mut b = board();
        b.segments[0].end = [0.025, 0.01];
        let mut last = b.segments[0].clone();
        last.id = "last".into();
        last.start = [0.025, 0.01];
        last.end = [0.045, 0.01];
        last.width = 0.0006;
        b.segments.push(last);
        let a = inspect(&b, "DATA", "GND").unwrap();
        let waves = s_parameters(&a, 1e9, 50.0).unwrap();
        b.segments.reverse();
        assert_eq!(s_parameters(&inspect(&b, "DATA", "GND").unwrap(), 1e9, 50.0).unwrap(), waves);
    }
}

/// A geometrically parallel overlapping pair. Points are SI metres; this is not a coupling coefficient.
#[derive(Clone, Debug)]
pub struct Pair {
    pub source: String,
    pub target: String,
    pub source_segment: String,
    pub target_segment: String,
    pub layer: String,
    pub a: [[f64; 2]; 2],
    pub b: [[f64; 2]; 2],
    pub gap: f64,
    pub length: f64,
}

/// Find same-layer parallel overlaps, including wide spacing for before/after illustration.
/// ```
/// let b=emcraft_model::pcb::Board{name:"empty".into(),layers:vec![],segments:vec![],vias:vec![],copper:vec![],notes:vec![],region:None,classes:vec![],masks:vec![]};
/// assert!(emcraft_theory::pcb::pairs(&b,"DATA","GND").is_err());
/// ```
pub fn pairs(board: &Board, source: &str, ground: &str) -> Result<Vec<Pair>, String> {
    board.check().map_err(|e| e.to_string())?;
    let selected: Vec<_> = board.segments.iter().filter(|s| s.net == source).collect();
    if selected.is_empty() || selected.len() > 512 {
        return Err("net must contain 1..512 routed segments".into());
    }
    let mut out = Vec::new();
    let mut work = 0usize;
    for s in selected {
        let d = sub(s.end, s.start);
        let len = distance(s.start, s.end);
        let u = [d[0] / len, d[1] / len];
        for t in board.segments.iter().filter(|t| t.net != source && t.net != ground && t.layer == s.layer) {
            work += 1;
            if work > 1_000_000 {
                return Err("pair screening work limit exceeded".into());
            }
            let e = sub(t.end, t.start);
            let el = distance(t.start, t.end);
            // Only genuinely parallel straight sections get a schematic; angled/transition cases are withheld.
            if cross(d, e).abs() / (len * el) > 1e-6 {
                continue;
            }
            let dot = |p: [f64; 2]| {
                let q = sub(p, s.start);
                q[0] * u[0] + q[1] * u[1]
            };
            let x = dot(t.start);
            let y = dot(t.end);
            let lo = x.min(y).max(0.);
            let hi = x.max(y).min(len);
            if hi - lo < 1e-9 {
                continue;
            }
            let a = |z: f64| [s.start[0] + z * u[0], s.start[1] + z * u[1]];
            let b = |z: f64| {
                let f = (z - x) / (y - x);
                [t.start[0] + f * e[0], t.start[1] + f * e[1]]
            };
            let gap = distance(a(lo), b(lo)) - (s.width + t.width) / 2.;
            if gap <= 0. {
                continue;
            }
            if out.len() >= 256 {
                return Err("too many parallel pairs; select a smaller net".into());
            }
            out.push(Pair {
                source: s.net.clone(),
                target: t.net.clone(),
                source_segment: s.id.clone(),
                target_segment: t.id.clone(),
                layer: s.layer.clone(),
                a: [a(lo), a(hi)],
                b: [b(lo), b(hi)],
                gap,
                length: hi - lo,
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod pair_tests {
    use super::*;
    use emcraft_model::pcb::{Layer, Segment};
    #[test]
    fn overlaps_are_oriented_and_nonparallel_or_different_layers_are_withheld() {
        let mut b = Board {
            name: "pair".into(),
            layers: vec![
                Layer { name: "top".into(), z: 0., eps_below: 4., loss_below: 0.0, copper: 0.0 },
                Layer { name: "bottom".into(), z: 0.001, eps_below: 1., loss_below: 0.0, copper: 0.0 },
            ],
            segments: vec![
                Segment { id: "a".into(), net: "A".into(), layer: "top".into(), start: [0., 0.], end: [0.04, 0.], width: 0.0003 },
                Segment { id: "b".into(), net: "B".into(), layer: "top".into(), start: [0.03, 0.001], end: [0.01, 0.001], width: 0.0003 },
            ],
            vias: vec![],
            copper: vec![],
            notes: vec![],
            region: None,
            classes: vec![],
            masks: vec![],
        };
        let p = pairs(&b, "A", "GND").unwrap();
        assert_eq!(p.len(), 1);
        approx::assert_abs_diff_eq!(p[0].length, 0.02, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(p[0].gap, 0.0007, epsilon = 1e-12);
        assert_eq!(p[0].a[0], [0.01, 0.]);
        b.segments[1].layer = "bottom".into();
        assert!(pairs(&b, "A", "GND").unwrap().is_empty());
        b.segments[1].layer = "top".into();
        b.segments[1].end[1] = 0.003;
        assert!(pairs(&b, "A", "GND").unwrap().is_empty());
    }
}

/// The screening's geometry against plain geometry worked out by hand, and its line against
/// the textbook's microstrip.
#[cfg(test)]
mod geometry_tests {
    use super::*;
    use emcraft_model::pcb::{Copper, Layer};

    const MM: f64 = 1e-3;

    fn seg(id: &str, net: &str, layer: &str, start: [f64; 2], end: [f64; 2], width: f64) -> Segment {
        Segment { id: id.into(), net: net.into(), layer: layer.into(), start, end, width }
    }

    fn plane(net: &str, layer: &str, x: [f64; 2], y: [f64; 2]) -> Copper {
        Copper { net: net.into(), layer: layer.into(), polygon: vec![[x[0], y[0]], [x[1], y[0]], [x[1], y[1]], [x[0], y[1]]] }
    }

    /// Four layers 0.18, 1.22 and 0.18 mm apart, each dielectric its own permittivity; ground
    /// 60 × 40 mm on the layers named.
    fn board(segments: Vec<Segment>, grounds: &[&str]) -> Board {
        let layer = |name: &str, z: f64, eps_below: f64| Layer { name: name.into(), z, eps_below, loss_below: 0.0, copper: 0.0 };
        Board {
            name: "test".into(),
            layers: vec![layer("top", 0.0, 4.2), layer("g1", 0.18 * MM, 3.0), layer("g2", 1.4 * MM, 2.2), layer("bottom", 1.58 * MM, 1.0)],
            segments,
            vias: vec![],
            copper: grounds.iter().map(|l| plane("GND", l, [0.0, 0.06], [0.0, 0.04])).collect(),
            notes: vec![],
            region: None,
            classes: vec![],
            masks: vec![],
        }
    }

    fn codes(r: &Report) -> Vec<Code> {
        r.findings.iter().map(|f| f.code).collect()
    }

    /// A 3-4-5 segment from (1, 1) to (4, 5): a point 2 along it and 1.5 to its left is 1.5
    /// away at 2/5; one behind its start is nearest the start.
    #[test]
    fn the_nearest_point_of_a_slanted_segment() {
        let (a, b) = ([1.0, 1.0], [4.0, 5.0]);
        let (d, t) = near([1.0, 3.5], a, b);
        assert!((d - 1.5).abs() < 1e-14 && (t - 0.4).abs() < 1e-14, "{d} {t}");
        let (d, t) = near([0.4, 0.2], a, b);
        assert!((d - 1.0).abs() < 1e-14 && t == 0.0, "{d} {t}");
        let (d, t) = near([7.0, 9.0], a, b);
        assert!((d - 5.0).abs() < 1e-14 && t == 1.0, "{d} {t}");
        // A segment of no length: the distance to its point.
        assert_eq!(near([4.0, 5.0], a, a), (5.0, 0.0));
    }

    /// An L of unit squares, a diamond met at its corners' heights and a slanted triangle.
    #[test]
    fn points_in_and_out_of_polygons() {
        let l = [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]];
        for p in [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.0, 1.0], [2.0, 0.5], [0.3, 2.0], [0.0, 0.0]] {
            assert!(inside(p, &l), "{p:?}");
        }
        for p in [[1.5, 1.5], [2.5, 0.5], [-0.5, 0.5], [0.5, 2.5], [0.5, -0.5], [1.001, 1.001], [3.0, 1.0], [-1.0, 1.0], [-1.0, 2.0]] {
            assert!(!inside(p, &l), "{p:?}");
        }
        let diamond = [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]];
        assert!(inside([-0.5, 0.0], &diamond) && inside([0.5, 0.0], &diamond) && inside([0.0, 0.5], &diamond));
        assert!(!inside([-2.0, 0.0], &diamond) && !inside([2.0, 0.0], &diamond) && !inside([-0.6, 0.6], &diamond) && !inside([0.6, -0.6], &diamond));
        let triangle = [[1.0, 1.0], [5.0, 2.0], [2.0, 6.0]];
        assert!(inside([3.0, 3.0], &triangle) && inside([2.0, 4.0], &triangle));
        assert!(!inside([3.0, 1.2], &triangle) && !inside([4.5, 4.0], &triangle) && !inside([1.2, 3.0], &triangle) && !inside([6.0, 3.0], &triangle));
    }

    /// A line from x = −1 to 4 over the unit square and the square from 2 to 3: uncovered on
    /// its fifths 0–1, 2–3 and 4–5. The work is counted: an edge each, then each polygon's
    /// edges for each stretch until one covers it (8 + 8 + 4 + 8 + 8 + 8).
    #[test]
    fn the_stretches_no_polygon_covers() {
        let squares = [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], [[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]]];
        let polys: Vec<&[[f64; 2]]> = squares.iter().map(|p| &p[..]).collect();
        let mut budget = 44;
        let open = gaps([-1.0, 0.5], [4.0, 0.5], &polys, &mut budget).unwrap();
        assert_eq!(budget, 0);
        assert_eq!(open.len(), 3);
        for (got, want) in open.iter().zip([[0.0, 0.2], [0.4, 0.6], [0.8, 1.0]]) {
            assert!((got[0] - want[0]).abs() < 1e-12 && (got[1] - want[1]).abs() < 1e-12, "{got:?}");
        }
        for short in [43, 40, 8, 7, 0] {
            assert!(gaps([-1.0, 0.5], [4.0, 0.5], &polys, &mut { short }).is_err(), "{short}");
        }
        let run = |a: [f64; 2], b: [f64; 2]| gaps(a, b, &polys, &mut 1000).unwrap();
        // Within one square, along its edge and from corner to corner: covered.
        assert!(run([0.1, 0.2], [0.9, 0.7]).is_empty() && run([0.0, 0.0], [1.0, 0.0]).is_empty() && run([0.0, 0.0], [1.0, 1.0]).is_empty());
        // Along the edges' line across the gap between the squares: thirds of 0.5 to 2.5 over
        // the gap from 1 to 2.
        let open = run([0.5, 1.0], [2.5, 1.0]);
        assert!(open.len() == 1 && (open[0][0] - 0.25).abs() < 1e-12 && (open[0][1] - 0.75).abs() < 1e-12, "{open:?}");
        // Slanted, from the first square's middle to the second's: out between x = 1 and 2.
        let open = run([0.5, 0.25], [2.5, 0.75]);
        assert!(open.len() == 1 && (open[0][0] - 0.25).abs() < 1e-12 && (open[0][1] - 0.75).abs() < 1e-12, "{open:?}");
        // Off every polygon: all of it.
        assert_eq!(run([0.0, 2.0], [3.0, 2.0]), [[0.0, 1.0]]);
    }

    /// A route is a line when its segments make one chain whose copper meets nowhere else.
    #[test]
    fn routes_that_are_one_line_and_routes_that_are_not() {
        let w = 0.3 * MM;
        let chain = |points: &[[f64; 2]], widths: &[f64]| -> Vec<Segment> {
            points.windows(2).enumerate().map(|(i, p)| seg(&format!("s{i}"), "DATA", "top", p[0], p[1], widths[i % widths.len()])).collect()
        };
        let look = |segments: Vec<Segment>| inspect(&board(segments, &["g1"]), "DATA", "GND").unwrap();
        // Up, across and down, given out of order: 15 + 25 + 12 mm, walked from an end.
        let mut bent = chain(&[[0.005, 0.005], [0.005, 0.02], [0.03, 0.02], [0.03, 0.008]], &[w]);
        bent.rotate_left(1);
        let r = look(bent);
        assert!(r.model_available && codes(&r).is_empty(), "{:?}", r.findings);
        assert!(r.path == [2, 0, 1] || r.path == [1, 0, 2], "{:?}", r.path);
        approx::assert_abs_diff_eq!(r.length, 0.052, epsilon = 1e-15);
        // Two legs 0.2 and 0.6 mm wide that are not joined to each other touch when their
        // centres are less than 0.4 mm apart.
        let hairpin =
            |apart: f64| chain(&[[0.005, 0.01], [0.03, 0.01], [0.03, 0.01 + apart], [0.005, 0.01 + apart]], &[0.2 * MM, 0.1 * MM, 0.6 * MM]);
        assert!(!codes(&look(hairpin(0.41 * MM))).contains(&Code::RouteTopology));
        assert!(codes(&look(hairpin(0.39 * MM))).contains(&Code::RouteTopology));
        // Legs that would cross if they went on, but stop short: a line. Going on, they cross.
        let open = [[0.005, 0.005], [0.03, 0.005], [0.03, 0.012], [0.02, 0.008]];
        assert!(look(chain(&open, &[w])).model_available);
        let crossing = [[0.005, 0.005], [0.03, 0.005], [0.03, 0.012], [0.015, 0.003]];
        assert!(codes(&look(chain(&crossing, &[w]))).contains(&Code::RouteTopology));
        // A closed triangle has no ends; a ring with two tails has two, and branches.
        let closed = look(chain(&[[0.005, 0.005], [0.03, 0.005], [0.02, 0.02], [0.005, 0.005]], &[w]));
        assert!(codes(&closed).contains(&Code::RouteTopology) && !closed.model_available);
        let mut ring = chain(&[[0.005, 0.01], [0.015, 0.01], [0.025, 0.02], [0.035, 0.01], [0.045, 0.01]], &[w]);
        ring.extend([seg("r0", "DATA", "top", [0.015, 0.01], [0.025, 0.0], w), seg("r1", "DATA", "top", [0.025, 0.0], [0.035, 0.01], w)]);
        assert!(codes(&look(ring)).contains(&Code::RouteTopology));
        // 512 segments are looked at, 513 refused.
        let many = |n: usize| chain(&(0..=n).map(|i| [0.005 + i as f64 * 5e-5, 0.01]).collect::<Vec<_>>(), &[w]);
        assert!(inspect(&board(many(512), &["g1"]), "DATA", "GND").is_ok());
        assert!(inspect(&board(many(513), &["g1"]), "DATA", "GND").is_err());
    }

    /// A microstrip's impedance and effective permittivity by the textbook's closed form
    /// (Pozar, Microwave Engineering, 4th ed., eqs. 3.195–3.196), within a few per cent of
    /// Hammerstad and Jensen's.
    fn pozar(u: f64, er: f64) -> (f64, f64) {
        let ee = (er + 1.0) / 2.0 + (er - 1.0) / 2.0 / (1.0 + 12.0 / u).sqrt();
        let z = if u <= 1.0 {
            60.0 / ee.sqrt() * (8.0 / u + u / 4.0).ln()
        } else {
            120.0 * std::f64::consts::PI / (ee.sqrt() * (u + 1.393 + 0.667 * (u + 1.444).ln()))
        };
        (z, ee)
    }

    /// The reference is the nearest layer with ground copper, the line the microstrip of the
    /// dielectric between the two; what is not an outer microstrip over the next layer has no
    /// model.
    #[test]
    fn the_reference_is_the_nearest_ground_and_the_line_its_microstrip() {
        let w = 0.3 * MM;
        let route = |layer: &str, width: f64| vec![seg("route", "DATA", layer, [0.005, 0.01], [0.045, 0.01], width)];
        let line = |layer: &str, grounds: &[&str], width: f64| inspect(&board(route(layer, width), grounds), "DATA", "GND").unwrap();
        for (layer, grounds, reference, er) in
            [("top", ["g1", "g2"], "g1", 4.2), ("bottom", ["g1", "g2"], "g2", 2.2), ("bottom", ["g2", "top"], "g2", 2.2)]
        {
            let r = line(layer, &grounds, w);
            assert!(r.model_available && codes(&r).is_empty(), "{layer}: {:?}", r.findings);
            let s = &r.sections[0];
            let (z, ee) = pozar(w / (0.18 * MM), er);
            assert_eq!(s.reference, reference);
            approx::assert_abs_diff_eq!(s.h, 0.18 * MM, epsilon = 1e-15);
            approx::assert_abs_diff_eq!(s.length, 0.04, epsilon = 1e-15);
            assert!((s.impedance / z - 1.0).abs() < 0.03, "{layer}: {} against {z}", s.impedance);
            assert!((s.delay / (0.04 * ee.sqrt() / C0) - 1.0).abs() < 0.03, "{layer}: {}", s.delay);
            assert!(s.uncovered.is_empty());
        }
        // Ground two layers down: found, 1.4 mm away, but no microstrip of this model.
        let far = line("top", &["g2"], w);
        assert_eq!(codes(&far), [Code::UnsupportedCrossSection]);
        assert!(far.sections[0].reference == "g2" && (far.sections[0].h - 1.4 * MM).abs() < 1e-15 && !far.model_available);
        // A trace on an inner layer, and strips far too narrow and too wide.
        assert_eq!(codes(&line("g1", &["g2"], w)), [Code::UnsupportedCrossSection]);
        assert_eq!(codes(&line("g2", &["g1"], w)), [Code::UnsupportedCrossSection]);
        assert_eq!(codes(&line("top", &["g1"], 0.005 * 0.18 * MM)), [Code::UnsupportedCrossSection]);
        assert!(codes(&line("top", &["g1"], 200.0 * 0.18 * MM)).contains(&Code::UnsupportedCrossSection));
        // No ground but on the trace's own layer: no reference.
        assert_eq!(codes(&line("top", &["top"], w)), [Code::NoReference]);
    }

    /// A slot from x = 20 to 21 mm in the nearest ground is a gap from 3/8 to 2/5 of a route
    /// from 5 to 45 mm, whatever lies over it: ground on a farther layer, another net's copper
    /// on the same one.
    #[test]
    fn a_slot_is_not_covered_by_other_copper() {
        let mut b = board(vec![seg("route", "DATA", "top", [0.005, 0.01], [0.045, 0.01], 0.3 * MM)], &["g2"]);
        b.copper.extend([
            plane("GND", "g1", [0.0, 0.02], [0.0, 0.04]),
            plane("GND", "g1", [0.021, 0.06], [0.0, 0.04]),
            plane("PWR", "g1", [0.019, 0.022], [0.0, 0.04]),
            plane("GND", "bottom", [0.0, 0.06], [0.0, 0.04]),
        ]);
        let r = inspect(&b, "DATA", "GND").unwrap();
        assert!(codes(&r).contains(&Code::ReturnGap) && !r.model_available);
        let open = &r.sections[0].uncovered;
        assert!(open.len() == 1 && (open[0][0] - 0.375).abs() < 1e-12 && (open[0][1] - 0.4).abs() < 1e-12, "{open:?}");
        assert_eq!(r.sections[0].reference, "g1");
    }

    /// The return current's corridor is h + W/2 to either side of the trace's centre (here
    /// 0.18 + 0.25 mm): ground ending 2 % nearer on one side is an edge, 2 % farther is not,
    /// for a route along x, along y and at 37° and its reverse.
    #[test]
    fn the_return_corridor_is_h_and_half_the_width_to_either_side() {
        let reach = 0.18 * MM + 0.25 * MM;
        for degrees in [0.0_f64, 90.0, 37.0, 217.0] {
            let (u, n) = ([degrees.to_radians().cos(), degrees.to_radians().sin()], [-degrees.to_radians().sin(), degrees.to_radians().cos()]);
            let at = |along: f64, across: f64| [0.05 + along * u[0] + across * n[0], 0.05 + along * u[1] + across * n[1]];
            let look = |left: f64, right: f64| {
                let mut b = board(vec![seg("route", "DATA", "top", at(0.0, 0.0), at(0.02, 0.0), 0.5 * MM)], &[]);
                b.copper = vec![Copper {
                    net: "GND".into(),
                    layer: "g1".into(),
                    polygon: vec![at(-0.005, -right), at(0.025, -right), at(0.025, left), at(-0.005, left)],
                }];
                codes(&inspect(&b, "DATA", "GND").unwrap())
            };
            assert!(look(1.02 * reach, 1.02 * reach).is_empty(), "{degrees}");
            assert_eq!(look(0.98 * reach, 5.0 * reach), [Code::ReferenceEdge], "{degrees}");
            assert_eq!(look(5.0 * reach, 0.98 * reach), [Code::ReferenceEdge], "{degrees}");
        }
    }

    /// Another signal on the layer within 3h (0.54 mm) of the trace's edge and within about
    /// 6° of its direction is named with the gap between the edges; the same net, the ground,
    /// another layer, a crossing trace and one farther off are not.
    #[test]
    fn a_close_parallel_signal_is_named_with_its_gap() {
        let victim = seg("route", "DATA", "top", [0.005, 0.01], [0.045, 0.01], 0.3 * MM);
        let look = |other: Segment| {
            let r = inspect(&board(vec![other, victim.clone()], &["g1"]), "DATA", "GND").unwrap();
            r.findings.into_iter().find(|f| f.code == Code::NearbySignal).map(|f| f.detail)
        };
        // Centres 0.9 mm apart, the traces 0.3 and 0.5 mm wide: edges 0.5 mm apart.
        let beside = |net: &str, layer: &str, apart: f64| seg("other", net, layer, [0.01, 0.01 + apart], [0.03, 0.01 + apart], 0.5 * MM);
        let detail = look(beside("CLK", "top", 0.9 * MM)).unwrap();
        assert!(detail.starts_with("CLK ") && detail.contains("gap 5.000e-4 m"), "{detail}");
        assert!(look(beside("CLK", "top", -0.9 * MM)).is_some());
        // It leaves the line its model.
        assert!(inspect(&board(vec![beside("CLK", "top", 0.9 * MM), victim.clone()], &["g1"]), "DATA", "GND").unwrap().model_available);
        assert_eq!(look(beside("CLK", "top", 0.96 * MM)), None);
        assert_eq!(look(beside("GND", "top", 0.9 * MM)), None);
        assert_eq!(look(beside("DATA", "top", 0.9 * MM)), None);
        assert_eq!(look(beside("CLK", "bottom", 0.9 * MM)), None);
        // Overlapping copper: no gap left.
        assert!(look(beside("CLK", "top", 0.3 * MM)).unwrap().contains("gap 0.000e0 m"));
        // Turned about a point 0.95 mm off: by 3° still alongside, by 10° and 90° not.
        let turned = |degrees: f64| {
            let (c, s) = (degrees.to_radians().cos(), degrees.to_radians().sin());
            seg("other", "CLK", "top", [0.02 - 0.002 * c, 0.01095 - 0.002 * s], [0.02 + 0.002 * c, 0.01095 + 0.002 * s], 0.5 * MM)
        };
        assert!(look(turned(3.0)).is_some() && look(turned(183.0)).is_some());
        assert_eq!(look(turned(10.0)), None);
        assert_eq!(look(seg("other", "CLK", "top", [0.02, 0.0106], [0.02, 0.015], 0.5 * MM)), None);
        // The work is counted, one for each other signal's segment on the layer.
        let b = board(vec![beside("CLK", "top", 0.9 * MM), beside("GND", "top", 0.9 * MM), victim.clone()], &["g1"]);
        let mut budget = 1;
        assert_eq!(nearby_signal(&b, &victim, "GND", 0.18 * MM, &mut budget).unwrap().map(|(net, _)| net), Some("CLK"));
        assert_eq!(budget, 0);
        assert!(nearby_signal(&b, &victim, "GND", 0.18 * MM, &mut budget).is_err());
        assert_eq!(nearby_signal(&b, &victim, "CLK", 0.18 * MM, &mut 5).unwrap().map(|(net, _)| net), Some("GND"));
    }

    /// A quarter wave of 100 Ω between 50 Ω ends shows 100²/50 = 200 Ω: S11 = 0.6 and
    /// S21 = −0.8j (Pozar §2.5, §4.4). At 0 Hz the line is not there. Numbers out of range
    /// are refused.
    #[test]
    fn a_quarter_wave_transformer_and_refused_numbers() {
        let section = Section { id: "line".into(), reference: "gnd".into(), length: 0.1, h: 0.001, impedance: 100.0, delay: 1e-9, uncovered: vec![] };
        let r = Report { findings: vec![], sections: vec![section], path: vec![0], model_available: true, length: 0.1 };
        let (s11, s21) = s_parameters(&r, 0.25e9, 50.0).unwrap();
        assert!((s11 - C::new(0.6, 0.0)).norm() < 1e-14 && (s21 - C::new(0.0, -0.8)).norm() < 1e-14, "{s11} {s21}");
        let (s11, s21) = s_parameters(&r, 0.0, 50.0).unwrap();
        assert!(s11.norm() < 1e-15 && (s21 - 1.0).norm() < 1e-15);
        assert!(s_parameters(&r, 1e12, 0.1).is_ok() && s_parameters(&r, 1e9, 10000.0).is_ok());
        for (f, z) in
            [(1.001e12, 50.0), (-1.0, 50.0), (f64::INFINITY, 50.0), (1e9, 0.09), (1e9, 10001.0), (1e9, f64::NAN), (1e9, f64::INFINITY), (1e9, -50.0)]
        {
            assert!(s_parameters(&r, f, z).is_err(), "{f} {z}");
        }
        assert!(s_parameters(&Report { path: vec![1], ..r.clone() }, 1e9, 50.0).is_err());
        assert!(s_parameters(&Report { model_available: false, ..r }, 1e9, 50.0).is_err());
    }

    /// A pair along a 3-4-5 direction: the target 1 mm to the left of a 50 mm source, drawn
    /// backwards from 30 to 10 mm along it. The stretch they share is 20 mm, the edges
    /// 1 − (0.3 + 0.5)/2 mm apart; a target reaching beyond an end is cut at the end.
    #[test]
    fn a_slanted_pair_is_cut_to_what_the_two_share() {
        let (u, n) = ([0.6, 0.8], [-0.8, 0.6]);
        let at = |along: f64, across: f64| [0.01 + along * u[0] + across * n[0], 0.005 + along * u[1] + across * n[1]];
        let source = seg("s", "A", "top", at(0.0, 0.0), at(0.05, 0.0), 0.3 * MM);
        let look = |targets: Vec<Segment>| {
            let mut segments = vec![source.clone()];
            segments.extend(targets);
            pairs(&board(segments, &["g1"]), "A", "GND")
        };
        let target = |from: f64, to: f64, across: f64| seg("t", "B", "top", at(from, across), at(to, across), 0.5 * MM);
        let close = |p: [f64; 2], q: [f64; 2]| distance(p, q) < 1e-12;
        let p = look(vec![target(0.03, 0.01, MM)]).unwrap();
        assert_eq!(p.len(), 1);
        approx::assert_abs_diff_eq!(p[0].length, 0.02, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(p[0].gap, 0.6 * MM, epsilon = 1e-12);
        assert!(close(p[0].a[0], [0.016, 0.013]) && close(p[0].a[1], [0.028, 0.029]), "{:?}", p[0].a);
        assert!(close(p[0].b[0], [0.0152, 0.0136]) && close(p[0].b[1], [0.0272, 0.0296]), "{:?}", p[0].b);
        assert!(p[0].source == "A" && p[0].target == "B" && p[0].source_segment == "s" && p[0].target_segment == "t" && p[0].layer == "top");
        // Beyond the source's end and before its start.
        let p = look(vec![target(0.04, 0.07, -MM)]).unwrap();
        approx::assert_abs_diff_eq!(p[0].length, 0.01, epsilon = 1e-12);
        assert!(close(p[0].a[1], at(0.05, 0.0)) && close(p[0].b[0], at(0.04, -MM)) && close(p[0].b[1], at(0.05, -MM)));
        let p = look(vec![target(-0.02, 0.01, MM)]).unwrap();
        assert!(close(p[0].a[0], at(0.0, 0.0)) && close(p[0].b[0], at(0.0, MM)) && (p[0].length - 0.01).abs() < 1e-12);
        // Nothing shared: past the end, copper on copper, the ground, and a target a hundred
        // thousandth of a radian off the source's direction.
        assert!(look(vec![target(0.05, 0.06, MM)]).unwrap().is_empty());
        assert!(look(vec![target(0.01, 0.03, 0.3 * MM)]).unwrap().is_empty());
        assert!(look(vec![Segment { net: "GND".into(), ..target(0.01, 0.03, MM) }]).unwrap().is_empty());
        assert!(look(vec![seg("t", "B", "top", at(0.01, MM), at(0.03, MM + 2e-7), 0.5 * MM)]).unwrap().is_empty());
        // 256 pairs are listed, 257 refused.
        let stack = |count: usize| (0..count).map(|i| Segment { id: format!("t{i}"), ..target(0.01, 0.03, (i + 1) as f64 * MM) }).collect::<Vec<_>>();
        assert_eq!(look(stack(256)).unwrap().len(), 256);
        assert!(look(stack(257)).is_err());
    }

    /// The pairs' work is bounded: 500 source segments against 2000 others is a million
    /// looks, one more than that is refused; so are 513 segments of the net.
    #[test]
    fn the_pairs_work_is_bounded() {
        let sources = |count: usize| (0..count).map(|i| seg(&format!("s{i}"), "A", "top", [i as f64 * 1e-5, 0.0], [i as f64 * 1e-5, 0.01], 0.3 * MM));
        let others = |count: usize| {
            (0..count).map(|i| seg(&format!("t{i}"), "B", "top", [0.0, 0.02 + i as f64 * 1e-6], [0.01, 0.02 + i as f64 * 1e-6], 0.3 * MM))
        };
        let look = |s: usize, t: usize| pairs(&board(sources(s).chain(others(t)).collect(), &["g1"]), "A", "GND");
        assert!(look(500, 2000).unwrap().is_empty());
        assert!(look(500, 2001).is_err());
        assert!(look(512, 1).is_ok() && look(513, 1).is_err());
    }
}
