//! Copper zones filled where a board's file keeps their outlines only (PCBCraft's: `pcb.import`).
//!
//! Each pour, the highest priority first (ties in the file's order), fills its outline on its
//! layer less every other net's copper there grown by its clearance — traces, vias' pads, pads and
//! the pours filled before it — by [`emcraft_mesh::planar::fill`]. Its own net's traces and vias
//! join it solid; its own pads by its connection: solid, a thermal relief (a gap round the pad
//! crossed by four spokes along the pad's sides) or not at all (the clearance round them). An
//! island with none of the net's copper on it is kept unless smaller than the pour's minimum
//! island. Each island becomes filled copper of the net, as a board's saved fill is read.

use emcraft_mesh::planar::fill::{self, fill};
use emcraft_mesh::planar::offset;
use emcraft_model::pcb::{Board, Connection, Copper, Pour};

/// The sagitta of the chords round copper is traced and grown with (m).
const SAG: f64 = 1e-6;

/// Fills `pours` into `board`'s copper, noting each pour it could not fill (why) and how many
/// islands the rest made.
pub(crate) fn pour(board: &mut Board, pours: &[Pour]) {
    let mut order: Vec<&Pour> = pours.iter().collect();
    order.sort_by_key(|p| std::cmp::Reverse(p.priority));
    let pads = board.copper.clone();
    let mut islands = 0usize;
    for p in order {
        match islands_of(board, &pads, p) {
            Ok(filled) => {
                islands += filled.len();
                board.copper.extend(filled.into_iter().map(|polygon| Copper { net: p.net.clone(), layer: p.layer.clone(), polygon }));
            }
            Err(why) => note(board, format!("A copper zone of {} on {} could not be filled: {why}", p.net, p.layer)),
        }
    }
    if !pours.is_empty() {
        note(
            board,
            format!(
                "{} copper zone(s) filled by EMCraft into {islands} island(s) (the file keeps their outlines only): each zone's outline less other nets' copper grown by its clearance, the highest priority first; its own pads by its connection (thermal reliefs, solid or none).",
                pours.len()
            ),
        );
    }
}

fn note(board: &mut Board, text: String) {
    if board.notes.len() < 128 && !board.notes.contains(&text) {
        board.notes.push(text);
    }
}

/// The islands pour `p` fills on `board` (`pads`: the board's copper before any pour was filled).
fn islands_of(board: &Board, pads: &[Copper], p: &Pour) -> Result<Vec<Vec<[f64; 2]>>, String> {
    let index = |name: &str| board.layers.iter().position(|l| l.name == name);
    let layer = index(&p.layer).ok_or_else(|| format!("the board has no layer {}", p.layer))?;
    let c = p.clearance;
    let (mut keepout, mut own, mut relief, mut spokes) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for s in board.segments.iter().filter(|s| s.layer == p.layer) {
        let grow = if s.net == p.net { 0.0 } else { 2.0 * c };
        let ring = offset::trace(s.start, s.end, s.width + grow, SAG);
        if s.net == p.net { own.push(ring) } else { keepout.push(ring) }
    }
    for v in &board.vias {
        let (a, b) = (index(&v.layers[0]), index(&v.layers[1]));
        let spans = matches!((a, b), (Some(a), Some(b)) if a.min(b) <= layer && layer <= a.max(b));
        if !spans {
            continue;
        }
        if v.net == p.net {
            own.push(offset::disc(v.at, v.diameter, SAG));
        } else {
            keepout.push(offset::disc(v.at, v.diameter + 2.0 * c, SAG));
        }
    }
    for k in board.copper.iter().filter(|k| k.layer == p.layer) {
        if k.net != p.net {
            keepout.extend(grown(&k.polygon, c));
            continue;
        }
        own.push(k.polygon.clone());
        if !pads.contains(k) {
            continue;
        }
        match p.connection {
            Connection::Solid => {}
            Connection::None => relief.extend(grown(&k.polygon, c)),
            Connection::Thermal { gap, spoke } => {
                relief.extend(grown(&k.polygon, gap));
                spokes.extend(spokes_of(&k.polygon, gap, spoke));
            }
        }
    }
    let request = fill::Pour { outline: &p.outline, keepout: &keepout, relief: &relief, spokes: &spokes, own: &own, min_island: p.min_island };
    fill(&request).map_err(|e| e.to_string())
}

/// `polygon` grown by `d` (m): itself and a trace `2 d` wide along each of its edges (their
/// union is the polygon's Minkowski sum with a disc). Edges that run back along themselves — the
/// bridges joining a filled polygon's holes — bound nothing and are left out.
fn grown(polygon: &[[f64; 2]], d: f64) -> Vec<Vec<[f64; 2]>> {
    let n = polygon.len();
    let edge = |k: usize| (polygon[k], polygon[(k + 1) % n]);
    let mut rings = vec![polygon.to_vec()];
    if d > 0.0 {
        let edges: Vec<([f64; 2], [f64; 2])> = (0..n).map(edge).collect();
        rings.extend(edges.iter().filter(|(a, b)| a != b && !edges.contains(&(*b, *a))).map(|&(a, b)| offset::trace(a, b, 2.0 * d, SAG)));
    }
    rings
}

/// The four spokes `width` wide across the gap `gap` round a pad: two bars through its middle
/// along its first side and across it, each reaching a spoke's width past the gap.
fn spokes_of(pad: &[[f64; 2]], gap: f64, width: f64) -> Vec<Vec<[f64; 2]>> {
    let n = pad.len().max(1) as f64;
    let mid = pad.iter().fold([0.0, 0.0], |m, p| [m[0] + p[0] / n, m[1] + p[1] / n]);
    let (Some(a), Some(b)) = (pad.first(), pad.get(1)) else { return Vec::new() };
    let len = (b[0] - a[0]).hypot(b[1] - a[1]);
    if !(len > 0.0) {
        return Vec::new();
    }
    let u = [(b[0] - a[0]) / len, (b[1] - a[1]) / len];
    [u, [-u[1], u[0]]]
        .iter()
        .map(|d| {
            let reach = pad.iter().map(|p| ((p[0] - mid[0]) * d[0] + (p[1] - mid[1]) * d[1]).abs()).fold(0.0, f64::max) + gap + width;
            let (e, s) = ([-d[1] * 0.5 * width, d[0] * 0.5 * width], [d[0] * reach, d[1] * reach]);
            vec![
                [mid[0] - s[0] - e[0], mid[1] - s[1] - e[1]],
                [mid[0] + s[0] - e[0], mid[1] + s[1] - e[1]],
                [mid[0] + s[0] + e[0], mid[1] + s[1] + e[1]],
                [mid[0] - s[0] + e[0], mid[1] - s[1] + e[1]],
            ]
        })
        .collect()
}
