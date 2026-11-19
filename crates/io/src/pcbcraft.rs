//! PCBCraft boards (github.com/tangency-labs/pcbcraft, MIT OR Apache-2.0): a project folder, its
//! `project.json` or a packed `.pc`, opened read-only through PCBCraft's own storage and engine
//! (`pcbcraft-storage`, `pcbcraft-engine`, pinned to a commit; docs/adr/0013-pcbcraft-boards.md),
//! so its versioned formats, footprint libraries and copper connectivity are PCBCraft's to resolve.
//!
//! **What is read.** The project's one physical board: its copper layers in stack order (their
//! heights from the finished dielectric thicknesses, each gap's permittivity and loss tangent from
//! the one dielectric report of its material), the footprints' copper pads (rectangles, rounded
//! rectangles and ovals by chords; a plated hole a via through the board), the routed tracks and
//! through vias, each with the net PCBCraft resolves for its copper island. Coordinates are
//! PCBCraft's board coordinates (Y down on screen, as KiCad's), in metres.
//!
//! The copper zones come back beside the board as pours still to fill ([`Pour`]: their outlines
//! and fill settings — clearance, the board's when the zone gives none, thermal reliefs, minimum
//! island, priority): PCBCraft keeps a zone's outline and settings, not filled copper yet, so
//! the caller fills them (`emcraft_engine`'s `pcb.import`).
//!
//! **What is not.** Copper that PCBCraft resolves to no single net, pads without copper or a
//! net, and zones whose net or layer is not resolved are left out, each noted. A stackup whose dielectric thickness or permittivity is unknown is refused unless the
//! caller supplies the copper layers itself: no electrical value is guessed.

use std::path::Path;

use emcraft_model::pcb::{Board, Connection, Copper, Layer, Pour, Segment, Via};
use pcbcraft_engine::Project;
use pcbcraft_engine::board::definition::PadShape;
use pcbcraft_engine::board::materials::{Construction, Material, MaterialRef};
use pcbcraft_engine::board::setup::{BoardSetup, Stratum};
use pcbcraft_engine::board::view::{CopperLayers, PadView};
use pcbcraft_engine::board::zones;
use pcbcraft_engine::schematic::Point;

use crate::{Error, Result};

fn bad(s: &str) -> Error {
    Error::Invalid(format!("PCBCraft import: {s}"))
}

/// Metres of PCBCraft's integer nanometres.
fn m(nm: i64) -> f64 {
    nm as f64 * 1e-9
}

fn xy(p: Point) -> [f64; 2] {
    [m(p.x_nm), m(p.y_nm)]
}

/// Whether `path` names a PCBCraft project: a folder holding `project.json`, that entry point
/// itself, or a packed `.pc`.
///
/// ```
/// use std::path::Path;
/// assert!(emcraft_io::pcbcraft::is_project(Path::new("board.pc")));
/// assert!(!emcraft_io::pcbcraft::is_project(Path::new("board.kicad_pcb")));
/// ```
pub fn is_project(path: &Path) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
    ext.as_deref() == Some("pc") || path.file_name().is_some_and(|n| n == "project.json") || path.join("project.json").is_file()
}

/// The board of the PCBCraft project at `path` and its copper zones as pours still to fill, its
/// copper layers the caller's (`override_layers`, in SI metres, named as the board names its
/// copper layers) or its own stackup's. Refused: no physical board, a board with no layer setup,
/// an unknown dielectric thickness or permittivity (without the caller's layers), no routed
/// copper.
///
/// ```
/// assert!(emcraft_io::pcbcraft::read(std::path::Path::new("no/such/project.json"), None).is_err());
/// ```
pub fn read(path: &Path, override_layers: Option<Vec<Layer>>) -> Result<(Board, Vec<Pour>)> {
    let project = pcbcraft_storage::preview(path).map_err(|e| bad(&e))?;
    board(&project, override_layers)
}

/// [`read`] of a project already open: its first physical board.
pub fn board(project: &Project, override_layers: Option<Vec<Layer>>) -> Result<(Board, Vec<Pour>)> {
    let view = project.boards().next().ok_or_else(|| bad("the project has no physical board"))?;
    let setup = view.setup().ok_or_else(|| bad("the board has no layer setup (create it in PCBCraft's Board setup)"))?;
    let copper: Vec<(uuid::Uuid, String)> =
        setup.copper_stack().filter_map(|c| setup.layers().iter().find(|l| l.id == c.layer).map(|l| (l.id, l.name.clone()))).collect();
    let name_of = |id: uuid::Uuid| copper.iter().find(|(l, _)| *l == id).map(|(_, n)| n.clone());
    let layers = match override_layers {
        Some(layers) if !layers.is_empty() => {
            if let Some(missing) = copper.iter().find(|(_, n)| !layers.iter().any(|l| &l.name == n)) {
                return Err(bad(&format!("the stackup given has no layer `{}` of the board's copper", missing.1)));
            }
            layers
        }
        _ => stackup(setup)?,
    };
    let (first, last) = match (copper.first(), copper.last()) {
        (Some(f), Some(l)) => (f.1.clone(), l.1.clone()),
        _ => return Err(bad("the board has no copper layers")),
    };
    let mut board = Board {
        name: view.name().chars().take(128).collect(),
        layers,
        segments: Vec::new(),
        vias: Vec::new(),
        copper: Vec::new(),
        notes: vec!["Packages, solder mask and copper thickness are not included in this routing model; pads are copper of their nets.".into()],
        region: None,
        classes: vec![],
    };
    let note = |board: &mut Board, text: &str| {
        if board.notes.len() < 128 && !board.notes.iter().any(|n| n == text) {
            board.notes.push(text.into());
        }
    };
    let nets = view.copper_nets();
    for t in view.tracks() {
        let (Some(net), Some(layer)) = (nets.get(&t.id), name_of(t.layer)) else {
            note(&mut board, "Copper that PCBCraft resolves to no single net (unconnected or conflicting) was skipped.");
            continue;
        };
        if t.start == t.end {
            continue;
        }
        board.segments.push(Segment { id: t.id.to_string(), net: net.clone(), layer, start: xy(t.start), end: xy(t.end), width: m(t.width_nm) });
    }
    for v in view.vias() {
        let Some(net) = nets.get(&v.id) else {
            note(&mut board, "Copper that PCBCraft resolves to no single net (unconnected or conflicting) was skipped.");
            continue;
        };
        board.vias.push(Via {
            net: net.clone(),
            at: xy(v.at),
            layers: [first.clone(), last.clone()],
            diameter: m(v.diameter_nm),
            drill: Some(m(v.drill_nm)),
        });
    }
    for fp in view.footprints() {
        for pad in &fp.pads {
            let (Some(net), Some(corners), Some(CopperLayers::Resolved { layers, .. })) = (&pad.net, pad.copper, &pad.copper_layers) else {
                note(&mut board, "Pads without copper or a net were skipped.");
                continue;
            };
            let polygon = pad_polygon(pad, corners);
            for layer in layers.iter().filter_map(|l| name_of(*l)) {
                board.copper.push(Copper { net: net.clone(), layer, polygon: polygon.clone() });
            }
            if let (true, Some(drill)) = (pad.through_hole, pad.drill_nm) {
                let side = |a: Point, b: Point| m(a.x_nm - b.x_nm).hypot(m(a.y_nm - b.y_nm));
                let diameter = side(corners[0], corners[1]).min(side(corners[1], corners[2]));
                board.vias.push(Via { net: net.clone(), at: xy(pad.center), layers: [first.clone(), last.clone()], diameter, drill: Some(m(drill)) });
            }
        }
    }
    let pours = pours(view.zones(), &zone_nets(&view), view.effective_rules().values.clearance_nm, &name_of);
    if pours.len() < view.zones().len() {
        note(&mut board, "Copper zones whose net or copper layer is not resolved were skipped.");
    }
    board.check().map_err(|e| bad(&e.to_string()))?;
    Ok((board, pours))
}

/// The names of the nets the board's registry knows by id (zones name their net by its id).
fn zone_nets(view: &pcbcraft_engine::board::view::BoardView) -> Vec<(uuid::Uuid, String)> {
    let Ok(registry) = view.inspect_zone_nets() else { return Vec::new() };
    let nets = registry.get("nets").and_then(|n| n.as_array()).map_or(&[][..], Vec::as_slice);
    nets.iter()
        .filter_map(|n| {
            let id = n.get("id").and_then(|v| v.as_str()).and_then(|s| s.parse().ok())?;
            Some((id, n.get("name")?.as_str()?.to_string()))
        })
        .collect()
}

/// The zones as pours (metres), each on its net's name and its copper layer's; a zone without
/// its own clearance takes the board's (`board_clearance`, nm). Zones whose net or layer is
/// unknown are left out.
fn pours(zones: &[zones::Zone], nets: &[(uuid::Uuid, String)], board_clearance: i64, layer: &impl Fn(uuid::Uuid) -> Option<String>) -> Vec<Pour> {
    zones
        .iter()
        .filter_map(|z| {
            let net = nets.iter().find(|(id, _)| *id == z.net)?.1.clone();
            let o = &z.options;
            let connection = match o.connection {
                zones::Connection::Thermal => Connection::Thermal { gap: m(o.thermal_gap_nm), spoke: m(o.spoke_width_nm) },
                zones::Connection::Solid => Connection::Solid,
                zones::Connection::None => Connection::None,
            };
            Some(Pour {
                net,
                layer: layer(z.layer)?,
                outline: z.points.iter().map(|p| xy(*p)).collect(),
                clearance: m(o.clearance_nm.unwrap_or(board_clearance)),
                connection,
                min_island: o.min_island_area_nm2 as f64 * 1e-18,
                priority: u32::from(o.priority),
            })
        })
        .collect()
}

/// The copper layers from the stackup, top down: each one's depth z (m) below the top copper,
/// the sum of the finished dielectric thicknesses above it, and the permittivity and loss tangent
/// of the gap below it. Copper thickness is not counted (the conductors are sheets).
fn stackup(setup: &BoardSetup) -> Result<Vec<Layer>> {
    let name = |id: uuid::Uuid| setup.layers().iter().find(|l| l.id == id).map(|l| l.name.clone());
    let mut layers: Vec<Layer> = Vec::new();
    let mut z = 0.0;
    for stratum in &setup.stackup().strata {
        match stratum {
            Stratum::Copper { layer, .. } => {
                let name = name(*layer).ok_or_else(|| bad("a copper stratum names no layer"))?;
                layers.push(Layer { name, z, eps_below: 1.0, loss_below: 0.0, copper: 0.0 });
            }
            Stratum::Dielectric { construction, thickness_nm, .. } => {
                let unknown =
                    || bad("a dielectric's finished thickness or permittivity is unknown: give it in PCBCraft's Board setup, or supply `stackup`");
                let thickness = thickness_nm.filter(|t| *t > 0).ok_or_else(unknown)?;
                let (eps, loss) = dielectric(setup.materials(), construction.as_ref()).ok_or_else(unknown)??;
                let above = layers.last_mut().ok_or_else(|| bad("the stackup starts with a dielectric"))?;
                (above.eps_below, above.loss_below) = (eps, loss);
                z += m(thickness);
            }
        }
    }
    Ok(layers)
}

/// A gap's permittivity and loss tangent from its material's dielectric reports: the one Dk the
/// reports agree on (with its Df, else lossless). None when no material or Dk is known; refused
/// when the plies' materials or reports disagree (no homogenization is guessed).
fn dielectric(materials: &[Material], construction: Option<&Construction>) -> Option<Result<(f64, f64)>> {
    let refs: Vec<MaterialRef> = match construction? {
        Construction::Core { material, .. } => vec![(*material)?],
        Construction::Prepreg { plies } => plies.iter().map(|p| p.material).collect::<Option<Vec<_>>>()?,
    };
    let mut values: Vec<(f64, f64)> = Vec::new();
    for r in &refs {
        let material = materials.iter().find(|x| x.id == r.id && x.revision == r.revision)?;
        for report in &material.dielectric {
            if let Some(dk) = report.dk_millionths {
                values.push((dk as f64 * 1e-6, report.df_millionths.map_or(0.0, |d| d as f64 * 1e-6)));
            }
        }
    }
    let (eps, loss) = *values.first()?;
    if values.iter().any(|(e, _)| (e - eps).abs() > 1e-9) {
        return Some(Err(bad("a dielectric's materials or reports give different permittivities: supply an explicitly homogenized `stackup`")));
    }
    Some(Ok((eps, loss)))
}

/// A pad's copper outline (m): its rectangle, a circle or an oval (stadium) by chords within a
/// hundredth of its size; rounded rectangles as their rectangle.
fn pad_polygon(pad: &PadView, corners: [Point; 4]) -> Vec<[f64; 2]> {
    let c = corners.map(xy);
    let rect = c.to_vec();
    let (u, v) = ([c[1][0] - c[0][0], c[1][1] - c[0][1]], [c[3][0] - c[0][0], c[3][1] - c[0][1]]);
    let (a, b) = (u[0].hypot(u[1]), v[0].hypot(v[1]));
    if a <= 0.0 || b <= 0.0 {
        return rect;
    }
    let centre = xy(pad.center);
    match pad.shape {
        PadShape::Rectangle | PadShape::RoundedRectangle => rect,
        PadShape::Circle | PadShape::Oval => {
            // The long axis's unit vector and the stadium's half-length between its arcs' centres.
            let (long, short, axis) = if a >= b { (a, b, [u[0] / a, u[1] / a]) } else { (b, a, [v[0] / b, v[1] / b]) };
            let (r, half) = (0.5 * short, 0.5 * (long - short));
            let per_end = 16usize;
            let mut out = Vec::with_capacity(2 * (per_end + 1));
            let along = axis[1].atan2(axis[0]);
            // Each end's half circle, counterclockwise: about +axis from along − π/2, about −axis
            // from along + π/2.
            for end in [1.0, -1.0] {
                let o = [centre[0] + end * half * axis[0], centre[1] + end * half * axis[1]];
                let start = along - end * std::f64::consts::FRAC_PI_2;
                for k in 0..=per_end {
                    let t = start + std::f64::consts::PI * k as f64 / per_end as f64;
                    out.push([o[0] + r * t.cos(), o[1] + r * t.sin()]);
                }
            }
            out
        }
    }
}
