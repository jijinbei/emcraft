//! The electrons `results.track` sent last, drawn in the model (docs/adr/0012-particles.md):
//! each trajectory a line through its points, coloured by the energy gained so far (the
//! theme's warning) or lost (its success), its lines' colour for none — not the field's red and
//! blue, whose plane the trajectories cross; the legend saying the largest change. And the
//! particles a particle-in-cell run ended with ([`cloud`]): each a short stroke along its
//! motion, coloured by its energy.

use emcraft_engine::Session;
use emcraft_render::view3d::{Line, Scene};

use super::Built;

/// A trajectory's lines are drawn this opaque.
const ALPHA: u8 = 255;

/// The session's last tracking while the eigenmodes it was run on are still the ones solved:
/// what the window draws and charts.
pub fn tracking(session: &Session) -> Option<&emcraft_engine::Tracking> {
    session.views().tracked.as_deref().filter(|t| t.is_current(session))
}

/// The trajectories of the session's last tracking, in metres, while the eigenmodes it was run
/// on are still the ones solved; nothing otherwise.
///
/// ```
/// use emcraft_ui_core::scene3d::{Colours, trajectories};
/// let theme = emcraft_ui_core::theme::Themes::load(None).get("dark");
/// let s = emcraft_engine::Session::new();
/// // Nothing tracked: nothing drawn.
/// assert!(trajectories(&Colours::of(&theme), &s).scene.lines.is_empty());
/// ```
pub fn trajectories(colours: &super::Colours, session: &Session) -> Built {
    let Some(tracked) = tracking(session) else { return Built::default() };
    let start = tracked.energy;
    let runs: Vec<Vec<([f64; 3], f64)>> = tracked.particles.iter().map(|p| p.points.iter().map(|q| (q.x, q.energy)).collect()).collect();
    let most = runs.iter().flatten().map(|(_, e)| (e - start).abs()).fold(0.0, f64::max);
    let scale = if most > 0.0 { 1.0 / most } else { 0.0 };
    let lines = runs
        .iter()
        .flat_map(|run| {
            run.windows(2).map(|w| {
                let gained = 0.5 * (w[0].1 + w[1].1) - start;
                let k = (gained * scale).clamp(-1.0, 1.0);
                let color = if k >= 0.0 { colours.wire.mix(colours.gained, k) } else { colours.wire.mix(colours.lost, -k) };
                Line { a: w[0].0, b: w[1].0, color, alpha: ALPHA }
            })
        })
        .collect();
    let said = format!("electrons: ±{} keV", crate::number::value(most / 1e3));
    Built { scene: Scene { lines, ..Scene::default() }, legend: vec![(said, emcraft_render::colormap::diverging(1.0))], ..Built::default() }
}

/// A stroke's length, as a fraction of the run's largest cell.
const STROKE: f64 = 0.6;

/// The particles the session's particle run ended with, each a short stroke along its momentum
/// (a small cross at rest), from the lines' colour at none to the theme's warning at the most
/// energy; the legend saying the most.
///
/// ```
/// use emcraft_ui_core::scene3d::{Colours, cloud};
/// let theme = emcraft_ui_core::theme::Themes::load(None).get("dark");
/// // No run: nothing drawn.
/// assert!(cloud(&Colours::of(&theme), &emcraft_engine::Session::new()).scene.lines.is_empty());
/// ```
pub fn cloud(colours: &super::Colours, session: &Session) -> Built {
    let Some(r) = session.results().particles.as_deref() else { return Built::default() };
    let most = r.snapshot.iter().map(|p| p.energy).fold(0.0, f64::max);
    let scale = if most > 0.0 { 1.0 / most } else { 0.0 };
    let len = STROKE * r.h.iter().copied().fold(0.0, f64::max);
    let mut lines = Vec::with_capacity(r.snapshot.len() * 3);
    for p in &r.snapshot {
        let color = colours.wire.mix(colours.gained, (p.energy * scale).clamp(0.0, 1.0));
        let n = p.u.iter().map(|c| c * c).sum::<f64>().sqrt();
        let strokes: Vec<[f64; 3]> = if n > 0.0 { vec![p.u.map(|c| c / n)] } else { vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] };
        for d in strokes {
            let half = if n > 0.0 { 0.0 } else { 0.5 };
            let a = std::array::from_fn(|k| p.x[k] - half * len * d[k]);
            let b = std::array::from_fn(|k| p.x[k] + (1.0 - half) * len * d[k]);
            lines.push(Line { a, b, color, alpha: ALPHA });
        }
    }
    let said = format!("particles: {} keV", crate::number::value(most / 1e3));
    Built { scene: Scene { lines, ..Scene::default() }, legend: vec![(said, emcraft_render::colormap::diverging(1.0))], ..Built::default() }
}
