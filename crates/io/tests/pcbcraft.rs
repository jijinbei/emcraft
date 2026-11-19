//! PCBCraft's sensor-node example (crates/io/tests/data/pcbcraft/sensor-node, from PCBCraft's
//! examples, MIT OR Apache-2.0), read through PCBCraft's own storage and engine; and the same
//! board with a known stackup (sensor-node-stackup: a 1.5 mm core whose material's dielectric
//! report gives Dk 4.3 and Df 0.017).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use emcraft_io::pcbcraft::{board, is_project, read};
use emcraft_model::pcb::Layer;

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/pcbcraft").join(name)
}

/// Two copper layers 1.53 mm apart over FR-4, named as the example names them.
fn fr4() -> Vec<Layer> {
    vec![
        Layer { name: "Top".into(), z: 0.0, eps_below: 4.4, loss_below: 0.02, copper: 0.0 },
        Layer { name: "Bottom".into(), z: 1.53e-3, eps_below: 1.0, loss_below: 0.0, copper: 0.0 },
    ]
}

#[test]
fn the_sensor_node_reads_with_its_nets_pads_and_vias() {
    let path = data("sensor-node");
    assert!(is_project(&path) && is_project(&path.join("project.json")));
    let (b, _) = read(&path, Some(fr4())).unwrap();
    // Every track of the example on a net PCBCraft resolves, its ten nets.
    let nets: BTreeSet<&str> = b.segments.iter().map(|s| s.net.as_str()).collect();
    assert_eq!(b.segments.len(), 123);
    assert_eq!(nets, BTreeSet::from(["+3V3", "ACTIVITY_N", "ALERT_N", "GND", "LED_A", "LED_B", "SCL", "SDA", "STATUS_N", "UPDI"]));
    // Its 17 vias, and the 9 plated holes of its headers through the board.
    assert_eq!(b.vias.len(), 26);
    assert!(b.vias.iter().all(|v| v.layers == ["Top".to_string(), "Bottom".to_string()] && v.drill.is_some_and(|d| d > 0.0 && d < v.diameter)));
    assert_eq!(b.copper.len(), 52);
    // Every track's end lies on copper of its own net: a pad, a via or another track (its end
    // or along it, a T).
    let along = |p: [f64; 2], a: [f64; 2], b: [f64; 2], r: f64| {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
        (p[0] - a[0] - t * dx).hypot(p[1] - a[1] - t * dy) <= r
    };
    let near = |p: [f64; 2], q: [f64; 2], r: f64| (p[0] - q[0]).hypot(p[1] - q[1]) <= r;
    let inside = |p: [f64; 2], poly: &[[f64; 2]]| {
        let (x0, y0, x1, y1) =
            poly.iter().fold((f64::MAX, f64::MAX, f64::MIN, f64::MIN), |(a, b, c, d), q| (a.min(q[0]), b.min(q[1]), c.max(q[0]), d.max(q[1])));
        p[0] >= x0 - 1e-9 && p[0] <= x1 + 1e-9 && p[1] >= y0 - 1e-9 && p[1] <= y1 + 1e-9
    };
    for s in &b.segments {
        for end in [s.start, s.end] {
            let joined = b.copper.iter().any(|c| c.net == s.net && c.layer == s.layer && inside(end, &c.polygon))
                || b.vias.iter().any(|v| v.net == s.net && near(end, v.at, 0.5 * v.diameter))
                || b.segments.iter().any(|t| t.id != s.id && t.net == s.net && t.layer == s.layer && along(end, t.start, t.end, 0.5 * t.width));
            assert!(joined, "{} ends at {end:?} on no copper of {}", s.id, s.net);
        }
    }
}

#[test]
fn a_stackup_is_read_from_its_materials_or_refused_when_unknown() {
    // The example's own stackup gives no thickness or material: refused, not guessed.
    let err = read(&data("sensor-node"), None).unwrap_err().to_string();
    assert!(err.contains("unknown") && err.contains("stackup"), "{err}");
    // With a known core: the bottom copper 1.5 mm down, the core's Dk and Df between.
    let (b, _) = read(&data("sensor-node-stackup"), None).unwrap();
    assert_eq!(b.layers.iter().map(|l| l.name.as_str()).collect::<Vec<_>>(), ["Top", "Bottom"]);
    approx::assert_abs_diff_eq!(b.layers[1].z, 1.5e-3, epsilon = 1e-15);
    approx::assert_abs_diff_eq!(b.layers[0].eps_below, 4.3, epsilon = 1e-12);
    approx::assert_abs_diff_eq!(b.layers[0].loss_below, 0.017, epsilon = 1e-12);
    // A stackup given that lacks one of the board's copper layers is refused.
    let err = read(&data("sensor-node"), Some(vec![fr4().remove(0), Layer { name: "B.Cu".into(), ..fr4().remove(1) }])).unwrap_err().to_string();
    assert!(err.contains("Bottom"), "{err}");
    assert!(read(&data("no-such-project"), None).is_err());
}

/// The stackup example with a ground zone over its bottom layer round every track, added through
/// PCBCraft's own `pcb.add_zone` (the example has none): a thermal zone at the board's clearance.
fn with_ground_zone() -> std::sync::Arc<pcbcraft_engine::Project> {
    let mut s = pcbcraft_engine::Session::default();
    s.open_snapshot(pcbcraft_storage::preview(&data("sensor-node-stackup")).unwrap());
    let p = s.snapshot().unwrap();
    let view = p.boards().next().unwrap();
    let bottom = view.setup().unwrap().layers().iter().find(|l| l.name == "Bottom").unwrap().id;
    let nets = p.inspect_net_registry().unwrap();
    let gnd = nets["nets"].as_array().unwrap().iter().find(|n| n["name"] == "GND").unwrap().clone();
    let (b, _) = board(&p, None).unwrap();
    let (lo, hi) = b
        .segments
        .iter()
        .flat_map(|s| [s.start, s.end])
        .fold(([f64::MAX; 2], [f64::MIN; 2]), |(lo, hi), q| ([lo[0].min(q[0]), lo[1].min(q[1])], [hi[0].max(q[0]), hi[1].max(q[1])]));
    let nm = |v: f64| (v * 1e9).round() as i64;
    let (x0, y0, x1, y1) = (nm(lo[0] - 2e-3), nm(lo[1] - 2e-3), nm(hi[0] + 2e-3), nm(hi[1] + 2e-3));
    let points = serde_json::json!([{"x_nm":x0,"y_nm":y0},{"x_nm":x1,"y_nm":y0},{"x_nm":x1,"y_nm":y1},{"x_nm":x0,"y_nm":y1}]);
    let mut params = serde_json::json!({"layer":bottom,"points":points,"expected_revision":s.revision()});
    if gnd["id"].is_null() {
        params["target"] = serde_json::json!({"kind":"derived","anchor":gnd["anchor"]});
    } else {
        params["net"] = gnd["id"].clone();
    }
    s.apply("pcb.add_zone", &params).unwrap();
    s.snapshot().unwrap()
}

#[test]
fn a_zone_comes_back_as_a_pour_to_fill() {
    let p = with_ground_zone();
    let (b, pours) = board(&p, None).unwrap();
    assert_eq!(pours.len(), 1);
    let z = &pours[0];
    assert!(z.net == "GND" && z.layer == "Bottom" && z.outline.len() == 4 && z.priority == 0 && z.min_island == 0.0, "{z:?}");
    // The board's clearance (no override), and PCBCraft's default relief: 0.3 mm gap and spokes.
    assert!(z.clearance > 0.0 && z.clearance < 1e-3, "{}", z.clearance);
    assert_eq!(z.connection, emcraft_model::pcb::Connection::Thermal { gap: 300_000.0 * 1e-9, spoke: 300_000.0 * 1e-9 });
    assert!(!b.notes.iter().any(|n| n.contains("zone")), "{:?}", b.notes);
}

/// Writes `with_ground_zone` as the project `tests/data/pcbcraft/sensor-node-pour` (through
/// PCBCraft's own codec and storage), which the engine's tests import:
/// `cargo test -p emcraft-io --test pcbcraft -- --ignored write_the_pour_example`.
#[test]
#[ignore = "writes a fixture"]
fn write_the_pour_example() {
    let documents = pcbcraft_engine::persistence::encode_documents(with_ground_zone()).unwrap();
    pcbcraft_storage::save(&data("sensor-node-pour"), None, &documents, &mut |_| Ok(())).unwrap();
}
