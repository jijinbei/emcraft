use super::*;
use crate::Session;
use serde_json::json;
#[test]
fn board_import_is_embedded_saved_undoable_and_diagnostics_are_explicit() {
    let mut s = Session::new();
    let path = scratch("routing.kicad_pcb");
    std::fs::write(&path, include_str!("../../../io/tests/fixtures/route.kicad_pcb")).unwrap();
    s.execute("pcb.import", &json!({"path": path})).unwrap();
    assert_eq!(s.history().last().unwrap().command, "pcb.load");
    std::fs::remove_file(path).unwrap();
    let report = s.execute("pcb.inspect", &json!({"net": "DATA"})).unwrap();
    assert_eq!(report["model_available"], true);
    assert_eq!(report["si"]["method"], "analytical_transmission_line");
    assert_eq!(report["sections"].as_array().unwrap().len(), 2);
    let project = scratch("pcb.emcraft");
    s.execute("file.save", &json!({"path": project})).unwrap();
    let mut t = Session::new();
    t.execute("file.open", &json!({"path": project})).unwrap();
    assert_eq!(s.model().pcb(), t.model().pcb());
    assert_eq!(t.views().pcb.net(), Some("DATA"));
    t.execute("pcb.sample", &json!({"split": true})).unwrap();
    let bad = t.execute("pcb.inspect", &json!({"net": "DATA"})).unwrap();
    assert!(bad["si"].is_null());
    assert_eq!(bad["model_available"], false);
    t.execute("edit.undo", &json!({})).unwrap();
    assert_eq!(s.model().pcb(), t.model().pcb());
    let _ = std::fs::remove_file(project);
}

#[test]
fn a_pcbcraft_project_imports_with_its_nets_and_its_stackup() {
    // PCBCraft's sensor-node example with a known core (crates/io/tests/data/pcbcraft): its
    // board through PCBCraft's own engine, embedded as a KiCad board is, its nets inspected.
    let project = concat!(env!("CARGO_MANIFEST_DIR"), "/../io/tests/data/pcbcraft/sensor-node-stackup");
    let mut s = Session::new();
    let r = s.execute("pcb.import", &json!({"path": project})).unwrap();
    assert_eq!(s.history().last().unwrap().command, "pcb.load");
    let board = s.model().pcb().unwrap();
    assert_eq!((board.segments.len(), board.vias.len(), board.layers.len()), (123, 26, 2));
    approx::assert_abs_diff_eq!(board.layers[0].eps_below, 4.3, epsilon = 1e-12);
    assert_eq!((r["board"].as_str(), r["segments"].as_u64()), (Some("Board"), Some(123)));
    let report = s.execute("pcb.inspect", &json!({"net": "SDA"})).unwrap();
    assert_eq!(report["net"], "SDA");
    // The example's own unknown stackup is refused unless one is given, in display units.
    let plain = concat!(env!("CARGO_MANIFEST_DIR"), "/../io/tests/data/pcbcraft/sensor-node/project.json");
    let err = Session::new().execute("pcb.import", &json!({"path": plain})).unwrap_err().to_string();
    assert!(err.contains("stackup"), "{err}");
    let stackup = json!([{"name": "Top", "z": 0, "eps_below": 4.4}, {"name": "Bottom", "z": 1.53, "eps_below": 1}]);
    let mut t = Session::new();
    t.execute("pcb.import", &json!({"path": plain, "stackup": stackup})).unwrap();
    approx::assert_abs_diff_eq!(t.model().pcb().unwrap().layers[1].z, 1.53e-3, epsilon = 1e-15);
}

/// Whether `q` is inside `polygon` by the even–odd rule.
fn inside(polygon: &[[f64; 2]], q: [f64; 2]) -> bool {
    let n = polygon.len();
    (0..n)
        .filter(|&k| {
            let (a, b) = (polygon[k], polygon[(k + 1) % n]);
            (a[1] > q[1]) != (b[1] > q[1]) && q[0] < a[0] + (q[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        })
        .count()
        % 2
        == 1
}

/// The stackup example with a ground zone over its bottom layer (PCBCraft keeps its outline
/// only): filled on import, every other net's bottom copper kept its clearance away, so SDA's top
/// traces have a plane under them and lines' impedances; the board meshes (no two nets overlap).
#[test]
fn a_pcbcraft_zone_is_filled_clear_of_other_nets() {
    let project = concat!(env!("CARGO_MANIFEST_DIR"), "/../io/tests/data/pcbcraft/sensor-node-pour");
    let mut s = Session::new();
    s.execute("pcb.import", &json!({"path": project})).unwrap();
    let board = s.model().pcb().unwrap().clone();
    let fills: Vec<&[[f64; 2]]> =
        board.copper.iter().filter(|c| c.net == "GND" && c.layer == "Bottom" && c.polygon.len() > 40).map(|c| c.polygon.as_slice()).collect();
    assert!(!fills.is_empty(), "{:?}", board.notes);
    assert!(board.notes.iter().any(|n| n.contains("filled by EMCraft")), "{:?}", board.notes);
    let filled = |q: [f64; 2]| fills.iter().any(|f| inside(f, q));
    // Along every other net's bottom trace, at its edges and its middle: no ground there.
    let others: Vec<_> = board.segments.iter().filter(|t| t.layer == "Bottom" && t.net != "GND").collect();
    assert!(!others.is_empty());
    for t in &others {
        let (d, len) = ([t.end[0] - t.start[0], t.end[1] - t.start[1]], (t.end[0] - t.start[0]).hypot(t.end[1] - t.start[1]));
        let n = [-d[1] / len * 0.5 * t.width, d[0] / len * 0.5 * t.width];
        for k in 0..=8 {
            let f = k as f64 / 8.0;
            let m = [t.start[0] + f * d[0], t.start[1] + f * d[1]];
            for q in [m, [m[0] + n[0] * 1.2, m[1] + n[1] * 1.2], [m[0] - n[0] * 1.2, m[1] - n[1] * 1.2]] {
                assert!(!filled(q), "ground over {} ({}) at {q:?}", t.id, t.net);
            }
        }
    }
    // SDA's top traces over the ground: impedances now, where the zone's outline alone gave none.
    let r = s.execute("pcb.impedance", &json!({"net": "SDA"})).unwrap();
    let solved = r["sections"].as_array().unwrap().iter().filter(|x| x["z0"].as_f64().is_some()).count();
    assert!(solved > 0, "{r}");
    // A region 6 mm square round an SDA trace meshes: no two nets' copper overlaps on a layer.
    let sda = board.segments.iter().find(|t| t.net == "SDA").unwrap();
    let mid = [500.0 * (sda.start[0] + sda.end[0]), 500.0 * (sda.start[1] + sda.end[1])];
    s.execute("pcb.region", &json!({"lo": [mid[0] - 3.0, mid[1] - 3.0], "hi": [mid[0] + 3.0, mid[1] + 3.0], "above": 3, "below": 0})).unwrap();
    s.execute("mesh.generate", &json!({"size": 1, "strip_edge": 0.5})).unwrap();
}

/// A board in mm: a 0.2 mm FR-4 core (εr 4.3) between TOP and a GND plane on L2, 0.4 mm on to
/// L3, 0.4 mm more to a GND plane on BOT; a 0.35 mm microstrip, a differential pair named to pair,
/// a stripline centred between L2 and BOT, and a trace with no plane under it.
fn lines_board() -> serde_json::Value {
    let plane = |layer: &str| json!({"net": "GND", "layer": layer, "polygon": [[-5, -10], [45, -10], [45, 10], [-5, 10]]});
    let seg =
        |id: &str, net: &str, layer: &str, y: f64, w: f64| json!({"id": id, "net": net, "layer": layer, "start": [0, y], "end": [40, y], "width": w});
    json!({
        "name": "lines",
        "layers": [
            {"name": "TOP", "z": 0, "eps_below": 4.3},
            {"name": "L2", "z": 0.2, "eps_below": 4.3},
            {"name": "L3", "z": 0.6, "eps_below": 4.3},
            {"name": "BOT", "z": 1.0, "eps_below": 1}
        ],
        "segments": [
            seg("ms", "MS", "TOP", 0.0, 0.35),
            seg("p", "USB_D+", "TOP", 5.0, 0.2),
            seg("n", "USB_D-", "TOP", 5.35, 0.2),
            seg("sl", "SL", "L3", 0.0, 0.2),
            json!({"id": "far", "net": "FAR", "layer": "TOP", "start": [50, 0], "end": [60, 0], "width": 0.3})
        ],
        "copper": [plane("L2"), plane("BOT")]
    })
}

/// The microstrip of `lines_board` with 35 µm of copper on TOP (and on the L2 plane under it):
/// lower than the sheet, as Hammerstad and Jensen's thick strip over the thinner substrate is
/// (to 1.5 %); under a 20 µm mask of εr 3.5 lower again, with no closed form. A mask on an
/// inner layer, or of no thickness, is refused.
#[test]
fn copper_thickness_and_a_mask_lower_a_microstrips_impedance() {
    let z = |copper: f64, mask: Option<serde_json::Value>| {
        let mut b = lines_board();
        b["layers"][0]["copper"] = json!(copper);
        b["layers"][1]["copper"] = json!(copper);
        if let Some(m) = mask {
            b["masks"] = json!([m]);
        }
        let mut s = Session::new();
        s.execute("pcb.load", &json!({"data": b})).unwrap();
        s.execute("pcb.impedance", &json!({"net": "MS"})).unwrap()["sections"][0].clone()
    };
    let sheet = z(0.0, None)["z0"].as_f64().unwrap();
    let thick = z(0.035, None);
    let (zt, closed) = (thick["z0"].as_f64().unwrap(), thick["closed_form"].as_f64().unwrap());
    assert!(zt < sheet - 1.0 && (zt / closed - 1.0).abs() < 0.015, "{sheet} {thick}");
    // The closed form's substrate is the dielectric less the plane's half thickness: 0.1825 mm.
    let (zf, _) = emcraft_theory::microstrip::thick(0.35 / 0.1825, 0.035 / 0.1825, 4.3);
    approx::assert_abs_diff_eq!(closed, zf, epsilon = 1e-3);
    let masked = z(0.035, Some(json!({"layer": "TOP", "thickness": 0.02, "eps": 3.5})));
    assert!(masked["z0"].as_f64().unwrap() < zt - 0.5 && masked["closed_form"].is_null(), "{thick} {masked}");
    for bad in [json!({"layer": "L2", "thickness": 0.02, "eps": 3.5}), json!({"layer": "TOP", "thickness": 0, "eps": 3.5})] {
        let mut b = lines_board();
        b["masks"] = json!([bad]);
        assert!(Session::new().execute("pcb.load", &json!({"data": b})).is_err(), "{bad}");
    }
}

#[test]
fn line_impedances_come_from_their_cross_sections() {
    let mut s = Session::new();
    s.execute("pcb.load", &json!({"data": lines_board()})).unwrap();
    let row = |r: &serde_json::Value| r["sections"][0].clone();
    // The outer microstrip over L2's plane: Hammerstad and Jensen's fit to 1.5 %.
    let ms = row(&s.execute("pcb.impedance", &json!({"net": "MS"})).unwrap());
    let (z, closed) = (ms["z0"].as_f64().unwrap(), ms["closed_form"].as_f64().unwrap());
    assert!(ms["kind"] == "microstrip" && ms["below"] == "L2" && (z / closed - 1.0).abs() < 0.015, "{ms}");
    // The stripline centred between L2's and BOT's planes: Cohn's exact Z₀ to 1 %.
    let sl = row(&s.execute("pcb.impedance", &json!({"net": "SL"})).unwrap());
    let (z, closed) = (sl["z0"].as_f64().unwrap(), sl["closed_form"].as_f64().unwrap());
    assert!(sl["kind"] == "stripline" && sl["above"] == "L2" && sl["below"] == "BOT" && (z / closed - 1.0).abs() < 0.01, "{sl}");
    // The pair found by its names: its odd mode below its even, z_diff twice z_odd.
    let r = s.execute("pcb.impedance", &json!({"net": "USB_D+"})).unwrap();
    let d = row(&r);
    assert_eq!(r["pair"], "USB_D-");
    let (odd, even, diff) = (d["z_odd"].as_f64().unwrap(), d["z_even"].as_f64().unwrap(), d["z_diff"].as_f64().unwrap());
    assert!(odd < even && (diff - 2.0 * odd).abs() < 1e-2 && (d["gap"].as_f64().unwrap() - 0.15).abs() < 1e-9, "{d}");
    assert_eq!(r["impedance"]["mean"], d["z_diff"]);
    // No plane under the far trace: no impedance, said so.
    let far = row(&s.execute("pcb.impedance", &json!({"net": "FAR"})).unwrap());
    assert!(far["z0"].is_null() && far["note"].as_str().is_some_and(|n| n.contains("no reference")), "{far}");
    for bad in [json!({"net": "NOPE"}), json!({"net": "MS", "pair": "MS"}), json!({"net": "MS", "pair": "NOPE"})] {
        assert!(s.execute("pcb.impedance", &bad).is_err(), "{bad}");
    }
}

/// Net classes' targets checked segment by segment: the microstrip (52.8 Ω) within 10 % of 50 Ω,
/// the stripline (67.0 Ω) off it, the far trace with no plane under it; the pair against its own
/// z_diff within 1 %. Each finding names its segment by its id.
#[test]
fn net_classes_pass_and_fail_their_segments_by_id() {
    let mut s = Session::new();
    s.execute("pcb.load", &json!({"data": lines_board()})).unwrap();
    let diff = s.execute("pcb.impedance", &json!({"net": "USB_D+"})).unwrap()["sections"][0]["z_diff"].as_f64().unwrap();
    s.execute("pcb.class", &json!({"name": "50", "nets": ["MS", "SL", "FA*"], "z0": 50})).unwrap();
    s.execute("pcb.class", &json!({"name": "USB", "nets": ["USB_D*"], "z_diff": diff, "tolerance": 1})).unwrap();
    let r = s.execute("pcb.check", &json!({})).unwrap();
    assert_eq!(r["verdict"], "fail");
    let class = |name: &str| r["classes"].as_array().unwrap().iter().find(|c| c["class"] == name).cloned().unwrap();
    assert!(class("50")["verdict"] == "fail" && class("50")["failing"] == 2 && class("50")["segments"] == 3, "{r}");
    // The pair checked once, from USB_D+ beside USB_D-.
    assert!(class("USB")["verdict"] == "pass" && class("USB")["segments"] == 1, "{r}");
    let found = |id: &str| r["findings"].as_array().unwrap().iter().find(|f| f["segment"] == id).cloned();
    let sl = found("sl").unwrap();
    assert!(sl["problem"] == "off target" && sl["net"] == "SL" && sl["deviation_percent"].as_f64().unwrap() > 30.0, "{sl}");
    assert_eq!(found("far").unwrap()["problem"], "no reference plane");
    assert!(found("ms").is_none());
    // Loosened, the class passes but for the far trace; one class checked alone.
    s.execute("pcb.class", &json!({"name": "50", "tolerance": 40})).unwrap();
    let r = s.execute("pcb.check", &json!({"class": "50"})).unwrap();
    assert!(r["classes"].as_array().unwrap().len() == 1 && r["findings_total"] == 1, "{r}");
    // Saved with the board; removed again.
    assert_eq!(s.model().pcb().unwrap().classes.len(), 2);
    s.execute("pcb.class.delete", &json!({"name": "USB"})).unwrap();
    assert_eq!(s.model().pcb().unwrap().classes.len(), 1);
    let before = s.model().clone();
    for (command, bad) in [
        ("pcb.class", json!({"name": "new", "z0": 50})),
        ("pcb.class", json!({"name": "new", "nets": ["MS"]})),
        ("pcb.class", json!({"name": "new", "nets": ["MS"], "z0": -5})),
        ("pcb.class", json!({"name": "new", "nets": ["MS"], "z0": 1e9})),
        ("pcb.class", json!({"name": "new", "nets": ["MS"], "z0": 50, "tolerance": 0})),
        ("pcb.class", json!({"name": "new", "nets": ["MS"], "z0": 50, "tolerance": 500})),
        ("pcb.class", json!({"name": "new", "nets": [3], "z0": 50})),
        ("pcb.class", json!({"name": " ", "nets": ["MS"], "z0": 50})),
        ("pcb.class.delete", json!({"name": "nope"})),
        ("pcb.check", json!({"class": "nope"})),
    ] {
        assert!(s.execute(command, &bad).is_err(), "{command} {bad}");
        assert_eq!(s.model(), &before);
    }
    s.execute("pcb.class.delete", &json!({"name": "50"})).unwrap();
    assert!(s.execute("pcb.check", &json!({})).is_err());
}

#[test]
fn board_commands_refuse_hostile_inputs_and_keep_the_model() {
    let mut s = Session::new();
    s.execute("pcb.sample", &json!({"split": false})).unwrap();
    let before = s.model().clone();
    for p in [
        json!({"net": "missing"}),
        json!({"net": "DATA", "ground": "DATA"}),
        json!({"net": "DATA", "reference": 0}),
        json!({"net": "DATA", "points": 100000}),
        json!({"net": "DATA", "stop": 1e100}),
    ] {
        assert!(s.execute("pcb.inspect", &p).is_err());
        assert_eq!(s.model(), &before);
    }
    let mut board = s.model().pcb().cloned().unwrap();
    board.segments[0].width = -1.0;
    assert!(s.execute("pcb.load", &json!({"data": board})).is_err());
    assert_eq!(s.model(), &before);
    s.execute("units.set", &json!({"length": "m", "frequency": "MHz"})).unwrap();
    let r = s.execute("pcb.inspect", &json!({"net": "DATA", "stop": 1000})).unwrap();
    approx::assert_abs_diff_eq!(r["length"].as_f64().unwrap(), 0.04, epsilon = 1e-12);
    assert_eq!(r["si"]["points"][63]["frequency"], 1000.0);
}

#[test]
fn pcb_templates_declare_their_workflow_and_rebuild_from_parameters() {
    for (name, model, nearby) in [("pcb-microstrip", true, false), ("pcb-split-ground", false, false), ("pcb-parallel", true, true)] {
        let mut s = Session::new();
        s.execute("file.sample", &json!({"name": name, "values": {"l": 80, "w": 0.4}})).unwrap();
        assert_eq!(s.model.analysis(), emcraft_model::Analysis::Pcb);
        assert!(s.model.mesh.is_none());
        let status = s.execute("project.status", &json!({})).unwrap();
        assert_eq!(status["analysis"]["solve"], "pcb.inspect");
        assert!(status["not_needed"].as_array().unwrap().contains(&json!("mesh")));
        let r = s.execute("pcb.inspect", &json!({})).unwrap();
        assert_eq!(r["model_available"], model);
        approx::assert_abs_diff_eq!(r["length"].as_f64().unwrap(), 80., epsilon = 1e-10);
        assert_eq!(r["stop"], 1.);
        assert_eq!(r["findings"].as_array().unwrap().iter().any(|f| f["code"] == "nearby_signal"), nearby);
        s.execute("project.variable", &json!({"name": "l", "value": 60})).unwrap();
        let r = s.execute("pcb.inspect", &json!({"net": "DATA"})).unwrap();
        approx::assert_abs_diff_eq!(r["length"].as_f64().unwrap(), 60., epsilon = 1e-10);
        let before = s.summary();
        assert!(s.execute("project.variable", &json!({"name": "h", "value": -1})).is_err());
        assert_eq!(s.summary(), before);
    }
}

#[test]
fn a_board_loaded_with_expressions_is_built_again_from_its_variables() {
    // The microstrip circuit templates draw their boards with `pcb.load`, their numbers
    // expressions of the template's parameters: a parameter changed moves the copper.
    let mut s = Session::new();
    s.execute("file.sample", &json!({"name": "pcb-stub-filter", "values": {"l_stub": 6, "er": 9.8}})).unwrap();
    assert_eq!(s.model.analysis(), emcraft_model::Analysis::Frequency);
    let board = s.model.pcb().unwrap();
    let stub = board.segments.iter().find(|g| g.id == "stub").unwrap();
    approx::assert_abs_diff_eq!(stub.end[1], 0.006, epsilon = 1e-12);
    assert_eq!(board.layers[0].eps_below, 9.8);
    s.execute("project.variable", &json!({"name": "l_stub", "value": 8})).unwrap();
    let stub = s.model.pcb().unwrap().segments.iter().find(|g| g.id == "stub").unwrap().clone();
    approx::assert_abs_diff_eq!(stub.end[1], 0.008, epsilon = 1e-12);
    // An expression of a name the session does not hold: refused, nothing changed.
    let before = s.summary();
    let bad = json!({"data": {"name": "x", "layers": [{"name": "F", "z": 0, "eps_below": "=nothing"}, {"name": "B", "z": 1, "eps_below": 1}], "segments": []}});
    assert!(s.execute("pcb.load", &bad).is_err());
    assert_eq!(s.summary(), before);
}

#[test]
fn pcb_workflow_navigation_is_checked_saved_and_does_not_run_diagnosis() {
    let mut s = Session::new();
    s.execute("view.pcb", &json!({"stage": "board"})).unwrap();
    assert!(s.execute("view.pcb", &json!({"stage": "net"})).is_err());
    s.execute("pcb.sample", &json!({"split": false})).unwrap();
    assert_eq!(crate::pcb::stage(&s), crate::pcb::Stage::Board);
    assert!(s.execute("view.pcb", &json!({"stage": "settings"})).is_err());
    assert!(s.execute("view.pcb", &json!({"net": "missing"})).is_err());
    s.execute("view.pcb", &json!({"stage": "net", "net": "DATA"})).unwrap();
    s.execute("view.pcb", &json!({"stage": "settings"})).unwrap();
    let file = scratch("pcb-stage.emcraft");
    s.execute("file.save", &json!({"path": file})).unwrap();
    let mut reopened = Session::new();
    reopened.execute("file.open", &json!({"path": file})).unwrap();
    assert_eq!(crate::pcb::stage(&reopened), crate::pcb::Stage::Settings);
    assert_eq!(reopened.views().pcb.net(), Some("DATA"));
    reopened.execute("pcb.inspect", &json!({})).unwrap();
    assert_eq!(crate::pcb::stage(&reopened), crate::pcb::Stage::Results);
    reopened.execute("pcb.inspect", &json!({"reference": 75, "stop": 10})).unwrap();
    reopened.execute("file.save", &json!({"path": file})).unwrap();
    let mut again = Session::new();
    again.execute("file.open", &json!({"path": file})).unwrap();
    assert_eq!(again.views().pcb.conditions().unwrap().reference, 75.);
    assert_eq!(again.views().pcb.conditions().unwrap().stop_hz, 1e10);
    let _ = std::fs::remove_file(file);
}

#[test]
fn coupling_pairs_roles_and_geometry_spacing_are_explicit() {
    let mut s = Session::new();
    s.execute("file.sample", &json!({"name": "pcb-parallel"})).unwrap();
    s.execute("view.pcb", &json!({"net": "DATA", "victim": "CLK", "mode": "magnetic"})).unwrap();
    let r = s.execute("pcb.inspect", &json!({"net": "DATA"})).unwrap();
    let pairs = r["coupling_pairs"].as_array().unwrap();
    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0]["target"], "CLK");
    approx::assert_abs_diff_eq!(pairs[0]["gap"].as_f64().unwrap(), 0.25, epsilon = 1e-10);
    assert_eq!(pairs[0]["overlap"], 20.);
    assert_eq!(pairs[0]["method"], "geometry_only");
    assert!(s.execute("view.pcb", &json!({"victim": "DATA"})).is_err());
    assert!(s.execute("view.pcb", &json!({"victim": "missing"})).is_err());
    assert!(s.execute("view.pcb", &json!({"mode": "radiation"})).is_err());
    let file = scratch("pcb-coupling.emcraft");
    s.execute("file.save", &json!({"path": file})).unwrap();
    let mut again = Session::new();
    again.execute("file.open", &json!({"path": file})).unwrap();
    assert_eq!(s.views().pcb_illustration, again.views().pcb_illustration);
    s.execute("project.variable", &json!({"name": "gap", "value": 1.5})).unwrap();
    let r = s.execute("pcb.inspect", &json!({"net": "DATA"})).unwrap();
    approx::assert_abs_diff_eq!(r["coupling_pairs"][0]["gap"].as_f64().unwrap(), 1.5, epsilon = 1e-10);
    let _ = std::fs::remove_file(file);
}

/// The diagnosis's conditions: refused before a route and for hostile values (nothing
/// changed), confirmed for the route, kept when another route is chosen, the defaults of
/// pcb.inspect, and reset by another board.
#[test]
fn the_diagnosis_conditions_are_confirmed_for_a_route() {
    use crate::pcb::Diagnosis;
    let mut s = Session::new();
    s.execute("pcb.sample", &json!({})).unwrap();
    assert!(s.execute("pcb.conditions", &json!({"reference": 75})).is_err(), "a route first");
    s.execute("view.pcb", &json!({"net": "DATA"})).unwrap();
    assert_eq!(s.views().pcb, Diagnosis::Routed { net: "DATA".into() });
    for bad in [
        json!({"ground": "NOPE"}),
        json!({"ground": ""}),
        json!({"reference": 0}),
        json!({"reference": "x"}),
        json!({"stop": -1}),
        json!({"stop": 1e9}),
        json!({"stop": f64::MAX}),
    ] {
        assert!(s.execute("pcb.conditions", &bad).is_err(), "{bad}");
        assert_eq!(s.views().pcb, Diagnosis::Routed { net: "DATA".into() }, "{bad}");
    }
    let r = s.execute("pcb.conditions", &json!({"reference": 75, "stop": 5})).unwrap();
    assert_eq!(r, json!({"ground": "GND", "reference": 75.0, "stop": 5.0}));
    s.execute("view.pcb", &json!({"net": "CLK"})).unwrap();
    assert_eq!((s.views().pcb.net(), s.views().pcb.conditions().map(|c| c.stop_hz)), (Some("CLK"), Some(5e9)));
    let report = s.execute("pcb.inspect", &json!({})).unwrap();
    assert_eq!((report["net"].as_str(), report["reference"].as_f64(), report["stop"].as_f64()), (Some("CLK"), Some(75.0), Some(5.0)));
    s.execute("pcb.sample", &json!({})).unwrap();
    assert_eq!(s.views().pcb, Diagnosis::Unrouted);
}
