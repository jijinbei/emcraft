//! `results.field.export`: a plane as CSV and the mesh as a VTK unstructured grid.

use serde_json::json;

use super::fixtures::scratch;
use crate::Session;

/// The high-pass guide (a = 22.86 mm) solved at 7 GHz, first order.
fn solved() -> Session {
    let mut s = Session::new();
    s.execute("file.sample", &json!({"name": "highpass"})).unwrap();
    s.execute("solve.frequency", &json!({"frequencies": [7], "order": 1})).unwrap();
    s
}

#[test]
fn a_planes_rows_are_the_fields_probed_there() {
    let mut s = solved();
    let path = scratch("plane.csv");
    let r = s.execute("results.field.export", &json!({"path": path, "frequency": 7, "plane": "h"})).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    assert_eq!(header.len(), 15);
    assert_eq!(header[0], "x (mm)");
    assert_eq!(header[4], "Ex im (V/m)");
    let rows: Vec<Vec<f64>> = lines.map(|l| l.split(',').map(|v| v.parse().unwrap()).collect()).collect();
    assert_eq!(rows.len() as u64, r["csv"]["rows"].as_u64().unwrap());
    assert!(rows.len() > 1000);
    // A row in the middle: results.probe at its point gives its numbers.
    let row = &rows[rows.len() / 2];
    let probe = s.execute("results.probe", &json!({"frequency": 7, "at": [[row[0], row[1], row[2]]]})).unwrap();
    let e = &probe["points"][0]["e"];
    for (k, part) in [(1, 0), (1, 1)] {
        let ours = row[3 + 2 * k + part];
        let probed = e[k][part].as_f64().unwrap();
        assert!((ours - probed).abs() <= 1e-9 * probed.abs().max(1.0), "{ours} {probed}");
    }
}

#[test]
fn the_mesh_goes_out_whole_with_te10s_field_at_its_nodes() {
    // A plain WR-90 guide: TE10 alone at 10 GHz.
    let mut s = Session::new();
    s.execute("geometry.waveguide", &json!({"width": 22.86, "height": 10.16, "sections": [{"length": 30, "material": "vacuum"}]})).unwrap();
    s.execute("mesh.generate", &json!({"size": 4})).unwrap();
    s.execute("model.port.create", &json!({"name": "1", "end": "start"})).unwrap();
    s.execute("model.port.create", &json!({"name": "2", "end": "end"})).unwrap();
    s.execute("solve.frequency", &json!({"frequencies": [10], "order": 2})).unwrap();
    let path = scratch("guide.vtu");
    let r = s.execute("results.field.export", &json!({"path": path, "frequency": 10})).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let text = String::from_utf8_lossy(&bytes);
    let points = r["vtu"]["points"].as_u64().unwrap();
    assert!(text.contains(&format!("NumberOfPoints=\"{points}\"")));
    assert!(text.contains(&format!("NumberOfCells=\"{}\"", r["vtu"]["tetrahedra"])));
    for name in ["E re (V/m)", "E im (V/m)", "H re (A/m)", "H im (A/m)", "|E| (V/m)", "|H| (A/m)", "material"] {
        assert!(text.contains(&format!("Name=\"{name}\"")), "{name}");
    }
    assert!(text.ends_with("</VTKFile>\n"));
    // TE10's E (E_y ∝ sin(πx/a), Pozar, Microwave Engineering, §3.3) vanishes on the side walls
    // x = 0 and x = a and peaks between them (1 V/m, the port's unit wave).
    let xyz = array(&bytes, "points");
    let e = array(&bytes, "|E| (V/m)");
    let peak = e.iter().copied().fold(0.0, f64::max);
    let a = xyz.chunks(3).map(|p| p[0]).fold(0.0, f64::max);
    let walls = xyz.chunks(3).zip(&e).filter(|(p, _)| p[0].abs() < 1e-9 || (p[0] - a).abs() < 1e-9).map(|(_, v)| *v);
    let middle = xyz.chunks(3).zip(&e).filter(|(p, _)| (p[0] - a / 2.0).abs() < a / 8.0).map(|(_, v)| *v);
    // On average over the wall nodes, that is: a node's field is the mean of its tetrahedra's
    // values at their corners, which second-order elements 4 mm across get within a few percent.
    let walls: Vec<f64> = walls.collect();
    assert!(walls.iter().sum::<f64>() / (walls.len() as f64) < 0.1 * peak);
    assert!(middle.fold(0.0, f64::max) > 0.8 * peak);
}

/// A Float64 array of a .vtu file read back from its appended block.
fn array(bytes: &[u8], name: &str) -> Vec<f64> {
    let text = String::from_utf8_lossy(bytes);
    let at = text.find(&format!("Name=\"{name}\"")).unwrap();
    let offset: usize = text[at..].split("offset=\"").nth(1).unwrap().split('"').next().unwrap().parse().unwrap();
    let mark = b"<AppendedData encoding=\"raw\">\n   _";
    let start = bytes.windows(mark.len()).position(|w| w == mark).unwrap() + mark.len() + offset;
    let len = u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap()) as usize;
    bytes[start + 8..start + 8 + len].chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn exports_without_a_solution_or_to_an_unknown_kind_of_file_are_refused() {
    let mut s = Session::new();
    s.execute("file.sample", &json!({"name": "highpass"})).unwrap();
    assert!(s.execute("results.field.export", &json!({"path": scratch("a.csv"), "frequency": 7})).is_err());
    let mut s = solved();
    for path in [scratch("a.txt"), scratch("noext"), String::new()] {
        assert!(s.execute("results.field.export", &json!({"path": path, "frequency": 7})).is_err(), "{path}");
    }
    let missing = scratch("no/such/dir/a.vtu");
    assert!(s.execute("results.field.export", &json!({"path": missing, "frequency": 7})).is_err());
}
