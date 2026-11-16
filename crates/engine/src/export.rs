//! `results.field.export`: a solution's fields as numbers for other programs — a plane's grid as
//! CSV (the plane `view.field` draws, one row a point) or the whole mesh as a VTK unstructured
//! grid (`.vtu`: E and H at every node, the material of every tetrahedron) for ParaView, VisIt
//! or meshio. Points in the display's length unit, fields in V/m and A/m, complex as real and
//! imaginary parts (e^{jωt}).

use std::io::Write;

use emcraft_model::CVec3;
use serde_json::{Value, json};

use crate::view::{FieldView, field_view, node_fields, sample_plane};
use crate::{EngineError, Params, Result, Session};

emcraft_macros::named_enum! {
    /// The file formats written, by the path's extension.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Format {
        Csv = "csv",
        Vtu = "vtu",
    }
}

/// `results.field.export`.
pub fn field(s: &mut Session, p: &Params) -> Result<Value> {
    let path = p.string("path").required()?;
    let extension = std::path::Path::new(path).extension().and_then(|e| e.to_str()).map_or_else(String::new, str::to_ascii_lowercase);
    let format = Format::parse(&extension).ok_or_else(|| p.error("path", "ends in .csv (a plane) or .vtu (the whole mesh)"))?;
    let view = field_view(s, p)?;
    let file = std::fs::File::create(path).map_err(|e| EngineError::cannot_write(path, e))?;
    let mut out = std::io::BufWriter::new(file);
    let reply = match format {
        Format::Csv => csv(s, &view, &mut out),
        Format::Vtu => vtu(s, &view, &mut out),
    }?;
    out.flush().map_err(|e| EngineError::cannot_write(path, e))?;
    Ok(json!({"path": path, "frequency": s.show_frequency(reply.1), format.id(): reply.0}))
}

/// The six complex components of (E, H) as twelve numbers: each real part, then its imaginary.
fn parts((e, h): (CVec3, CVec3)) -> [f64; 12] {
    let c = [e.x, e.y, e.z, h.x, h.y, h.z];
    std::array::from_fn(|k| if k % 2 == 0 { c[k / 2].re } else { c[k / 2].im })
}

/// The plane's grid: a header, then a row per point inside the mesh. The rows written and the
/// frequency (Hz).
fn csv(s: &Session, view: &FieldView, out: &mut impl Write) -> Result<(Value, f64)> {
    let samples = sample_plane(s, view)?;
    let l = s.display.length.symbol();
    let columns = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"].map(|f| {
        let unit = if f.starts_with('E') { "V/m" } else { "A/m" };
        format!("{f} re ({unit}),{f} im ({unit})")
    });
    let mut text = format!("x ({l}),y ({l}),z ({l}),{}\n", columns.join(","));
    let mut rows = 0usize;
    for (c, e, h) in samples.points(|m| s.show_length(m)) {
        let numbers = c.iter().copied().chain(parts((e, h))).map(|v| v.to_string()).collect::<Vec<_>>();
        text += &numbers.join(",");
        text.push('\n');
        rows += 1;
    }
    out.write_all(text.as_bytes()).map_err(|e| EngineError::failed(format!("cannot write the CSV: {e}")))?;
    Ok((json!({"rows": rows}), samples.frequency))
}

/// The whole mesh with the fields at its nodes. The counts written and the frequency (Hz).
fn vtu(s: &Session, view: &FieldView, out: &mut impl Write) -> Result<(Value, f64)> {
    let (mesh, fields, frequency) = node_fields(s, view)?;
    let points = gathered(mesh.nodes.iter().map(|p| p.to_array().map(|c| s.show_length(c))), mesh.nodes.len(), "the points")?;
    // E and H, each real and imaginary, then |E| and |H| (peak), at every node.
    let vector = |which: usize, part: fn(&emcraft_solver::Complex64) -> f64| {
        let pick = move |f: &(CVec3, CVec3)| if which == 0 { f.0 } else { f.1 };
        gathered(fields.iter().flat_map(move |f| [pick(f).x, pick(f).y, pick(f).z].map(|c| part(&c))), fields.len() * 3, "a field at the nodes")
    };
    let (re, im): (fn(&emcraft_solver::Complex64) -> f64, fn(&emcraft_solver::Complex64) -> f64) = (|c| c.re, |c| c.im);
    let (e_re, e_im, h_re, h_im) = (vector(0, re)?, vector(0, im)?, vector(1, re)?, vector(1, im)?);
    let peak = |v: CVec3| (v.x.norm_sqr() + v.y.norm_sqr() + v.z.norm_sqr()).sqrt();
    let e_abs = gathered(fields.iter().map(|f| peak(f.0)), fields.len(), "|E| at the nodes")?;
    let h_abs = gathered(fields.iter().map(|f| peak(f.1)), fields.len(), "|H| at the nodes")?;
    let material = gathered(mesh.tet_material.iter().map(|&m| f64::from(m)), mesh.tets.len(), "the materials")?;
    let array = |name, components, values| emcraft_io::vtu::Array { name, components, values };
    let grid = emcraft_io::vtu::Grid {
        points: &points,
        tets: &mesh.tets,
        point_data: vec![
            array("E re (V/m)", 3, &e_re),
            array("E im (V/m)", 3, &e_im),
            array("H re (A/m)", 3, &h_re),
            array("H im (A/m)", 3, &h_im),
            array("|E| (V/m)", 1, &e_abs),
            array("|H| (A/m)", 1, &h_abs),
        ],
        cell_data: vec![array("material", 1, &material)],
    };
    emcraft_io::vtu::write(&grid, out)?;
    Ok((json!({"points": points.len(), "tetrahedra": mesh.tets.len()}), frequency))
}

/// `n` items into a vector made by [`emcraft_solver::memory`]: refused, not aborted, when the
/// memory is not there.
fn gathered<T>(items: impl Iterator<Item = T>, n: usize, what: &'static str) -> Result<Vec<T>> {
    let mut v = emcraft_solver::memory::with_capacity(n, what).map_err(|e| EngineError::failed(e.to_string()))?;
    v.extend(items);
    Ok(v)
}
