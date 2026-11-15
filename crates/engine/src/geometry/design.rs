//! The SolidCraft designs a project imports (ADR 0010): which they are, their parameters, and
//! the model built again with one of them changed (`geometry.design`) or every file read again
//! (`geometry.reload`).

use serde_json::{Value, json};

use super::{DesignParameter, MAX_IMPORT_BYTES, read_capped};
use crate::{Params, Result, Session};

/// A SolidCraft design the project imports (ADR 0010): its file as the import names it, the
/// parameters the import sets (name → the number or expression given) and the design's own
/// parameters (or why it cannot be read now).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct ImportedDesign {
    pub path: String,
    /// A template's design (`template:<file>`), compiled in: set by the template's parameters.
    pub template: bool,
    pub set: serde_json::Map<String, Value>,
    pub parameters: std::result::Result<Vec<DesignParameter>, String>,
}

/// The SolidCraft designs `s`'s history imports, each once, in the order first imported, with
/// the parameters its last import sets.
pub fn designs(s: &Session) -> Vec<ImportedDesign> {
    let imports: Vec<(&str, serde_json::Map<String, Value>)> = s
        .history()
        .iter()
        .filter(|st| st.command == "geometry.import")
        .filter_map(|st| {
            let path = st.param("path").and_then(Value::as_str).filter(|p| emcraft_io::solidcraft::is_design(p))?;
            Some((path, st.param("parameters").and_then(Value::as_object).cloned().unwrap_or_default()))
        })
        .collect();
    let mut paths: Vec<&str> = imports.iter().map(|(path, _)| *path).collect();
    // Each once, where first imported.
    let mut seen = std::collections::BTreeSet::new();
    paths.retain(|p| seen.insert(*p));
    paths
        .into_iter()
        .map(|path| ImportedDesign {
            path: path.to_owned(),
            template: path.starts_with(emcraft_templates::DESIGN_SCHEME),
            set: imports.iter().rev().find(|(p, _)| *p == path).map_or_default(|(_, set)| set.clone()),
            parameters: design_parameters(path).map_err(|e| e.to_string()),
        })
        .collect()
}

/// The parameters of the design at `path` (a template's, compiled in, or a file read now).
fn design_parameters(path: &str) -> Result<Vec<DesignParameter>> {
    let bytes = match emcraft_templates::design(path) {
        Some(b) => std::borrow::Cow::Borrowed(b),
        None => std::borrow::Cow::Owned(read_capped(path, MAX_IMPORT_BYTES)?),
    };
    Ok(emcraft_io::solidcraft::parameters(&bytes)?.into_iter().map(DesignParameter::from).collect())
}

/// The session built again from `history`, every imported file read again; the history's own
/// state (undo, the file, the runs) kept.
fn build_again(s: &mut Session, history: &[crate::Step]) -> Result<()> {
    let rebuilt = crate::study::variables::rebuild(s, history)?;
    crate::study::variables::adopt(s, rebuilt);
    Ok(())
}

/// `geometry.reload`: the model built again from its history, every imported file read again (a
/// design saved in SolidCraft since); the history, the runs and the file kept.
pub fn reload(s: &mut Session, _: &Params) -> Result<Value> {
    let history = s.history.to_vec();
    build_again(s, &history)?;
    Ok(json!({"designs": designs(s).len()}))
}

/// `geometry.design`: sets parameter `name` of the imported design `design` to `value` where it
/// is imported (a number in the unit the design shows it in, or an expression), the model built
/// again with it.
pub fn design(s: &mut Session, p: &Params) -> Result<Value> {
    let path = p.string("design").required()?;
    let name = p.string("name").required()?;
    let value = design_value(p)?;
    let all = designs(s);
    let known = all.iter().map(|d| d.path.clone()).collect::<Vec<_>>();
    let found = all.into_iter().find(|d| d.path == path).ok_or_else(|| {
        p.error(
            "design",
            crate::text::unknown(format_args!("names no design the project imports: `{}`", crate::text::clip(path, 80)), "designs: ", known),
        )
    })?;
    let parameters = found.parameters.map_err(|e| p.error("design", e))?;
    if !parameters.iter().any(|q| q.name == name) {
        let known = parameters.iter().map(|q| q.name.clone());
        return Err(p.error(
            "name",
            crate::text::unknown(format_args!("names no parameter of the design: `{}`", crate::text::clip(name, 64)), "parameters: ", known),
        ));
    }
    build_again(s, &setting(s.history(), path, name, &value))?;
    Ok(json!({"design": path, "name": name, "value": value}))
}

/// `value`, as `geometry.design` takes it: a finite number, else the expression as given.
fn design_value(p: &Params) -> Result<Value> {
    let given = p.string("value").required()?.trim();
    if given.is_empty() || given.chars().count() > crate::expr::MAX_LEN + 1 {
        return Err(p.error("value", format!("give a number or an expression of at most {} characters", crate::expr::MAX_LEN)));
    }
    Ok(given.parse::<f64>().ok().filter(|x| x.is_finite()).map_or_else(|| json!(given), |x| json!(x)))
}

/// `history` with every import of the design at `path` setting its parameter `name` to `value`.
fn setting(history: &[crate::Step], path: &str, name: &str, value: &Value) -> Vec<crate::Step> {
    history
        .iter()
        .map(|st| {
            let imports = st.command == "geometry.import" && st.param("path").and_then(Value::as_str) == Some(path);
            let mut st = st.clone();
            if imports && let Some(map) = st.params.as_object_mut() {
                let set = map.entry("parameters").or_insert_with(|| json!({}));
                if let Some(set) = set.as_object_mut() {
                    set.insert(name.to_owned(), value.clone());
                }
            }
            st
        })
        .collect()
}
