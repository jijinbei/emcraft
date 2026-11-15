//! The names dialogs show for command parameters and for their choices' values, translated like
//! every other text; the parameter ids and values stay what commands, scripts and automation use.
//! A parameter's own name is its command's (`ParamSpec::label`, checked against the
//! translations when the engine compiles).

/// The label of parameter `name` in the dialog of `command` (its command's; its name for a
/// command or parameter there is not).
///
/// ```
/// use emcraft_ui_core::param_labels::param_label;
/// assert_eq!(param_label("mesh.generate", "size"), "Element size");
/// assert_eq!(param_label("no.such", "size"), "size");
/// ```
pub fn param_label(command: &str, name: &'static str) -> &'static str {
    emcraft_engine::find_command(command).and_then(|c| c.params.iter().find(|p| p.name == name)).map_or(name, |p| p.label)
}

/// The words a choice's values show as (the value itself is what the command takes): the ends,
/// solvers, shapes and kinds commands share. Names (materials, ports, bodies) show as they are.
pub const VALUE_LABELS: &[(&str, &str)] = &[
    ("start", t!("Start (low side)").en()),
    ("end", t!("End (high side)").en()),
    ("auto", t!("Automatic").en()),
    ("numeric", t!("Numeric").en()),
    ("modal", t!("First mode only").en()),
    ("metal", t!("Metal").en()),
    ("interior", t!("Interior (the region)").en()),
    ("box", t!("Box").en()),
    ("cylinder", t!("Cylinder").en()),
    ("sphere", t!("Sphere").en()),
    ("prism", t!("Prism").en()),
    ("revolved", t!("Revolved").en()),
    ("cavity", t!("Air the metal encloses").en()),
    ("add", t!("Add").en()),
    ("subtract", t!("Subtract").en()),
    ("intersect", t!("Intersect").en()),
    ("electric", t!("Electric wall").en()),
    ("magnetic", t!("Magnetic wall").en()),
    ("enclosed", t!("Every enclosed piece").en()),
    ("largest", t!("The largest piece").en()),
    ("linear", t!("Linear").en()),
    ("log", t!("Logarithmic").en()),
    ("single", t!("Single").en()),
    ("double", t!("Double").en()),
    ("fast", t!("Fast").en()),
    ("full", t!("Full").en()),
];

/// What a choice's value shows as.
pub fn value_label(v: &str) -> Option<&'static str> {
    VALUE_LABELS.iter().find(|(id, _)| *id == v).map(|(_, l)| *l)
}
