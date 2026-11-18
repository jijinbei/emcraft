//! What the File menu, the file dialogs and the questions about unsaved work ask of the window.

/// What the File menu, the start page, the file dialog, the path prompt and the question
/// before changes are lost ask for.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum FileAction {
    /// New, open, a recent project or a sample, asking first when there are unsaved changes.
    Request(Then),
    Save,
    SaveAs,
    /// A path chosen in the file dialog or the prompt (`None`: cancelled).
    Chosen(Purpose, Option<String>),
    /// The answer to "save the changes?".
    Confirm(Answer),
    /// The import dialog's file, chosen in the file dialog (or the path prompt without one).
    PickPart,
    /// A file dropped on the window: a part to import, anything else a project to open.
    Dropped(String),
    /// A new SolidCraft design: where to save it asked, then it opens in SolidCraft and is
    /// imported once saved there (docs/adr/0010-geometry-from-solidcraft.md).
    NewDesign,
}

/// What a path asked for is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Purpose {
    Open,
    SaveAs,
    /// A part for the import dialog.
    Import,
    /// Where a new SolidCraft design is saved.
    NewDesign,
}

/// What to do once the unsaved changes are dealt with.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Then {
    New,
    /// A blank project for a kind of analysis (the start page's Blank, a purpose chosen).
    Blank(emcraft_engine::workflow::Analysis),
    Open,
    OpenPath(String),
    /// A template, its parameters changed to `values`, for `purpose` (the kind of analysis).
    Sample {
        name: String,
        values: serde_json::Value,
        purpose: Option<emcraft_engine::workflow::Analysis>,
    },
    /// An autosaved copy recovered: `file.open` of the copy, the project's own file `original`.
    Recover {
        file: String,
        original: Option<String>,
    },
    /// A new project with the import dialog open in step 1 (the start page's Import).
    Import,
    /// The window closes.
    Quit,
}

/// An answer to "save the changes?".
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Answer {
    Save,
    Discard,
    Cancel,
    /// A job runs when the window would close: stop it and carry on.
    StopJob,
}

impl FileAction {
    /// Whether it waits while a job runs (`Action::waits_for_job`): what opens,
    /// begins or saves a project. Closing asks about the job instead, an answer or a path
    /// chosen goes on (the commands it runs refuse themselves).
    pub fn waits_for_job(&self) -> bool {
        match self {
            FileAction::Request(then) => *then != Then::Quit,
            FileAction::Save | FileAction::SaveAs | FileAction::PickPart | FileAction::Dropped(_) | FileAction::NewDesign => true,
            FileAction::Chosen(..) | FileAction::Confirm(_) => false,
        }
    }
}

/// The path field shown when there is no file dialog.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prompt {
    pub purpose: Purpose,
    pub text: String,
}

/// The one question over the window, which takes the keyboard while it is asked.
#[derive(Clone, Debug, PartialEq)]
pub enum Ask {
    /// A file's path, typed (there is no file dialog).
    Path(Prompt),
    /// "Save the changes first?" before going on with `Then`.
    Save(Then),
    /// Where the history goes (File › Export history), as typed.
    Export(String),
}

/// A file dialog, asked without blocking: `done` is called with the path chosen, `None` when
/// cancelled, from any thread.
pub trait Picker: Send + Sync {
    /// A project to open.
    fn open(&self, done: Box<dyn FnOnce(Option<String>) + Send>);
    /// Where to save a project, `suggested` the file name offered.
    fn save(&self, suggested: &str, done: Box<dyn FnOnce(Option<String>) + Send>);
    /// A part to import: a SolidCraft design, a 3MF package or an STL file; or a board (KiCad,
    /// a packed PCBCraft project).
    fn import(&self, done: Box<dyn FnOnce(Option<String>) + Send>);
    /// Where to make a new SolidCraft design, `suggested` the file name offered.
    fn design(&self, suggested: &str, done: Box<dyn FnOnce(Option<String>) + Send>);
}

/// The extensions of the parts `geometry.import` reads, and of the boards `pcb.import` reads.
pub const PART_EXTENSIONS: &[&str] = &["sc", "3mf", "stl", "kicad_pcb", "pc"];

/// The extensions of the boards among them: a KiCad board, a packed PCBCraft project.
const BOARD_EXTENSIONS: &[&str] = &["kicad_pcb", "pc"];

/// Whether `path` names a board `pcb.import` reads rather than a part, by its extension.
///
/// ```
/// use emcraft_ui_core::files::is_board;
/// assert!(is_board("/tmp/route.kicad_pcb") && is_board("/tmp/sensor-node.PC"));
/// assert!(!is_board("/tmp/post.sc"));
/// ```
pub fn is_board(path: &str) -> bool {
    let ext = std::path::Path::new(path).extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
    ext.is_some_and(|e| BOARD_EXTENSIONS.contains(&e.as_str()))
}

/// The file name a new SolidCraft design is offered under: `.sc`, what SolidCraft writes.
pub const NEW_DESIGN: &str = "design.sc";

/// Where a new design is written: `path`, `.sc` added unless it ends so (any case).
///
/// ```
/// use emcraft_ui_core::files::{NEW_DESIGN, design_path, is_part};
/// assert_eq!(design_path("/tmp/pillbox"), "/tmp/pillbox.sc");
/// assert_eq!(design_path("/tmp/pillbox.SC"), "/tmp/pillbox.SC");
/// assert_eq!(design_path(NEW_DESIGN), NEW_DESIGN);
/// assert!(is_part(NEW_DESIGN));
/// ```
pub fn design_path(path: &str) -> String {
    let named = std::path::Path::new(path).extension().is_some_and(|e| e.eq_ignore_ascii_case("sc"));
    if named { path.to_owned() } else { format!("{path}.sc") }
}

/// Whether `path` names a part to import, by its extension.
pub fn is_part(path: &str) -> bool {
    let ext = std::path::Path::new(path).extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
    ext.is_some_and(|e| PART_EXTENSIONS.contains(&e.as_str()))
}
