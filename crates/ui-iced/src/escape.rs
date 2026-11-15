//! Escape on the iced window: the shared rules ([`emcraft_ui_core::window::Open`]) pick the
//! layer it closes from what the window has open; here, the message that closes it. A grip being
//! dragged lets go first.

use emcraft_ui_core::files::{Answer, FileAction};
use emcraft_ui_core::layer::Layer;
use emcraft_ui_core::window::{DialogOpen, Open};

use crate::app::{App, Message};

/// What is open in the window now, as [`emcraft_ui_core::window::Open`] weighs it.
pub fn open(app: &App) -> Open {
    Open {
        cover: app.cover,
        ask: app.asking.is_some(),
        template: app.template.card.is_some(),
        search: !app.search.is_empty(),
        dialog: app.dialog.as_ref().map(|d| if d.item.is_some() { DialogOpen::ItemForm } else { DialogOpen::Command }),
        tool: app.tool.is_some(),
        selection: !app.selection.is_clear(),
        reports: app.crashes.shown.is_some(),
    }
}

/// The layers open now, top first.
pub fn stack(app: &App) -> Vec<Layer> {
    open(app).stack()
}

/// The layer Escape closes ([`Open::escape_target`]).
pub fn target(app: &App, captured: bool) -> Option<Layer> {
    open(app).escape_target(captured)
}

/// What closes layer `l` (the tool and the selection are the window's own, closed there).
pub fn close(l: Layer) -> Option<Message> {
    match l {
        Layer::Ask => Some(Message::File(FileAction::Confirm(Answer::Cancel))),
        Layer::Template => Some(Message::Template(crate::template_dialog::TemplateAct::Open(None))),
        Layer::Search => Some(Message::Search(String::new())),
        Layer::Dialog => Some(Message::Dialog(crate::dialog::Edit::Cancel)),
        Layer::Settings => Some(Message::Settings(false)),
        Layer::Reports => Some(Message::Reports(crate::record::ReportAct::Show(false))),
        Layer::Tool | Layer::Selection | Layer::Pen | Layer::Box => None,
    }
}
