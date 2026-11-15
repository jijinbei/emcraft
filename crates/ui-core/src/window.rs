//! What covers the project's pages, and which layer Escape closes (docs/adr/0007-iced-window.md:
//! the window's rules are ui-core's; the iced window only lays them out and sends the message
//! that closes a layer).

use crate::layer::Layer;

/// What fills the window over the project's pages: nothing, the start page, or the settings —
/// one at a time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Cover {
    /// The project's pages are seen.
    #[default]
    Project,
    Start,
    Settings,
}

impl Cover {
    /// Whether the project's pages are seen.
    pub fn project(self) -> bool {
        self == Cover::Project
    }

    pub fn start(self) -> bool {
        self == Cover::Start
    }

    pub fn settings(self) -> bool {
        self == Cover::Settings
    }

    /// The start page shown (`on`), which takes the settings' place, or left.
    ///
    /// ```
    /// use emcraft_ui_core::window::Cover;
    /// assert_eq!(Cover::Settings.with_start(true), Cover::Start);
    /// assert_eq!(Cover::Start.with_start(false), Cover::Project);
    /// // Leaving the start page leaves the settings shown.
    /// assert_eq!(Cover::Settings.with_start(false), Cover::Settings);
    /// ```
    #[must_use]
    pub fn with_start(self, on: bool) -> Cover {
        match (on, self) {
            (true, _) => Cover::Start,
            (false, Cover::Start) => Cover::Project,
            (false, other) => other,
        }
    }

    /// The settings shown (`on`), which take the start page's place, or left.
    #[must_use]
    pub fn with_settings(self, on: bool) -> Cover {
        match (on, self) {
            (true, _) => Cover::Settings,
            (false, Cover::Settings) => Cover::Project,
            (false, other) => other,
        }
    }
}

/// A dialog open: a command's, or an item's form (which waits unseen off the project's pages).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DialogOpen {
    Command,
    ItemForm,
}

/// What is open in the window now, as Escape weighs it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Open {
    pub cover: Cover,
    pub ask: bool,
    pub template: bool,
    pub search: bool,
    pub dialog: Option<DialogOpen>,
    pub tool: bool,
    pub selection: bool,
    pub reports: bool,
}

impl Open {
    /// The layers open, top first ([`Layer::ALL`]'s order). The tool and the selection are the
    /// project's pages' own, so not while something covers them; the pen and the box being
    /// dragged are the 3D view's (let go there).
    ///
    /// ```
    /// use emcraft_ui_core::layer::Layer;
    /// use emcraft_ui_core::window::{Cover, DialogOpen, Open};
    /// let open = Open { cover: Cover::Settings, dialog: Some(DialogOpen::Command), selection: true, ..Open::default() };
    /// assert_eq!(open.stack(), [Layer::Dialog, Layer::Settings]);
    /// ```
    pub fn stack(&self) -> Vec<Layer> {
        let project = self.cover.project();
        let open = |l: Layer| match l {
            Layer::Ask => self.ask,
            Layer::Template => self.template,
            Layer::Search => self.search,
            Layer::Dialog => match self.dialog {
                Some(DialogOpen::Command) => true,
                Some(DialogOpen::ItemForm) => project,
                None => false,
            },
            Layer::Settings => self.cover.settings(),
            Layer::Tool => project && self.tool,
            Layer::Selection => project && self.selection,
            Layer::Reports => self.reports,
            Layer::Pen | Layer::Box => false,
        };
        Layer::ALL.into_iter().filter(|l| open(*l)).collect()
    }

    /// The layer Escape closes: the top one, unless a field took the key (`captured`) and the top
    /// layer is not one whose own fields it holds (a question, the search, a card or a dialog over
    /// the window: what is under them takes no keys).
    ///
    /// ```
    /// use emcraft_ui_core::layer::Layer;
    /// use emcraft_ui_core::window::{Cover, Open};
    /// let open = Open { cover: Cover::Settings, ..Open::default() };
    /// assert_eq!(open.escape_target(false), Some(Layer::Settings));
    /// // A field of the settings has the key: Escape only lets go of it.
    /// assert_eq!(open.escape_target(true), None);
    /// ```
    pub fn escape_target(&self, captured: bool) -> Option<Layer> {
        let top = *self.stack().first()?;
        let owned = matches!(top, Layer::Ask | Layer::Search | Layer::Template | Layer::Reports | Layer::Dialog);
        (!captured || owned).then_some(top)
    }
}
