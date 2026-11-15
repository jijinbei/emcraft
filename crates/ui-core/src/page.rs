//! The settings' sections and the app bar's tabs, as the window's handles and recorded actions
//! name them (what covers the project's pages is [`crate::window::Cover`]).

emcraft_macros::named_enum! {
    /// A section of the settings page.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum Section {
        #[default]
        Appearance = "appearance" => "Appearance",
        Language = "language" => "Language",
        Units = "units" => "Units",
        View3d = "3d" => "3D view",
        Dialogs = "dialogs" => "Dialogs",
        Files = "files" => "Files",
    }
}
emcraft_macros::named_serde!(Section);

emcraft_macros::named_enum! {
    /// A page of the window, a tab of the app bar.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub enum Tab {
        /// The start page: new projects, recent ones, templates.
        Start = "start" => "Start",
        /// The open project's workspace (its tab reads the project's name).
        #[default]
        Project = "project" => "Project",
        /// The settings, while they are open.
        Settings = "settings" => "Settings",
    }
}
emcraft_macros::named_serde!(Tab);
