//! The EMCraft desktop app: `emcraft [PROJECT.emcraft] [--sample NAME] [--control PORT]
//! [--replay REPORT] [--shot PNG]`. A project file or a sample is opened; with neither the start
//! page shows. `--control` listens for an agent's JSON-line requests on 127.0.0.1:PORT
//! (docs/control-protocol.md; `EMCRAFT_CONTROL_PORT` too); `--replay` plays back a recording or
//! a crash report's folder; `--shot` writes a picture of the window once it has drawn, then
//! closes it. `--help` and `--version` print and exit.
//!
//! On Windows the release build is a window program with no console (a file opened from
//! Explorer opens no terminal); what it prints shows only when its output is redirected.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let env_port = std::env::var("EMCRAFT_CONTROL_PORT").ok().and_then(|p| p.parse().ok());
    let Cli { sample, project, replay, control, shot } = match parse_args(&args, env_port) {
        Parsed::Run(cli) => cli,
        Parsed::Print(text) => {
            println!("{text}");
            return ExitCode::SUCCESS;
        }
        Parsed::Usage => {
            eprintln!("{}", usage_text());
            return ExitCode::FAILURE;
        }
    };
    // The control channel; its token file lives as long as the app (the window looks at its
    // requests every frame).
    let (requests, token_file) = match control.map(emcraft_control::server::start) {
        Some(Ok(started)) => {
            eprintln!("emcraft: {}", started.note);
            (Some(started.requests), started.token_file)
        }
        Some(Err(e)) => {
            eprintln!("emcraft: control channel not started: {e}");
            (None, None)
        }
        None => (None, None),
    };
    if let Some(why) = no_display(|name| std::env::var(name).ok()) {
        eprintln!("error: {why}");
        return ExitCode::FAILURE;
    }
    let picker: std::sync::Arc<dyn emcraft_ui_core::files::Picker> = std::sync::Arc::new(emcraft_picker::SystemPicker);
    let launch = emcraft_ui_iced::Launch { sample, project, shoot: shot, replay, picker: Some(picker), control: requests };
    // The window's toolkit panics when it cannot make its event loop (a display named but not
    // running): caught here, before any of EMCraft's work, and told as an error.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| emcraft_ui_iced::run(launch)))
        .map_err(|_| "the window could not open (see the message above): is the display server running? emcraft-cli works without one".to_owned())
        .and_then(|run| run.map_err(|e| e.to_string()));
    drop(token_file);
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            if control.is_none() {
                show_start_error(&e);
            }
            ExitCode::FAILURE
        }
    }
}

/// Why no window can open here, when no display is named at all: on Linux and the BSDs the
/// window needs a Wayland compositor (`WAYLAND_DISPLAY`) or an X server (`DISPLAY`). `var` reads
/// the environment. Elsewhere (Windows, macOS) the system always has one.
fn no_display(var: impl Fn(&str) -> Option<String>) -> Option<String> {
    if !cfg!(all(unix, not(target_os = "macos"))) {
        return None;
    }
    let named = |name: &str| var(name).is_some_and(|v| !v.trim().is_empty());
    (!named("WAYLAND_DISPLAY") && !named("DISPLAY")).then(|| {
        "no display to open the window on: neither WAYLAND_DISPLAY nor DISPLAY is set. Run EMCraft in a desktop session, or use emcraft-cli (the same commands, no window)".to_owned()
    })
}

/// The command line's choices: a sample or a project file to open, a crash report or recording
/// to replay, the control channel's port, a picture to take.
#[derive(Debug, Default, PartialEq)]
struct Cli {
    sample: Option<String>,
    project: Option<String>,
    replay: Option<std::path::PathBuf>,
    control: Option<u16>,
    shot: Option<std::path::PathBuf>,
}

/// What the command line asks: run with its choices, print a text and stop (help, version),
/// or it is wrong (print the usage and fail).
#[derive(Debug, PartialEq)]
enum Parsed {
    Run(Cli),
    Print(String),
    Usage,
}

/// The usage line with the samples' names.
fn usage_text() -> String {
    format!(
        "usage: emcraft [PROJECT.emcraft] [--sample NAME] [--control PORT] [--replay REPORT] [--shot PNG]   (samples: {})",
        emcraft_engine::templates::names().join(", ")
    )
}

/// Reads the arguments (without the program's name); `control` is the port from the
/// environment, which `--control` overrides.
fn parse_args(args: &[String], control: Option<u16>) -> Parsed {
    let mut cli = Cli { control, ..Cli::default() };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => return Parsed::Print(usage_text()),
            "-V" | "--version" => return Parsed::Print(format!("emcraft {}", env!("CARGO_PKG_VERSION"))),
            "--sample" => match it.next() {
                Some(name) if emcraft_engine::templates::find(name).is_some() => cli.sample = Some(name.clone()),
                _ => return Parsed::Usage,
            },
            // A crash report's folder (or a recording): played back in the window.
            "--replay" => match it.next() {
                Some(path) => cli.replay = Some(std::path::PathBuf::from(path)),
                None => return Parsed::Usage,
            },
            "--shot" => match it.next() {
                Some(path) => cli.shot = Some(std::path::PathBuf::from(path)),
                None => return Parsed::Usage,
            },
            "--control" => match it.next().map(|p| p.parse::<u16>()) {
                Some(Ok(p)) => cli.control = Some(p),
                _ => return Parsed::Usage,
            },
            path if !path.starts_with('-') && cli.project.is_none() => cli.project = Some(path.to_owned()),
            _ => return Parsed::Usage,
        }
    }
    Parsed::Run(cli)
}

/// The window could not open (typically no graphics driver: a virtual machine, a remote
/// desktop). A window program on Windows has no console for the error, so it is shown in a
/// message box; elsewhere the terminal shows it. Not with `--control`: an agent reads stderr.
#[cfg(all(windows, not(debug_assertions)))]
fn show_start_error(error: &str) {
    rfd::MessageDialog::new()
        .set_level(rfd::MessageLevel::Error)
        .set_title("EMCraft")
        .set_description(format!("EMCraft could not open its window.\n\n{error}\n\nInstall or update the graphics driver."))
        .set_buttons(rfd::MessageButtons::Ok)
        .show();
}

#[cfg(not(all(windows, not(debug_assertions))))]
fn show_start_error(_error: &str) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str], env: Option<u16>) -> Parsed {
        parse_args(&args.iter().map(ToString::to_string).collect::<Vec<_>>(), env)
    }

    #[test]
    fn arguments_choose_what_opens_and_bad_ones_give_the_usage() {
        assert_eq!(parse(&[], Some(7)), Parsed::Run(Cli { control: Some(7), ..Cli::default() }));
        let Parsed::Run(cli) = parse(&["model.emcraft", "--sample", "highpass", "--control", "9000", "--shot", "a.png"], Some(7)) else {
            panic!("not run")
        };
        let want = Cli {
            sample: Some("highpass".into()),
            project: Some("model.emcraft".into()),
            control: Some(9000),
            shot: Some("a.png".into()),
            replay: None,
        };
        assert_eq!(cli, want);
        assert!(matches!(parse(&["--version"], None), Parsed::Print(v) if v.starts_with("emcraft ")));
        assert!(matches!(parse(&["--help"], None), Parsed::Print(u) if u.starts_with("usage:")));
        for bad in [&["--sample", "no-such"][..], &["--control", "x"], &["--control"], &["a", "b"], &["--what"], &["--replay"], &["--shot"]] {
            assert_eq!(parse(bad, None), Parsed::Usage, "{bad:?}");
        }
    }

    #[test]
    fn without_a_display_the_app_says_so_instead_of_panicking() {
        // Issue #8: no WAYLAND_DISPLAY and no DISPLAY (or blank ones) is refused up front.
        let env = |pairs: &'static [(&'static str, &'static str)]| move |name: &str| pairs.iter().find(|p| p.0 == name).map(|p| p.1.to_owned());
        let refused = no_display(env(&[]));
        let blank = no_display(env(&[("DISPLAY", " "), ("WAYLAND_DISPLAY", "")]));
        if cfg!(all(unix, not(target_os = "macos"))) {
            assert!(refused.as_deref().is_some_and(|w| w.contains("DISPLAY") && w.contains("emcraft-cli")), "{refused:?}");
            assert!(blank.is_some());
        } else {
            assert_eq!((refused, blank), (None, None));
        }
        assert_eq!(no_display(env(&[("DISPLAY", ":0")])), None);
        assert_eq!(no_display(env(&[("WAYLAND_DISPLAY", "wayland-0")])), None);
    }
}
