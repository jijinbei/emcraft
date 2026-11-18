//! `emcraft-cli --version` prints the version a report or a run's record cites, and in a git
//! checkout the commit it was built from.

use std::process::Command;

#[test]
fn the_version_is_printed_and_nothing_else_runs() {
    for flag in ["--version", "-V", "version"] {
        let out = Command::new(env!("CARGO_BIN_EXE_emcraft-cli")).arg(flag).output().unwrap();
        assert!(out.status.success(), "{flag}");
        let line = String::from_utf8(out.stdout).unwrap();
        let line = line.trim();
        let rest = line.strip_prefix(&format!("emcraft-cli {}", env!("CARGO_PKG_VERSION"))).unwrap_or_else(|| panic!("{line}"));
        // A checkout's build names its commit: `(v0.1.0-25-gc0d75490)` or, before any tag, the
        // commit alone.
        let commit = std::process::Command::new("git").args(["rev-parse", "--short=8", "HEAD"]).output().ok().filter(|o| o.status.success());
        match commit {
            Some(c) => assert!(rest.starts_with(" (") && rest.contains(String::from_utf8_lossy(&c.stdout).trim()), "{line}"),
            None => assert_eq!(rest, "", "{line}"),
        }
    }
}
