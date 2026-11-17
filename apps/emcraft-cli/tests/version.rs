//! `emcraft-cli --version` prints the version a report or a run's record cites.

use std::process::Command;

#[test]
fn the_version_is_printed_and_nothing_else_runs() {
    for flag in ["--version", "-V", "version"] {
        let out = Command::new(env!("CARGO_BIN_EXE_emcraft-cli")).arg(flag).output().unwrap();
        assert!(out.status.success(), "{flag}");
        assert_eq!(String::from_utf8(out.stdout).unwrap().trim(), format!("emcraft-cli {}", env!("CARGO_PKG_VERSION")));
    }
}
