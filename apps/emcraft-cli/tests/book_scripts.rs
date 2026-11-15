//! The scripts the book shows (docs/book/scripts, included in its pages) run as they are
//! written: a page cannot show a script the commands no longer take.

use std::path::Path;
use std::process::Command;

#[test]
fn the_scripts_of_the_book_run() {
    let scripts = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/book/scripts");
    // The scripts write their files (a Touchstone export) beside nothing of ours: a folder of
    // the test's own.
    let scratch = Path::new(env!("CARGO_TARGET_TMPDIR")).join("book-scripts");
    std::fs::create_dir_all(&scratch).unwrap();
    let mut ran = 0;
    for entry in std::fs::read_dir(&scripts).unwrap() {
        let script = entry.unwrap().path();
        if script.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let out = Command::new(env!("CARGO_BIN_EXE_emcraft-cli")).arg("run").arg(&script).current_dir(&scratch).output().unwrap();
        assert!(out.status.success(), "{}: {}", script.display(), String::from_utf8_lossy(&out.stderr));
        ran += 1;
    }
    assert!(ran > 0, "no scripts in {}", scripts.display());
}
