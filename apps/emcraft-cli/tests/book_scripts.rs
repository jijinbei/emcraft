//! The scripts the book shows (docs/book/scripts, included in its pages) run as they are
//! written: a page cannot show a script the commands no longer take. Those that solve for more
//! than a few seconds are in its `heavy` folder.

use std::path::Path;
use std::process::Command;

/// The design the SolidCraft tutorial's script imports, which its reader downloads beside it.
const DESIGN: &str = "tests/oracle/design-post/post.sc";

/// Runs every script of `folder` (under docs/book/scripts) with `emcraft-cli run`.
fn run_scripts(folder: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let scripts = root.join("docs/book/scripts").join(folder);
    // The scripts write their files (pictures, a Touchstone export, a project) where they run:
    // a folder of the test's own.
    let scratch = Path::new(env!("CARGO_TARGET_TMPDIR")).join("book-scripts").join(folder);
    std::fs::create_dir_all(&scratch).unwrap();
    std::fs::copy(root.join(DESIGN), scratch.join("post.sc")).unwrap();
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

#[test]
fn the_scripts_of_the_book_run() {
    run_scripts("");
}

#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn the_heavy_scripts_of_the_book_run() {
    run_scripts("heavy");
}
