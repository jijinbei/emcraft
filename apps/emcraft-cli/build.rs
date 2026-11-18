//! The commit `emcraft-cli --version` names after its version (`git describe`: the release tag,
//! the commits since it and the commit, `v0.1.0-25-gc0d75490`), so a run's record says which
//! build solved it; none outside a git checkout (a source archive). Built again when the
//! checkout moves to another commit.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    (out.status.success() && !text.trim().is_empty()).then(|| text.trim().to_owned())
}

fn main() {
    let commit = git(&["describe", "--tags", "--always", "--abbrev=8"]).unwrap_or_default();
    println!("cargo:rustc-env=EMCRAFT_COMMIT={commit}");
    // HEAD moves to another branch; the branch's ref (or the packed refs) to another commit.
    let refs = ["HEAD".to_owned(), git(&["symbolic-ref", "-q", "HEAD"]).unwrap_or_default(), "packed-refs".to_owned()];
    for name in refs.iter().filter(|r| !r.is_empty()) {
        if let Some(path) = git(&["rev-parse", "--path-format=absolute", "--git-path", name]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
}
