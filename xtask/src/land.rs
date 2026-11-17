//! `cargo xtask land [--dry-run] [--oracle] PR...`: the maintainer's merge queue, run locally.
//! The pull requests are stacked in order on the latest `develop` (each rebased on the ones
//! before it), the stack is checked as CI's gate checks it (fmt, layers, style, clippy, the tests
//! of what changed, the oracle with `--oracle`), and only then is each rebased pull request pushed
//! back to its own branch and `develop` fast-forwarded to the stack — so GitHub sees every one as
//! merged and closes the issues it names. A pull request that does not rebase cleanly, or a check
//! that fails, stops it before anything is pushed. `--dry-run` stops after the checks.
//!
//! Pushing to a contributor's fork needs their pull request's "Allow edits by maintainers"
//! (GitHub's default); one without it is refused up front.

use std::path::Path;
use std::process::Command;

/// The trunk pull requests are landed on.
const TRUNK: &str = "develop";
/// The branch the stack is built on.
const STACK: &str = "xtask-land";

/// What the command line asks.
#[derive(Debug, PartialEq)]
struct Ask {
    prs: Vec<u32>,
    dry_run: bool,
    oracle: bool,
}

fn parse(args: &[&str]) -> Result<Ask, String> {
    let mut ask = Ask { prs: Vec::new(), dry_run: false, oracle: false };
    for a in args {
        match *a {
            "--dry-run" => ask.dry_run = true,
            "--oracle" => ask.oracle = true,
            n => match n.trim_start_matches('#').parse::<u32>() {
                Ok(n) if !ask.prs.contains(&n) => ask.prs.push(n),
                Ok(n) => return Err(format!("pull request #{n} is named twice")),
                Err(_) => return Err(format!("`{n}` is not a pull request number (usage: cargo xtask land [--dry-run] [--oracle] PR...)")),
            },
        }
    }
    if ask.prs.is_empty() {
        return Err("name the pull requests to land: cargo xtask land [--dry-run] [--oracle] PR...".into());
    }
    Ok(ask)
}

/// A pull request as GitHub describes it.
#[derive(Debug)]
struct Pull {
    number: u32,
    title: String,
    /// Its branch and the commit at its head when fetched.
    head: String,
    head_oid: String,
    /// Where its branch lives: `origin`, or a contributor's fork's URL.
    remote: String,
}

/// Reads pull request `n` and refuses one that cannot land: closed, aimed elsewhere than the
/// trunk, or from a fork the maintainer may not push to.
fn pull(root: &Path, n: u32) -> Result<Pull, String> {
    let fields = "number,title,state,baseRefName,headRefName,headRefOid,isCrossRepository,maintainerCanModify,headRepository,headRepositoryOwner";
    let out = Command::new("gh")
        .args(["pr", "view", &n.to_string(), "--json", fields])
        .current_dir(root)
        .output()
        .map_err(|e| format!("gh (the GitHub CLI) is needed: {e}"))?;
    if !out.status.success() {
        return Err(format!("pull request #{n}: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).map_err(|e| format!("pull request #{n}: {e}"))?;
    let text = |k: &str| v[k].as_str().unwrap_or_default().to_owned();
    if text("state") != "OPEN" {
        return Err(format!("pull request #{n} is {}", text("state").to_lowercase()));
    }
    if text("baseRefName") != TRUNK {
        return Err(format!("pull request #{n} is aimed at `{}`, not `{TRUNK}`", text("baseRefName")));
    }
    let fork = v["isCrossRepository"].as_bool().unwrap_or(false);
    if fork && !v["maintainerCanModify"].as_bool().unwrap_or(false) {
        return Err(format!("pull request #{n} comes from a fork that does not allow edits by maintainers: ask for it, or merge it on GitHub"));
    }
    let remote = if fork {
        let owner = v["headRepositoryOwner"]["login"].as_str().unwrap_or_default();
        let repo = v["headRepository"]["name"].as_str().unwrap_or_default();
        format!("https://github.com/{owner}/{repo}.git")
    } else {
        "origin".to_owned()
    };
    Ok(Pull { number: n, title: text("title"), head: text("headRefName"), head_oid: text("headRefOid"), remote })
}

fn git(root: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git").args(args).current_dir(root).output().map_err(|e| format!("git: {e}"))?;
    if !out.status.success() {
        return Err(format!("git {}: {}", args.join(" "), String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

/// The local branch of pull request `n`'s rebased commits.
fn branch(n: u32) -> String {
    format!("{STACK}-{n}")
}

pub fn run(root: &Path, args: &[&str]) -> Result<(), String> {
    let ask = parse(args)?;
    if !git(root, &["status", "--porcelain", "--untracked-files=no"])?.is_empty() {
        return Err("commit or put aside the working tree's changes first".into());
    }
    let back = git(root, &["rev-parse", "--abbrev-ref", "HEAD"])?;
    let pulls = ask.prs.iter().map(|&n| pull(root, n)).collect::<Result<Vec<_>, _>>()?;
    git(root, &["fetch", "--quiet", "origin", TRUNK])?;
    let trunk = git(root, &["rev-parse", &format!("origin/{TRUNK}")])?;
    let stacked = stack(root, &pulls, &trunk);
    let checked = stacked.and_then(|()| check(root, &trunk, ask.oracle));
    if let Err(e) = checked {
        // The stack stays on its branch for a look; the checkout goes back to where it was.
        let _ = git(root, &["switch", "--quiet", &back]);
        return Err(format!("{e}\nnothing was pushed; the stack so far is on branch `{STACK}`"));
    }
    if ask.dry_run {
        git(root, &["switch", "--quiet", &back])?;
        eprintln!("land: {} pull request(s) stacked and checked on `{STACK}`; --dry-run pushed nothing", pulls.len());
        return Ok(());
    }
    // Each pull request's branch first, so the trunk's push finds their commits in it and GitHub
    // marks them merged; a branch moved since it was fetched is not overwritten.
    for p in &pulls {
        let lease = format!("--force-with-lease=refs/heads/{}:{}", p.head, p.head_oid);
        git(root, &["push", "--quiet", &lease, &p.remote, &format!("{}:refs/heads/{}", branch(p.number), p.head)])?;
    }
    git(root, &["push", "--quiet", "origin", &format!("{STACK}:refs/heads/{TRUNK}")])?;
    // The local trunk follows; the checkout goes back to where it was.
    git(root, &["switch", "--quiet", &back])?;
    if back == TRUNK {
        git(root, &["merge", "--quiet", "--ff-only", STACK])?;
    } else {
        git(root, &["branch", "--force", TRUNK, STACK])?;
    }
    for p in &pulls {
        git(root, &["branch", "--quiet", "-D", &branch(p.number)])?;
        eprintln!("land: #{} {}", p.number, p.title);
    }
    git(root, &["branch", "--quiet", "-D", STACK])?;
    eprintln!("land: {} pull request(s) on {TRUNK}", pulls.len());
    Ok(())
}

/// Stacks the pull requests on `trunk` in order, each rebased on the ones before it.
fn stack(root: &Path, pulls: &[Pull], trunk: &str) -> Result<(), String> {
    git(root, &["switch", "--quiet", "-C", STACK, trunk])?;
    for p in pulls {
        let fetched = format!("refs/land/pr-{}", p.number);
        git(root, &["fetch", "--quiet", "origin", &format!("+pull/{}/head:{fetched}", p.number)])?;
        let from = git(root, &["merge-base", &fetched, trunk])?;
        let mine = branch(p.number);
        git(root, &["switch", "--quiet", "-C", &mine, &fetched])?;
        if let Err(e) = git(root, &["rebase", "--quiet", "--onto", STACK, &from, &mine]) {
            let _ = git(root, &["rebase", "--abort"]);
            return Err(format!("pull request #{} ({}) does not rebase cleanly on {TRUNK} and the ones before it: {e}", p.number, p.title));
        }
        git(root, &["switch", "--quiet", STACK])?;
        git(root, &["merge", "--quiet", "--ff-only", &mine])?;
        eprintln!("land: stacked #{} {}", p.number, p.title);
    }
    Ok(())
}

/// The stack checked as CI's gate does: formatting, the machine-checked rules, clippy, the tests
/// of the packages changed since `trunk`, and the oracle when asked.
fn check(root: &Path, trunk: &str, oracle: bool) -> Result<(), String> {
    let mut fmt = crate::cargo();
    fmt.args(["fmt", "--all", "--check"]);
    crate::run(fmt, "fmt")?;
    crate::cmd_layers()?;
    crate::colours::run(root)?;
    crate::style::run(root)?;
    let mut clippy = crate::cargo();
    clippy.args(["clippy", "--workspace", "--all-targets", "--", "-D", "warnings"]);
    crate::run(clippy, "clippy")?;
    crate::cmd_test(&["--changed", trunk])?;
    if oracle {
        let mut o = crate::cargo();
        o.args(["run", "--release", "-p", "emcraft-testkit", "--bin", "oracle", "--", "--no-write"]);
        crate::run(o, "oracle")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_command_line_names_pull_requests_once_and_its_options() {
        assert_eq!(parse(&["12", "#15", "--dry-run"]), Ok(Ask { prs: vec![12, 15], dry_run: true, oracle: false }));
        assert_eq!(parse(&["--oracle", "3"]), Ok(Ask { prs: vec![3], dry_run: false, oracle: true }));
        for bad in [&[][..], &["--dry-run"], &["12", "12"], &["x"], &["-1"]] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
    }
}
