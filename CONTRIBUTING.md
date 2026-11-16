# Contributing to EMCraft

EMCraft is an open-source electromagnetic field simulator written in pure Rust: model a microwave
component (or open a SolidCraft part), mesh it, solve Maxwell's equations with our own finite
element solvers and look at verified S-parameters and fields — from the desktop app, the command
line, or an AI agent over the control channel.

Help is welcome, whether that's a one-line fix, a verification benchmark, a file format or owning a
whole solver. This page covers how to build it, the few rules that aren't negotiable, where help is
needed most, and how to claim work. The rules every change follows, by a person or an agent, are in
[AGENTS.md](AGENTS.md); how the code is written — functional style first, names, small functions,
docs, errors, tests — is in [docs/style.md](docs/style.md). Read both before your first change.

## Get it running

You need stable Rust (see `rust-version` in [Cargo.toml](Cargo.toml)).

```sh
git clone https://github.com/jijinbei/emcraft && cd emcraft
cargo run --release -p emcraft -- --sample highpass        # the desktop app on a sample model
cargo run -p emcraft-cli -- commands                       # every command and its parameters
cargo test -p emcraft-solver                               # the tests of one crate
cargo xtask oracle --no-write                              # the verification cases (docs/oracle.md)
```

`develop` is the trunk and `main` holds released code. The layers and crates are in
[docs/architecture.md](docs/architecture.md); every user-visible capability is a command in
`crates/engine/src/cmd/table.rs` ([docs/features.md](docs/features.md) lists them).

## Where help is wanted most

See the [`help wanted`](https://github.com/jijinbei/emcraft/labels/help%20wanted) and
[`good first issue`](https://github.com/jijinbei/emcraft/labels/good%20first%20issue) labels, and
the `area:` labels for your field. Broadly, in order of how much they would help users:

1. **Verification benchmarks.** An engineer with a published benchmark or a measurement (a filter,
   an antenna, a cavity) can add it as an oracle case in `tests/oracle/` without touching solver
   code. Every case cites its source; the reference is computed independently of EMCraft.
2. **Geometry and meshing**: curved and thin parts, imported CAD, inclusions that touch.
3. **Solvers**: the open milestones in [ROADMAP.md](ROADMAP.md) — discontinuities, time domain,
   radiation, periodic structures, performance on large problems.
4. **Getting results in and out**: Touchstone, field exports, PCB formats.
5. **The app**: forms, charts and the 3D view.

Bigger areas are open to anyone who wants to own one: open an issue describing what you would
like to build and the maintainer replies.

## Before you start: claim an issue

Every change starts from an issue, and only the maintainer merges.

1. Pick an open issue labelled **help wanted** or **good first issue** that is not labelled
   **claimed**. If what you want to do has no issue, open one first and wait for a reply.
2. Comment `/claim` on it. The bot assigns it to you, labels it **claimed** and replies; then it is
   yours. It refuses an issue someone else holds (naming them and when their claim lapses), an
   issue an open pull request already covers, and an issue without **help wanted** or **good
   first issue**: on those, say what you would like to do and the maintainer replies. Work on
   unassigned issues may not be accepted.
3. Open a draft pull request early, linked with `Closes #N`. A pull request that references an
   open-for-contributors issue nobody holds (`Closes #N` in the body, `(#N)` in the title) claims
   it the same way; if someone holds it already, or an older open pull request references it,
   the bot says so on your pull request instead.
4. An issue with no visible progress for 14 days goes back to the pool: a daily sweep frees a claim
   with no linked pull request 14 days after it was made, and one with a pull request after 14
   days without pushes. `/unclaim` releases it early (you or a maintainer).

Issues labelled **in progress** are being worked on by the maintainer; please don't start on them.
The bot never lets them be claimed, and never frees an issue a maintainer holds.

## Labels and triage

The triage bot labels every new issue and pull request (`.github/workflows/triage.yml`). It only
adds labels; correct them freely and it won't put them back.

- **Pull requests** get `area:*` from the paths they change: `crates/mesh`, `crates/tetmesh` →
  `area:mesh`; `crates/solver`, `fem`, `fdtd`, `theory`, `particles`, `gpu` → `area:solver`;
  `tests/oracle` → `area:verification`; `crates/ui-core`, `ui-iced`, `gpu3d`, `render`, `picker`,
  `i18n`, `apps/`, `locales/` → `area:ui`; `crates/io` → `area:io`; `docs/` and top-level `*.md` →
  `area:docs`; `.github/`, `xtask/`, `packaging/`, `scripts/`, `tools/` → `area:infra`.
- **Issues** get `bug` or `enhancement` from the form they were filed with (or a crash or
  "Feature request" title), and an `area:*` from words in the title or the command an error
  names (`step 8 (solve.frequency): …`). Until it has a priority, an issue carries
  `needs-triage`; adding a `priority:` label removes it.
- **Maintainers** comment `/triage area:mesh high S` to sort an issue or pull request: any
  existing label (case, and `-` or `_` for a space, don't matter: `in-progress`), `high` for
  `priority: high`, `S`/`M`/`L` for a size (which replaces the old one), `-label` removes one.
  Any `/triage` with a change clears `needs-triage`.
- **Short bug reports without steps to reproduce** get `needs-info` and one comment asking for
  the steps (and the version, when it's missing). The reporter's reply, or an edit that adds the
  steps, removes the label; an issue that stays in `needs-info` for 14 days is closed, and can be
  reopened any time. Feature requests and maintainers' issues are never asked.
- `needs-triage` is the queue: [open issues with it](https://github.com/jijinbei/emcraft/issues?q=is%3Aopen+label%3Aneeds-triage).

## Reporting a problem or an idea

Use the templates under **New issue**. Show it: for anything you can see — the view, a mesh, a
field picture, a chart — add a screenshot or a short screen recording (drag it into the issue)
and mark the problem on it. For a wrong number, say what it should be and why: a closed form, a
published benchmark or a measurement — never output from a commercial EM tool.

## Making the change

1. Branch from `develop` (the trunk; `main` holds released code only) and open the pull request
   against `develop`. Keep it to one topic.
2. Before pushing: `cargo fmt --all`, and `cargo clippy -p <crate> --all-targets -- -D warnings`
   and `cargo test -p <crate>` for the crates you touched (`cargo xtask test --changed` runs the
   tests of everything your change reaches). CI runs the gate on your pull request once the
   maintainer approves the run (a draft starts none).
3. Write the code as [docs/style.md](docs/style.md) says, and look in
   [docs/architecture.md](docs/architecture.md) for a shared building block before writing a helper
   (`cargo xtask dupes` finds copies).
4. User-visible behaviour is a command (`crates/engine/src/cmd/table.rs`) with tests, including
   hostile parameters. Nothing panics: see "Never crash" in [AGENTS.md](AGENTS.md).
5. Solver and meshing algorithms cite the paper they come from in the module docs and in
   [docs/solvers/literature.md](docs/solvers/literature.md). A solver result that changes moves
   an oracle fixture: say why in the pull request.
6. A change you can see (the view, meshes, panels, field pictures, icons) shows a picture before
   and after in the pull request.
7. Never contribute code, text or data from GPL-family projects or from commercial EM tools
   (CST, HFSS, FEKO, COMSOL, …), and never drive those tools to make test data (see
   [docs/adr/](docs/adr/) and [AGENTS.md](AGENTS.md)).

By contributing you agree your work is licensed under MIT OR Apache-2.0.

`develop` and `main` are protected: changes land on `develop` only through pull requests that the
maintainer has reviewed, as rebase merges (keep each commit building and passing its tests). Be
kind: see the [code of conduct](CODE_OF_CONDUCT.md).
