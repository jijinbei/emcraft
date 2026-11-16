Closes #

## What and why

## Before and after
<!-- For any visible change (the view, meshes, panels, field pictures, icons): a picture before and after. Drag them in here. -->

## How it was verified
<!-- Tests added or changed; oracle cases (tests/oracle) for solver and meshing results, with their cited references. A moved oracle fixture says why. -->

## Checklist
- [ ] The issue was assigned to me before I started (see CONTRIBUTING.md)
- [ ] The code follows [docs/style.md](https://github.com/jijinbei/emcraft/blob/develop/docs/style.md) and [AGENTS.md](https://github.com/jijinbei/emcraft/blob/develop/AGENTS.md)
- [ ] `cargo fmt --all`, `cargo clippy -p <crates I touched> --all-targets -- -D warnings` and their tests pass
- [ ] No panics in non-test code; hostile parameters are refused (AGENTS.md, "Never crash")
- [ ] Before/after pictures for any visible change
- [ ] No code, text or data from GPL-family projects or commercial EM tools
