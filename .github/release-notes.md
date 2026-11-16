The first release of EMCraft: an open-source electromagnetic field simulator in pure Rust for
microwave, RF and PCB work — finite element and time-domain solvers, S-parameters, fields and far
fields, every capability checked against exact solutions or published benchmarks
(585 of 585 verified quantities pass, [docs/oracle.md](https://github.com/jijinbei/emcraft/blob/main/docs/oracle.md)).

## Download

| Platform | File |
|---|---|
| Linux x86_64 | `emcraft-v0.1.0-linux-x86_64.tar.gz` |
| Windows x86_64 | `emcraft-v0.1.0-windows-x86_64.zip`, or the installer `.exe` |
| macOS arm64 (Apple silicon) | `emcraft-v0.1.0-macos-arm64.tar.gz` |

Each archive holds the desktop app (`emcraft`) and the command line (`emcraft-cli`), with the
licences; `SHA256SUMS` lists the archives' checksums. The binaries are not signed: on macOS,
remove the quarantine first (`xattr -dr com.apple.quarantine emcraft-v0.1.0-macos-arm64`); on
Windows, SmartScreen asks once.

```sh
./emcraft --sample highpass                          # the desktop app on a sample
./emcraft-cli sample highpass --out pictures         # S-parameters against the exact solution
./emcraft-cli run script.json                        # a script of commands
```

## What it does

- **Geometry:** waveguides with steps and irises, boxes, cylinders, spheres and their Boolean
  combinations, STL, 3MF and SolidCraft designs, symmetry planes.
- **Boards:** KiCad boards read whole or drawn in 3D, route screening, lumped ports at trace ends.
- **Frequency domain:** Nédélec elements of orders 1 and 2, modal, terminal, lumped and Floquet
  ports, a fast frequency sweep, adaptive refinement; a multifrontal direct solver that goes out
  of core and compresses its largest blocks (block low-rank) past the machine's memory, and GMRES
  with p-multigrid when even that does not fit.
- **Eigenmodes, materials, radiation, time domain, studies and projects** — see the
  [README](https://github.com/jijinbei/emcraft#what-it-does) and the
  [documentation](https://site.jijinbei.jp/emcraft/book/).
- **Everything is a command:** the app, scripts and agents (a JSON control channel) reach the
  same commands. The window speaks English and Japanese.

A worked example: [the leakage through a microwave oven's door](https://github.com/jijinbei/emcraft-reports/tree/main/microwave-oven-leakage),
a whole-oven model of 2.3 million unknowns solved with this release on a 30 GB machine.

## Contributing

Issues and pull requests are welcome: see
[CONTRIBUTING.md](https://github.com/jijinbei/emcraft/blob/main/CONTRIBUTING.md). Claim an issue
with `/claim` and branch from `develop`.
