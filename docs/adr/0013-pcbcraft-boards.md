# ADR 0013: PCBCraft boards through PCBCraft's storage and engine

Status: accepted (2026-11-19).

## Decision

`pcb.import` opens a PCBCraft project (github.com/tangency-labs/pcbcraft) itself: its folder,
its `project.json` entry point or a packed `.pc`. It goes through two PCBCraft crates,
`pcbcraft-storage` (read-only `preview`: no locks, no recovery, nothing written) and
`pcbcraft-engine` (the project's board view), and calls nothing else of PCBCraft. The code is
`emcraft-io`'s `pcbcraft` module (L1), next to the KiCad reader; both give an
`emcraft_model::pcb::Board`, so everything after the import — diagnostics, the planar mesher,
the finite elements — is the same for either source.

- **Why the engine, not the files.** A PCBCraft board's files do not say which net a track or
  a via is on: PCBCraft resolves each copper island's net from the pads it touches and the
  schematic's intent, through its versioned formats and its pinned footprint libraries. Reading
  the JSON ourselves would mean writing that resolution again and following every format
  version; the engine's `BoardView` already gives the pads (shape, layers, net), the tracks and
  vias, `copper_nets` and the setup (layers, stackup, materials).
- **One narrow use.** As with SolidCraft (ADR 0003): one module reads plain data out of the
  board view; nothing else of EMCraft sees a PCBCraft type.
- **Pinned.** Both crates come from PCBCraft's repository at one commit (`rev` in the workspace's
  Cargo.toml). Their exact pins (`zip =9.1.3`, `rustix =1.1.6`, `tempfile =3.27.0`) hold the
  workspace's lock at those versions; a change in PCBCraft reaches EMCraft only when the commit
  is moved, in an EMCraft change whose tests run against it.
- **Nothing guessed.** A dielectric's thickness and permittivity come from the board's own setup
  (the finished thickness, the one Dk its material's reports agree on, their Df the loss
  tangent); when unknown — PCBCraft leaves them blank by default — the import is refused unless
  the caller gives the copper layers (`stackup`). A copper layer's finished thickness is its
  copper's (a sheet when not given); PCBCraft keeps no solder mask's thickness or permittivity,
  so none is read.
- **Zones filled by EMCraft, until PCBCraft fills them.** PCBCraft keeps a zone's outline and
  fill settings (clearance, thermal gap and spokes, connection, minimum island, priority), not
  filled copper (its fill is PCBCraft's issue #232). Without the fill a board's ground pour is
  missing and its lines have no plane under them, so `emcraft_io::pcbcraft` hands the zones back
  as pours (`emcraft_model::pcb::Pour`) and `pcb.import` fills them
  (`emcraft_mesh::planar::fill`, the Boolean found on a constrained triangulation as the planar
  mesher finds its union of copper): the outline less every other net's copper grown by the
  clearance, the highest priority first, the net's own pads by their connection. Board-edge
  clearance and net classes' larger clearances are not applied. When PCBCraft saves or exposes
  its own fill, the import reads that instead and this fill goes.
- **Targets are EMCraft's.** PCBCraft's net classes carry dimensions (clearances, widths,
  vias), not impedances, and bind to nets through its identity registry. The impedances a net
  should have are EMCraft's own net classes (`pcb.class`, saved with the board, nets by name);
  `pcb.check`'s findings name segments by the ids the import kept — a PCBCraft track's UUID — so
  PCBCraft can point at them.

## Licences

PCBCraft is MIT OR Apache-2.0; what its engine and storage add (`geo-index`, `geo-types`,
`geo-traits`, `dunce`) is MIT or Apache-2.0. None is GPL or LGPL. The test data in
`crates/io/tests/data/pcbcraft` is PCBCraft's sensor-node example (MIT OR Apache-2.0), with a copy
whose stackup EMCraft's tests fill in, and a copy of that with a ground zone added through
PCBCraft's own `pcb.add_zone` (`sensor-node-pour`, written by the ignored test
`write_the_pour_example`).
