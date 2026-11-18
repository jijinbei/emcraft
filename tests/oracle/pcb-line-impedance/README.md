# PCB line impedances from their cross-sections

An original four-layer board (εr 4.3; TOP, a GND plane on L2 0.2 mm down, L3 0.4 mm further,
a GND plane on BOT 0.4 mm below that) with an outer microstrip over L2, a stripline centred on
L3 between the two planes and a coupled pair beside it, named to pair (`CS_P`, `CS_N`).
`pcb.impedance` finds each segment's reference planes and partner, and solves its cross-section
quasi-statically by the finite elements. The microstrip is compared with Hammerstad and Jensen's
fit, the stripline and the pair with Cohn's exact even, odd and single impedances of strips of no
thickness. It does not validate copper thickness, losses, dispersion or the board's
discontinuities.
