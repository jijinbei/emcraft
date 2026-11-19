# A microstrip's copper thickness

An original two-layer board: a 0.35 mm trace on TOP over 0.2 mm of εr 4.3 to a ground plane on
BOT, its copper 17.5, 35 and 70 µm thick (the layer's `copper`, standing up into the air).
`pcb.impedance` solves each cross-section quasi-statically by the finite elements, the strip a
conductor of its thickness. The impedances are compared with Hammerstad and Jensen's fit for a
strip of finite thickness (their effective widths). It does not validate solder masks, losses,
dispersion or the board's discontinuities.
