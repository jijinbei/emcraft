# A relativistic ring beam in a drift tube, as a body of revolution

Particle-in-cell in a body of revolution (`solve.particles` with `axisymmetric`, the r–z grid of
`emcraft_fdtd::axi`, docs/adr/0012-particles.md): a steady ring of electrons (`particles.beam`,
5.75 to 6.25 mm in radius) is injected at 1.022 MeV (γ₀ = 3) through the grounded end of a
perfectly conducting tube 10 mm in radius and 80 mm long, held on its lines by a 10 T axial
field. Its own charge sinks its potential: at 0.7 of the limiting current (13.109 kA of
18.728 kA) the electrons slow to γ_b = 2.10734 (Bogdankevich & Rukhadze 1971; Miller 1982, §3.2),
and all of the current passes.

At γ₀ = 3 the beam (β = 0.94) outruns the Yee grid's shortest waves on square cells and drives
its numerical Cherenkov instability, which deepens the depression by about a fifth. On cells
8 times finer along z (`fine`) the time step's c Δt nears Δz, where the grid's waves along z
travel at c (the magic time step along z, Taflove & Hagness §4.3), and the depression is the
closed form's to 0.4 %.
