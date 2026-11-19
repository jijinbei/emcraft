# A stripline's losses and delay

An original three-layer board: a strip on MID centred between ground planes on TOP and BOT 1 mm
apart in εr 4.3 of loss tangent 0.02, 0.5 mm wide and 70 µm thick, and 0.4 mm wide and 35 µm
thick, 40 mm long. `pcb.impedance` at 1 GHz gives each segment's delay and its dielectric and
conductor losses: the dielectric's from the complex permittivity's share of the electric energy,
the copper's by Wheeler's incremental inductance rule (the conductors' faces receded by the finite
elements) at copper's surface resistance. The delay and the dielectric loss are exact in one
dielectric; the conductor loss is compared with Pozar's fit of Wheeler's rule. It does not
validate a rough foil's loss, a microstrip's conductor loss, dispersion or the board's
discontinuities.
