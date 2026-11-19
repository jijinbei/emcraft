# Crosstalk between two striplines

An original three-layer board: two 0.2 mm traces of different nets on MID, 0.2 and 0.4 mm apart,
centred between ground planes on TOP and BOT 0.8 mm apart in εr 4.3, 40 mm long. `pcb.crosstalk`
solves the victim's segment beside the aggressor's as two coupled lines and gives their
capacitive and inductive couplings, the near-end coefficient (k_L + k_C)/4 and the far-end
crosstalk. In one dielectric the near end is Cohn's (Z_e − Z_o)/2(Z_e + Z_o) and the far end
none. It does not validate a microstrip's far-end crosstalk (its sign is checked by the engine's
tests), strong coupling or unmatched lines.
