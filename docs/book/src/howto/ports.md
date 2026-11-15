# Choose and place ports

A port is where a wave enters or leaves the model, and what S is measured between. Pick the
kind by what feeds the structure.

| The structure is fed by | Use | Command |
|---|---|---|
| a waveguide, a coaxial line, a stripline running out of the model | a **waveguide port** on the end face | `model.port.create` |
| a trace end on a board | a **PCB port** | `pcb.port` |
| a gap between two conductors (an antenna's feed, a chip component's pads) | a **lumped port** | `model.port.lumped` |

In the window all three are on the **＋** of the Excitation step; the 3D view's **Ports**
overlay shows each port's face and its direction.

## Waveguide ports

```json
{"command": "model.port.create", "params": {"name": "1", "end": "start", "modes": 4}}
```

- **`end`** is `start` or `end` of the guide. On a solid with several arms (a bend, a tee), give
  the rectangle with `face` and `axis` instead; on an imported SolidCraft design, name the face
  with `face_name` and the port follows it when the design changes.
- **`modes`** is how many modes the port absorbs (default 4). Only the first is excited and
  read. Too few modes near a discontinuity reflect its evanescent field back into the model:
  if S changes when you raise `modes`, it was too low, or the port is too close.
- **Keep a port away from discontinuities**, about a guide width, or raise `modes`.
- **The first mode must propagate** over the whole band. A band that starts below the port's
  cut-off is flagged in the Setup step.

## See the modes before solving

`solve.port_modes` computes each port's modes on the mesh in seconds, without solving the
volume: propagation constant, effective permittivity, cut-off, and for a line with conductors
its characteristic impedance. In the window, the **Port modes** page shows their patterns. Use
it to check that the port carries the mode you think it does.

## Move the reference plane

S's phase is referred to the port's face. `deembed` on the port, or the **Ref. plane** boxes
under the S-parameter chart, shift the reference along the line; S follows at once, with no new
solve.

## Lumped ports

```json
{"command": "model.port.lumped", "params": {"name": "feed", "from": [30, 30, 29.5], "to": [30, 30, 30.5], "impedance": 50}}
```

A source in series with `impedance` across the gap from `from` to `to`. A radiation solve
reports the port's input impedance and S11 against `impedance`; a closed model with only lumped
ports gives S between them.
