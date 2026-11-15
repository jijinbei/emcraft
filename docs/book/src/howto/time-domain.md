# Use the time-domain solver

EMCraft has two solvers for S-parameters. The frequency-domain finite elements are the
default; the time-domain solver (finite integration on a grid of hexahedra) sends a pulse
through the model and transforms what comes out.

## When to use it

| Use the time domain when | Stay in the frequency domain when |
|---|---|
| the band is very wide | the structure is resonant (narrow filters, high Q): the pulse rings for a long time |
| the model is large in wavelengths and memory is the limit | walls are curved or details are fine: the grid staircases them |
| you want an independent check of a finite element result | you need the accuracy of second-order curved elements |
| you have an NVIDIA GPU | |

## Run it

```json
[
  {"command": "file.sample", "params": {"name": "highpass", "solve": true}},
  {"command": "solve.time_domain", "params": {"start": 5, "stop": 9, "cell": 1}},
  {"command": "view.sparameters", "params": {"path": "both.png"}}
]
```

`cell` is the grid's cell size (default 1 mm): about ten cells per wavelength at the top of the
band is a start, and halving it should not move the answer. The chart then shows both solvers'
curves over each other.

![The high-pass filter by both solvers, over the exact solution](../media/highpass-fem-fdtd.png)

In the window, choose **Time domain** as the kind of analysis on the project's Home, confirm
the band and the cell in Setup, and run.

## On a GPU

`solve.devices` lists what the machine has; `solve.device` with `"device": "cuda"` moves the
time stepping to an NVIDIA GPU, in single precision by default. The setting belongs to the
machine, not the project. The Windows build has no GPU kernels yet.

## What it does to losses

A loss tangent becomes a conductivity matched at the band's centre, so the loss is exact there
and drifts towards the band's ends; for a wide band with lossy dielectrics, give the material a
Debye or Lorentz model instead (`model.material.create`).
