# Get a mesh you can trust

The mesh decides the accuracy and the cost. The goal is not the finest mesh but the coarsest
one whose answer no longer moves.

## Start from the wavelength

```json
{"command": "mesh.generate", "params": {"per_wavelength": 6, "frequency": 12}}
```

`per_wavelength` sizes the elements in each material from the wavelength there at `frequency`
(use the top of your band). With second-order elements, the default, four to six per
wavelength is a sound start. `size` gives a length directly instead.

In the window, the Mesh step sizes the mesh from the band you confirmed in Setup, and shows
the element count and an estimate of the unknowns, the memory and the time before you run.

## Let the solver refine it

```json
{"command": "mesh.adapt", "params": {"size": 6, "frequency": 10, "passes": 4, "tolerance": 0.005, "order": 2}}
```

`mesh.adapt` solves at one frequency, refines where its error indicator is largest, and
repeats until S changes by less than `tolerance` between passes. The reply lists every pass
with its ΔS: that is your error estimate. Adapt at the frequency where the structure is most
sensitive, usually a resonance or the top of the band.

## Know the cheap gains

- **Order 2 before a finer mesh.** Second-order elements converge much faster than first-order
  ones on the same mesh. Drop to `order: 1` only when memory forces it.
- **Edges.** Fields are singular at metal edges (irises, steps, strip edges). `edge` and
  `strip_edge` grade the mesh towards them; that buys more than refining everywhere.
- **Curved walls** are followed by curved elements, so a cylinder does not need many facets.
- **One layer where nothing varies.** In an H-plane waveguide device nothing changes across the
  height: `size_y` equal to the height is exact there.

## Check it

1. **Convergence:** solve, refine, solve again, compare. **Set up mesh convergence** in the
   Results step does this; from a script, run twice with `size` halved, or read `mesh.adapt`'s
   passes.
2. **Physics:** `results.check` (the **Passivity and reciprocity** button). A lossless model
   must be unitary; an error well above 1e-9 points at the model, not the mesh.
3. **Memory:** the direct solver's memory grows faster than the unknowns. When a model does not
   fit, the factorization spills to scratch files (the factor, then the updates waiting for
   the largest fronts) and slows down; a coarser mesh with `mesh.adapt` usually beats a
   uniformly fine one. For a model of a million unknowns or more, `solve.frequency` with
   `"compress": 1e-8` keeps the factor's large blocks as low-rank tiles: less disk and
   memory, the same S-parameters after iterative refinement.
