# Sweep frequencies, and sweep a dimension

## A frequency sweep

```json
{"command": "solve.frequency", "params": {"start": 8, "stop": 12, "points": 201, "order": 2}}
```

With three or more frequencies the sweep is *fast*: a few frequencies are solved in full, the
others in the basis their fields span, and full solves are added where the residual is largest
until every point is within tolerance. So 201 points cost little more than 11, and a narrow
resonance is not missed between points. `"sweep": "full"` solves every point in full if you
want to compare.

In the window: set the band in the Setup step, confirm it, and press **▶** on Run.

## More points where it matters

```json
{"command": "solve.refine", "params": {"start": 9.6, "stop": 9.9, "points": 101}}
```

`solve.refine` adds frequencies to the sweep you have, on the same mesh and without
assembling again. In the window, drag across a part of the S-parameter chart.

## Sweep a dimension

Make the dimension a variable, use it in the model, then sweep it:

```json
[
  {"command": "file.sample", "params": {"name": "highpass"}},
  {"command": "study.sweep", "params": {
    "variable": "l_gap", "start": 10, "stop": 30, "points": 5,
    "solve": {"command": "solve.frequency", "params": {"start": 5, "stop": 9, "points": 41}},
    "outputs": ["s21"]
  }}
]
```

For each value the model is rebuilt, meshed and solved, and the results are collected in one
table (the **Study** page in the window). Your own model gets variables with
`project.variable`; any number in a later command may then be an expression of them, written
as a string starting with `=`, such as `"=l_gap/2 + 1"`.

- Several variables at once go in `variables`, on a grid.
- When each solve is expensive, `"surrogate": {"budget": N}` solves only N points of the grid
  and predicts the rest.
- `study.optimize` searches for the values that meet a goal instead of tabulating them.
