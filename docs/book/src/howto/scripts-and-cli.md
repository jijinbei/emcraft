# Run EMCraft from scripts

Everything the window does is a command with documented parameters, and `emcraft-cli` runs
commands without a window. A script is a JSON array of them.

## The command line

```
emcraft-cli commands                 # every command and its parameters
emcraft-cli exec file.templates      # one command on an empty session
emcraft-cli run script.json          # a script: prints each reply, then the model's summary
emcraft-cli sample highpass --out d  # solve a sample, print it against the exact solution
```

`emcraft-cli` needs no display or graphics adapter, so it runs on a server or in CI.

## A script

```json
{{#include ../../scripts/cavity.json}}
```

- Commands run in order on one session. The first error stops the script and is printed;
  nothing panics on bad input.
- Each reply is printed as JSON, one object per command, so a script's output can be read by
  another program.
- Lengths are in millimetres and frequencies in gigahertz, unless the script starts with
  `units.set`. See [units](../reference/units.md).
- File paths in a script are relative to where you run it.

## Parameters and expressions

```json
[
  {"command": "project.variable", "params": {"name": "d", "value": 30}},
  {"command": "geometry.waveguide", "params": {"width": 22.86, "height": 10.16,
    "sections": [{"length": "=d", "material": "vacuum"}]}}
]
```

A number may be an expression of the variables defined before it: a string starting with `=`,
with `+ − * / ^`, parentheses, `pi`, and `sqrt`, `abs`, `sin`, `cos`, `tan`, `exp`, `ln`.

## Get a script from the window

A session's history is a script. Build the model in the window, save the project, and the
`.emcraft` file holds the commands that made it; the [samples](../reference/samples.md) are
scripts too, in
[`crates/templates/scripts`](https://github.com/jijinbei/emcraft/tree/main/crates/templates/scripts).
The verification cases under
[`tests/oracle`](https://github.com/jijinbei/emcraft/tree/main/tests/oracle) are scripts with
their expected results, one for nearly every capability: a good place to copy from.

## The same result every time

The same script gives the same mesh and the same numbers bit for bit, on any machine and any
number of threads. A script in version control is therefore a regression test of your design.
