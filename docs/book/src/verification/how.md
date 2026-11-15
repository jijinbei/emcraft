# How results are checked

A field solver that is wrong looks exactly like one that is right: a smooth curve either way.
So EMCraft does not ask you to take its numbers on trust. Every solver capability is compared
with a reference the code cannot influence, and the comparison is public:
[every checked quantity](results.md), with its reference value, EMCraft's value, the error and
the tolerance.

## What a result is compared with

| Reference | What it shows | Examples |
|---|---|---|
| **Theory** | the code solves the equations right | closed-form waveguide modes, transmission-line cascades, cavity resonances and Q, the Mie series; power conservation and reciprocity; convergence rates |
| **Published benchmarks** | agreement on standard problems | filter and discontinuity designs from papers and textbooks, converged mode matching |
| **Experiment** | the model predicts reality | measurements published in papers |

The reference is always evaluated separately from the solver: the closed form is computed on
its own, not by the code under test. No reference comes from another simulator.

![The high-pass filter's S-parameters: the exact solution, the finite elements and the time domain](../media/highpass-fem-fdtd.png)

*One case: a waveguide high-pass filter. The exact cascade, the finite elements and the time
domain, three independent routes to the same curve.*

## How a case stays right

Each case is a folder under
[`tests/oracle/`](https://github.com/jijinbei/emcraft/tree/main/tests/oracle) holding three
files:

- `recipe.json`: the model, as a script of ordinary EMCraft commands. You can run it yourself:
  `emcraft-cli run tests/oracle/highpass-filter/recipe.json`.
- `expected.json`: each quantity's reference value, its tolerance, and where the reference comes
  from (the formula or the paper).
- `fixture.json`: the accepted results, frozen. A later change to the code that moves one of
  them fails the build, even if it stays within the tolerance, so accuracy cannot drift
  unnoticed.

The same input gives the same mesh and the same result bit for bit, on any number of threads.
That is what makes freezing results possible.

## What this does not tell you

The table says the solvers are right on the checked models. Your own model has its own mesh,
and a coarse mesh gives a coarse answer in any solver. Three habits cover it:

1. **Refine and compare.** The Results step offers a mesh convergence run: solve again finer (or
   at element order 1 against 2) and see how far S moves.
2. **Check the physics.** *Passivity and reciprocity* in the Results step: a lossless structure
   must give |S11|² + |S21|² = 1, and S12 must equal S21.
3. **Start from a sample.** A [sample](../reference/samples.md) close to your structure is a
   model you can trust before you change it.

The full policy, with every family of tests, is
[docs/verification.md](https://github.com/jijinbei/emcraft/blob/main/docs/verification.md).
