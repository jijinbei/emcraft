# EMCraft

EMCraft is an open-source electromagnetic field simulator for microwave, RF and PCB work. You
describe a structure (a waveguide filter, a cavity, an antenna, a board), it solves Maxwell's
equations with its own finite element and time-domain solvers, and it shows S-parameters,
resonances, fields and far fields.

![The S-parameters of a two-post waveguide resonator in the EMCraft window](media/app-sparameters.png)

## Is it for you?

It is, if you want to:

- **get S-parameters, resonances or radiation patterns** of a 3D structure without a licence
  server, on Linux, macOS or Windows;
- **know why to trust the numbers**: every solver capability is compared with a closed form or a
  published benchmark, and [the whole table is public](verification/results.md);
- **drive the solver from a script or an agent**: everything the window does is a
  [command](reference/commands.md) you can also send as JSON.

## Where to go

| You want to | Read |
|---|---|
| Get it running | [Install](start/install.md) |
| See one model solved end to end | [Your first solve](start/first-solve.md) |
| Find a model close to yours | [Samples](reference/samples.md) |
| Look up a command and its parameters | [Commands](reference/commands.md) |
| Judge the accuracy | [How results are checked](verification/how.md) |

## What it does not do yet

EMCraft is young (version 0.1). Before you plan work around it:

- **No packaged release for Linux and macOS yet.** You build from source, or take the build of a
  commit from the repository's Actions page. Windows has an installer.
- **CAD import is STL, 3MF and SolidCraft designs** (and KiCad boards). STEP and IGES are not
  read.
- **The Windows build has no GPU kernels.** The time domain runs on the CPU there.
- **Checked means the checked cases.** A capability is verified on the models of the
  [verification table](verification/results.md); your own model still needs a mesh convergence
  check, which the Results step offers.

The source, the issues and the roadmap are on
[GitHub](https://github.com/jijinbei/emcraft). People who want to change the code start from
[AGENTS.md](https://github.com/jijinbei/emcraft/blob/main/AGENTS.md) and
[docs/architecture.md](https://github.com/jijinbei/emcraft/blob/main/docs/architecture.md) there.
