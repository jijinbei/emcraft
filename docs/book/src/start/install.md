# Install

EMCraft is two programs: `emcraft`, the desktop app, and `emcraft-cli`, the same engine on the
command line. There is no packaged release yet, so today you either build from source (any
system) or take a ready build from the repository's Actions page.

## Build from source (Linux, macOS, Windows)

You need [Rust](https://rustup.rs) (stable) and Git. On Debian or Ubuntu the desktop app also
needs three system libraries:

```
sudo apt-get install -y libxkbcommon-dev libwayland-dev libgl1-mesa-dev
```

Then:

```
git clone https://github.com/jijinbei/emcraft
cd emcraft
cargo build --release -p emcraft -p emcraft-cli
```

The first build takes several minutes. The programs are `target/release/emcraft` and
`target/release/emcraft-cli`; copy them anywhere on your `PATH`.

Check it:

```
target/release/emcraft-cli commands | head     # the first commands of the reference
target/release/emcraft --sample highpass       # the window, a filter open
```

## A ready build

Every commit on `main` is built for Linux (x86_64), Windows (x86_64) and macOS (Apple silicon).
On GitHub, open [Actions → Build](https://github.com/jijinbei/emcraft/actions/workflows/build.yml),
pick the latest green run and download `emcraft-linux-x86_64`, `emcraft-windows-x86_64` or
`emcraft-macos-arm64` (GitHub asks you to sign in). Each archive holds the two programs and the
licences.

## Windows installer

`EMCraft-VERSION-setup.exe` installs for your user without administrator rights, adds the Start
menu entries and opens `.emcraft` files in EMCraft. Until the first release it is an artifact of
the [Windows installer workflow](https://github.com/jijinbei/emcraft/actions/workflows/release-windows.yml).

- The installer is not code-signed yet: Windows SmartScreen may say "Windows protected your PC".
  Choose **More info → Run anyway**.
- `emcraft-cli` is not added to your `PATH`. Open **EMCraft Command Prompt** from the Start menu,
  a terminal where it works.
- The Windows build has no GPU kernels; everything runs on the CPU.

Where it puts its files, silent install and uninstall:
[docs/install-windows.md](https://github.com/jijinbei/emcraft/blob/main/docs/install-windows.md).

## What it needs

- **Memory** decides how large a model you can solve: the direct solver keeps its factors in
  memory and spills to scratch files when they do not fit. The Mesh step shows the unknowns,
  memory and time of a run before you start it.
- **A graphics adapter** for the 3D view (Vulkan, Metal, DirectX 12 or OpenGL; a software one
  will do). `emcraft-cli` needs none.

Next: [your first solve](first-solve.md).
