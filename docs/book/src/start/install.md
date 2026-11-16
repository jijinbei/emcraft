# Install

EMCraft is two programs: `emcraft`, the desktop app, and `emcraft-cli`, the same engine on the
command line. Download a release (Linux, Windows, macOS), build from source (any system), or take
a ready build of any commit from the repository's Actions page.

## Download a release

The [releases page](https://github.com/jijinbei/emcraft/releases) has, for each version:

| Platform | File |
|---|---|
| Linux x86_64 | `emcraft-VERSION-linux-x86_64.tar.gz` |
| Windows x86_64 | `emcraft-VERSION-windows-x86_64.zip`, or the installer `EMCraft-VERSION-setup.exe` |
| macOS arm64 (Apple silicon) | `emcraft-VERSION-macos-arm64.tar.gz` |

Unpack it and run `emcraft` (the window) or `emcraft-cli` (the command line) from the folder;
`SHA256SUMS` lists the archives' checksums. The binaries are not signed: on macOS remove the
quarantine first (`xattr -dr com.apple.quarantine emcraft-VERSION-macos-arm64`).

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

## A ready build of any commit

Every commit on `develop` and `main` is built for Linux (x86_64), Windows (x86_64) and macOS (Apple silicon).
On GitHub, open [Actions → Build](https://github.com/jijinbei/emcraft/actions/workflows/build.yml),
pick the latest green run and download `emcraft-linux-x86_64`, `emcraft-windows-x86_64` or
`emcraft-macos-arm64` (GitHub asks you to sign in). Each archive holds the two programs and the
licences.

## Windows installer

`EMCraft-VERSION-setup.exe` installs for your user without administrator rights, adds the Start
menu entries and opens `.emcraft` files in EMCraft. It is attached to every
[release](https://github.com/jijinbei/emcraft/releases); for a commit between releases, it is an
artifact of the [Windows installer workflow](https://github.com/jijinbei/emcraft/actions/workflows/release-windows.yml).

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
