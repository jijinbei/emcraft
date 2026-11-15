# Drive the window from an agent

An agent (or a test) can drive the running desktop app over a local socket: run commands,
press what a person would press, and look at the window. People and agents then work on the
same model with the same commands.

For batch work without a window, use [`emcraft-cli`](scripts-and-cli.md) instead: it is
simpler and needs no display.

## Start the app with a control channel

```
emcraft --sample highpass --control 7878
```

The app listens on `127.0.0.1:7878` only, and makes a random token for this launch, written to
a file only you can read:

| System | Token file |
|---|---|
| Linux | `~/.local/share/emcraft/control/7878.token` |
| macOS | `~/Library/Application Support/EMCraft/control/7878.token` |
| Windows | `%LOCALAPPDATA%\EMCraft\control\7878.token` |

## Talk to it

Each request is one line of JSON, each reply one line. The first request on a connection
carries the token.

```python
import json, pathlib, socket

token = (pathlib.Path.home() / ".local/share/emcraft/control/7878.token").read_text().strip()
link = socket.create_connection(("127.0.0.1", 7878)).makefile("rw")

def call(method, **params):
    link.write(json.dumps({"id": 1, "method": method, "params": params, "token": token}) + "\n")
    link.flush()
    return json.loads(link.readline())

call("engine.execute", command="solve.frequency", params={"start": 5, "stop": 9})
call("ui.render", path="/tmp/window.png")        # the whole window as a PNG
print(call("ui.inspect")["result"]["page"])      # what the window shows now
```

## What it can do

| Method | Does |
|---|---|
| `engine.execute` | runs one command, the same ones as in the [reference](../reference/commands.md) |
| `engine.script` | runs a list of commands as one job |
| `engine.commands` | lists every command with its parameters |
| `ui.inspect` | says what the window shows: the page, the workflow's steps and their status, the open dialog, the running job |
| `ui.handles`, `ui.click` | lists what can be pressed now, and presses it |
| `ui.dialog` | opens a command's dialog, fills its fields, confirms |
| `ui.render` | draws the whole window to a PNG, without needing a visible window |
| `ui.cancel` | stops the running job |

Long commands (solves, meshing) run in the background while the window keeps drawing; the
reply comes when they end, or at once with `"wait": false`. Commands run this way never open
dialogs.

## Safety

The channel is local and needs the token, and it treats what it receives as untrusted: request
sizes, connection counts and coordinates are capped, and a bad request gets an error, not a
crash. The full protocol, with every method and limit, is
[docs/control-protocol.md](https://github.com/jijinbei/emcraft/blob/main/docs/control-protocol.md).
