# Security policy

EMCraft is young; only `main` and the latest release get fixes.

Please **don't open a public issue** for a security problem. Report it privately through the
repository's [Security tab](https://github.com/jijinbei/emcraft/security/advisories/new) with
what you found, how to reproduce it (a model, project file or script), the version and platform,
and the impact you expect.

Command scripts, parameters and files are treated as hostile input (AGENTS.md, "Never crash").
Reports are especially useful for opening untrusted projects, models or scripts (crashes, runaway
memory or CPU, path tricks) and for the app's control channel (docs/control-protocol.md).
