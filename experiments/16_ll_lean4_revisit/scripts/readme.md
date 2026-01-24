# VSCode: Local Environment for Workspace

This folder, `experiments/16_ll_lean4_revisit/scripts/readme.md`, also contains an instance of application of local env for workspace.

Semarate these in your mind: Distinguish between:
* Folder scope
* Workspace scope
* Window’s platform
* (vscode env)

* (repo root folder)

On a Remote-SSH window to Linux, the window’s platform  is linux.
The local client, is not relevant, but it is part of the bigger picture.

The "Settings file" can be either:
* workspace settings file
* folder settings file
active settings file can be one of the above, only.

The source of settings can be:
* `./.vscode/settings.json`
* `linlog.code-workspace`
* (MONO) REPO ROOT`/.vscode/settings.json`

The starting point:
.... xyz... depends whether "if you open the workspace file" vs "if you open the folder directly".

Here, several orthoginal semantics emerged:
* the settings
* the folder
* the env
* the scope
* the starting point (route)?

The "window" (reload window).
The filder, the workspaece, the folder (as in "open folder"?).


In the `~/.bashrc`, add:

```bash
# Choose initial directory
if [[ -n "$INITIAL_CWD" ]]; then
  cd -- "$INITIAL_CWD" || echo "cd failed: $INITIAL_CWD"
else
  cd -- /myvol || echo "cd failed: /myvol"
fi
```

Note `VSCODE_AGENT_FOLDER`:
```bash
export VSCODE_AGENT_FOLDER="/dataneura/dothome/dot.vscode-server" ;
```
And some setting on client vscode settings.


Some useful commands:
```bash
batcat -lbash  -pp my_source_file
```

A comment:
```bash
echo "Note: the startups are:

~/.profile
~/.bashrc
~/sspanel/dgocean/ephemssss.envs.bash.source
~/neopiler/scripts/bashrc.bash.source
"
```

The `~/.bash_profile` should remain non-existent.

A shortcut:
`batcat -pp -lbash ~/.profile ~/.bashrc ~/sspanel/dgocean/ephemssss.envs.bash.source ~/neopiler/scripts/bashrc.bash.source`
