# linlog1proj

to experiment Lean4 with linear logic, Mathlib, CSLib.

Containts a working configuration (version tuple to pin).

Three layers:
```.
elan
↓
lake
↓
lean
```

Each layer, it's own typical commands:
```shell
# ------ elan layer:
elan
# ------ lake layer:
lake update
lake build
# lake init # only once
# ----- lean layer
lake env lean filename.lean
```

## Some key commands:
```bash
# init afresh the scaffold
lake init linlog math

# check
cat .elan/lean-toolchain

# key command
lake build
lake env
```

How to rebuild after change odf package name, change of version, etc:
```bash
# rebuild fresh (e.g., after renaming your module's name/dir, or, editing lakefile.toml)
rm -rf .lake lake-manifest.json
lake update
lake build
```

## The version of lean itself
Set using `elan`:
```bash
elan toolchain install leanprover/lean4:nightly
elan toolchain list
elan override set leanprover/lean4:nightly
# elan override unset
```
see/check/change file:
```txt
lean-toolchain
```

Incorrect note:
`$ELAN_HOME/lean-toolchain`:
Is created by
```bash
elan override set leanprover/lean4:nightly
```


# The verison triplet:

This combination worked.

## This triplet worked:
* `leanprover/lean4:v4.27.0-rc1`
* `v4.27.0-rc1`
* `b55a607`

Set them in: here, and ./lean-toolchain
Source:
based on: https://github.com/leanprover/cslib/blob/b55a6073c8d000f59c6db812a6974017d1be0858/lakefile.toml
and https://github.com/leanprover/cslib/blob/b55a6073c8d000f59c6db812a6974017d1be0858/lean-toolchain
also maybe: https://github.com/leanprover/cslib/blob/b55a6073c8d000f59c6db812a6974017d1be0858/lake-manifest.json


## Other triplets:

( leanprover/lean4:v4.27.0-rc1, v4.27.0-rc1, b55a607 )
( leanprover/lean4:v4.26.0, "v4.26.0", - )


# How to set the version triplet:
```bash
elan self update
elan override set leanprover/lean4:v4.27.0-rc1
elan toolchain list
elan show
lean --version
# Then, manually set versions in `lakefile.toml` and `./lean-toolchain`
rm -rf .lake lake-manifest.json
lake update
lake build
# success!

lake env lean LinlogMy.lean
```

# The working configuration
in lakefile.toml:
```toml
[[require]]
name = "mathlib"
scope = "leanprover-community"
# git = "https://github.com/leanprover-community/mathlib4"
# rev = "leanprover/lean4:nightly"
# use github of CSLib to extract with the exact commit that mathlib4’s lean-toolchain uses for `4.26.0`
# from cslib: commit b55a607:
rev = "v4.27.0-rc1"

[[require]]
name = "cslib"
scope = "leanprover"
# rev = "main"
git = "https://github.com/leanprover/cslib"
rev = "b55a607"
# See: https://github.com/leanprover/cslib/commit/b55a607
# eg https://github.com/leanprover/cslib/blob/b55a6073c8d000f59c6db812a6974017d1be0858/lakefile.toml
```
and in
File: `./lean-toolchain`:
```txt
leanprover/lean4:v4.27.0-rc1
```

## Anohter old sane valid configuration:
But CSLib did not work:
```toml
[[require]]
name = "mathlib"
scope = "leanprover-community"
rev = "v4.26.0"
```
This combination worked on this lean:
File: `./lean-toolchain`:
```txt
leanprover/lean4:v4.26.0
```


## Some draft notes
(failed) Experiments for installing cslib
Failed (on `lean-toolchain` = `lean4:v4.27.0-rc1`)
```toml
[[require]]
name = "CSLib"
# git = "git@github.com:leanprover/cslib.git"
git = "https://github.com/leanprover/cslib"
# rev = "main"
# rev = "nightly-testing-2026-01-20"
# rev = "v4.26.0"
# rev = "v4.27.0-rc1"
# rev = "v4.26.0"
# rev = "main"
rev = "v4.27.0-rc1"
```


# General, to explore

The lean-lsp-mcp:
Follow instructions at:
https://github.com/oOo0oOo/lean-lsp-mcp
For extesion:
https://vscode.dev/redirect/mcp/install?name=lean-Isp&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22lean-Isp-mcp%22%5D%7D

```bash
claude mcp add lean-lsp uvx lean-lsp-mcp
```

Id: leanprover.lean4

lumina

