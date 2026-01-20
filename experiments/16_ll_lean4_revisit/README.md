# linlog1proj

exit 1

# cd here

# key command: init afresh
lake init linlog math

# check
cat .elan/lean-toolchain

# key command
lake build

lake env


## Maintenance: rebuild fresh (e.g., after renaming your module's name/dir, or, editing lakefile.toml)

rm -rf .lake lake-manifest.json
lake update
lake build



This combination worked on :
```toml
[[require]]
name = "mathlib"
scope = "leanprover-community"
rev = "v4.26.0"
```

File: `./lean-toolchain`
```txt
leanprover/lean4:v4.27.0-rc1
```



# Version of lean itself
is set using `elan`

# elan toolchain install .... (?)

```bash
elan toolchain install leanprover/lean4:nightly
elan toolchain list
elan override set leanprover/lean4:nightly
# elan override unset
```

`$ELAN_HOME/lean-toolchain`:
Is created by
```bash
elan override set leanprover/lean4:nightly
```


# The verison triplet:

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


## How to set the version triplet:
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
```

## The working configuration
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
and in `./lean-toolchain`:
```txt
leanprover/lean4:v4.27.0-rc1
```

## Anohter old sane configuration:
This combination worked on `lean4:v4.27.0-rc1` (see file `lean-toolchain`)
But CSLib did not work.
```toml
[[require]]
name = "mathlib"
scope = "leanprover-community"
rev = "v4.26.0"
```


## (failed) Experiments for installing cslib
Failed (on lean4:v4.27.0-rc1)
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
