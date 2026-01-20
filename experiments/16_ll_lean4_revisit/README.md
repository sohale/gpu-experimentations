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
