#!/bin/bash
set -eu

# From: sohale/ifc2brep-0/scripts/lsp.bash
# This script is actually not evexuted, but can be used for test/verify instatllation/setup of LSP

echo "Standalone clangd LSP"
# language-server
# Language Server Protocol (LSP) server
# clangd

[ -n "${1:-}" ] \
 || { echo -e "Usage: script <particular cpp file>"; exit 1; }
echo "ok"
PARTICULAR_FILE="$1"
ls -alth $PARTICULAR_FILE

# input parameters
CLANGD_BIN="/usr/lib/llvm-18/bin/clangd"
#CLANGD_FLAGS=

# $CLANGD_BIN --help
$CLANGD_BIN --version
echo "$PARTICULAR_FILE"


# With --check=<filename>, attempts to parse a particular file.
# Parse one file in isolation instead of acting as a language server.
# Useful to investigate/reproduce crashes or configuration problems.
$CLANGD_BIN \
    --pretty \
    --log=verbose \
    --ckeck=$PARTICULAR_FILE

#  --pch-storage=<value>               - Storing PCHs in memory increases memory usages, but may improve performance
#    =disk                             -   store PCHs on disk
#    =memory                           -   store PCHs in memory


# cool:
#   --path-mappings=<string>            - Translates between client paths (as seen by a remote editor) and server paths (where clangd sees files on disk). Comma separated list of '<client_path>=<server_path>' pairs, the first entry matching a given path is used. e.g. /home/project/incl=/opt/include,/home/project=/workarea/project


# Thu May 16 18:41:27 UTC 2024
