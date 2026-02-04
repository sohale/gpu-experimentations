#!/bin/bash
set -eux

export MONOREPO_ROOT="$HOME/gpu-experimentations"

# paths
# VSCODE_WSDIR
WS="$MONOREPO_ROOT/experiments/21_cuda_another_warmup_exercise"
WSFILE=$WS/exper21.generated.code-workspace

cd "$WS"

# installation
: || {
  # clang++ installation as of 21.1.7
  # CUDA instation: as of: 13-1
  :
}

# each time
cat scripts/code-workspace.yaml \
  | yq -o=json eval \
    > $WSFILE

# to convert conversly: `cat jsonfile.json | yq -P
# or ` | yq -P eval -`

# For documentation. Not automating.
echo "On vscode, click on $(realpath --relative-to="$WS" "$WSFILE")"
