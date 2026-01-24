
set -x
echo "Details:"
echo  "doin: workspace-env.bash arg: $1"
uname -a
echo "TERM_PROGRAM=$TERM_PROGRAM"
echo "TERM_PROGRAM_VERSION=$TERM_PROGRAM_VERSION"

set -x
cd dataneura/gpu-experimentations/experiments/16_ll_lean4_revisit$
pwd

# The terminal will not stay if I don't add the following command:
# The `--noprofile` `--norc` needs to be both here and in $MONO_REPO_ROOT/experiments/16_ll_lean4_revisit/.vscode/settings.json
# Will reset the `set -ex` etc
exec bash --noprofile --norc
