
set -eux

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
INITIAL_PWD="$(realpath --no-symlinks "$(pwd)")"

EXPERDIR="$(realpath "$SCRIPT_DIR/.")"

SRC=$(realpath "$EXPERDIR/.")
BUILD=$(realpath "$EXPERDIR/build")

mkdir -p "$BUILD" && rm "$BUILD/*" && mkdir -p "$BUILD"
# cd "$SRC"


clang++ -fopenmp -O2 -std=c++17 \
    "$SRC/rng_skipahead_v2.cpp" \
    -o "$BUILD/exec_rng_skipahead_v2"



cd "$INITIAL_PWD"
# relative to $INITIAL_PWD
RBUILD="$(realpath --relative-to="$INITIAL_PWD" "$BUILD")"

echo "Built executable in: $RBUILD"
echo "to run:"
ls -alth  --color=always "./$RBUILD/"*



# forked from experiments/11_matrix_cuda/scripts/build_and_run.bash
# compiler cli code from  /home/ephemssss/cs-glossaries/compilers/openmp.md

