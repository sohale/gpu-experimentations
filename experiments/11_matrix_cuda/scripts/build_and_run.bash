# sudo apt install nvidia-cuda-toolkit

echo 'To run the terraform:

bash provisioning_scripts/terraform/common/localmachine/up-matmul-cuda-experiment.bash tfapply
'
# echo "which in turn"
# find provisioning_scripts/terraform/common/localmachine/upload_scripts_to_there.bash
# echo "which in turn"
# find provisioning_scripts/terraform/environments/cuda-ptx-hardcoded-dev-experiments/environment-box/local_manual__setup_at_creation.bash

set -eux

# nvcc -o matrix_mult main.cpp matrix_kernel.cu

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
INITIAL_PWD="$(realpath --no-symlinks "$(pwd)")"

EXPERDIR="$(realpath "$SCRIPT_DIR/..")"

SRC=$(realpath "$EXPERDIR/./src")
BUILD=$(realpath "$EXPERDIR/./build")

mkdir -p "$BUILD" && rm "$BUILD/*" && mkdir -p "$BUILD"
cd "$SRC"


nvcc  \
    -arch=sm_86 \
    -x cu \
    -o $BUILD/matrix_mult \
    \
    experiment_driver.cu \
    naive_matmul.cu

# use `-x cu` \ to tell nvcc that the file is a cuda file
# use `-arch=sm_86` \ to nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).

cd "$INITIAL_PWD"
# relative to $INITIAL_PWD
RBUILD="$(realpath --relative-to="." "$BUILD")"
echo "to run:"
ls -alth  --color=always "./$RBUILD/matrix_mult"
