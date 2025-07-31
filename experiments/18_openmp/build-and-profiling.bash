
set -eux

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
INITIAL_PWD="$(realpath --no-symlinks "$(pwd)")"

EXPERDIR="$(realpath "$SCRIPT_DIR/.")"

SRC=$(realpath "$EXPERDIR/.")
BUILD=$(realpath "$EXPERDIR/build")

mkdir -p "$BUILD" && rm "$BUILD/*" && mkdir -p "$BUILD"
# cd "$SRC"

# installation
# sudo apt get clang libomp-dev. # usually installs clang14
: || {
    clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -O3 -march=native -std=c++20 \
    experiment6.cpp -o experiment6.exec


    clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -O3 -march=native -std=c++20 ./practice_tm1.cpp -o omp.exec
    clang: error: cannot find libdevice for sm_35; provide path to different CUDA installation via '--cuda-path', or pass '-nocudalib' to build without linking with libdevice

    see Also, but dont use:
    provisioning_scripts/terraform/environments/cuda-ptx-hardcoded-dev-experiments/environment-box/scripts_to_push/install_nvidia_sh.bash

    sudo apt install nvidia-container-toolkit
    sudo ubuntu-drivers devices

    Latest supported:
    * https://www.nvidia.com/en-us/drivers/
    * search & get URL
    * wget ...
    * sudo sh NVIDIA-Linux-x86_64-570.169.run

    wget -qO- https://apt.llvm.org/llvm.sh | bash -s
}




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

