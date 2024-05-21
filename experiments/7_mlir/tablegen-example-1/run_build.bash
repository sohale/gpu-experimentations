set -eux
# B=/mlir/llvm-project/build/bin
# Where it is built and left (no "install")
export BUILT="/mlir/llvm-project/build"
export PATH="$B:$PATH"
# export LLVM_DIR="/mlir"
export LLVM_DIR="$BUILT/lib/cmake/llvm"
export MLIR_DIR="$BUILT/lib/cmake/mlir"

# get ready
mlir-opt --version
which llvm-tblgen
llvm-tblgen --version


cd /home/ephemssss/gpu-experimentations/experiments/7_mlir/tablegen-example-1

mkdir build -p
cd build

cmake -G Ninja ..
ninja

# and run

mlir-opt ../test/test.mlir
