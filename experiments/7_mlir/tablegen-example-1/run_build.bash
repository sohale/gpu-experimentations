set -eux
export PATH="$B:$PATH"
export LLVM_DIR="/mlir"

# get ready
mlir-opt --version
which llvm-tblgen
llvm-tblgen --version



mkdir build -p
cd build
cmake -G Ninja ..
ninja

# and run

mlir-opt test/test.mlir


