set -ex
echo "VERIFYING envs:"
echo "\$MLIR_BASE: $MLIR_BASE"
echo "\$MLIR_BINARY_DIR: $MLIR_BINARY_DIR"

export HM="/home/ephemssss"
export REPO="$HM/gpu-experimentations"


export PATH="$PATH:$MLIR_BINARY_DIR"


: || {
mkdir -p $REPO/experiments/14_minisa1/build
rm -rf $REPO/experiments/14_minisa1/build
cd $REPO/experiments/14_minisa1
mkdir build && cd build

export PATH=$PATH:$MLIR_BINARY_DIR
cmake -DMLIR_DIR=$MLIR_BASE/lib/cmake/mlir ..
}


which llvm-tblgen

# MLIR_BASE means "/mlir"

export LLVM_SRC_DIR=$MLIR_BASE/llvm-project/llvm
export MLIR_SRC_DIR=$MLIR_BASE/llvm-project/mlir



llvm-tblgen -gen-instr-info \
  -I . \
  -I $LLVM_SRC_DIR/include \
  -I $MLIR_SRC_DIR/include \
  minisa01.td \
  -o build/Minisa01GenInstrInfo.inc

