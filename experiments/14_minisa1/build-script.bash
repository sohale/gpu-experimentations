# run this after (inside) bash ... provision_scripts/mlir_env/mlir_env.bash

set -ex
echo "VERIFYING envs:"
echo "\$MLIR_BASE: $MLIR_BASE"
echo "\$MLIR_BINARY_DIR: $MLIR_BINARY_DIR"

export HM="/home/ephemssss"
export REPO="$HM/gpu-experimentations"


export PATH="$PATH:$MLIR_BINARY_DIR"

function veify_folder() {

  if [ -z "$1" ]; then   # if $1 is not specified
    echo "incorrect usage"
    exit 1
  fi
  if [ ! -d "$1" ]; then
    echo "Folder $1 does not exist."
    exit 1
  fi
  if [ ! -d "$1/" ]; then
    echo "Folder $1/ does not exist."
    exit 1
  fi
}

: || {
mkdir -p $REPO/experiments/14_minisa1/build
rm -rf $REPO/experiments/14_minisa1/build
cd $REPO/experiments/14_minisa1
mkdir build && cd build

export PATH=$PATH:$MLIR_BINARY_DIR
cmake -DMLIR_DIR=$MLIR_BASE/lib/cmake/mlir ..
}


which llvm-tblgen



export LLVM_SRC_DIR=$MLIR_BASE/llvm-project/llvm
export MLIR_SRC_DIR=$MLIR_BASE/llvm-project/mlir
# MLIR_BASE means "/mlir"
: || ```tree

/mlir/llvm-project/
├── llvm/
│   └── include/
│       └── llvm/Target/Target.td
├── mlir/
│   └── include/
```



veify_folder "$MLIR_BASE"
veify_folder "$LLVM_SRC_DIR"
veify_folder "$MLIR_SRC_DIR"
veify_folder  "$LLVM_SRC_DIR/include"
veify_folder  "$MLIR_SRC_DIR/include"

# tree -L 2 "$MLIR_SRC_DIR/include"
# find "$MLIR_SRC_DIR/include" | grep "llvm/Target/TargetInstrInfo.td"
# find "$LLVM_SRC_DIR/include" | grep "TargetInstrInfo.td"
find "$LLVM_SRC_DIR/include" | grep --color=always ".td"
# LONG:
find "$MLIR_SRC_DIR/include" | grep --color=always ".td"

mkdir -p build

llvm-tblgen -gen-instr-info \
  -I . \
  -I "$LLVM_SRC_DIR/include" \
  -I "$MLIR_SRC_DIR/include" \
  minisa01.td \
  -o build/Minisa01GenInstrInfo.inc

