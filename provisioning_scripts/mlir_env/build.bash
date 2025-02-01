set -ex
# Run in docker, via: experiments/8_mlir_nn/mlir_env.bash
export MLIR="$B"

# verify the syntax of your MLIR file,
$MLIR/mlir-opt single_layer_perceptron.mlir



# notes:
# [1] see: experiments/7_mlir/tablegen-example-1/run_build.bash

% $MLIR/mlir-translate -mlir-to-llvmir single_layer_perceptron.mlir -o single_layer_perceptron.ll

clang single_layer_perceptron.ll -o single_layer_perceptron -lm

./single_layer_perceptron
