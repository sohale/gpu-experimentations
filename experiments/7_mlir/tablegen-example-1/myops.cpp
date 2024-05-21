#include "my_mlir_project/MyOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

namespace my_mlir_project {
  #define GET_OP_CLASSES
  // #include "my_mlir_project/MyOps.cpp.inc"
  #include "myops.cpp.inc"
}
