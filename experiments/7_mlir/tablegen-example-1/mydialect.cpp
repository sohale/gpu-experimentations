#include "my_mlir_project/MyOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

namespace my_mlir_project {
  class MyDialect : public Dialect {
  public:
    explicit MyDialect(MLIRContext *context) : Dialect("mydialect", context, TypeID::get<MyDialect>()) {
      addOperations<
        #define GET_OP_LIST
        // #include "my_mlir_project/MyOps.cpp.inc"
        #include "myops.cpp.inc"
      >();
    }
  };
} // end namespace my_mlir_project

static DialectRegistration<my_mlir_project::MyDialect> MyDialect;


