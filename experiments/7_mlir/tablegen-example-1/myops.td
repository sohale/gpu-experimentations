// MyOps.td - TableGen definitions for MyDialect

include "mlir/IR/OpBase.td"

// todo: use "module"
// module my_mlir_project {

  def MyDialect : Dialect {
    let name = "mydialect";
  }

  class MyOp<dag operands, type resultType> : Op<"mydialect.MyOp", []> {
    let arguments = (ins operands);
    let results = (outs resultType);

    let assemblyFormat = "attr-dict";
  }

  def MyAddOp : MyOp<(ins F32:$lhs, F32:$rhs), (outs F32:$result)> {
    let summary = "A simple addition operation";
    let description = [{
      This operation takes two floating-point numbers and returns their sum.
    }];
  }
// }

