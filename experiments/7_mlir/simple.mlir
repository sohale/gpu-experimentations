// simple.mlir
module {
  func @main() -> i32 {
    %c1 = constant 1 : i32
    %c2 = constant 2 : i32
    %sum = addi %c1, %c2 : i32
    return %sum : i32
  }
}
