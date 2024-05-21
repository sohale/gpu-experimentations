module {
  func @main() {
    %0 = "mydialect.MyAddOp"(%0, %1) : (f32, f32) -> f32
    return
  }
}


