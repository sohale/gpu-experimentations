// MLIR code for a single-layer perceptron with RGB input data.

module {
  // Constants for dimensions.
  %l1 = constant 27 : index   // L1 = W * H * R (e.g., 3x3x3 for a toy example).
  %rho_12 = constant 0.3 : f32 // Downsampling factor.

  // Compute L2 = L1 * rho_12, downsampled size.
  %l2_f32 = arith.mulf %rho_12, arith.sitofp %l1 : index to f32
  %l2 = arith.fptosi %l2_f32 : f32 to index

  // Allocate and initialize input tensor: [L1, R] = [27, 3] for RGB.
  %input = memref.alloc() : memref<27x3xf32>
  affine.for %i = 0 to 27 {
    affine.for %j = 0 to 3 {
      %value = arith.constant 1.0 : f32 // Example: Initialize all inputs to 1.0.
      memref.store %value, %input[%i, %j] : memref<27x3xf32>
    }
  }

  // Allocate and initialize weights tensor: [L1, L2] = [27, 8].
  %weights = memref.alloc() : memref<27x8xf32>a
  affine.for %i = 0 to 27 {
    affine.for %j = 0 to 8 {
      %value = arith.constant 0.5 : f32 // Example: Initialize all weights to 0.5.
      memref.store %value, %weights[%i, %j] : memref<27x8xf32>
    }
  }

  // Allocate and initialize biases tensor: [L2] = [8].
  %biases = memref.alloc() : memref<8xf32>
  affine.for %i = 0 to 8 {
    %value = arith.constant 0.1 : f32 // Example: Initialize all biases to 0.1.
    memref.store %value, %biases[%i] : memref<8xf32>
  }

  // Allocate output tensor: [L2].
  %output = memref.alloc() : memref<8xf32>

  // Compute perceptron output: output = activation(weights * input + biases).
  affine.for %i = 0 to 8 {
    %sum = arith.constant 0.0 : f32
    affine.for %j = 0 to 27 {
      affine.for %k = 0 to 3 {
        %input_val = memref.load %input[%j, %k] : memref<27x3xf32>
        %weight_val = memref.load %weights[%j, %i] : memref<27x8xf32>
        %prod = arith.mulf %input_val, %weight_val : f32
        %sum = arith.addf %sum, %prod : f32
      }
    }
    %bias_val = memref.load %biases[%i] : memref<8xf32>
    %sum = arith.addf %sum, %bias_val : f32

    // Apply ReLU activation: max(0, sum).
    %zero = arith.constant 0.0 : f32
    %relu = arith.maxf %sum, %zero : f32

    // Store result in the output tensor.
    memref.store %relu, %output[%i] : memref<8xf32>
  }

  // Output tensor now contains the perceptron result for this layer.
}
