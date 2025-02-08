// #include <cstdlib>
#include <cuda_runtime.h>

#define N 1024 // Matrix dimension (adjust as needed)
#define M 256  // Number of input vectors in a batch

__global__ void matrixMultiplyNaive(float *W, float *X, float *Y, int n,
                                    int m) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < m) {
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
      sum += W[row * n + k] * X[k * m + col];
    }
    Y[row * m + col] = sum;
  }
}
