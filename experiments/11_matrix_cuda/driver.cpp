#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#define N 1024 // Matrix dimension (adjust as needed)
#define M 256  // Number of input vectors in a batch


int main() {
  // Allocate memory on host
  float *h_W = (float *)malloc(N * N * sizeof(float));
  float *h_X = (float *)malloc(N * M * sizeof(float));
  float *h_Y = (float *)malloc(N * M * sizeof(float));

  // Initialize W and X with random values
  for (int i = 0; i < N * N; ++i)
    h_W[i] = static_cast<float>(rand()) / RAND_MAX;
  for (int i = 0; i < N * M; ++i)
    h_X[i] = static_cast<float>(rand()) / RAND_MAX;

  // Allocate memory on device
  float *d_W, *d_X, *d_Y;
  cudaMalloc((void **)&d_W, N * N * sizeof(float));
  cudaMalloc((void **)&d_X, N * M * sizeof(float));
  cudaMalloc((void **)&d_Y, N * M * sizeof(float));

  // Copy data from host to device
  cudaMemcpy(d_W, h_W, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_X, h_X, N * M * sizeof(float), cudaMemcpyHostToDevice);

  // Define grid and block sizes
  dim3 blockSize(16, 16);
  dim3 gridSize((M + blockSize.x - 1) / blockSize.x,
                (N + blockSize.y - 1) / blockSize.y);

  // Launch naive matrix multiplication kernel
  matrixMultiplyNaive<<<gridSize, blockSize>>>(d_W, d_X, d_Y, N, M);

  // Copy result back to host
  cudaMemcpy(h_Y, d_Y, N * M * sizeof(float), cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_W);
  cudaFree(d_X);
  cudaFree(d_Y);
  free(h_W);
  free(h_X);
  free(h_Y);

  std::cout << "Naive CUDA matrix multiplication completed." << std::endl;
  return 0;
}
