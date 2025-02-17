#include "profilers.h"

#include "runtime_profiling_reporter.h"

#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void matrixMultiplyNaive(float *W, float *X, float *Y, int n, int m);

#define M 25600 // Number of input vectors in a batch
// #define N 1024 // Matrix dimension (adjust as needed)

// stores
InMemoryStructuredReporter reporter;

// measures time
Profiler profiler;

void executeTrial(float *d_W, float *d_X, float *d_Y, float *h_W, float *h_X,
                  float *h_Y, int N, int Nrep, int t) {
  auto s = profiler.start();

  // Copy data from host to device
  cudaMemcpy(d_W, h_W, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_X, h_X, N * M * sizeof(float), cudaMemcpyHostToDevice);

  // Define grid and block sizes
  dim3 blockSize(16, 16);
  dim3 gridSize((M + blockSize.x - 1) / blockSize.x,
                (N + blockSize.y - 1) / blockSize.y);

  for (int r = 0; r < Nrep; ++r) {
    // Launch naive matrix multiplication kernel
    matrixMultiplyNaive<<<gridSize, blockSize>>>(d_W, d_X, d_Y, N, M);
  }

  // Copy result back to host
  cudaMemcpy(h_Y, d_Y, N * M * sizeof(float), cudaMemcpyDeviceToHost);

  double elapsed = s.stop();

  // reporter.report_measurement(N, Nrep, t, elapsed.count() );
  reporter.report_measurement(
      InMemoryStructuredReporter::ProfilingEntry{N, M, Nrep, t, elapsed});
}

void runExperiment(int N, int Nrep, int Ntrials) {
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

  for (int t = 0; t < Ntrials; ++t) {
    executeTrial(d_W, d_X, d_Y, h_W, h_X, h_Y, N, Nrep, t);
  }

  // Cleanup
  cudaFree(d_W);
  cudaFree(d_X);
  cudaFree(d_Y);
  free(h_W);
  free(h_X);
  free(h_Y);
}

void runProfiling(std::vector<int> N_k, int Nrep, int Ntrials) {
  for (int N : N_k) {
    runExperiment(N, Nrep, Ntrials);
  }
}

int main() {
  std::vector<int> N_k = {256, 512, 1024, 2048}; // Example sequence of N values
  int Nrep = 10;   // Number of kernel executions per measurement (once data is transferred into (device) GPU global memory)
  // this is separated, to separate the time taken to transfer data between CPU memory & GPU global memory,
  // with the time taken to execute the kernel.
  int Ntrials = 5; // Number of repeated measurements per N (includes transfer time)
  // M: batch size (one trnsaciton, whole calculation.).
  // Matrix A is of size N x N,
  // matrix B is of size N x M.

  std::cout << "This may take a while, please wait..." << std::endl;
  runProfiling(N_k, Nrep, Ntrials);

  std::cout << "Profiling completed." << std::endl;
  return 0;
}
