#include "profilers.h"

#include "runtime_profiling_reporter.h"

#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <vector>
using std::string;

__global__ void matrixMultiplyNaive(float *W, float *X, float *Y, int n, int m);

// Takes a good number of time:  M =25600
#define M 2560 // Number of input vectors in a batch
// #define N 1024 // Matrix dimension (adjust as needed)

/*
struct ProfilingEntryFormatter {
  // std::string
  // operator()(const InMemoryStructuredReporter::ProfilingEntry &e) const {
  std::string operator()(const ProfilingEntry &e) const {

    std::ostringstream os{};
    os << "A:" << e.N << "×" << e.N << ", B: " << e.N << "×" << e._M << ", x"
       << e.Nrep; // << ", trial: " << e.trial;
    return os.str();
  }
};
*/

/*
// InMemoryStructuredReporter::ProfilingEntry
template<typename REntry>
struct ProfilingEntryFormatter {
  std::string
  operator()(const REntry &e) const {

    std::ostringstream os{};
    os << "A:" << e.N << "×" << e.N << ", B: " << e.N << "×" << e._M << ", x"
       << e.Nrep << ", trial: " << e.t;
    return os.str();
  }
};
*/

template <class Entry> string describer_lambda(Entry e) {
  // InMemoryStructuredReporter::ProfilingEntry

  std::ostringstream os{};
  os << "A:" << e.N << "×" << e.N << ", B: " << e.N << "×" << e._M << ", x"
     << e.Nrep; // << ", trial: " << e.trial;
  return os.str();
}
// problem:
// experiment_driver.cu(66): error: function template "describer_lambda" is not
// a type name InMemoryStructuredReporter<describer_lambda<ProfilingEntry>>
// reporter;

// stores
// InMemoryStructuredReporter<describer_lambda> reporter;

// InMemoryStructuredReporter<ProfilingEntryFormatter> reporter;

// InMemoryStructuredReporter<ProfilingEntryFormatter<InMemoryStructuredReporter::ProfilingEntry>>
// reporter;

// InMemoryStructuredReporter<ProfilingEntryFormatter> reporter;
InMemoryStructuredReporter<describer_lambda<ProfilingEntry>> reporter;

// measures time
Profiler profiler;

// float, VT, FT
template <typename VT>
void executeTrial(VT *d_W, VT *d_X, VT *d_Y, VT *h_W, VT *h_X, VT *h_Y, int N,
                  int Nrep, int t) {
  auto s = profiler.start();

  // Copy data from host to device
  cudaMemcpy(d_W, h_W, N * N * sizeof(VT), cudaMemcpyHostToDevice);
  cudaMemcpy(d_X, h_X, N * M * sizeof(VT), cudaMemcpyHostToDevice);

  // Define grid and block sizes
  dim3 blockSize(16, 16);
  dim3 gridSize((M + blockSize.x - 1) / blockSize.x,
                (N + blockSize.y - 1) / blockSize.y);

  for (int r = 0; r < Nrep; ++r) {
    // Launch naive matrix multiplication kernel
    matrixMultiplyNaive<<<gridSize, blockSize>>>(d_W, d_X, d_Y, N, M);
  }

  // Copy result back to host
  cudaMemcpy(h_Y, d_Y, N * M * sizeof(VT), cudaMemcpyDeviceToHost);

  double elapsed = s.stop();

  // reporter.record_measurement(N, Nrep, t, elapsed.count() );
  reporter.record_measurement(
      // InMemoryStructuredReporter<ProfilingEntryFormatter>::ProfilingEntry{
      ProfilingEntry{N, M, Nrep, t, elapsed});
}

template <typename VT> void runExperiment(int N, int Nrep, int Ntrials) {
  // Allocate memory on host
  VT *h_W = (VT *)malloc(N * N * sizeof(VT));
  VT *h_X = (VT *)malloc(N * M * sizeof(VT));
  VT *h_Y = (VT *)malloc(N * M * sizeof(VT));

  // Initialize W and X with random values
  for (int i = 0; i < N * N; ++i)
    h_W[i] = static_cast<VT>(rand()) / RAND_MAX;
  for (int i = 0; i < N * M; ++i)
    h_X[i] = static_cast<VT>(rand()) / RAND_MAX;

  // Allocate memory on device
  VT *d_W, *d_X, *d_Y;
  cudaMalloc((void **)&d_W, N * N * sizeof(VT));
  cudaMalloc((void **)&d_X, N * M * sizeof(VT));
  cudaMalloc((void **)&d_Y, N * M * sizeof(VT));

  for (int t = 0; t < Ntrials; ++t) {
    executeTrial<VT>(d_W, d_X, d_Y, h_W, h_X, h_Y, N, Nrep, t);
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
    runExperiment<float>(N, Nrep, Ntrials);
  }
}

int main() {
  std::vector<int> N_k = {256, 512, 1024, 2048}; // Example sequence of N values
  int Nrep = 10; // Number of kernel executions per measurement (once data is
                 // transferred into (device) GPU global memory)
  // this is separated, to separate the time taken to transfer data between CPU
  // memory & GPU global memory, with the time taken to execute the kernel.
  int Ntrials =
      5; // Number of repeated measurements per N (includes transfer time)
  // M: batch size (one trnsaciton, whole calculation.).
  // Matrix A is of size N x N,
  // matrix B is of size N x M.

  std::cout << "This may take a while, please wait..." << std::endl;
  runProfiling(N_k, Nrep, Ntrials);

  std::cout << "Profiling completed." << std::endl;

  // profiling measung time
  // recodring: storing in in-memory DB
  // reporting: saving in CSV file, and printing to console

  // impliciy calls: reporter.report_and_save_to_file();
  return 0;
}
