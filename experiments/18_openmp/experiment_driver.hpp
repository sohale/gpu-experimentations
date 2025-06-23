
#include <cstdlib>
// #include <iostream>
// #include <sstream>
#include <string>
#include <vector>
using std::string;

#include "profilers.hpp"
#include "runtime_profiling_reporter.hpp"

// std::tuple<M>(params)


struct MyParams {
  int M;
};

InMemoryStructuredReporter<ProfilingEntryFormatter1, MyParams> reporter;

// measures time
Profiler profiler;





template<typename ParamsType,  typename PreparationFunc, typename ExperimentExecutionFunc, typename CleanupFunc>
int runProfiling(ExperimentExecutionFunc experiment_lambda) {
  std::vector<int> N_k = {256, 512, 1024, 2048}; // Example sequence of N values
  int Ntrials =   5; // Number of repeated measurements per N (includes transfer time)
  // no `Nrep`

  std::cout << "This may take a while, please wait..." << std::endl;
  // runProfiling(N_k, Nrep, Ntrials);
  // void runProfiling(std::vector<int> N_k, int Nrep, int Ntrials) {
  for (int N : N_k) {

    // runExperiment
    // template <typename VT> void runExperiment(int N, int Nrep, int Ntrials) {

    // Allocate memory on host, device, etc
    // Transfer data
    PreparationFunc prep;
    prep();

    for (int trial = 0; trial < Ntrials; ++trial) {

      // void executeTrial(...) :

      auto s = profiler.start();

      // other parameters: M
      experiment_lambda(trial, N );

      double elapsed = s.stop();
      auto e = ProfilingEntry<ParamsType>{params, N, trial, elapsed};
      reporter.record_measurement(e);
    }
  }

  // Cleanup / free Allocated memory
  // Transfer back results data
  CleanupFunc f;
  f();
  // end of runProfiling()

  std::cout << "Profiling completed." << std::endl;

  // profiling measung time
  // recodring: storing in in-memory DB
  // reporting: saving in CSV file, and printing to console

  // impliciy calls: reporter.report_and_save_to_file();
  return 0;
}

// forked from https://github.com/sohale/gpu-experimentations/blob/main/experiments/11_matrix_cuda/src/experiment_driver.cu
