#include "profilers.h"

#include "runtime_profiling_reporter.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>
using std::string;

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

/*
template<class Entry>
string describer_lambda(Entry e) {
     // InMemoryStructuredReporter::ProfilingEntry
    std::ostringstream os{};
    os << "A:" << e.N << "×" << e.N << ", B: " << e.N << "×" << e.M << ", x"
       << e.Nrep << ", trial: " << e.t;
    return os.str();
  }
*/

// stores
// InMemoryStructuredReporter<describer_lambda> reporter;

// InMemoryStructuredReporter<ProfilingEntryFormatter> reporter;

// InMemoryStructuredReporter<ProfilingEntryFormatter<InMemoryStructuredReporter::ProfilingEntry>>
// reporter;


InMemoryStructuredReporter<ProfilingEntryFormatter> reporter;

// measures time
Profiler profiler;


/* Typical usage:
void executeTrial(...) {
  auto s = profiler.start();
  ...
  double elapsed = s.stop();

  // reporter.record_measurement(N, Nrep, t, elapsed.count() );
  reporter.record_measurement(
      // InMemoryStructuredReporter<ProfilingEntryFormatter>::ProfilingEntry{
      ProfilingEntry{N, M, Nrep, t, elapsed});
}

template <typename VT> void runExperiment(int N, int Nrep, int Ntrials) {
  // Allocate memory on host, device, etc
  // Transfer data

  for (int t = 0; t < Ntrials; ++t) {
    executeTrial(...);
  }

  
  // Cleanup / free Allocated memory
  // Transfer back results data
  ...
}
*/

void runProfiling(std::vector<int> N_k, int Nrep, int Ntrials) {
  for (int N : N_k) {
    runExperiment(N, Nrep, Ntrials);
  }
}
*/

int main() {
  std::vector<int> N_k = {256, 512, 1024, 2048}; // Example sequence of N values
  int Nrep = 10; // Number of kernel executions per measurement (once data is
                 // transferred into (device) GPU global memory)
  // this is separated, to separate the time taken to transfer data between CPU
  // memory & GPU global memory, with the time taken to execute the kernel.
  int Ntrials =
      5; // Number of repeated measurements per N (includes transfer time)



  std::cout << "This may take a while, please wait..." << std::endl;
  runProfiling(N_k, Nrep, Ntrials);

  std::cout << "Profiling completed." << std::endl;

  // profiling measung time
  // recodring: storing in in-memory DB
  // reporting: saving in CSV file, and printing to console

  // impliciy calls: reporter.report_and_save_to_file();
  return 0;
}

// forked from https://github.com/sohale/gpu-experimentations/blob/main/experiments/11_matrix_cuda/src/experiment_driver.cu
