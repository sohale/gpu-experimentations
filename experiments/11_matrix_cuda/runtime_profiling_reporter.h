#ifndef REPORTER_H
#define REPORTER_H

// scaffolding.cpp
// cuda_profiler.h
// runtime_profiling_reporter.h
// cuda_results.csv runtime_results.csv

#include <fstream>
#include <memory>

class Reporter {
public:
  Reporter() { report_begin(); }
  ~Reporter() { end_report(); }

protected:
  std::unique_ptr<std::ofstream> outfile;

private:
  void report_begin() {
    outfile = std::make_unique<std::ofstream>("runtime_results.csv");
    *outfile << "# CUDA Profiling Results\n";
    // CSV header:
    *outfile << "N, Nrep, Trial, Time(s)\n";
  }

public:
  void report_measurement(int N, int Nrep, int trial, double dtime) {
    std::string SEP = ", ";
    *outfile << N << SEP << Nrep << SEP << trial << SEP << dtime << "\n";
  }

private:
  void end_report() {
    outfile->close();
    outfile = nullptr;
  }
};

#endif // REPORTER_H
