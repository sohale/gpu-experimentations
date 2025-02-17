#ifndef REPORTER_H
#define REPORTER_H

// scaffolding.cpp
// cuda_profiler.h
// runtime_profiling_reporter.h
// cuda_results.csv runtime_results.csv

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

// Directly into the file. Not in-memory.
// Direct file-based logging
// class FileReporter
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
    // CSV header: avoid blank spaces
    *outfile << "N,Nrep,Trial,Time\n";
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

// In-memory logging, MemoryReporter, InMemoryStringStreamReporter,
// InMemoryStringBufferReporter
class InMemoryStringBufferReporter {

public:
  InMemoryStringBufferReporter() { report_begin(); }
  ~InMemoryStringBufferReporter() { save_to_file(); }

  void report_measurement(int N, int Nrep, int trial, double dtime) {
    mbuffer << N << "," << Nrep << "," << trial << "," << dtime << "\n";
  }

private:
  std::ostringstream mbuffer{};
  std::string get_report() const { return mbuffer.str(); }

  void save_to_file(const std::string &filename = "runtime_results.csv") {
    std::ofstream file(filename);
    if (file.is_open()) {
      file << mbuffer.str();
      file.close();
    }
  }

  void report_begin() {
    mbuffer << "# CUDA Profiling Results\n";
    mbuffer << "N,Nrep,Trial,Time\n";
  }
};

#endif // REPORTER_H
