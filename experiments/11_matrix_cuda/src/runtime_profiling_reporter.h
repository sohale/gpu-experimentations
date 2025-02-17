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
#include <tuple>
#include <vector>

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

class InMemoryStructuredReporter {
// or: Recorder: InMemory Structured Recorder & reporter

public:
  struct ProfilingEntry {
    int N;
    int M;
    int Nrep;
    int trial;
    double dtime;

    ProfilingEntry(int n, int m, int nrep, int t, double time)
        : N(n), M(m), Nrep(nrep), trial(t), dtime(time) {}
  };

  InMemoryStructuredReporter(int hint_max_count = 8000) {

    report_begin(hint_max_count);
  }
  ~InMemoryStructuredReporter() { save_to_file(); }

  void report_measurement(const ProfilingEntry &profiling_entry_struct) {
    report_entries.emplace_back(profiling_entry_struct);
    // Uses ProfilingEntry struct
  }

  /*
  void report_measurement(int N, int Nrep, int trial, double dtime) {
    report_entries.emplace_back(N, Nrep, trial, dtime);
    // Uses ProfilingEntry struct
  }
  */

private:
  std::vector<ProfilingEntry> report_entries;

private:
  void save_to_file(const std::string &filename = "runtime_results.csv") {
    std::ofstream file(filename);
    if (file.is_open()) {
      file << "# CUDA Profiling Results\n";
      file << "N,M,Nrep,Trial,Time\n";
      for (const auto &entry : report_entries) {
        const std::string SEP = ", ";
        file << entry.N << SEP << entry.M << SEP << entry.Nrep << SEP
             << entry.trial << SEP << entry.dtime << "\n";
        std::cout << "N: " << entry.N << " M: " << entry.M
                  << " Trial: " << entry.trial << " Time: " << entry.dtime
                  << "s" << std::endl;
      }
      file.close();
      std::cout << "Profiling results saved." << std::endl;
    }
  }

  void report_begin(int hint_max_count) {
    report_entries.reserve(hint_max_count);
    report_entries.clear();

    // We don't want to hiccups in the middle of the experiment.
    // std::cout << "Profiling results." << std::endl;
  }
};

#endif // REPORTER_H
