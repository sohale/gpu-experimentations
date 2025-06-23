#ifndef REPORTER_H
#define REPORTER_H

// scaffolding.cpp
// cuda_profiler.h
// runtime_profiling_reporter.h
// cuda_results.csv runtime_results.csv

#include <fstream>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>


/*
// Prepares a CSV in memory
template<typename Params>
class InMemoryStringBufferReporter {

public:
  InMemoryStringBufferReporter() { report_begin(); }
  ~InMemoryStringBufferReporter() { save_to_file(); }

  void record_measurement( int N, int trial, double dtime, std::string label) {
    std::string SEP = ", ";
    auto escape = [](const std::string &s){return "\"" + s + "\""; };
    mbuffer << N << SEP << trial << SEP << dtime << SEP << escape(label) << "\n";
  }

private:
  std::ostringstream mbuffer{};
  std::string get_report() const { return mbuffer.str(); }

  void save_to_file(const std::string &csv_filename = "runtime_results.csv") {
    std::ofstream file(csv_filename);
    if (file.is_open()) {
      file << mbuffer.str();
      file.close();
    }
  }

  void report_begin() {
    mbuffer << "# Profiling Results\n";
    mbuffer << "N,Trial,Time,Label\n";
  }
};
*/

/*
// template<typename Params>
struct ProfilingEntryFormatter1 {

  std::string operator()(const ProfilingEntry<std::tuple<int>> &e) const {
    std::ostringstream os{};
    // no "trial", unique label shared by all trials
    //os << "A:" << e.N << "×" << e.N << ", B: " << e.N << "×" << e.get(0)
    //   << e.Nrep; // << ", trial: " << e.trial;
    os << "N:" << e.N  << "×" << std::get<0>(e._params);
   return os.str();
  }

};
*/

template<typename Params>
struct ProfilingEntry {
  int N;
  int trial;
  double dtime;
  Params _params;

  ProfilingEntry(Params params, int n, int trial_, double time)
      : _params(params), N(n), trial(trial_), dtime(time) {}
};

// DescriberFunc === Formatter
// template <typename DescriberFunc>

template <typename ProfilingEntry>
class InMemoryStructuredReporter {

public:
  InMemoryStructuredReporter(const std::string &csv_filename, int hint_max_count = 8000)
      : csv_filename(csv_filename) {
    ready_to_die = false;
    report_begin(hint_max_count);
  }
  ~InMemoryStructuredReporter() {
    report_and_save_to_file(this->csv_filename);
    // assert(this->ready_to_die, "You forgot to call
    // report_and_save_to_file()");
  }

  void record_measurement(const ProfilingEntry &profiling_entry_struct) {
    report_entries.emplace_back(profiling_entry_struct);
    // Uses ProfilingEntry struct
  }

private:
  std::vector<ProfilingEntry> report_entries;
  bool ready_to_die = false;
  std::string csv_filename;

private:
  // template <typename DescriberFunc>
  void
  report_and_save_to_file(const std::string &csv_filename) {
    std::ofstream file(csv_filename);
    if (file.is_open()) {
      file << "# Profiling Results\n";
      file << "N,Trial,Time,Label\n";
      for (const auto &entry : report_entries) {

        // entry._params.get(0); // Assuming _params is a tuple or similar structure
        //   std::string label = formatter(entry);
        std::string label = entry._params.formatter();

        const std::string SEP = ", ";
        auto escape = [](const std::string &s){return "\"" + s + "\""; };

        file << entry.N << SEP << entry.trial << SEP << entry.dtime << SEP
             << escape(label)  <<  "\n";

        // on screen : stdout
        std::cout << "N: " << entry.N
                  << " Trial: " << entry.trial << " Time: " << entry.dtime
                  << " Label: " << label << "\n";
      }
      file << std::flush;
      file.close();
      std::cout << "Profiling results saved." << std::endl;
    }
    this->ready_to_die = true;
  }

  void report_begin(int hint_max_count) {
    report_entries.reserve(hint_max_count);
    report_entries.clear();

    // hint_max_count: We don't want to hiccups in the middle of the experiment.
    // std::cout << "Profiling results." << std::endl;
  }
};

#endif // REPORTER_H


// forked from: https://github.com/sohale/gpu-experimentations/blob/main/experiments/11_matrix_cuda/src/runtime_profiling_reporter.h
