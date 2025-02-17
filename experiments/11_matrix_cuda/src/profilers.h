#ifndef PROFILER_H
#define PROFILER_H

#include <chrono>

class Profiler {
public:
  auto start() {
    // Also, by its type, implies a "started" state.
    auto started_timepoint = std::chrono::high_resolution_clock::now();
    return started_timepoint;
  }

  auto stop(std::chrono::high_resolution_clock::time_point started_timepoint) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - started_timepoint;
    return elapsed.count();
    ;
  }

  static void _example_usage() {
    /*
    Profiler profiler;
    auto start = profiler.start();
    // do something
    auto elapsed = profiler.stop(start);
    double elapsed_time = profiler.elapsed_time(elapsed);

    double elapsed_time(const std::chrono::duration<double> &elapsed) {
        return elapsed.count();
    }
    double elapsed_time(const std::chrono::duration<double> &elapsed) {
        return elapsed.count();
    }
    */

    Profiler profiler;
    auto s = profiler.start();
    double elapsed = s.stop();
  }
};

#endif // PROFILER_H
