#ifndef PROFILER_H
#define PROFILER_H

#include <cassert>
#include <chrono>

class Profiler {
private:
  class StartedTimer {
  public:
    // a wrapper around:
    std::chrono::high_resolution_clock::time_point started_timepoint;
    StartedTimer(
        std::chrono::high_resolution_clock::time_point started_timepoint)
        : started_timepoint{started_timepoint} {

          // verified it aliases steady_clock (Measuring durations reliably, even if system clock changes)
          // assert(std::chrono::high_resolution_clock::is_steady == true);
          // fails, but it's fine.
        }

    // Using my "state chain" pattern:
    // gives a state-machine feel to it, when we add the stop her,
    // as opposed to stop() being directly in the Profiler class.
    double stop();
  };

public:
  StartedTimer start() {
    // Also, by its type, implies a "started" state.
    auto started_timepoint = std::chrono::high_resolution_clock::now();
    return Profiler::StartedTimer{started_timepoint};
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

double Profiler::StartedTimer::stop() {
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - started_timepoint;
  return elapsed.count();
}

#endif // PROFILER_H

// forked from gpu-experimentations/experiments/11_matrix_cuda/src/profilers.h
