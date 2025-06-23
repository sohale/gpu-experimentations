
#include <cmath>
#include <cstdint>
#include <utility>
#include <iostream>
#include <functional>

#include <cassert>
#define assertm(exp, msg) assert((void(msg), exp))

#include <omp.h>

#include "profilers.hpp"
#include "runtime_profiling_reporter.hpp"

typedef uint32_t rng_state_t;
typedef uint32_t rng_value_t;

constexpr std::pair<rng_value_t, rng_state_t> lcg_rand(rng_state_t seed) {
    rng_state_t next = (1103515245u * seed + 12345u) & 0x7fffffff; // mod 2^31
    rng_value_t val = next % 10000; // Example: limit to 0-99 range
    return {val, next};
}

// template<uint32_t M=15>

// compile-time repear M times
template<int M=15>
constexpr std::pair<rng_value_t, rng_state_t> lcg_rand_skipahead(rng_state_t seed) {
    if constexpr (M > 1) {
        auto [_, next] = lcg_rand_skipahead<M - 1>(seed);
        return lcg_rand(next);
    } else {
        return lcg_rand(seed);
    }
}

void openmp_rng_serial(rng_state_t seed, int N, rng_value_t* array) {

  rng_state_t next = seed;  // 0 ≤ seed < 2^31
  for (int i = 0; i < N; ++i) {
        auto [random_number, next_] = lcg_rand(next);
        *array = random_number;
        next = next_;
        array++;
  }
}
template<int M=15>
void openmp_rng_skipahead(rng_state_t seed, int N, rng_value_t* array) {

  rng_state_t next = seed;
  rng_state_t nexts[M];
  rng_value_t *array_slots[M];
  for (int i = 0; i < M; ++i) {

    auto r1 = lcg_rand_skipahead<1>(next);
    auto [random_number, next_] = r1;

    assertm( r1 == lcg_rand(next), "error");
    *array = random_number;
    next = next_;
    nexts[i] = next_;
    array++;
    // i++
    array_slots[i] = array;
  }

  // parallel:
  for (int i = 0; i < M; ++i) {

    // serial:
    for (int j = 1, jM = M; j < N/M; ++j) {

      auto [random_number, next_] = lcg_rand_skipahead<M>(nexts[i]);

      // std::cout << "i = " << i << ", j = " << j << ", jM = " << jM << std::endl;

      assertm( jM == j * M , "j*M consistency failed");
      if (i + jM < N) {
         // *array = random_number;
         *array_slots[i] = random_number;
      } else {
        //
      }
      nexts[i] = next_;
      // array += M; // j
      array_slots[i] += M; // j
      // array += 1; // i
      jM += M;
    }
  }

  /*
  rng_state_t next = seed;  // 0 ≤ seed < 2^31
  for (int i = 0; i < M; ++i) {
    #pragma omp parallel
    {
      rng_state_t local_next = next;
      // Each thread gets its own local copy of next
      for (int j = 0; j < 10; ++j) { // Each thread generates 10 numbers
        rng_value_t random_number = lcg_rand(seed, local_next);
        #pragma omp critical
        {
          std::cout << "Thread " << omp_get_thread_num() << ": " << random_number << "\n";
        }
      }
    }
  }
  */

}

void print_values(rng_value_t* array, int N) {
  auto array_end = array + N;

  for (auto p = array; p<array_end; ++p) {
    std::cout << " " << *p << " ";
  }
  std::cout << "\n" << std::flush;
}


struct MyParams {
  int M;
  std::string formatter() const {
    return "M = " + std::to_string(M);
  }
};

int heuristic_nrep(int n) {
  const double NANOSEC = 1e-9; // 1 ns
  const double MICROSEC = 1e-6; // 1 us
  // generous

  /*
  double min_measurable_time_ub = 0.0001; // 0.00001; // 10 us
  double expected_runtime_per_n = 0.00000000001; // 0.01 ns
  */
  /*
  double min_measurable_time_ub = 100 * MICROSEC; // -  0.0001; // 0.0001; // 0.00001; // 10 us
  double expected_runtime_per_n = 0.01 * NANOSEC; //  - 0.00000000001;
  */
  double min_measurable_time_ub = 100 * MICROSEC;
  double expected_runtime_per_n = 0.1 * NANOSEC;

  //std::cout << min_measurable_time_ub << std::endl;
  //std::cout << expected_runtime_per_n << std::endl;

  // At least 5 of these `min_measurable_time_ub` should (expected to) fit:
  const double threshold_ratio = 5.0;
  /*
  N: 10 Time: 1.01702e-08 raw_dtime: 0.00462283 nsamples:454546
  N: 90 Time: 1.47589e-07 raw_dtime: 0.0081094 nsamples:54946
  */

  double expected_runtime_lb = expected_runtime_per_n * (std::fabs(n)+1);
  // double ratio = min_measurable_time_ub / expected_runtime_lb;
  // how many rulers, dents, etc
  double ratio = expected_runtime_lb / min_measurable_time_ub;

  if (ratio > threshold_ratio) {
    // already ok
    return 1;
  } else {
    // expected_runtime_lb is too small
    return (int)(std::ceil(threshold_ratio / ratio ));
  }
}

void experiment(int N, std::function<void(int)> execute_callback) {

  Profiler profiler;
  InMemoryStructuredReporter<ProfilingEntry<MyParams>> csv_reporter("runtime_results.csv");


  MyParams params{15}; // Example parameter, can be adjusted


  // runProfiling

  for(int n = 0; n <= N-1; n+=10) {
    for (int trial = 0; trial < 10; ++trial) {

      auto s = profiler.start();
      double time_start = omp_get_wtime();


      // for times that are too short. But divide later on. For longer running times, you can just repeat trials (not adding them up).
      int Nrep = heuristic_nrep(n);
      // std::cout << "Nrep for n = " << Nrep << std::endl;
      for (int rep = 0; rep < Nrep; ++rep) {
          execute_callback(n);
      }

      double time_end = omp_get_wtime();
      double elapsed = s.stop();
      auto e = ProfilingEntry<MyParams>{params, n, Nrep, trial, elapsed};

      csv_reporter.record_measurement(e);
    }
  }

}
int main() {

  const int N = 100;
  rng_value_t array[N];

  experiment(N, [&array](int n){
      openmp_rng_serial(42, n, array);
  });
  print_values(array, N);

  rng_value_t array2[N];
  openmp_rng_skipahead<1>(42, N, array2);
  print_values(array2, N);

  return 0;
}

