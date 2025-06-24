/*
Inspired by Tim Mattson's Introduction to OpenMP: 07 Module 4
https://www.youtube.com/watch?v=WcPZLJKtywc
https://github.com/tgmattso/OpenMP_intro_tutorial
https://github.com/tgmattso/OpenMP_intro_tutorial/blob/master/omp_hands_on.pdf

clang++ -fopenmp -O2 -std=c++20 practice_tm1.cpp  -o practice_tm1.exec

*/

#include <vector>
#include <iostream>
#include <numbers>
#include <omp.h>
using std::vector;
using std::cout;
using std::endl;
#include "histogram.hpp"

#define PARAM

// static long num_steps = 100000000;
// static long num_steps = 1000000;
static long num_steps = 10000000;



struct ResultReportType {
  // in-givens:
  int param_nthreads;
  int param_experno;

  // outcomes:
  double result_value;
  double run_time;
  int actual_numthreads;
};

ResultReportType experiment1(int param_nthreads)
{
    omp_set_num_threads(param_nthreads);
    double start_time =  omp_get_wtime();
    const double dx_step = 1.0/(double) num_steps;
    int actual_numthreads = -1;
    double naive_sum = 0.0;
    #pragma omp parallel
    {
        #pragma omp single
        actual_numthreads = omp_get_num_threads();

    // #pragma omp for reduction(+:naive_sum)
        for ( int xi = 0; xi < num_steps; xi++)
        {
          double x = ( xi + 0.5 ) * dx_step;
          naive_sum = naive_sum + 4.0 / ( 1.0 + x * x );
        }
    }
	  double result_value = dx_step * naive_sum;
    double run_time = omp_get_wtime() - start_time;
    ResultReportType result = {
      .param_nthreads = param_nthreads, .param_experno = 1,
      .result_value = result_value, .run_time = run_time, .actual_numthreads = actual_numthreads
    };
    return result;
}

constexpr int MAX_SLOTS = 1000; // maximum number of threads
/*
Deliberately incorrect.
*/
ResultReportType experiment2(int param_nthreads)
{
    omp_set_num_threads(param_nthreads);
    double start_time =  omp_get_wtime();
    const double dx_step = 1.0/(double) num_steps;
    int actual_numthreads = -1;
    constexpr int STACK_STRIDE = 64; // make sure satcks done overlap
    double naive_sum[MAX_SLOTS][STACK_STRIDE];
    double total_sum = 0.0;

    #pragma omp parallel
    {

        // private?
        int id = omp_get_thread_num();

        int nthreads = omp_get_num_threads();

        int M = nthreads;
        int offset = id;
        naive_sum[offset][0] = 0.0;
        for ( int xi = offset; xi < num_steps; xi+=M)
        {
          double x = ( xi + 0.5 ) * dx_step;
          naive_sum[offset][0] += 4.0 / ( 1.0 + x * x );
        }

        #pragma omp barrier


        #pragma omp critical
        total_sum += naive_sum[offset][0];
        // todo: this is serial
        // a way to prarallelise it is, like merge-sort, first pairwise (half number of threads), then half, etc
        // why not do this in the first place in the first one?

        // if (id == 0)
        #pragma omp single
        actual_numthreads = nthreads;

    } // omp-parallel
	  double result_value = dx_step * total_sum;
    double run_time = omp_get_wtime() - start_time;
    ResultReportType result = {
      .param_nthreads = param_nthreads, .param_experno = 2,
      .result_value = result_value, .run_time = run_time, .actual_numthreads = actual_numthreads
    };
    return result;
}


int main() {

  int MAX_NUMTHREADS = 4*4;
  PARAM const int NTRIALS = 10;
  vector<ResultReportType> results;
  for (int param_nthreads = 1; param_nthreads <= MAX_NUMTHREADS; param_nthreads++ )
  {
    for(int trial = 0 ; trial < NTRIALS; trial++) {
      cout << trial << " "; // << std::flush;
      auto r = experiment2(param_nthreads);

      results.push_back(r);

    }
  }
  cout << endl;

  for(const auto &r : results) {
     cout << "result= " << r.result_value << "  " << r.run_time <<"(s) " << " threads:" << r.actual_numthreads << "/" << r.param_nthreads <<  "   \t ε=" << (r.result_value-std::numbers::pi) << "\n";
  }
  cout << endl;


  vector<double> times = myv_map(results, [](const ResultReportType&r) -> double {return r.run_time;});
  /*
  auto h1 =     print_histogram(times, HistogramSpecs{.num_bins=8*2});
  cout << endl;
  */


  // h1 = HistogramCooked::from_minmaxwidth(0,2, 0.1);
  // h1 = HistogramCooked::from_minmaxwidth(0.0129, 0.0148, 0.001);
  // stress test:
  // h1 = HistogramCooked::from_minmaxwidth(0.0120+0.00299, 0.0150, 0.0005/2);
  // works:
  // auto h1 = HistogramCooked::from_minmaxwidth(0.0120, 0.0150, 0.0005/2);

  // auto h1 = HistogramCooked::from_minmaxwidth(0.0120, 0.0150, 0.0005/2);
  auto h1 = HistogramCooked(HistogramSpecs{.num_bins=8*2}, times);

  print_histogram(times, h1);

  for(int nth = 1; nth <= MAX_NUMTHREADS; nth++ )
  {
    cout << "nthreads=" << nth << endl;

  vector<double> times1 = myv_map(results,
    [](const ResultReportType&r) -> double {return r.run_time;},
    [nth](const ResultReportType&r) -> bool {return r.actual_numthreads == nth;}
  );
  // print_histogram(times1, HistogramSpecs{.num_bins=8*2}, h1);
  // print_histogram(times1, (HistogramSpecs)h1); // fails: HistogramSpecs(h1));
  // after a lot of search in various patterns, it went back to the same old typical C++ pattern.
  // C++ design is very specific.
  // print_histogram(times1, HistogramSpecs::fromCooked(h1));
  print_histogram(times1, h1);
  report_basic_stats(times1);
  cout << endl;
}

}
