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


// static long num_steps = 100000000;

static long num_steps = 1000000;



struct ResultReportType {
  // in-givens:
  int param_nthreads;
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
    double sum = 0.0;
    #pragma omp parallel
    {
    // #pragma omp single
        // printf(" num_threads = %d",omp_get_num_threads());
        actual_numthreads = omp_get_num_threads();

    // #pragma omp for reduction(+:sum)
        for ( int xi = 0; xi < num_steps; xi++)
        {
          double x = ( xi + 0.5 ) * dx_step;
          sum = sum + 4.0 / ( 1.0 + x * x );
        }
    }
	  double result_value = dx_step * sum;
    double run_time = omp_get_wtime() - start_time;
    ResultReportType result = { .param_nthreads = param_nthreads, .result_value = result_value, .run_time = run_time, .actual_numthreads = actual_numthreads};
    return result;
}

#define param

int main() {

  param const int NTRIALS = 10;
  vector<ResultReportType> results;
  for (int param_nthreads = 1; param_nthreads <= 4; param_nthreads++ )
  {
    for(int trial = 0 ; trial < NTRIALS; trial++) {
      cout << trial << endl;
      auto r = experiment1(param_nthreads);

      results.push_back(r);

    }
  }
  cout << endl;
  for(const auto &r : results) {
     cout << "result= " << r.result_value << "  " << r.run_time <<"(s) " << " threads:" << r.actual_numthreads << "/" << r.param_nthreads <<  "   \t ε=" << (r.result_value-std::numbers::pi) << "\n";
  }
  cout << endl;
}
