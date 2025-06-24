#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <ranges>
#include <iomanip>

struct HistogramSpecs {
  std::size_t num_bins = 10;

};


void print_histogram(const std::vector<double>& data, HistogramSpecs params) {
    if (data.empty()) {
        std::cerr << "Empty data.\n";
        return;
    }



    // length of the maximum histogram bar (histogram bars are horizontal)
    std::size_t graph_width = 50;
    
    // HistogramSpecs -> HistogramCooked
    // params -> cooked

    struct HistogramCooked  {
        double min;
        double max;
        double bin_width;
        HistogramCooked(HistogramSpecs params, const std::vector<double>& data) {
            auto [min_it, max_it] = std::ranges::minmax_element(data);
            this-> min = *min_it;
            this-> max = *max_it;
            bin_width = (max - min) / params.num_bins;

            if (min == max) {
                std::cerr << "Data has no range.\n";
                return;
            }
        }
    };
    // cooked, baked
    HistogramCooked cooked (params, data);

    std::vector<std::size_t> bins(params.num_bins, 0);

    // Fill bins
    for (double val : data) {
        
        auto discretise =[cooked, params](double val) ->  std::size_t  {
          return std::min(static_cast<std::size_t>((val - cooked.min) / cooked.bin_width), params.num_bins - 1);
        };
        std::size_t idx = discretise(val);
        
        // std::size_t idx = std::min(static_cast<std::size_t>((val - min) / cooked.bin_width), params.num_bins - 1);
        ++bins[idx];
    }

    // Determine scaling
    std::size_t max_count = *std::ranges::max_element(bins);

    std::cout << "Histogram (" << params.num_bins << " bins):\n\n";
    for (std::size_t i = 0; i < params.num_bins; ++i) {
        double bin_start = cooked.min + i * cooked.bin_width;
        double bin_end = bin_start + cooked.bin_width;
        std::size_t count = bins[i];
        std::size_t bar_len = static_cast<std::size_t>((static_cast<double>(count) / max_count) * graph_width);

        std::cout << std::fixed << std::setprecision(2)
                  << "[" << bin_start << ", " << bin_end << "): "
                  << std::string(bar_len, '#') << " (" << count << ")\n";
    }
}

/*
int main() {
    std::vector<double> data = {
        1.2, 2.4, 2.8, 3.0, 3.1, 4.2, 4.4, 5.0, 5.1, 5.3, 5.5, 6.1, 6.5, 6.8, 7.0, 7.2, 8.0, 8.4, 9.5
    };

    print_histogram(data, HistogramSpecs{.num_bins=8});
    return 0;
}
*/
