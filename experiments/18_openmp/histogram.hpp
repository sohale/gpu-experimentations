#include <cstddef>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <iomanip>
#include <optional>
#include <cassert>
#include <numeric>


namespace ansi {

constexpr const char* reset   = "\033[0m";
constexpr const char* bold    = "\033[1m";
constexpr const char* dim     = "\033[2m";
constexpr const char* italic  = "\033[3m";
constexpr const char* underline = "\033[4m";

//namespace fg {
// Foreground colors
constexpr const char* red     = "\033[31m";
constexpr const char* green   = "\033[32m";
constexpr const char* yellow  = "\033[33m";
constexpr const char* blue    = "\033[34m";
constexpr const char* magenta = "\033[35m";
constexpr const char* cyan    = "\033[36m";
constexpr const char* white   = "\033[37m";

// Bright foreground
constexpr const char* bright_red     = "\033[91m";
constexpr const char* bright_green   = "\033[92m";
constexpr const char* bright_yellow  = "\033[93m";
constexpr const char* bright_blue    = "\033[94m";
// }
}

/*
struct HistogramSpecs;  // forward declare
// forward-declared
// struct HistogramCooked;
// HistogramCooked::null();
*/

struct HistogramSpecs {
  std::size_t num_bins = 10;
  // length of the maximum histogram bar (histogram bars are horizontal)
  std::size_t graph_width = 50;
  /*
  std::optional<HistogramCooked> precooked = std::nullopt;
  static HistogramSpecs fromCooked(const HistogramCooked& c);
  */
};


struct HistogramCooked  {
    double min;
    // double max;
    double bin_width;
    HistogramSpecs hscopy;

    HistogramCooked(const HistogramSpecs &params, const std::vector<double>& data);


    /*
    HistogramCooked( const HistogramSpecs &with_precooked);
    */
    static HistogramCooked from_minmaxwidth(double min, double max, double bin_width);
private:
    HistogramCooked(std::nullptr_t){};
};



HistogramCooked HistogramCooked::from_minmaxwidth(double min, double max, double bin_width) {
// HistogramCooked histogram_base_range(double min, double max, double bin_width) {

    HistogramSpecs temp_specs_params;
    //temp_specs.num_bins = ; // default value
    //temp_specs.graph_width = ; // keep default value
    int numbins = std::ceil((max - min)/ bin_width);
    assert(numbins > 0 && "Number of bins must be positive");
    cout << "Number of bins: " << numbins << "\n";
    temp_specs_params.num_bins = numbins;

    const double eps = (bin_width/10.0 + std::fabs(max-min)/2000000.0 + 1e-9)/3.0;
    assert(eps > 0);

    HistogramCooked cooked(nullptr);
    cooked.hscopy = temp_specs_params;
    cooked.min = min - eps;
    // cooked.max = std::max( max, min + bin_width * numbins);
    double max_ =  min + bin_width * numbins;
    cooked.bin_width = bin_width;
    assert(cooked.min < max_ && "Min must be less than max");
    assert(max_ >= max && "code: heuristic adjustments should not shrink the range");

    return cooked;
}




/*
struct HistogramSpecs {
  std::size_t num_bins = 10;
*/
  // std::unique_ptr<HistogramCooked> precooked = nullptr;
  // std::shared_ptr<HistogramCooked> precooked = nullptr;
  // HistogramCooked precooked = HistogramCooked::null();
  // HistogramCooked precooked;
  /*
  std::optional<HistogramCooked> precooked = std::nullopt;
  */

  /*
  HistogramSpecs() = default;
  // default designated initializer
  HistogramSpecs(const HistogramSpecs&s) {
    if (s.precooked != nullptr) {
        throw std::invalid_argument("Cannot copy HistogramSpecs with precooked data.");
    }
    this->num_bins = 10;
    // copies
    this->precooked = nullptr; // s.precooked ? std::make_unique<HistogramCooked>(*s.precooked) : nullptr;
  };
  */
  // HistogramSpecs(const HistogramSpecs&) = default;
  /*
  HistogramSpecs& operator=(const HistogramCooked& c)
    {
        this->num_bins = -1;
        this->precooked = std::make_unique<HistogramCooked>(c);
        return *this;
    }
    */
    /*
    // static
    HistogramSpecs HistogramSpecs::fromCooked(const HistogramCooked& c) {
        HistogramSpecs specs;
        specs.num_bins = -1; // or some default value
        specs.precooked = c; // std::make_unique<HistogramCooked>(c);
        return specs;
    }
    */
// };



// struct HistogramCooked  {
    /*
    moved to header:
    double min;
    double max;
    double bin_width;
    */

    HistogramCooked::HistogramCooked(const HistogramSpecs &params, const std::vector<double>& data) {


        this->hscopy = params; // have a copy of the specss

        auto [min_it, max_it] = std::ranges::minmax_element(data);
        this-> min = *min_it;
        // this-> max = *max_it;
        double max = *max_it;
        bin_width = (max - min) / params.num_bins;

        if (min == max) {
            std::cerr << "Data has no variation.\n";
            throw std::invalid_argument("Data has no variation.");
        }
    }
    //HistogramCooked( const HistogramCooked &precooked)
    //: min(precooked.min), max(precooked.max), bin_width(precooked.bin_width) {}
    /*
    HistogramCooked::HistogramCooked( const HistogramSpecs &with_precooked)
    {
        // if (with_precooked.precooked != nullptr) {
        if (!with_precooked.precooked.has_value()) {
            throw std::invalid_argument("No precooked histogram data provided.");
        }
        // copy into this
        *this = *with_precooked.precooked;
        // : min(precooked->min), max(precooked->max), bin_width(precooked->bin_width)
    }
    */

    /*
    // need to abandon ( due to cyclic dependency ) and use default contructor. which is error-prone for user.
    // idea: instead of precooked, we can specify it as an input to HistogramSpecs
    // sentinel:
    bool is_null() const {
        return std::isnan(min) || std::isnan(max) || std::isnan(bin_width);
    }
    static HistogramCooked null() {
        HistogramCooked n{nullptr}; // = {.min = std::nan(""), .max = std::nan(""), .bin_width = std::nan("")};
        return n;
    }
    private:
    HistogramCooked(std::nullptr_t) : min(std::nan("")), max(std::nan("")), bin_width(std::nan("")) {};
    // usage: as arg:
    // HistogramCooked precooked=HistogramCooked::null()
    */

// };


HistogramCooked print_histogram(const std::vector<double>& data, HistogramSpecs params_) {
    if (data.empty()) {
        std::cerr << "Empty data.\n";
        // return HistogramCooked();
        throw std::invalid_argument("Empty data provided for histogram.");
    }

    // HistogramSpecs -> HistogramCooked
    // params -> cooked

    // todo: flow like statements.
    // struct HistogramCooked; // ...

    // cooked, baked
    // HistogramCooked cooked0 (params, data);
    /*
    if (!precooked.is_null()) {
        cooked = precooked; // use precooked if provided
    }
    */
    // use precooked if provided
    /*
    // if (params.precooked != nullptr) {
    if (params.precooked.has_value() ) {
        cooked = params.precooked.value();
    } else {
        // cooked = HistogramCooked(params, data);
        cooked = cooked0;
    }
    */
    /*
    cout << "precooked.has_value: " << params_.precooked.has_value() << endl;
    HistogramCooked cooked = params_.precooked.has_value() ? params_.precooked.value() : HistogramCooked(params_, data);
    cout << cooked.min << " " << cooked.max << " " << cooked.bin_width << "\n";
    */

    HistogramCooked cooked(params_, data);

    // watertight
    // from now on, barr the old params
    HistogramSpecs params = cooked.hscopy; // copy the specs from cooked, in case it is

    void print_histogram(const std::vector<double>& data, const HistogramCooked &cooked);
    print_histogram(data, cooked);
    return cooked;
}

void print_histogram(const std::vector<double>& data, const HistogramCooked &cooked) {
    /* ideal syntax:
    if ...
    HistogramCooked cooked (params, data);
    else
    HistogramCooked cooked (*params.precooked);
    */
    // std::string num_bins;

    // HistogramSpecs origparams = cooked.hscopy;
    auto orig_num_bins = cooked.hscopy.num_bins;
    auto orig_graph_width = cooked.hscopy.graph_width;

    std::vector<std::size_t> bins(orig_num_bins, 0);
    std::size_t outof_bounds_count_max = 0;
    std::size_t outof_bounds_count_min = 0;

    // std::cout << "min: " << cooked.min << ", max: " << cooked.max << ", bin_width: " << cooked.bin_width << "\n";

    // Fill bins
    for (double val : data) {

        // cout << val << " ";
        // cout << "min: " << cooked.min << ", max: " << cooked.max << ", bin_width: " << cooked.bin_width << "\n";
        int OUTOFBOUND_MAX = -1;
        int OUTOFBOUND_MIN = -2;
        /*
        auto discretise =[cooked, orig_num_bins](double val) ->  std::size_t  {
          return std::min(
            static_cast<std::size_t>((val - cooked.min) / cooked.bin_width),
            orig_num_bins - 1
        );
        };
        std::size_t idx = discretise(val);
        */
        int bin_idx = static_cast<int>((val - cooked.min) / cooked.bin_width);
        if (bin_idx < 0 ) {
            bin_idx = OUTOFBOUND_MIN; // mark as out of bounds
        }
        if (bin_idx >= bins.size()) {
            bin_idx = OUTOFBOUND_MAX; // mark as out of bounds
        }

        // std::size_t idx = std::min(static_cast<std::size_t>((val - min) / cooked.bin_width), orig_num_bins - 1);
        if (bin_idx == OUTOFBOUND_MAX) {
            ++outof_bounds_count_max;
        } else if (bin_idx == OUTOFBOUND_MIN) {
            ++outof_bounds_count_min;
        } else {
            assert( bin_idx >= 0);
            assert( bin_idx < bins.size());

            std::size_t idx = static_cast<std::size_t>(bin_idx);
            ++bins[idx];
        }

    }

    // Determine scaling
    std::size_t max_count = *std::ranges::max_element(bins);
    if (max_count == 0) {
        // std::cerr << "All bins are empty.\n";
        max_count = 1; // avoid division by zero
    }
    assert(orig_graph_width > 0);
    assert(bins.size() > 0);

    double calculated_max = cooked.min + cooked.bin_width * orig_num_bins;

    std::cout << "Histogram (" << orig_num_bins << " bins):\n";
    std::cout << std::setprecision(6);
    // std::cout << "Min: " << cooked.min << ", Max*: " << calculated_max << ", Bin Width: " << cooked.bin_width << "\n";

    for (std::size_t i = 0; i < orig_num_bins; ++i) {
        double bin_start = cooked.min + i * cooked.bin_width;
        double bin_end = bin_start + cooked.bin_width;
        std::size_t count = bins[i];
        // cout << i << " jss" << max_count << " count " << count << " orig_graph_width: " << orig_graph_width << "\n";
        std::size_t bar_len = static_cast<std::size_t>((static_cast<double>(count) / max_count) * orig_graph_width);
        // cout << i << " jasd " << bar_len << " \n";

        std::cout << std::fixed << std::setprecision(4)
                << "[" << bin_start << ", " << bin_end << "): ";

        if (bar_len > 0) {
            std::cout
                << ansi::cyan
                << std::string(bar_len, '#')
                << ansi::reset;
        } else {
            std::cout
                << ansi::cyan << ansi::dim
                << "|"
                << ansi::reset;
        }

        std::cout
                << ansi::dim << " (" << count << ")" <<  ansi::reset << "\n";

    }
    if (outof_bounds_count_max > 0 || outof_bounds_count_min > 0) {
        std::cout << ansi::yellow  << "Out of bounds: ";
        if (outof_bounds_count_max > 0) {
            std::cout <<  " >max: " << outof_bounds_count_max;
        }
        if (outof_bounds_count_min > 0) {
            std::cout <<  " <min: " << outof_bounds_count_min;
        }
        std::cout << ansi::reset << "\n";
    }
}

template<typename T>
void report_basic_stats(const std::vector<T>&data) {
    std::cout << "Basic Statistics: ";
    if (data.empty()) {
        std::cerr << "Is empty.\n";
        return;
    }
    auto [min_it, max_it] = std::ranges::minmax_element(data);
    double min = *min_it;
    double max = *max_it;
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / data.size();
    double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / data.size() - mean * mean);

    std::cout << "min: " << min << ", ";
    std::cout << "max: " << max << ", ";
    std::cout << "avg: " << mean << ", ";
    std::cout << "std: " << stdev << "\n";
}

// for future
/*
C++26:
template <auto MemberPtr>
constexpr auto project = std::views::transform([](auto&& obj) -> decltype(auto) {
    return std::forward<decltype(obj)>(obj).*MemberPtr;
});
  auto run_times = reports
    | project<&ResultReportType::run_time>
    | std::ranges::to<std::vector>();
*/
/*
template <std::ranges::input_range R>
auto to_vector(R&& r) {
    return std::vector(std::ranges::begin(r), std::ranges::end(r));
}
*/

/*
template <std::ranges::input_range R>
auto to_vector(R&& range) {
    return std::vector<std::ranges::range_value_t<R>>(std::ranges::begin(range), std::ranges::end(range));
}
*/


// attempts for statements that do the mapping in main():

  /*
  auto data = auto sorted_times = results
    | std::views::filter([](auto& r) { return r.actual_numthreads > 1; })
    | std::views::transform(&ResultReportType::run_time)
    | std::ranges::to<std::vector>();
  */
  /* C++26
  auto run_times = reports
    | std::views::transform(&ResultReportType::run_time)
    | std::ranges::to<std::vector>();
  */
  /*
  auto view = reports | std::views::transform(&ResultReportType::run_time);
  std::vector<double> run_times(view.begin(), view.end());
  */
  // auto run_times = to_vector(reports | std::views::transform(&ResultReportType::run_time));

  /*
  auto run_times = to_vector(
    reports | std::views::transform(&ResultReportType::run_time)
  );
  */
  //auto view = reports | std::views::transform(&ResultReportType::run_time);
  //   std::vector<double> run_times(view.begin(), view.end());
//


template<typename ST, typename MapFunc>
auto myv_map(const std::vector<ST>& array, MapFunc map) {
    // C++ cannot infer template-arg from return type of a given lambda
    using DT = std::invoke_result_t<MapFunc, ST>;
    std::vector<DT> res;
    for( const ST& e : array) {
        DT copy = map(e);
        res.push_back(copy);
    }
    return res;
}
template<typename ST, typename MapFunc, typename FilterFunc>
auto myv_map(const std::vector<ST>& array, MapFunc map, FilterFunc filt) {
    // C++ cannot infer template-arg from return type of a given lambda
    using DT = std::invoke_result_t<MapFunc, ST>;
    std::vector<DT> res;
    for( const ST& e : array) {
        if (filt(e)) {
            DT copy = map(e);
            res.push_back(copy);
        }
    }
    return res;
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
