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
    double max;
    double bin_width;
    HistogramSpecs hscopy;

    HistogramCooked(const HistogramSpecs &params, const std::vector<double>& data);
    /*
    HistogramCooked( const HistogramSpecs &with_precooked);
    */
};







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
        this-> max = *max_it;
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

    HistogramCooked print_histogram(const std::vector<double>& data, HistogramCooked cooked);
    return print_histogram(data, cooked);

}

HistogramCooked print_histogram(const std::vector<double>& data, HistogramCooked cooked) {
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

    // Fill bins
    for (double val : data) {

        // cout << val << " ";
        // cout << "min: " << cooked.min << ", max: " << cooked.max << ", bin_width: " << cooked.bin_width << "\n";

        auto discretise =[cooked, orig_num_bins](double val) ->  std::size_t  {
          return std::min(
            static_cast<std::size_t>((val - cooked.min) / cooked.bin_width),
            orig_num_bins - 1
        );
        };
        std::size_t idx = discretise(val);

        // std::size_t idx = std::min(static_cast<std::size_t>((val - min) / cooked.bin_width), orig_num_bins - 1);
        ++bins[idx];

    }

    // Determine scaling
    std::size_t max_count = *std::ranges::max_element(bins);

    std::cout << "Histogram (" << orig_num_bins << " bins):\n";
    std::cout << std::setprecision(6);
    std::cout << "Min: " << cooked.min << ", Max: " << cooked.max << ", Bin Width: " << cooked.bin_width << "\n";

    for (std::size_t i = 0; i < orig_num_bins; ++i) {
        double bin_start = cooked.min + i * cooked.bin_width;
        double bin_end = bin_start + cooked.bin_width;
        std::size_t count = bins[i];
        std::size_t bar_len = static_cast<std::size_t>((static_cast<double>(count) / max_count) * orig_graph_width);

        std::cout << std::fixed << std::setprecision(4)
                  << "[" << bin_start << ", " << bin_end << "): "
                  << std::string(bar_len, '#') << " (" << count << ")\n";
    }
    return cooked;
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
