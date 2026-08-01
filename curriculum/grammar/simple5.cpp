// Grammar curriculum: functions separate data transformations into named steps.
// Conditions and loops below make the order of those transformations explicit.
#include <algorithm>
#include <cstddef>
#include <future>
#include <map>
#include <print>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
namespace sample5 {
bool has_expected_size(const std::vector<int>& values, std::size_t expected_size) {
const bool matches = values.size() == expected_size;
std::println("Expected size: {}, actual size: {}", expected_size, values.size());
return matches;
}
bool contains_expected_count(const std::vector<std::string>& names,
std::size_t expected_count) {
const bool contains_expected = names.size() == expected_count;
std::println("Expected count: {}, actual count: {}", expected_count, names.size());
return contains_expected;
}
std::string check_switch_case(int code) {
switch (code) {
case 0:
return "disabled";
case 1:
return "enabled";
case 2:
return "automatic";
default:
return "unknown";
}
}
void check_switch_cases(const std::vector<int>& codes) {
for (const int code : codes) {
std::println("Switch case {} means {}", code, check_switch_case(code));
}
}
void print_switch_acceptance(int code) {
std::println("Switch value accepted: {}", check_switch_case(code));
}
using NameTable = std::map<std::string, std::string>;
std::future<std::string> launch_name_lookup(const NameTable& table, std::string key) {
return std::async(std::launch::deferred, [&table, key = std::move(key)] {
const auto match = table.find(key);
return match == table.end() ? std::string{"missing"} : match->second;
});
}
void print_primary_name(const NameTable& table) {
auto name_future = launch_name_lookup(table, "primary");
std::println("Primary name: {}", name_future.get());
}
void print_summary(std::string_view label, int count, std::size_t total) {
std::println("{}: {} of {}", label, count, total);
}
void summarize_positive_values(const std::vector<int>& values) {
const int positive_count =
static_cast<int>(std::count_if(values.begin(), values.end(), [](int value) {
return value > 0;
}));
print_summary("positive values", positive_count, values.size());
}
namespace repeated_patterns {
std::string describe_switch(int code) {
switch (code) {
case 10:
return "ten";
case 20:
return "twenty";
default:
return "other";
}
}
void check_switch_cases(const std::vector<int>& codes) {
for (const int code : codes) {
std::println("Switch case {} means {}", code, describe_switch(code));
}
}
bool verify_measurement_count(const std::vector<int>& values,
std::size_t expected_size) {
const bool matches = values.size() == expected_size;
std::println("Measurement count matches: {}", matches);
return matches;
}
bool verify_label_count(const std::vector<std::string>& values,
std::size_t expected_size) {
const bool matches = values.size() == expected_size;
std::println("Label count matches: {}", matches);
return matches;
}
void report_switch_value(int code) {
std::println("Switch value accepted: {}", describe_switch(code));
}
void report_positive_count(const std::vector<int>& values) {
const int positive_count =
static_cast<int>(std::count_if(values.begin(), values.end(), [](int value) {
return value > 0;
}));
print_summary("positive values", positive_count, values.size());
}
}
namespace additional_patterns {
std::string check_switch_value(int code) {
return code > 0 ? "yes" : "no";
}
void check_switch_cases(const std::vector<int>& codes) {
for (const int code : codes) {
std::println("Switch value accepted: {}", check_switch_value(code));
}
}
bool validate_item_count(const std::vector<std::string>& values,
std::size_t expected_size) {
const bool matches = values.size() == expected_size;
if (!matches) {
std::println("Expected {} items but received {}", expected_size, values.size());
}
return matches;
}
bool validate_code_count(const std::vector<int>& values,
std::size_t expected_size) {
const bool matches = values.size() == expected_size;
return matches;
}
void summarize_signs(const std::vector<int>& values) {
const int positive_count =
static_cast<int>(std::count_if(values.begin(), values.end(), [](int value) {
return value > 0;
}));
print_summary("positive values", positive_count, values.size());
}
void read_primary_name(const NameTable& table) {
auto name_future = launch_name_lookup(table, "primary");
std::println("Resolved primary name: {}", name_future.get());
}
}
namespace final_patterns {
std::string check_switch_value(int code) {
return code == 1 ? "enabled" : "disabled";
}
void check_switch_cases(const std::vector<int>& codes) {
for (const int code : codes) {
std::println("Switch value accepted: {}", check_switch_value(code));
}
}
bool sizes_are_equal(const std::vector<int>& values,
std::size_t expected_size) {
const bool matches = values.size() == expected_size;
return matches;
}
void show_positive_summary(int positive_count, std::size_t total_count) {
print_summary("positive values", positive_count, total_count);
}
void fetch_primary_record(const NameTable& table) {
auto name_future = launch_name_lookup(table, "primary");
std::println("Fetched name: {}", name_future.get());
}
}
}
// namespace sample5
int extra_control_flow_sample(int limit) {
int total = 0;
for (int value = 0; value < limit; ++value) {
if (value % 2 == 0) total += value;
}
return total;
}
int main() {
const std::vector<int> values{-3, 2, 0, 7};
const std::vector<std::string> names{"primary", "secondary"};
const sample5::NameTable table{{"primary", "Ada"}, {"secondary", "Linus"}};
std::println("Sizes match: {}", sample5::has_expected_size(values, 4));
std::println("Counts match: {}", sample5::contains_expected_count(names, 2));
sample5::check_switch_cases({0, 1, 2, 9});
sample5::print_switch_acceptance(1);
sample5::print_primary_name(table);
sample5::summarize_positive_values(values);
sample5::repeated_patterns::check_switch_cases({10, 20, 30});
sample5::repeated_patterns::verify_measurement_count(values, 4);
sample5::repeated_patterns::verify_label_count(names, 2);
sample5::repeated_patterns::report_switch_value(20);
sample5::repeated_patterns::report_positive_count(values);
sample5::additional_patterns::check_switch_cases({-1, 0, 1});
sample5::additional_patterns::validate_item_count(names, 2);
sample5::additional_patterns::validate_code_count(values, 4);
sample5::additional_patterns::summarize_signs(values);
sample5::additional_patterns::read_primary_name(table);
sample5::final_patterns::check_switch_cases({0, 1});
sample5::final_patterns::sizes_are_equal(values, 4);
sample5::final_patterns::show_positive_summary(2, values.size());
sample5::final_patterns::fetch_primary_record(table);
}
