#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <expected>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <print>
#include <ranges>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>
namespace sample9 {
using NumberArray = std::array<int, 4>;
using TextFunction = std::function<std::string(std::string_view)>;
using NumberVector = std::vector<int>;
using ScoreMap = std::map<std::string, int>;
using NamePair = std::pair<int, std::string>;
using ParseResult = std::expected<int, std::string>;
using OptionalName = std::optional<std::string>;
std::array<int, 3> make_coordinates(int x, int y, int z) {
return std::array<int, 3>{x, y, z};
}
int sum_array(const std::array<int, 5>& values) {
int total = 0;
for (const int value : values) {
total += value;
}
return total;
}
std::function<int(int, int)> select_operation(bool multiply) {
if (multiply) {
return std::function<int(int, int)>{
[](int left, int right) { return left * right; }
};
}
return std::function<int(int, int)>{
[](int left, int right) { return left + right; }
};
}
void run_callback(const std::function<void(std::string_view)>& callback,
std::string_view message) {
callback(message);
}
std::vector<int> keep_positive(const std::vector<int>& values) {
std::vector<int> result;
for (const int value : values) {
if (value > 0) {
result.push_back(value);
}
}
return result;
}
std::vector<std::string> map_keys(const std::map<std::string, int>& values) {
std::vector<std::string> keys;
for (const auto& [key, value] : values) {
static_cast<void>(value);
keys.push_back(key);
}
return keys;
}
std::map<std::string, int> count_words(const std::vector<std::string>& words) {
std::map<std::string, int> counts;
for (const std::string& word : words) {
++counts[word];
}
return counts;
}
std::map<int, std::vector<std::string>> group_names(
const std::vector<std::pair<int, std::string>>& entries) {
std::map<int, std::vector<std::string>> groups;
for (const std::pair<int, std::string>& entry : entries) {
groups[entry.first].push_back(entry.second);
}
return groups;
}
std::pair<int, std::string> make_record(int id, std::string name) {
return std::pair<int, std::string>{id, std::move(name)};
}
std::pair<std::string, bool> validate_name(std::string name) {
const bool valid = !name.empty();
return std::pair<std::string, bool>{std::move(name), valid};
}
std::expected<int, std::string> parse_positive(int value) {
if (value <= 0) {
return std::unexpected(std::string{"value must be positive"});
}
return value;
}
std::expected<std::string, int> find_name(
const std::map<int, std::string>& names,
int id) {
const auto match = names.find(id);
if (match == names.end()) {
return std::unexpected(id);
}
return match->second;
}
std::optional<std::string> lookup_value(
const std::map<int, std::string>& values,
int key) {
const auto match = values.find(key);
if (match == values.end()) {
return std::nullopt;
}
return match->second;
}
std::optional<int> first_positive(const std::vector<int>& values) {
for (const int value : values) {
if (value > 0) {
return std::optional<int>{value};
}
}
return std::nullopt;
}
using StringOrNumber = std::variant<std::string, int>;
using UserTuple = std::tuple<int, std::string, bool>;
using LookupTable = std::unordered_map<std::string, std::size_t>;
using UniqueNames = std::set<std::string>;
std::variant<int, std::string> describe_result(bool success) {
if (success) {
return std::variant<int, std::string>{42};
}
return std::variant<int, std::string>{std::string{"failed"}};
}
std::tuple<int, std::string, bool> make_user(int id,
std::string name,
bool active) {
return std::tuple<int, std::string, bool>{id, std::move(name), active};
}
std::unordered_map<std::string, int> build_index(
const std::vector<std::string>& names) {
std::unordered_map<std::string, int> index;
for (std::size_t position = 0; position < names.size(); ++position) {
index[names[position]] = static_cast<int>(position);
}
return index;
}
std::set<int> unique_values(const std::vector<int>& values) {
return std::set<int>{values.begin(), values.end()};
}
int sum_span(std::span<const int> values) {
int total = 0;
for (const int value : values) {
total += value;
}
return total;
}
std::unique_ptr<std::string> make_unique_name(std::string name) {
return std::make_unique<std::string>(std::move(name));
}
std::shared_ptr<std::vector<int>> make_shared_values(
std::vector<int> values) {
return std::make_shared<std::vector<int>>(std::move(values));
}
std::chrono::milliseconds timeout_for_attempt(int attempt) {
return std::chrono::milliseconds{attempt * 100};
}
std::filesystem::path append_filename(const std::filesystem::path& directory,
std::string_view filename) {
return directory / filename;
}
std::vector<int> sorted_values(std::vector<int> values) {
std::ranges::sort(values);
return values;
}
bool contains_value(const std::vector<int>& values, int target) {
return std::ranges::find(values, target) != std::ranges::end(values);
}
std::size_t count_positive_values(const std::vector<int>& values) {
return static_cast<std::size_t>(
std::ranges::count_if(values, [](int value) { return value > 0; }));
}
std::optional<int> minimum_value(const std::vector<int>& values) {
if (values.empty()) {
return std::nullopt;
}
return *std::ranges::min_element(values);
}
std::vector<int> collect_even_values(const std::vector<int>& values) {
auto even_values =
values | std::views::filter([](int value) { return value % 2 == 0; });
return std::vector<int>{std::ranges::begin(even_values),
std::ranges::end(even_values)};
}
std::vector<int> square_positive_values(const std::vector<int>& values) {
auto positive_squares =
values
| std::views::filter([](int value) { return value > 0; })
| std::views::transform([](int value) { return value * value; });
return std::vector<int>{std::ranges::begin(positive_squares),
std::ranges::end(positive_squares)};
}
std::vector<int> first_three_values(const std::vector<int>& values) {
const auto first_values = values | std::views::take(3);
return std::vector<int>{std::ranges::begin(first_values),
std::ranges::end(first_values)};
}
int sum_integer_range(int beginning, int ending) {
int total = 0;
for (const int value : std::views::iota(beginning, ending)) {
total += value;
}
return total;
}
void print_reverse(const std::vector<int>& values) {
for (const int value : values | std::views::reverse) {
std::println("reverse value: {}", value);
}
}
}
// namespace sample9
int main() {
const std::array<int, 5> values{1, -2, 3, -4, 5};
const std::vector<int> dynamic_values{1, 2, 2, 4};
const std::map<int, std::string> names{{1, "Ada"}, {2, "Grace"}};
const std::function<void(std::string_view)> printer =
[](std::string_view message) { std::println("{}", message); };
sample9::run_callback(printer, "standard types");
std::println("array sum: {}", sample9::sum_array(values));
std::println("span sum: {}", sample9::sum_span(dynamic_values));
std::println("unique count: {}", sample9::unique_values(dynamic_values).size());
std::println("lookup: {}", sample9::lookup_value(names, 1).value_or("missing"));
std::println("operation: {}", sample9::select_operation(false)(4, 7));
std::println("contains four: {}", sample9::contains_value(dynamic_values, 4));
std::println("positive count: {}", sample9::count_positive_values(dynamic_values));
std::println("range sum: {}", sample9::sum_integer_range(1, 6));
sample9::print_reverse(dynamic_values);
}
