#include <algorithm>
#include <array>
#include <print>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>
namespace validation_examples {
std::array<int, 3> make_coordinates(int x, int y, int z) {
return {x, y, z};
}
std::array<int, 3> make_coordinates(int x, int y, short z) {
return {x, y, static_cast<int>(z)};
}
std::array<int, 3> make_coordinates(int x, short y, int z) {
return {x, static_cast<int>(y), z};
}
int count_values_above(std::span<const int> values, int threshold) {
return static_cast<int>(std::ranges::count_if(values, [threshold](int value) { return value > threshold; }));
}
long count_values_above(std::span<const long> values, long threshold) {
return std::ranges::count_if(values, [threshold](long value) { return value > threshold; });
}
std::size_t count_values_above(const std::vector<unsigned>& values, unsigned threshold) {
return std::ranges::count_if(values, [threshold](unsigned value) { return value > threshold; });
}
std::string repeat_character(char character, int count) {
return std::string(static_cast<std::size_t>(count), character);
}
std::string repeat_character(char character, unsigned count) {
return std::string(count, character);
}
void print_reversed(const std::vector<int>& values) {
for (const int value : values | std::views::reverse) {
std::println("reverse value: {}", value);
}
}
void print_reversed_long(const std::vector<long>& values) {
for (const long value : values | std::views::reverse) {
std::println("reverse long: {}", value);
}
}
void print_reversed_unsigned(const std::vector<unsigned>& values) {
for (const unsigned value : values | std::views::reverse) {
std::println("reverse unsigned: {}", value);
}
}
void print_even(const std::vector<int>& values) {
for (const int value : values | std::views::filter([](int item) { return item % 2 == 0; })) {
std::println("even value: {}", value);
}
}
void print_first_three(const std::vector<int>& values) {
for (const int value : values | std::views::take(3)) {
std::println("first value: {}", value);
}
}
union IntegerBits {
int signed_value;
unsigned unsigned_value;
};
union IntegerBytes {
int signed_value;
std::array<unsigned char, sizeof(int)> bytes;
};
void report(bool verbose, std::string_view message) {
if (verbose) {
std::println("{}", message);
}
}
void report_value(bool verbose, int value) {
if (verbose) {
std::println("value: {}", value);
}
}
void report_positive(bool verbose, int value) {
if (verbose && value > 0) {
std::println("positive: {}", value);
}
}
void report_empty(bool verbose, std::string_view text) {
if (verbose && text.empty()) {
std::println("empty");
}
}
}
int main() {
const std::vector<int> values{1, 4, 7, 10};
const auto point = validation_examples::make_coordinates(2, 3, 5);
std::println("above: {}", validation_examples::count_values_above(values, 4));
std::println("repeated: {}", validation_examples::repeat_character('*', 4));
std::println("point: {}, {}, {}", point[0], point[1], point[2]);
validation_examples::print_reversed(values);
validation_examples::print_even(values);
validation_examples::print_first_three(values);
validation_examples::report(true, "complete");
validation_examples::report_value(true, 7);
validation_examples::report_positive(true, 7);
validation_examples::report_empty(true, "");
}
