#include <cstddef>
#include <print>
#include <string>
#include <string_view>
#include <vector>
namespace sample8 {
int sum_until_limit(const std::vector<int>& values, int limit) {
std::size_t index = 0;
int sum = 0;
while (index < values.size() && sum < limit) {
sum += values[index];
++index;
}
return sum;
}
std::size_t count_leading_spaces(std::string_view text) {
std::size_t count = 0;
while (count < text.size() && text[count] == ' ') {
++count;
}
return count;
}
void print_countdown(int value) {
while (value > 0) {
std::println("countdown: {}", value);
--value;
}
}
int find_first_positive(const std::vector<int>& values) {
std::size_t index = 0;
while (index < values.size()) {
if (values[index] > 0) {
return values[index];
}
++index;
}
return 0;
}
void remove_trailing_spaces(std::string& text) {
while (!text.empty() && text.back() == ' ') {
text.pop_back();
}
}
int multiply_with_addition(int value, int times) {
int result = 0;
while (times > 0) {
result += value;
--times;
}
return result;
}
void print_grid(int rows, int columns) {
int row = 0;
while (row < rows) {
int column = 0;
while (column < columns) {
std::print("({}, {}) ", row, column);
++column;
}
std::println();
++row;
}
}
int request_valid_value(std::vector<int> attempts) {
std::size_t index = 0;
int value = 0;
do {
value = attempts[index];
++index;
} while (value < 0 && index < attempts.size());
return value;
}
void print_at_least_once(int count) {
int current = 0;
do {
std::println("iteration: {}", current);
++current;
} while (current < count);
}
int decimal_digit_count(unsigned int value) {
int digits = 0;
do {
++digits;
value /= 10;
} while (value != 0);
return digits;
}
std::string repeat_character(char character, int count) {
std::string result;
do {
result.push_back(character);
--count;
} while (count > 0);
return result;
}
int bounded_retry(const std::vector<bool>& outcomes) {
std::size_t attempt = 0;
bool succeeded = false;
do {
succeeded = outcomes[attempt];
++attempt;
} while (!succeeded && attempt < outcomes.size());
return static_cast<int>(attempt);
}
void advance_to_even(int& value) {
do {
++value;
} while (value % 2 != 0);
}
void process_batches(const std::vector<std::vector<int>>& batches) {
std::size_t batch_index = 0;
while (batch_index < batches.size()) {
const auto& batch = batches[batch_index];
std::size_t item_index = 0;
do {
if (!batch.empty()) {
std::println("item: {}", batch[item_index]);
++item_index;
}
} while (item_index < batch.size());
++batch_index;
}
}
}
// namespace sample8
int main() {
const std::vector<int> values{-4, -1, 3, 8, 12};
std::string padded = "training ";
int odd_value = 4;
std::println("sum: {}", sample8::sum_until_limit(values, 10));
std::println("positive: {}", sample8::find_first_positive(values));
std::println("digits: {}", sample8::decimal_digit_count(2048));
std::println("repeated: {}", sample8::repeat_character('*', 4));
std::println("attempts: {}", sample8::bounded_retry({false, false, true}));
sample8::remove_trailing_spaces(padded);
sample8::print_countdown(3);
sample8::print_grid(2, 3);
sample8::print_at_least_once(2);
sample8::advance_to_even(odd_value);
sample8::process_batches({{1, 2}, {}, {3, 4, 5}});
}
