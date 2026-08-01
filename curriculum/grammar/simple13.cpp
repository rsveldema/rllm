#include <algorithm>
#include <array>
#include <atomic>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <future>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#ifndef _LOCAL_INLINE
#define _LOCAL_INLINE static inline
#endif
#define FIELD_OFFSET(type, field) offsetof(type, field)
#define LOG_WARN(format, ...) std::fprintf(stderr, format, __VA_ARGS__)
#define METRIC_INC(name) metrics.name += 1
namespace sample13 {
using namespace std::literals;
struct Widget {
explicit Widget(std::string value = {}) : name(std::move(value)) {}
std::string name;
};
template <typename T>
using MyAllocList = std::list<T, std::allocator<T>>;
MyAllocList<Widget> allocator_examples() {
MyAllocList<Widget> alpha_widgets{Widget{"alpha"}};
MyAllocList<Widget> beta_widgets{Widget{"beta"}};
MyAllocList<Widget> gamma_widgets{Widget{"gamma"}};
alpha_widgets.splice(alpha_widgets.end(), beta_widgets);
alpha_widgets.splice(alpha_widgets.end(), gamma_widgets);
return alpha_widgets;
}
std::filesystem::path append_filename(const std::filesystem::path &directory,
std::string_view filename) {
return directory / filename;
}
std::array<std::filesystem::path, 3>
directory_examples(const std::filesystem::path &directory) {
auto alpha_path = directory / "alpha.txt";
auto beta_path = directory / "beta.txt";
auto gamma_path = directory / "gamma.txt";
return {alpha_path, beta_path, gamma_path};
}
struct SomeDataStructure {
int value = 0;
};
std::array<std::shared_ptr<SomeDataStructure>, 3> shared_pointer_examples() {
std::shared_ptr<SomeDataStructure> alpha_data =
std::make_shared<SomeDataStructure>();
std::shared_ptr<SomeDataStructure> beta_data =
std::make_shared<SomeDataStructure>();
std::shared_ptr<SomeDataStructure> gamma_data =
std::make_shared<SomeDataStructure>();
return {alpha_data, beta_data, gamma_data};
}
_LOCAL_INLINE unsigned shift_left(unsigned value, unsigned count) {
return value << count;
}
struct MacAddress {
std::array<unsigned char, 6> bytes{};
std::string to_string(char separator) const {
std::string result;
for (std::size_t index = 0; index < bytes.size(); ++index) {
if (index != 0)
result += separator;
result += std::to_string(bytes[index]);
}
return result;
}
};
std::array<std::string, 3> address_examples(const MacAddress &address) {
const std::string alpha_value = address.to_string(':');
const std::string beta_status = address.to_string('-');
const std::string gamma_result = address.to_string('.');
return {alpha_value, beta_status, gamma_result};
}
struct Sensor {
int value = 0;
};
std::string sensor_payload(const Sensor &sensor) {
std::string payload = "sensor";
payload += "," + std::to_string(sensor.value);
return payload;
}
class Jumpable {
public:
virtual ~Jumpable() = default;
virtual void jump() = 0;
};
class NamedJump final : public Jumpable {
public:
void jump() override {
jumped = true;
}
bool jumped = false;
};
void trampoline_alarm(int) {}
void install_alarm_handler() {
std::signal(SIGALRM, trampoline_alarm);
}
struct IoUringSqe {
unsigned personality = 0;
unsigned hardlink = 0;
};
void clear_submission(IoUringSqe *sqe) {
sqe->personality = 0;
sqe->hardlink = 0;
}
struct Iterator {
int *head = nullptr;
};
int iterator_example(int *first) {
Iterator alpha_iterator{first};
int *alpha_head = alpha_iterator.head;
Iterator beta_iterator{alpha_head};
int *beta_head = beta_iterator.head;
Iterator gamma_iterator{beta_head};
int *gamma_head = gamma_iterator.head;
return gamma_head == nullptr ? 0 : *gamma_head;
}
struct Storage {
explicit Storage(std::size_t size) : bytes(size) {}
void *data_at(std::size_t index) {
return bytes.data() + index;
}
std::vector<std::byte> bytes;
};
std::array<void *, 3> storage_examples(Storage &storage) {
void *beginning = storage.data_at(0);
void *middle = storage.data_at(storage.bytes.size() / 2);
void *ending = storage.data_at(storage.bytes.size() - 1);
return {beginning, middle, ending};
}
std::future<std::string> launch_server_request(std::string request) {
return std::async(std::launch::deferred,
[request = std::move(request)] { return request; });
}
std::optional<int> find_first_positive(const std::vector<int> &values) {
const auto found =
std::find_if(values.begin(), values.end(), [](const int value) {
return value > 0;
});
if (found == values.end())
return std::nullopt;
return *found;
}
void process(Widget &lvalue_argument) {
lvalue_argument.name += " processed";
}
void processWidget(std::shared_ptr<Widget> widget, int priority) {
if (widget)
widget->name += " priority " + std::to_string(priority);
}
std::array<int, 3> periodic_examples() {
auto periodic1s = 1;
auto periodic5s = 5;
auto periodic10s = 10;
return {periodic1s, periodic5s, periodic10s};
}
struct Record {
int identifier;
double value;
};
std::array<std::size_t, 3> offset_examples() {
const auto alpha_offset = FIELD_OFFSET(Record, identifier);
const auto beta_offset = FIELD_OFFSET(Record, value);
const auto gamma_offset = offsetof(Record, value);
return {alpha_offset, beta_offset, gamma_offset};
}
enum class TimeToLive : unsigned {
alpha = 1,
beta = 64,
gamma = 255,
};
std::array<unsigned, 3> ttl_examples() {
const auto alpha_value =
static_cast<std::underlying_type_t<TimeToLive>>(TimeToLive::alpha);
const auto beta_status =
static_cast<std::underlying_type_t<TimeToLive>>(TimeToLive::beta);
const auto gamma_result =
static_cast<std::underlying_type_t<TimeToLive>>(TimeToLive::gamma);
return {alpha_value, beta_status, gamma_result};
}
struct Metrics {
int alpha = 0;
int beta = 0;
int gamma = 0;
};
Metrics metric_examples() {
Metrics metrics;
METRIC_INC(alpha);
METRIC_INC(beta);
METRIC_INC(gamma);
return metrics;
}
std::array<int, 3> sum_examples() {
int sum1 = 1 + 2 + 3;
int sum2 = 4 + 5 + 6;
int sum3 = 7 + 8 + 9;
return {sum1, sum2, sum3};
}
std::array<std::string_view, 3> literal_examples() {
const auto alpha_value = "alpha"sv;
const auto beta_status = "beta"sv;
const auto gamma_result = "gamma"sv;
return {alpha_value, beta_status, gamma_result};
}
}
// namespace sample13
int extra_control_flow_sample(int limit) {
int total = 0;
for (int value = 0; value < limit; ++value) {
if (value % 2 == 0) total += value;
}
return total;
}
int main() {
const auto paths = sample13::directory_examples(".");
const auto periods = sample13::periodic_examples();
const auto sums = sample13::sum_examples();
return paths.empty() || periods[1] != 5 || sums[1] != 15;
}
