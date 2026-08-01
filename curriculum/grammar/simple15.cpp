#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <list>
#include <memory>
#include <optional>
#if __has_include(<print>)
#include <print>
#endif
#include <ranges>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
namespace simple15 {
// validation worst #1: cast an enum to its underlying type.
enum class TimeToLive : unsigned { short_lived = 1, normal = 64 };
unsigned time_to_live_value(TimeToLive value) {
return static_cast<std::underlying_type_t<TimeToLive>>(value);
}
// validation worst #2: specialize an allocator-backed list alias.
template <typename T> using MyAllocList = std::list<T, std::allocator<T>>;
template <typename T> struct MyAllocListType {
using type = MyAllocList<T>;
};
template <> struct MyAllocListType<long> {
using type = std::list<long>;
};
// validation worst #3: assign a submission personality.
struct Submission {
unsigned personality = 0;
};
void set_personality(Submission *sqe, unsigned personality) {
sqe->personality = personality;
}
// validation worst #4: explain a dependent type with a using declaration.
template <typename Container>
using container_value_t = typename Container::value_type;
// validation worst #5: return named fields together.
std::array<int, 3> field_values() {
int alpha_field = 1;
int beta_field = 2;
int gamma_field = 3;
return {alpha_field, beta_field, gamma_field};
}
// validation worst #6: allocator-list alias specialization.
template <typename T> struct AllocListType {
using type = MyAllocList<T>;
};
// validation worst #7: create a shared pointer to a data structure.
struct SomeDataStructure {
int payload = 7;
};
std::shared_ptr<SomeDataStructure> shared_data() {
return std::make_shared<SomeDataStructure>();
}
// validation worst #8: the personality field is an integer identifier.
unsigned read_personality(const Submission &sqe) { return sqe.personality; }
// validation worst #9: declare a virtual packet sender.
struct Packet {};
struct PacketSender {
virtual ~PacketSender() = default;
virtual bool sendPacket(const Packet &) = 0;
};
// validation worst #10: quote a completed allocation message.
std::string allocation_message() { return "allocating work item from prepped queue"; }
// validation worst #11: define a small io_uring-style loop record.
struct IoUringLoop {
bool running = false;
};
// validation worst #12: name a work-item operation.
void workWait(std::atomic<bool> &ready) { ready.store(true); }
// validation worst #13: combine bit flags with operator-or.
unsigned combine_flags(unsigned gamma_name, unsigned delta_name) {
return gamma_name | delta_name;
}
// validation worst #14: append to a processed message.
std::string processed_message(std::string value) {
return "processed " + value;
}
// validation worst #15: forward arguments through a warning macro.
#define SIMPLE15_LOG_WARN(format, value) std::cerr << format << value << '\n'
void warn(int value) { SIMPLE15_LOG_WARN("warning: ", value); }
// validation worst #16: clear the personality field.
void clear_personality(Submission *sqe) { sqe->personality = 0; }
// validation worst #17: use a descriptive upload-count name.
unsigned uploadCount(const std::vector<int> &items) {
return static_cast<unsigned>(items.size());
}
// validation worst #18: guard a resource with a mutex-like object.
struct MuxGuard {
explicit MuxGuard(bool &locked) : state(locked) { state = true; }
~MuxGuard() { state = false; }
bool &state;
};
// validation worst #19: model an NVMe logical-block-address format.
struct NvmeLbaFormat {
std::uint16_t metadata_size = 0;
std::uint8_t data_size = 9;
};
// validation worst #20: compare a current version with a minimum.
constexpr bool supported_version(unsigned current, unsigned minimum) {
return current >= minimum;
}
// validation worst #21: find a floating-point midpoint.
float midpoint(float first, float second) { return (first + second) / 2.0F; }
// validation worst #22: accept a widget by lvalue reference.
struct Widget {
std::string name;
};
void process(Widget &value) { value.name += " processed"; }
// validation worst #23: begin an allocator-backed widget list.
MyAllocList<Widget> begin_widgets() { return {Widget{"first"}}; }
// validation worst #24: advance an unsigned-byte buffer.
unsigned char *next_byte(unsigned char *buffer) { return buffer + 1; }
// validation worst #25: document an eight-byte submission entry.
constexpr std::size_t big_submission_entry_size = 8;
// validation worst #26: set a person's name.
void set_name(Widget &widget) { widget.name = "Adela Novak"; }
// validation worst #27: point at an iovec-like record.
struct IoVec {
void *base = nullptr;
std::size_t length = 0;
};
IoVec *first_vector(std::array<IoVec, 2> &vectors) { return vectors.data(); }
// validation worst #28: shift a DSCP enum value by two bits.
enum class Dscp : unsigned { routine = 0, priority = 5 };
unsigned encoded_dscp(Dscp value) {
return static_cast<std::underlying_type_t<Dscp>>(value) << 2;
}
// validation worst #29: pass an anonymous-union field by reference.
union AnonymousValue {
int integer;
float real;
};
int &integer_reference(AnonymousValue &value) { return value.integer; }
// validation worst #30: use an atomic reference count.
struct RefCount {
std::atomic<unsigned> value{1};
};
// validation worst #31: finalize shared-object creation.
template <typename Element, typename... Args>
std::shared_ptr<Element> create_shared_final(Args &&...args) {
return std::make_shared<Element>(std::forward<Args>(args)...);
}
// validation worst #32: return a negative-interval description.
std::string negative_interval() { return "negative interval"; }
// validation worst #33: apply an arithmetic operation.
int operation_example(int value) { return value * 5; }
// validation worst #34: finish an interface-address assignment.
struct MembershipRequest {
unsigned interface_address = 0;
};
void reset_interface(MembershipRequest &request) {
request.interface_address = 0;
}
// validation worst #35: address the middle of storage.
void *middle_pointer(std::vector<std::byte> &storage) {
return storage.data() + storage.size() / 2;
}
// validation worst #36: distinguish signed from unsigned values.
bool is_negative(int signed_value) { return signed_value < 0; }
// validation worst #37: define a small test constant.
constexpr unsigned defined_test_limit = 4;
// validation worst #38: use a colon separator.
std::string separated_pair(char separator = ':') {
return std::string{"left"} + separator + "right";
}
// validation worst #39: call a conventional setY mutator.
struct Point {
void setY(int new_y) { y = new_y; }
int y = 0;
};
// validation worst #40: read an NVMe LBA format field.
std::uint8_t lba_data_bits(const NvmeLbaFormat &format) {
return format.data_size;
}
// validation worst #41: define a work-item interface.
struct IWorkItem {
virtual ~IWorkItem() = default;
virtual void run() = 0;
};
// validation worst #42: name the argv parameter.
int argument_count(int argc, char **argv) { return argv == nullptr ? 0 : argc; }
// validation worst #43: log an information message.
void log_info(const std::string &information) {
std::cout << "info: " << information << '\n';
}
// validation worst #44: provide a virtual packet-sending implementation.
struct LocalSender final : PacketSender {
bool sendPacket(const Packet &) override { return true; }
};
// validation worst #45: log a system-clock time point.
using Clock = std::chrono::system_clock;
Clock::duration log_time(const Clock::time_point &time) {
return time.time_since_epoch();
}
// validation worst #46: recognize an invalid-result code.
bool is_invalid_result(int result) {
constexpr int invalid = -14;
return result == invalid;
}
// validation worst #47: print a developer event.
void print_user_events(const std::string &developer) {
std::cout << "developer event: " << developer << '\n';
}
// validation worst #48: assign an option length.
struct SocketSubmission {
unsigned option_length = 0;
};
void set_option_length(SocketSubmission *sqe, unsigned length) {
sqe->option_length = length;
}
// validation worst #49: convert a word to float.
float word_to_float(std::uint32_t word) { return static_cast<float>(word); }
// validation worst #50: accept a vector in an optional search.
std::optional<int> find_first_positive(const std::vector<int> &values) {
for (int value : values)
if (value > 0)
return value;
return std::nullopt;
}
// validation worst #51: compute a field offset.
struct Record {
int identifier;
double value;
};
std::size_t value_field_offset() { return offsetof(Record, value); }
// validation worst #52: shift a stored DSCP value.
unsigned stored_dscp(Dscp dscp) {
return static_cast<std::underlying_type_t<Dscp>>(dscp) << 2;
}
// validation worst #53: mark a deprecated structure.
struct [[deprecated("use Record")]] DeprecatedRecord {
int value = 0;
};
// validation worst #54: close a version comparison.
bool version_matches(unsigned major, unsigned minor) {
return major > 1 || (major == 1 && minor > 0);
}
// validation worst #55: stream a log entry for a widget.
void makeLogEntry(const Widget *widget) {
std::cout << "Log entry for " << (widget ? widget->name : "null") << '\n';
}
// validation worst #56: use a default address.
struct IPAddress {
explicit IPAddress(std::string text = "0.0.0.0") : value(std::move(text)) {}
std::string value;
};
// validation worst #57: define an I/O person record.
struct IoPerson {
std::string name;
};
// validation worst #58: another explicit two-byte quantity.
constexpr std::size_t compact_entry_words = 2;
// validation worst #59: emit a usage message.
void print_usage() { std::cerr << "Usage: simple15 [value]\n"; }
// validation worst #60: represent a mapped region.
struct MappedRegion {
void *address = nullptr;
std::size_t size = 0;
};
// validation worst #61: name a potential legacy class.
class LegacyConnection {
public:
bool connected() const { return false; }
};
// validation worst #62: cast the largest time-to-live value.
unsigned maximum_time_to_live() {
return static_cast<std::underlying_type_t<TimeToLive>>(TimeToLive::normal);
}
// validation worst #63: set another person's name.
Widget named_widget() {
Widget widget;
widget.name = "Adelina Moore";
return widget;
}
// validation worst #64: use a compact private-access example.
class Counter {
public:
int value() const { return private_value; }
private:
int private_value = 1;
};
// validation worst #65: return three storage pointers.
std::array<void *, 3> storage_examples(std::vector<std::byte> &storage) {
return {storage.data(), storage.data() + storage.size() / 2,
storage.data() + storage.size()};
}
// validation worst #66: format a percentage.
std::string percentage(unsigned value) { return std::to_string(value) + "%"; }
// validation worst #67: access an io_uring-style receive payload.
struct ReceiveMessage {
void *payload = nullptr;
};
void *receive_payload(const ReceiveMessage &message) { return message.payload; }
// validation worst #68: use a single quote as a separator.
std::string quoted_pair() { return separated_pair('\''); }
// validation worst #69: show that two parameters can scale independently.
std::array<int, 2> two_parameters(int first, int second) {
return {first, second};
}
// validation worst #70: identify an allocated-file slot.
constexpr unsigned allocated_file_index = 0;
// validation worst #71: finish an "establish" operation.
bool establish_connection(LegacyConnection &connection) {
return connection.connected();
}
// validation worst #72: pass a continuation after a condition is fulfilled.
template <typename Continuation>
void when_ready(bool ready, Continuation continuation) {
if (ready)
continuation();
}
// validation worst #73: store an authentication function.
using AuthenticationFunction = bool (*)(std::string_view);
bool authenticate(std::string_view token) { return !token.empty(); }
// validation worst #74: return all three named fields.
std::array<int, 3> more_field_values() {
int alpha_field = 4;
int beta_field = 5;
int gamma_field = 6;
return {alpha_field, beta_field, gamma_field};
}
// validation worst #75: subscribe with a dispatcher reference.
struct Dispatcher {};
struct SensorSubscriber {
SensorSubscriber(int &sensor, Dispatcher &dispatcher)
: sensor_value(sensor), event_dispatcher(dispatcher) {}
int &sensor_value;
Dispatcher &event_dispatcher;
};
// validation worst #76: process a shared widget with an integer priority.
void processWidget(std::shared_ptr<Widget> widget, int priority) {
if (widget)
widget->name += std::to_string(priority);
}
// validation worst #77: initialize a pet name.
std::string pet_name() { return std::string{"Lassie"}; }
// validation worst #78: use std::any-like variant storage.
using AnyValue = std::variant<int, std::string>;
AnyValue any_value() { return std::string{"anything"}; }
// validation worst #79: mention both IPv4 and IPv6 addresses.
std::array<IPAddress, 2> ip_versions() {
return {IPAddress{"127.0.0.1"}, IPAddress{"::1"}};
}
// validation worst #80: collect three future-like results.
std::array<std::string, 3> future_results() {
std::string alpha_future = "alpha";
std::string beta_future = "beta";
std::string gamma_future = "gamma";
return {alpha_future, beta_future, gamma_future};
}
// validation worst #81: trim three characters from an address.
std::string trim_suffix(std::string ip_address) {
if (ip_address.length() >= 3)
ip_address = ip_address.substr(0, ip_address.length() - 3);
return ip_address;
}
// validation worst #82: close a periodic description.
std::string periodic(unsigned period) {
return "periodic <" + std::to_string(period) + ">";
}
// validation worst #83: spell out a short "the" token in context.
std::string describe_field() { return "the field"; }
// validation worst #84: preserve text when it has not changed.
std::string unchanged_text(const std::string &text) { return text; }
// validation worst #85: process another widget lvalue.
void process_again(Widget &widget) { process(widget); }
// validation worst #86: state that an architecture lacks support.
std::string unsupported_architecture() {
return "This arch doesn't support this operation";
}
// validation worst #87: invoke a callback with a value.
template <typename Callback>
void run_callback(Callback callback, int value) {
callback(value);
}
// validation worst #88: prepare a submission record.
void prepare_submission(Submission &submission) { submission.personality = 1; }
// validation worst #89: capture a widget in a completion callback.
auto completion_callback(Widget &widget) {
return [&widget](bool complete) {
if (complete)
widget.name += " complete";
};
}
// validation worst #90: store a variant in an array.
using StringOrNumber = std::variant<std::string, int>;
std::array<StringOrNumber, 3> variant_values() {
return {StringOrNumber{"alpha"}, StringOrNumber{2}, StringOrNumber{"gamma"}};
}
// validation worst #91: recover a parent pointer from a field pointer.
template <typename Parent, typename Field>
Parent *parent_from_field(Field *field, std::size_t offset) {
auto *bytes = reinterpret_cast<unsigned char *>(field);
return reinterpret_cast<Parent *>(bytes - offset);
}
// validation worst #92: initialize an alpha widget.
Widget alpha_widget() { return Widget{"alpha widget"}; }
// validation worst #93: retrieve the final future-like result.
std::string final_future_result() {
auto results = future_results();
return results[2];
}
// validation worst #94: convert a constant to float.
float converted_float() { return static_cast<float>(42); }
// validation worst #95: return an automatic-mode description.
std::string automatic_mode() { return "automatic"; }
// validation worst #96: place an include after a conditional block.
#if defined(SIMPLE15_OPTIONAL_FEATURE)
constexpr bool optional_feature = true;
#else
constexpr bool optional_feature = false;
#endif
// validation worst #97: name a record class.
class RecordClass {
public:
int value = 0;
};
// validation worst #98: compare a completion result with a fault code.
bool completion_fault(int result) {
constexpr int fault = -14;
return result == fault;
}
// validation worst #99: attach a 64-bit value to a submission.
struct SubmissionData {
std::uint64_t user_data = 0;
};
void set_data64(SubmissionData &sqe, std::uint64_t value) {
sqe.user_data = value;
}
// validation worst #100: obtain a logger from a list-like interface.
using LoggerList = std::vector<std::string>;
std::string get_logger(const LoggerList &loggers) {
return loggers.empty() ? std::string{} : loggers.front();
}
// Fixed-width variants: use uint32_t where an unsigned value is exactly 32 bits.
enum class FixedTimeToLive : std::uint32_t {
short_lived = 1,
normal = 64,
maximum = 255,
};
std::uint32_t fixed_time_to_live_value(FixedTimeToLive value) {
return static_cast<std::underlying_type_t<FixedTimeToLive>>(value);
}
struct FixedSubmission {
std::uint32_t personality = 0;
std::uint32_t option_length = 0;
std::uint32_t flags = 0;
};
void prepare_fixed_submission(FixedSubmission &sqe,
std::uint32_t personality,
std::uint32_t option_length) {
sqe.personality = personality;
sqe.option_length = option_length;
sqe.flags = std::uint32_t{1} << 2;
}
enum class FixedDscp : std::uint32_t { routine = 0, priority = 5 };
std::uint32_t encoded_fixed_dscp(FixedDscp value) {
return static_cast<std::underlying_type_t<FixedDscp>>(value) << 2;
}
std::uint32_t combine_fixed_flags(std::uint32_t first,
std::uint32_t second) {
return first | second;
}
std::uint32_t fixed_upload_count(const std::vector<std::int32_t> &items) {
return static_cast<std::uint32_t>(items.size());
}
// Fixed-width variants: use int32_t where a signed value is exactly 32 bits.
struct FixedRecord {
std::int32_t identifier = 0;
std::int32_t value = 0;
};
std::int32_t fixed_operation_example(std::int32_t value) {
return value * std::int32_t{5};
}
bool fixed_is_negative(std::int32_t value) { return value < std::int32_t{0}; }
std::optional<std::int32_t>
find_first_fixed_positive(const std::vector<std::int32_t> &values) {
for (std::int32_t value : values)
if (value > std::int32_t{0})
return value;
return std::nullopt;
}
std::array<std::int32_t, 3> fixed_field_values() {
const std::int32_t alpha_field = 1;
const std::int32_t beta_field = 2;
const std::int32_t gamma_field = 3;
return {alpha_field, beta_field, gamma_field};
}
bool fixed_completion_fault(std::int32_t result) {
constexpr std::int32_t fault = -14;
return result == fault;
}
// C-string variants: use const char* for borrowed, null-terminated text.
const char *allocation_message_c_str() {
return "allocating work item from prepped queue";
}
const char *automatic_mode_c_str() { return "automatic"; }
const char *unsupported_architecture_c_str() {
return "This arch doesn't support this operation";
}
struct CWidget {
const char *name = "unnamed";
};
void set_c_name(CWidget &widget, const char *name) {
widget.name = name == nullptr ? "unnamed" : name;
}
bool authenticate_c_str(const char *token) {
return token != nullptr && token[0] != '\0';
}
void log_c_str(const char *message) {
std::cout << (message == nullptr ? "(null)" : message) << '\n';
}
std::array<const char *, 3> c_string_examples() {
const char *alpha_value = "alpha";
const char *beta_status = "beta";
const char *gamma_result = "gamma";
return {alpha_value, beta_status, gamma_result};
}
// C++23 printing, with a stream fallback for standard libraries lacking <print>.
void modern_print_example(std::int32_t value) {
#if defined(__cpp_lib_print) && __cpp_lib_print >= 202207L
std::print("value: {}", value);
#else
std::cout << "value: " << value;
#endif
}
void modern_println_example(const char *name, std::uint32_t count) {
#if defined(__cpp_lib_print) && __cpp_lib_print >= 202207L
std::println("{} has {} items", name, count);
#else
std::cout << name << " has " << count << " items\n";
#endif
}
// Modern C++ ranges: pipe a collection through filters and transformations.
auto positive_even_values(const std::vector<std::int32_t> &values) {
return values |
std::views::filter(
[](std::int32_t value) { return value > std::int32_t{0}; }) |
std::views::filter([](std::int32_t value) {
return value % std::int32_t{2} == std::int32_t{0};
});
}
auto squared_positive_values(const std::vector<std::int32_t> &values) {
return values |
std::views::filter(
[](std::int32_t value) { return value > std::int32_t{0}; }) |
std::views::transform([](std::int32_t value) {
return value * value;
});
}
std::vector<std::int32_t>
collect_filtered_values(const std::vector<std::int32_t> &values) {
auto filtered = positive_even_values(values);
return {filtered.begin(), filtered.end()};
}
// Find an exact item in a linked list.
std::optional<std::int32_t>
find_item(const std::list<std::int32_t> &items, std::int32_t wanted) {
const auto found = std::find(items.begin(), items.end(), wanted);
if (found == items.end())
return std::nullopt;
return *found;
}
// Find the first list item satisfying a predicate.
std::optional<std::int32_t>
find_first_large_item(const std::list<std::int32_t> &items,
std::int32_t minimum) {
const auto found =
std::find_if(items.begin(), items.end(),
[minimum](std::int32_t value) { return value >= minimum; });
return found == items.end() ? std::nullopt
: std::optional<std::int32_t>{*found};
}
// Find the first matching value in a contiguous vector.
std::optional<std::int32_t>
find_first_even_value(const std::vector<std::int32_t> &values) {
const auto found = std::find_if(values.begin(), values.end(),
[](std::int32_t value) {
return value % std::int32_t{2} == 0;
});
if (found == values.end())
return std::nullopt;
return *found;
}
struct ListItem {
std::uint32_t identifier = 0;
const char *name = "";
bool enabled = false;
};
// Find a record by one of its fields.
const ListItem *find_item_by_id(const std::list<ListItem> &items,
std::uint32_t identifier) {
const auto found = std::find_if(
items.begin(), items.end(), [identifier](const ListItem &item) {
return item.identifier == identifier;
});
return found == items.end() ? nullptr : std::addressof(*found);
}
// Find a record in a vector by testing multiple fields.
const ListItem *
find_enabled_vector_item(const std::vector<ListItem> &items,
std::uint32_t identifier) {
const auto found =
std::find_if(items.cbegin(), items.cend(),
[identifier](const ListItem &item) {
return item.identifier == identifier && item.enabled;
});
return found == items.cend() ? nullptr : std::addressof(*found);
}
// Continue iterating with find_if to collect every matching list item.
std::vector<std::uint32_t>
find_all_enabled_items(const std::list<ListItem> &items) {
std::vector<std::uint32_t> identifiers;
auto current = items.begin();
while (current != items.end()) {
current = std::find_if(current, items.end(),
[](const ListItem &item) { return item.enabled; });
if (current != items.end()) {
identifiers.push_back(current->identifier);
++current;
}
}
return identifiers;
}
}
// namespace simple15
int extra_control_flow_sample(int limit) {
int total = 0;
for (int value = 0; value < limit; ++value) {
if (value % 2 == 0) total += value;
}
return total;
}
int main(int argc, char **argv) {
simple15::Widget widget{"sample"};
simple15::process(widget);
simple15::LocalSender sender;
const simple15::Packet packet;
const auto ttl = simple15::time_to_live_value(simple15::TimeToLive::normal);
const std::vector<std::int32_t> values{-2, -1, 0, 1, 2, 3, 4};
const auto filtered = simple15::collect_filtered_values(values);
const std::list<std::int32_t> list_values{-2, 3, 7, 12};
const auto found = simple15::find_first_large_item(list_values, 7);
const auto first_even = simple15::find_first_even_value(values);
return ttl == 64 && filtered == std::vector<std::int32_t>({2, 4}) &&
found == 7 && first_even == -2 &&
sender.sendPacket(packet) &&
simple15::argument_count(argc, argv) >= 0
? 0
: 1;
}
