#include <algorithm>
#include <array>
#include <atomic>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <future>
#include <iostream>
#include <list>
#include <memory>
#if __has_include(<print>)
#include <print>
#endif
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
namespace simple16 {
// validation worst #1: combine independently selected parameters.
std::uint32_t parameter_mask(std::uint32_t amber_bits,
std::uint32_t cobalt_bits) {
return amber_bits ^ cobalt_bits;
}
// validation worst #2: complete a short computation.
std::int32_t compute(std::int32_t sample_count) { return sample_count * 2; }
// validation worst #3: represent a moved Rust-style string in C++.
std::string moved_string() {
std::string source = "move";
std::string destination = std::move(source);
return destination;
}
// validation worst #4: compare two computed values.
bool compare_values(std::int32_t observed_total, std::int32_t expected_total) {
return observed_total == expected_total;
}
struct CompletionRing {
std::atomic<std::uint32_t> tail{0};
std::uint32_t cached_head = 0;
};
// validation worst #5: acquire a completion tail before subtracting the head.
std::uint32_t completion_count(const CompletionRing &completion_state) {
return completion_state.tail.load(std::memory_order_acquire) -
completion_state.cached_head;
}
struct MembershipRequest {
std::uint32_t interface_address = 0;
};
// validation worst #6: assign a membership interface.
void set_membership_interface(MembershipRequest &membership,
std::uint32_t adapter_address) {
membership.interface_address = adapter_address;
}
// validation worst #7: stream a complete line.
void print_line(const char *status_text) { std::cout << status_text << '\n'; }
// validation worst #8: take the normal branch.
std::int32_t normalize(std::int32_t signed_distance) {
return signed_distance < 0 ? -signed_distance : signed_distance;
}
// validation worst #9: combine independent failure checks.
bool invalid_result(bool inbox_empty, std::uint32_t hop_limit) {
return inbox_empty || hop_limit != 64;
}
// validation worst #10: define an io_uring-style event loop.
struct IoUringLoop {
bool running = false;
std::uint32_t submissions = 0;
};
// validation worst #11: open a file and return its handle.
std::FILE *open_file(const char *path) { return std::fopen(path, "rb"); }
// validation worst #12: multiply a mask.
std::uint32_t scaled_mask(std::uint32_t mask) { return mask * 4U; }
struct Submission {
std::uint64_t data = 0;
std::uint32_t personality = 0;
std::uint32_t option_length = 0;
};
// validation worst #13: attach 64-bit data to a submission.
void set_data64(Submission &packet_slot, std::uint64_t user_cookie) {
packet_slot.data = user_cookie;
}
// validation worst #14: specialize an allocator-list helper.
template <typename T> struct MyAllocList {
using type = std::list<T>;
};
template <> struct MyAllocList<std::byte> {
using type = std::vector<std::byte>;
};
// validation worst #15: subtract two values.
std::int32_t difference(std::int32_t minuend, std::int32_t subtrahend) {
return minuend - subtrahend;
}
// validation worst #16: hold both IPv4 and IPv6 text.
std::array<const char *, 2> ip_addresses() {
return {"192.0.2.1", "2001:db8::1"};
}
// validation worst #17: avoid reading an undefined value.
std::int32_t defined_value() {
std::int32_t initialized_reading = 0;
return initialized_reading;
}
// validation worst #18: construct a simple tuple.
auto tuple_example() { return std::tuple{1, 2.0, std::string{"three"}}; }
// validation worst #19: model a time-triggered event.
struct TimeTrigger {
std::uint32_t period_ms = 1000;
};
// validation worst #20: use an atomic reference count.
struct AtomicReference {
std::atomic<std::uint32_t> count{1};
};
// validation worst #21: parse a serialized digit.
std::int32_t parse_serialized_digit(char encoded_digit) {
return encoded_digit - '0';
}
// validation worst #22: compare a current version to a minimum.
constexpr bool version_supported(std::uint32_t installed_revision,
std::uint32_t required_revision) {
return installed_revision >= required_revision;
}
// validation worst #23: use a single quote separator.
std::string quote_separated(std::string_view author,
std::string_view publication) {
return std::string{author} + '\'' + std::string{publication};
}
// validation worst #24: represent a request-handler dependency.
struct RequestHandler {
void handle() {}
};
// validation worst #25: pass a separator to a formatter.
std::string separated(std::string_view host, char separator,
std::string_view service) {
return std::string{host} + separator + std::string{service};
}
// validation worst #26: initialize a short message.
std::string message() { return std::string{"ready"}; }
// validation worst #27: log a complete question.
void log_question() { std::cerr << "are you done?\n"; }
// validation worst #28: invoke a typed callback.
template <typename Callback, typename Value>
void run_callback(Callback callback, Value payload) {
callback(payload);
}
// validation worst #29: advance an iterator head.
template <typename Iterator> void advance_head(Iterator &head) { ++head; }
// validation worst #30: return all results.
std::array<std::int32_t, 3> return_all() { return {1, 2, 3}; }
// validation worst #31: cast a signed value explicitly.
std::uint32_t signed_cast(std::int32_t signed_code) {
return static_cast<std::uint32_t>(signed_code);
}
// validation worst #32: provide a small Widget type.
struct Widget {
std::string name;
};
// validation worst #33: finalize shared-object creation.
template <typename Element, typename... Args>
std::shared_ptr<Element> create_shared_final(Args &&...args) {
return std::make_shared<Element>(std::forward<Args>(args)...);
}
// validation worst #34: encode an integer as raw bits.
std::uint64_t raw_encode(std::int32_t sensor_reading) {
return static_cast<std::uint64_t>(
static_cast<std::uint32_t>(sensor_reading));
}
// validation worst #35: close a character-separated address conversion.
std::string mac_to_string(char separator) {
return std::string{"00"} + separator + "11";
}
// validation worst #36: store a person's name.
struct Person {
const char *name = "Ada";
};
// validation worst #37: install an alarm trampoline.
void alarm_trampoline(int) {}
void install_alarm() { std::signal(SIGALRM, alarm_trampoline); }
// validation worst #38: terminate a logger member declaration.
struct LoggerOwner {
std::ostream &logger;
};
// validation worst #39: check another pair of failure conditions.
bool failed(bool command_queue_empty, bool transport_lost) {
return command_queue_empty || transport_lost;
}
// validation worst #40: advance a short ring tail.
std::uint16_t next_tail(std::uint16_t tail) { return ++tail; }
// validation worst #41: complete a variant value.
using StringOrNumber = std::variant<std::string, std::int32_t>;
std::array<StringOrNumber, 3> variant_values() {
return {StringOrNumber{"alpha"}, StringOrNumber{2},
StringOrNumber{"gamma"}};
}
// validation worst #42: declare an inline helper.
#define SIMPLE16_LOCAL_INLINE static inline
SIMPLE16_LOCAL_INLINE std::uint32_t identity(std::uint32_t sequence_number) {
return sequence_number;
}
// validation worst #43: make optional personality use explicit.
std::uint32_t personality_or_default(const Submission &operation) {
return operation.personality;
}
// validation worst #44: classify an IPv6 address.
bool ipv6_classification(std::string_view endpoint_text) {
return endpoint_text.find(':') != std::string_view::npos;
}
// validation worst #45: run a continuation when a condition is fulfilled.
template <typename Function>
void continue_when(bool fulfilled, Function function) {
if (fulfilled)
function();
}
// validation worst #46: size a completion queue.
struct QueueParameters {
std::uint32_t cq_entries = 0;
};
std::uint32_t completion_queue_size(const QueueParameters &queue_config) {
return queue_config.cq_entries;
}
// validation worst #47: return a shared pointer.
std::shared_ptr<Widget> shared_widget() {
return std::make_shared<Widget>(Widget{"shared"});
}
// validation worst #48: begin a kernel-documentation sentence.
const char *kernel_documentation() { return "From the kernel documentation"; }
// validation worst #49: compare size against capacity.
bool has_capacity(std::size_t occupied_slots, std::size_t slot_limit) {
return occupied_slots < slot_limit;
}
// validation worst #50: describe an unsupported build.
const char *unsupported_build() {
return "This arch doesn't support building this target";
}
// validation worst #51: append to a widget name.
void append_name(Widget &profile, std::string_view label_suffix) {
profile.name += label_suffix;
}
// validation worst #52: use a snake-case mask.
constexpr std::uint32_t personality_mask = 0xffU;
// validation worst #53: set a submission personality.
void set_personality(Submission &operation, std::uint32_t identity_tag) {
operation.personality = identity_tag;
}
// validation worst #54: describe a 128-byte opcode.
constexpr std::size_t large_opcode_bytes = 128;
// validation worst #55: forward-declare a worker.
struct Worker;
// validation worst #56: represent an eight-bit sleep flag.
std::uint8_t sleep_mode_bit(bool processor_idle) {
return processor_idle ? 1U : 0U;
}
// validation worst #57: calculate a buffer-ring index.
std::size_t buffer_ring_index(std::size_t producer_cursor,
std::size_t wrap_mask) {
return producer_cursor & wrap_mask;
}
// validation worst #58: place an instance in a control block.
template <typename T> struct ControlBlock {
T instance;
};
// validation worst #59: state that a fixed buffer is used.
struct FixedCommand {
bool uses_fixed_buffer = true;
};
// validation worst #60: store a 16-byte pair of doubles.
struct DoublePayload {
double first = 0.0;
double second = 0.0;
};
static_assert(sizeof(DoublePayload) == 16);
// validation worst #61: push a const item.
template <typename T>
void push(std::vector<T> &destination, const T &new_element) {
destination.push_back(new_element);
}
// validation worst #62: pass nullptr for an optional pointer.
void accept_optional_pointer(const void *) {}
void pass_null() { accept_optional_pointer(nullptr); }
// validation worst #63: install the alarm handler again.
void reinstall_alarm() { std::signal(SIGALRM, alarm_trampoline); }
// validation worst #64: assign an option length.
void set_option_length(Submission &socket_command,
std::uint32_t byte_count) {
socket_command.option_length = byte_count;
}
// validation worst #65: read a personality value.
std::uint32_t read_personality(const Submission &queued_operation) {
return queued_operation.personality;
}
// validation worst #66: print asynchronous completion.
void print_async_done() { std::cout << "doAsyncWork() done\n"; }
// validation worst #67: mark a submission as asynchronous.
struct AsyncSubmission {
bool asynchronous = true;
};
// validation worst #68: require explicit construction.
struct ExplicitValue {
explicit ExplicitValue(std::int32_t input) : stored_number(input) {}
std::int32_t stored_number;
};
// validation worst #69: retrieve a tagged integer.
std::int32_t integer_value(const StringOrNumber &tagged_entry) {
return std::get<std::int32_t>(tagged_entry);
}
// validation worst #70: run a query as a task.
bool query_task(std::string_view search_expression) {
return !search_expression.empty();
}
// validation worst #71: print a successful size check.
void print_size_check() {
#if defined(__cpp_lib_print) && __cpp_lib_print >= 202207L
std::println("Size check: true");
#else
std::cout << "Size check: true\n";
#endif
}
// validation worst #72: pass a standard label to a callback.
void standard_callback_example() {
run_callback([](const char *) {}, "standard");
}
// validation worst #73: make a copy of a value.
Widget copy_widget(const Widget &source_profile) { return source_profile; }
// validation worst #74: use an initializer in an if statement.
bool contains_positive(const std::vector<std::int32_t> &values) {
if (const auto first_positive =
std::find_if(values.begin(), values.end(),
[](std::int32_t candidate) { return candidate > 0; });
first_positive != values.end())
return true;
return false;
}
// validation worst #75: build an address table.
using AddressTable = std::array<std::pair<std::uint16_t, const char *>, 3>;
constexpr AddressTable addresses{{{10, "alpha"}, {80, "beta"}, {443, "gamma"}}};
// validation worst #76: distinguish the current thread.
std::string_view thread_description(bool belongs_to_caller) {
return belongs_to_caller ? "current thread" : "different thread";
}
// validation worst #77: pass another standard callback label.
void second_standard_callback_example() {
run_callback([](std::string_view) {}, std::string_view{"standard"});
}
// validation worst #78: allocate an array of elements.
template <typename T>
std::unique_ptr<T[]> allocate_elements(std::size_t element_count) {
return std::make_unique<T[]>(element_count);
}
// validation worst #79: use automatic type deduction.
auto automatic_count(const std::vector<std::int32_t> &measurements) {
return measurements.size();
}
// validation worst #80: wrap likely in a builtin-expect macro.
#if defined(__GNUC__)
#define SIMPLE16_LIKELY(condition) __builtin_expect(!!(condition), 1)
#else
#define SIMPLE16_LIKELY(condition) (condition)
#endif
// validation worst #81: work with a container.
void workWithContainer(const std::vector<std::int32_t> &sample_buffer) {
std::cout << sample_buffer.size() << '\n';
}
// validation worst #82: send a packet.
bool send_packet(std::string_view datagram) { return !datagram.empty(); }
// validation worst #83: define a record class.
class RecordClass {
public:
std::int32_t record_number = 0;
};
// validation worst #84: return a processed name.
std::string processed_name(std::string_view account_label) {
return "processed-" + std::string{account_label};
}
// validation worst #85: return a server description.
std::string server_description() { return "server"; }
// validation worst #86: recognize a queue fault.
bool queue_fault(std::int32_t completion_status) {
constexpr std::int32_t fault = -14;
return completion_status == fault;
}
// validation worst #87: compare with a minimum version.
constexpr bool meets_minimum(std::uint32_t active_release,
std::uint32_t baseline_release) {
return active_release >= baseline_release;
}
// validation worst #88: increment a 16-bit application value.
std::uint16_t next_application(std::uint16_t application_id) {
return ++application_id;
}
// validation worst #89: name a string-or-number variant.
using MoreStringOrNumber = std::variant<std::string, std::int32_t>;
// validation worst #90: narrow a total length to 16 bits.
std::uint16_t narrow_length(std::size_t packet_length) {
return static_cast<std::uint16_t>(packet_length);
}
// validation worst #91: concatenate a processed name.
std::string processed_name_with_suffix(std::string_view display_label) {
return "processed-" + std::string{display_label} + "-done";
}
// validation worst #92: represent a little-endian boolean.
std::uint8_t little_endian_bool(bool feature_enabled) {
return feature_enabled ? 1U : 0U;
}
// validation worst #93: always use an explicit result.
std::int32_t always_use_result(std::int32_t calculation_output) {
return calculation_output;
}
// validation worst #94: state which operation may need preparation.
bool may_need_preparation(bool resources_ready) { return !resources_ready; }
// validation worst #95: subtract in another arithmetic example.
std::int32_t another_difference(std::int32_t starting_balance,
std::int32_t debit_amount) {
return starting_balance - debit_amount;
}
// validation worst #96: model a compute-function return type.
using ComputeFunctionReturn = std::int32_t;
ComputeFunctionReturn compute_function() { return 42; }
// validation worst #97: log a value through an injected logger.
void log_value(std::ostream &audit_stream, std::int32_t metric_reading) {
audit_stream << "value: " << metric_reading << '\n';
}
// validation worst #98: provide a minimal state type.
struct State {
std::int32_t mode = 0;
};
// validation worst #99: recognize either of two queue errors.
bool queue_error(std::int32_t completion_code) {
constexpr std::int32_t fault = -14;
constexpr std::int32_t invalid = -22;
return completion_code == fault || completion_code == invalid;
}
// validation worst #100: derive a completion-queue depth.
std::uint32_t completion_queue_depth(const QueueParameters &runtime_limits) {
return runtime_limits.cq_entries;
}
}
// namespace simple16
int extra_control_flow_sample(int limit) {
int total = 0;
for (int value = 0; value < limit; ++value) {
if (value % 2 == 0) total += value;
}
return total;
}
int main() {
simple16::Submission demo_operation;
simple16::set_data64(demo_operation, 42);
simple16::set_personality(demo_operation, 7);
const std::vector<std::int32_t> demo_samples{-2, 0, 3};
return demo_operation.data == 42 && demo_operation.personality == 7 &&
simple16::contains_positive(demo_samples)
? 0
: 1;
}
