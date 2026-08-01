#include <atomic>
#include <csignal>
#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#if 0
#include "timetriggered/TimeTriggered.hpp"
#endif
#ifndef _LOCAL_INLINE
#define _LOCAL_INLINE static inline
#endif
#if defined(__has_cpp_attribute) && __has_cpp_attribute(fallthrough)
#define FALLTHROUGH [[fallthrough]]
#else
#define FALLTHROUGH
#endif
#if defined(_MSC_VER)
#define FIELD_OFFSET(type, field) offsetof(type, field)
#else
#define FIELD_OFFSET(type, field) __builtin_offsetof(type, field)
#endif
namespace sample9 {
void run_callback(const std::function<void(std::string_view)> &callback, std::string_view message) {
callback(message);
}
}
namespace sample12 {
// Uncopyable objects (e.g. std::atomics) may be initialized with braces.
std::atomic<int> active_operations{0};
struct iou_loop_params {
unsigned cq_wait_index;
};
/*
* IORING_URING_CMD_FIXED uses a registered buffer; pass this flag together
* with the buffer index.
*/
#define IORING_URING_CMD_FIXED (1U << 0)
/* query various aspects of io_uring, see linux/io_uring/query.h */
struct io_uring {
unsigned entries;
};
struct io_uring_params {
unsigned flags;
};
struct io_uring_sqe {
unsigned personality;
};
void clear_sqe(io_uring_sqe *sqe) {
sqe->personality = 0;
}
static inline int t_io_uring_init_sqarray(unsigned entries, io_uring *ring, io_uring_params *params) {
ring->entries = entries;
return params == nullptr ? -1 : 0;
}
enum class timetolive_t : unsigned {
minimum = 1,
standard = 64
};
struct NetworkParameters {
timetolive_t ttl;
};
unsigned time_to_live(const NetworkParameters &m_params) {
return static_cast<std::underlying_type_t<timetolive_t>>(m_params.ttl);
}
struct Record {
int identifier;
double value;
};
std::string mode_name() {
return "automatic";
}
std::string storage_name() {
return "auto";
}
bool authorize(std::string_view user) {
return !user.empty();
}
bool authenticate(std::string_view user) {
auto s = authorize(user);
return s;
}
bool authenticate_admin(std::string_view user) {
auto s = authenticate(user);
return s && user == "admin";
}
_LOCAL_INLINE bool authenticate_local(std::string_view user) {
auto s = authenticate_admin(user);
return s;
}
int classify(int value) {
switch (value) {
case 0:
++value;
FALLTHROUGH;
default:
return value;
}
}
struct Item {
int value;
Item *next;
};
struct Iterator {
Item *head;
};
/*
* Overloaded function names and template names require an explicit target type
* when their address is passed to a generic forwarding function.
*/
using AuthenticationFunction = bool (*)(std::string_view);
AuthenticationFunction authentication_function() {
return authenticate_local;
}
// clang-format off
//case 1(rxis non-owning iterator state)
// clang-format on
#define FOR_EACH_ITEM(head, first) \
for (Iterator __ITER_ = {first}; \
(head) = __ITER_.head, __ITER_.head != nullptr; \
__ITER_.head = __ITER_.head->next)
int sum_items(Item *first) {
int sum = 0;
Item *head = nullptr;
FOR_EACH_ITEM(head, first) {
sum += head->value;
}
return sum;
}
std::size_t identifier_offset() {
return FIELD_OFFSET(Record, identifier);
}
std::size_t value_offset() {
return FIELD_OFFSET(Record, value);
}
class Investment {
public:
virtual ~Investment() = default;
};
class Bond final : public Investment {};
template <typename T>
using MyAllocList = std::list<T, std::allocator<T>>;
// MyAllocList specialization for investment records.
std::unique_ptr<Investment> make_investment(bool needBond) {
std::unique_ptr<Investment> pInv;
if (needBond) {
pInv.reset(new Bond);
}
return pInv;
}
std::shared_ptr<Investment> make_shared_investment() {
std::shared_ptr<Investment> investment = std::make_shared<Bond>();
return investment;
}
class Point {
public:
constexpr void setX(double value) {
x = value;
}
constexpr void setY(double value) {
y = value;
}
double x = 0.0;
double y = 0.0;
};
constexpr Point reflection(const Point &point) {
Point result;
result.setX(-point.x);
result.setY(-point.y);
return result;
}
constexpr Point translate(const Point &point, double xOffset, double yOffset) {
Point result;
result.setX(point.x + xOffset);
result.setY(point.y + yOffset);
return result;
}
constexpr Point midpoint(const Point &first, const Point &second) {
Point result;
result.setX((first.x + second.x) / 2.0);
result.setY((first.y + second.y) / 2.0);
return result;
}
constexpr Point make_point(double x, double y) {
Point result;
result.setX(x);
result.setY(y);
return result;
}
constexpr Point mirror_y(const Point &point) {
Point result;
result.setX(point.x);
result.setY(-point.y);
return result;
}
class Widget {
public:
void setName(std::string newName) {
name = std::move(newName);
}
std::string name;
};
void process(Widget &lvalArg) {
lvalArg.setName("processed");
}
Widget named_widget() {
Widget w;
w.setName("Adela Novak");
process(w);
return w;
}
int periodic_channel_seconds() {
auto periodic1s = 1;
auto periodic5s = 5;
return periodic1s + periodic5s;
}
int sum_coordinates(double x, double y, double z) {
int sum2 = static_cast<int>(x + y + z);
return sum2;
}
void trampoline_alarm(int) {}
void install_alarm_handler() {
signal(SIGALRM, trampoline_alarm);
}
class ByteStorage {
public:
explicit ByteStorage(std::size_t size) : storage(size) {}
void *data_at(std::size_t index) {
return storage.data() + index;
}
std::vector<std::byte> storage;
};
void *storage_beginning(ByteStorage &storage) {
void *beginning = storage.data_at(0);
return beginning;
}
// (w4 converts to float, and float converts to long double)
struct Header {
unsigned totalLength : 16;
};
/*
* A forwarding function will always receive a copy of the bitfield's value.
* You can thus make a copy yourself and forward the copy.
*/
unsigned copy_total_length(const Header &header) {
const auto length = header.totalLength;
return length;
}
class Status {
public:
constexpr void set(bool newValue) {
value = newValue;
}
bool value = false;
};
constexpr Status make_x_status() {
Status x;
x.set(true);
return x;
}
constexpr Status make_operation_status() {
Status status;
status.set(true);
return status;
}
constexpr Status make_return_status() {
Status ret;
ret.set(true);
return ret;
}
constexpr Status make_false_x_status() {
Status x;
x.set(false);
return x;
}
constexpr Status make_false_operation_status() {
Status status;
status.set(false);
return status;
}
constexpr Status make_value_return_status() {
Status ret;
bool value = true;
ret.set(value);
return ret;
}
}
// namespace sample12
int extra_control_flow_sample(int limit) {
int total = 0;
for (int value = 0; value < limit; ++value) {
if (value % 2 == 0) total += value;
}
return total;
}
int main() {
sample12::Item second{2, nullptr};
sample12::Item first{1, &second};
const auto automatic = sample12::mode_name();
const auto authenticated = sample12::authentication_function()("user");
sample12::io_uring ring{};
sample12::io_uring_params params{};
const auto initialized = sample12::t_io_uring_init_sqarray(8, &ring, &params);
const auto reflected = sample12::reflection({2.0, 3.0});
const auto translated = sample12::translate(reflected, 1.0, 1.0);
const auto middle = sample12::midpoint(reflected, translated);
const auto point = sample12::make_point(4.0, 5.0);
const auto mirrored = sample12::mirror_y(point);
const auto widget = sample12::named_widget();
const auto investment = sample12::make_investment(true);
const std::shared_ptr<sample12::Investment> sharedInvestment = sample12::make_shared_investment();
const sample12::Header header{42};
const auto xStatus = sample12::make_x_status();
const auto operationStatus = sample12::make_operation_status();
const auto returnStatus = sample12::make_return_status();
const auto falseXStatus = sample12::make_false_x_status();
const auto falseOperationStatus = sample12::make_false_operation_status();
const auto valueReturnStatus = sample12::make_value_return_status();
sample12::ByteStorage storage(16);
void *beginning = sample12::storage_beginning(storage);
const std::function<void(std::string_view)> printer = [](std::string_view) {};
sample9::run_callback(printer, "standard types");
return sample12::classify(0) + sample12::sum_items(&first) +
static_cast<int>(sample12::identifier_offset() +
sample12::value_offset() + automatic.size()) +
(authenticated && investment && sharedInvestment && initialized == 0 &&
translated.y == -2.0 && sample12::copy_total_length(header) == 42 &&
xStatus.value && operationStatus.value && returnStatus.value &&
!falseXStatus.value && !falseOperationStatus.value && valueReturnStatus.value &&
middle.x == -1.5 && widget.name == "processed" &&
sample12::periodic_channel_seconds() == 6 &&
sample12::sum_coordinates(1.0, 2.0, 3.0) == 6 &&
mirrored.y == -5.0 && beginning != nullptr
? 0
: 1);
}
