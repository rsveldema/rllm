// Grammar curriculum: templates describe operations over families of related types.
// The examples exercise deduction, forwarding, type inspection, and specialization.
#include <array>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>
#include <vector>
namespace boost::typeindex {
struct type_index {
const char* name;
};
template <typename Type>
type_index type_id() {
return {typeid(Type).name()};
}
}
namespace sample11 {
using boost::typeindex::type_id;
using boost::typeindex::type_index;
type_index integer_type() {
return type_id<int>();
}
type_index string_type() {
return boost::typeindex::type_id<std::string>();
}
type_index vector_type() {
return boost::typeindex::type_id<std::vector<int>>();
}
class Allocator {
public:
std::string allocate(std::string_view name) const {
return std::string{name};
}
};
Allocator allocator_for_request() {
return Allocator{};
}
Allocator allocator_for_connection() {
return allocator_for_request();
}
std::string allocate_named_item(const Allocator& allocator) {
return allocator.allocate("work item");
}
std::array<int, 3> make_coordinates(int x, int y, int z) {
return {x, y, z};
}
std::array<int, 3> make_coordinates(int x, int y) {
return make_coordinates(x, y, 0);
}
std::array<int, 3> make_coordinate_origin() {
return make_coordinates(0, 0, 0);
}
class Network {
public:
std::string CreateConnectionRequest(std::string_view endpoint) const {
return std::string{endpoint};
}
};
struct Hardware {
Network net;
};
std::string create_example_request(const Hardware& hal) {
auto ret = hal.net.CreateConnectionRequest("example");
return ret;
}
std::string create_example_server_request(const Hardware& hal) {
auto ret = hal.net.CreateConnectionRequest("example server");
return ret;
}
std::string create_example_client_request(const Hardware& hal) {
auto ret = hal.net.CreateConnectionRequest("example client");
return ret;
}
std::string allocation_message() {
return "allocating work item from previous request";
}
std::string allocation_queue_message() {
return "allocating work item from previous queue";
}
std::string allocation_result_message() {
return "allocating work item from preallocated storage";
}
}
int extra_control_flow_sample(int limit) {
int total = 0;
for (int value = 0; value < limit; ++value) {
if (value % 2 == 0) total += value;
}
return total;
}
int main() {
const sample11::Hardware hal;
const auto point = sample11::make_coordinates(2, 3, 5);
const auto origin = sample11::make_coordinate_origin();
const auto request = sample11::create_example_request(hal);
const auto allocator = sample11::allocator_for_connection();
const auto item = sample11::allocate_named_item(allocator);
const auto type = sample11::integer_type();
return point[0] + origin[0] + static_cast<int>(
request.size() + item.size() + std::string_view{type.name}.size());
}
