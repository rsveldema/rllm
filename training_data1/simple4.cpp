#include "RequestHandler.hpp"
#include "ServerConnection.hpp"
#include "StringProcessor.hpp"
#include <algorithm>
#include <array>
#include <future>
#include <iterator>
#include <map>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
template <typename T>
T clamp_between(T value, T minimum, T maximum){
return std::max(minimum, std::min(value, maximum));
}
template <typename T>
T restrict_range(T value, T lower_bound, T upper_bound){
return std::max(lower_bound, std::min(value, upper_bound));
}
bool has_expected_size(const std::vector<int>& values, std::size_t expected_size){
const bool matches = values.size() == expected_size;
std::println("Expected size: {}, actual size: {}", expected_size, values.size());
return matches;
}
bool contains_expected_count(const std::vector<std::string>& names, std::size_t expected_count){
const bool contains_expected = names.size() == expected_count;
std::println("Expected count: {}, actual count: {}", expected_count, names.size());
return contains_expected;
}
std::string check_switch_case(int code){
switch (code){
case 100:
return "continue";
case 200:
return "success";
case 301:
return "redirect";
case 404:
return "not found";
default:
return "unknown";
}
}
void check_switch_cases(const std::vector<int>& codes){
for (const int code : codes){
std::println("Switch case {} means {}", code, check_switch_case(code));
}
}
class AddressTable{
public:
void insert(std::string name, std::string address){
addresses_.insert_or_assign(std::move(name), std::move(address));
}
std::optional<std::string> lookup(std::string_view name) const{
const auto iterator = addresses_.find(std::string{name});
if (iterator == addresses_.end()){
return std::nullopt;
}
return iterator->second;
}
void print_boundaries() const{
if (addresses_.empty()){
std::println("Address table is empty");
return;
}
const auto beginning = addresses_.begin();
const auto ending = std::prev(addresses_.end());
std::println("Address boundaries: beginning={}, ending={}", beginning->first, ending->first);
}
private:
std::map<std::string, std::string, std::less<>> addresses_;
};
std::future<std::string> launch_name_lookup(const AddressTable& table, std::string name){
return std::async(std::launch::async, [&table, name = std::move(name)] {
const auto address = table.lookup(name);
if (address){
return name + "=" + *address;
}
return name + "=missing";
});
}
std::future<std::string> launch_server_request(std::string server_name, std::string request){
return std::async(std::launch::async, [server_name = std::move(server_name), request = std::move(request)] {
std::println("Server {} received request {}", server_name, request);
return server_name + ":" + request;
});
}
void print_lookup_result(std::future<std::string>& lookup_future){
const std::string result = lookup_future.get();
std::println("Lookup result: {}", result);
}
void print_server_result(std::future<std::string>& server_future){
const std::string response = server_future.get();
std::println("Server response: {}", response);
}
void print_user_events(std::string_view user_name){
std::println("User {} connected", user_name);
std::println("User {} opened a session", user_name);
std::println("User {} submitted a command", user_name);
std::println("User {} closed the session", user_name);
}
void print_service_events(std::string_view service_name){
std::println("Service {} is starting", service_name);
std::println("Service {} is ready", service_name);
std::println("Service {} is processing work", service_name);
std::println("Service {} has stopped", service_name);
}
std::string describe_interval(int value){
if (value < 0){
return "negative interval";
}
if (value == 0){
return "empty interval";
}
if (value < 10){
return "short interval";
}
return "long interval";
}
int return_intermediate_value(int first, int second){
const int intermediate = first + second;
return intermediate;
}
std::string return_interface_name(std::string_view prefix, int index){
const std::string interface_name = std::string{prefix} + std::to_string(index);
return interface_name;
}
void exercise_numeric_boundaries(){
const std::array<int, 8> values{-12, -1, 0, 5, 10, 19, 25, 40};
for (const int value : values){
const int limited = clamp_between(value, 0, 20);
std::println("Input value {} becomes {}", value, limited);
}
const int restricted = restrict_range(73, 10, 50);
std::println("Restricted value: {}", restricted);
}
int main(){
const std::vector<int> values{2, 4, 6, 8, 10, 12};
has_expected_size(values, 6);
check_switch_cases({100, 200, 301, 404, 500});
AddressTable table;
table.insert("primary", "10.0.0.1");
table.insert("secondary", "10.0.0.2");
table.insert("backup", "10.0.0.3");
table.print_boundaries();
auto name_future = launch_name_lookup(table, "primary");
auto server_future = launch_server_request("build-server", "compile");
print_lookup_result(name_future);
print_server_result(server_future);
print_user_events("developer");
print_service_events("scheduler");
std::println("Interval description: {}", describe_interval(7));
std::println("Intermediate value: {}", return_intermediate_value(20, 22));
std::println("Interface name: {}", return_interface_name("network", 3));
exercise_numeric_boundaries();
return 0;
}
