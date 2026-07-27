#include <cstddef>
#include <future>
#include <map>
#include <print>
#include <string>
#include <utility>
#include <vector>
namespace sample6 {
enum class ConnectionState {
disconnected,
connecting,
connected,
failed
};
enum class AccessMode {
read_only,
read_write,
administrator
};
std::string describe_state(ConnectionState state) {
switch (state) {
case ConnectionState::disconnected:
return "disconnected";
case ConnectionState::connecting:
return "connecting";
case ConnectionState::connected:
return "connected";
case ConnectionState::failed:
return "failed";
}
return "unknown";
}
std::string describe_switch_case(int code) {
switch (code) {
case 100:
return "continue";
case 200:
return "complete";
case 404:
return "missing";
default:
return "invalid";
}
}
void check_switch_cases(const std::vector<int>& codes) {
for (const int code : codes) {
std::println("Switch case {} means {}", code, describe_switch_case(code));
}
}
bool has_expected_size(const std::vector<int>& values, std::size_t expected_size) {
const bool matches = values.size() == expected_size;
return matches;
}
bool has_expected_size(const std::vector<std::string>& values,
std::size_t expected_size) {
const bool matches = values.size() == expected_size;
return matches;
}
using NameTable = std::map<std::string, std::string>;
std::future<std::string> launch_name_lookup(const NameTable& table,
std::string name) {
return std::async(std::launch::deferred, [&table, name = std::move(name)] {
const auto entry = table.find(name);
return entry == table.end() ? std::string{"unknown"} : entry->second;
});
}
void show_primary_name(const NameTable& table) {
auto name_future = launch_name_lookup(table, "primary");
std::println("Primary result: {}", name_future.get());
}
using AddressTable = std::map<int, std::string>;
std::future<std::string> launch_name_lookup(const AddressTable& table,
int address) {
return std::async(std::launch::deferred, [&table, address] {
const auto entry = table.find(address);
return entry == table.end() ? std::string{"unassigned"} : entry->second;
});
}
void print_address_boundaries(const AddressTable& table) {
if (table.empty()) {
std::println("Address boundaries: empty");
return;
}
const auto beginning = table.begin();
const auto ending = table.rbegin();
std::println("Address boundaries: beginning={}, ending={}",
beginning->first,
ending->first);
}
namespace alternate {
enum class ResultCode {
success,
retry,
rejected
};
std::string check_switch_value(int code) {
return code >= 0 ? "accepted" : "rejected";
}
void check_switch_cases(const std::vector<int>& codes) {
for (const int code : codes) {
std::println("Switch value accepted: {}", check_switch_value(code));
}
}
bool validate_count(const std::vector<int>& values, std::size_t expected_size) {
const bool matches = values.size() == expected_size;
std::println("Expected size: {}, matches: {}", expected_size, matches);
return matches;
}
using NameTable = std::map<std::string, std::string>;
std::future<std::string> launch_name_lookup(const NameTable& table,
std::string name) {
return std::async(std::launch::deferred, [&table, name = std::move(name)] {
const auto entry = table.find(name);
return entry != table.end() ? entry->second : std::string{"missing"};
});
}
void resolve_primary(const NameTable& table) {
auto name_future = launch_name_lookup(table, "primary");
std::println("Resolved name: {}", name_future.get());
}
using AddressTable = std::map<int, std::string>;
void show_address_range(const AddressTable& table) {
if (!table.empty()) {
const auto beginning = table.begin();
const auto ending = table.rbegin();
std::println("Address boundaries: beginning={}, ending={}",
beginning->first,
ending->first);
}
}
}
// namespace alternate
}
// namespace sample6
int main() {
const std::vector<int> codes{100, 200, 301, 404};
const sample6::NameTable names{{"primary", "main"}, {"backup", "secondary"}};
const sample6::AddressTable addresses{{10, "alpha"}, {80, "omega"}};
const sample6::alternate::NameTable alternate_names{{"primary", "preferred"}};
const sample6::alternate::AddressTable alternate_addresses{
{20, "begin"},
{90, "end"}
};
sample6::check_switch_cases(codes);
std::println("Count matches: {}", sample6::has_expected_size(codes, 4));
sample6::show_primary_name(names);
sample6::print_address_boundaries(addresses);
sample6::alternate::check_switch_cases(codes);
sample6::alternate::validate_count(codes, 4);
sample6::alternate::resolve_primary(alternate_names);
sample6::alternate::show_address_range(alternate_addresses);
std::println("State: {}", sample6::describe_state(sample6::ConnectionState::connected));
}
