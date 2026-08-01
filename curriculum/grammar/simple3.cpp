// Grammar curriculum: optional values represent results that may be absent.
// The examples combine scheduling, branching, loops, callbacks, and timeouts.
#include "CommandBuffer.hpp"
#include "NetworkClient.hpp"
#include "TaskScheduler.hpp"
#include <array>
#include <chrono>
#include <cstddef>
#include <future>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
enum class Status{
idle,
running,
finished,
failed,
};
std::string_view status_name(Status status){
switch (status){
case Status::idle:
return "idle";
case Status::running:
return "running";
case Status::finished:
return "finished";
case Status::failed:
return "failed";
}
return "unknown";
}
void check_status(Status status){
if (status == Status::finished){
std::println("Work finished successfully");
}
else if (status == Status::failed){
std::println("Work failed before completion");
}
else{
std::println("Work is currently {}", status_name(status));
}
}
bool check_size(std::span<const int> values, std::size_t expected_size){
const bool matches = values.size() == expected_size;
std::println("Size check: actual={}, expected={}, matches={}", values.size(), expected_size, matches);
return matches;
}
bool check_switch_value(int value){
switch (value){
case 4:
case 8:
case 16:
return true;
default:
return false;
}
}
class ByteStorage{
public:
explicit ByteStorage(std::size_t size)
: bytes_(size)
{}
void* data_at(std::size_t offset){
if (offset >= bytes_.size()){
return nullptr;
}
void* pointer = bytes_.data() + offset;
return pointer;
}
const void* data_at(std::size_t offset) const{
if (offset >= bytes_.size()){
return nullptr;
}
const void* pointer = bytes_.data() + offset;
return pointer;
}
std::size_t size() const{
return bytes_.size();
}
private:
std::vector<std::byte> bytes_;
};
void print_storage_addresses(ByteStorage& storage){
void* beginning = storage.data_at(0);
void* middle = storage.data_at(storage.size() / 2);
std::println("Storage addresses: beginning={}, middle={}", beginning, middle);
}
int count_values_above(std::span<const int> values, int threshold){
int matching_count = 0;
for (const int value : values){
if (value > threshold){
++matching_count;
}
}
std::println("Found {} values above {}", matching_count, threshold);
return matching_count;
}
int count_even_values(std::span<const int> values){
int even_count = 0;
for (const int value : values){
if (value % 2 == 0){
++even_count;
}
}
return even_count;
}
std::future<Status> start_worker(std::string worker_name){
return std::async(std::launch::async, [worker_name = std::move(worker_name)] {
std::println("Worker {} is starting", worker_name);
std::this_thread::sleep_for(std::chrono::milliseconds(10));
std::println("Worker {} has finished", worker_name);
return Status::finished;
});
}
std::future<int> start_calculation(int first, int second){
return std::async(std::launch::async, [first, second] {
const int result = first * second;
std::println("Calculated {} times {} = {}", first, second, result);
return result;
});
}
Status wait_for_worker(std::future<Status>& worker_future){
std::println("Waiting for worker status");
const Status status = worker_future.get();
std::println("Worker status received: {}", status_name(status));
return status;
}
int wait_for_calculation(std::future<int>& calculation_future){
std::println("Waiting for calculation result");
const int result = calculation_future.get();
std::println("Calculation result received: {}", result);
return result;
}
void wait_for_all_tasks(){
auto first_worker = start_worker("alpha");
auto second_worker = start_worker("beta");
const Status first_status = wait_for_worker(first_worker);
const Status second_status = wait_for_worker(second_worker);
std::println("All tasks completed: first={}, second={}", status_name(first_status), status_name(second_status));
}
std::optional<int> find_first_positive(std::span<const int> values){
for (const int value : values){
if (value > 0){
return value;
}
}
return std::nullopt;
}
void print_user_message(std::string_view user_name, std::string_view message){
std::println("User {} sent: {}", user_name, message);
}
void print_numeric_samples(){
const std::array<int, 8> samples{-9, -4, 0, 3, 7, 12, 18, 25};
for (std::size_t index = 0; index < samples.size(); ++index){
std::println("Sample {} has value {}", index, samples[index]);
}
const int positive_count = count_values_above(samples, 0);
const int even_count = count_even_values(samples);
std::println("Sample counts: positive={}, even={}", positive_count, even_count);
}
int extra_control_flow_sample(int limit) {
int total = 0;
for (int value = 0; value < limit; ++value) {
if (value % 2 == 0) total += value;
}
return total;
}
int main(){
ByteStorage storage(1024);
print_storage_addresses(storage);
const std::array<int, 6> values{-3, 1, 5, 7, 11, 20};
check_size(values, 6);
std::println("Switch value accepted: {}", check_switch_value(8));
auto calculation = start_calculation(6, 7);
const int answer = wait_for_calculation(calculation);
std::println("The calculated answer is {}", answer);
wait_for_all_tasks();
check_status(Status::finished);
print_user_message("operator", "processing complete");
print_numeric_samples();
if (const auto positive = find_first_positive(values)){
std::println("First positive value: {}", *positive);
}
return 0;
}
