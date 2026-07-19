#include "TaskQueue.hpp"
#include "WorkerPool.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <future>
#include <memory>
#include <print>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
struct Allocation{
void* data;
std::size_t size;
};
void* allocate_bytes(std::size_t size){
return std::malloc(size);
}
void release_bytes(void* data){
std::free(data);
}
Allocation make_allocation(std::size_t size){
return Allocation{allocate_bytes(size), size};
}
class BufferAllocator{
public:
explicit BufferAllocator(std::size_t capacity)
: storage_(capacity)
{}
void* allocate(std::size_t size){
if (offset_ + size > storage_.size()){
return nullptr;
}
void* result = storage_.data() + offset_;
offset_ += size;
return result;
}
void reset(){
offset_ = 0;
}
private:
std::vector<std::byte> storage_;
std::size_t offset_ = 0;
};
int count_positive(const std::vector<int>& values){
int count = 0;
for (int value : values){
if (value > 0){
++count;
}
}
std::println("Positive count: {}", count);
return count;
}
std::size_t count_long_names(const std::vector<std::string>& names){
const auto count = std::count_if(names.begin(), names.end(), [](const std::string& name) {
return name.size() > 8;
});
std::println("Long-name count: {}", count);
return static_cast<std::size_t>(count);
}
void print_summary(std::string_view label, int count, int total){
std::println("{}: {} of {} items", label, count, total);
}
int run_local_lambda(int value){
const auto local_transform = [value](int offset) {
std::println("Local transform: value={}, offset={}", value, offset);
return value + offset;
};
return local_transform(3);
}
void invoke_local_callback(){
const auto local_callback = [] {
std::println("Local callback invoked");
};
local_callback();
}
std::future<int> launch_background_sum(std::vector<int> values){
return std::async(std::launch::async, [values = std::move(values)] {
int sum = 0;
for (int value : values){
sum += value;
}
std::println("Background sum completed: {}", sum);
return sum;
});
}
std::future<std::string> launch_name_task(std::string name){
return std::async(std::launch::async, [name = std::move(name)] {
std::this_thread::sleep_for(std::chrono::milliseconds(5));
std::println("Async name task completed for {}", name);
return "processed-" + name;
});
}
void wait_for_tasks(){
auto sum_future = launch_background_sum({1, 2, 3, 4});
auto name_future = launch_name_task("worker");
const int sum = sum_future.get();
const std::string name = name_future.get();
std::println("Task results: sum={}, name={}", sum, name);
}
class TaskCounter{
public:
void record_success(){
++successful_count_;
std::println("Successful tasks: {}", successful_count_);
}
void record_failure(){
++failed_count_;
std::println("Failed tasks: {}", failed_count_);
}
int total_count() const{
return successful_count_ + failed_count_;
}
private:
int successful_count_ = 0;
int failed_count_ = 0;
};
template <typename T>
std::unique_ptr<T> allocate_object(T value){
return std::make_unique<T>(std::move(value));
}
template <typename T, std::size_t Size>
void print_array(const std::array<T, Size>& values){
for (std::size_t index = 0; index < values.size(); ++index){
std::println("array[{}] = {}", index, values[index]);
}
}
int main(){
BufferAllocator allocator(512);
void* first_block = allocator.allocate(64);
void* second_block = allocator.allocate(128);
std::println("Allocated blocks: first={}, second={}", first_block, second_block);
const int positive_count = count_positive({-2, 0, 4, 7});
print_summary("positive values", positive_count, 4);
TaskCounter counter;
counter.record_success();
counter.record_success();
counter.record_failure();
std::println("Total task count: {}", counter.total_count());
const int transformed = run_local_lambda(39);
std::println("Local lambda result: {}", transformed);
invoke_local_callback();
wait_for_tasks();
auto message = allocate_object(std::string{"ready"});
std::println("Allocated message: {}", *message);
print_array(std::array{2, 4, 6, 8});
allocator.reset();
return 0;
}
