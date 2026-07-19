#pragma once
#include "Corpus.hpp"
#include "NeuralNetwork.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <print>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#define LOG_TRAINING_PROGRESS(...)
static constexpr bool verbose = true;
if (verbose){
std::println(__VA_ARGS__);
}
enum class A{
car,
monkey,
banana
};
enum class B{
amsterdam,
berlin,
copenhagen
};
class C{
public:
void foo(A a)
{}
void foo(B b)
{}
};
class D : public C{
public:
void foo(A a){
std::println("D::foo(A)");
}
void foo(B b){
std::println("D::foo(B)");
}
};
class E{
protected:
int value;
public:
E(int v)
: value(v)
{}
virtual ~E() = default;
virtual void print() const{
std::println("E: {}", value);
}
};
struct FTA : public E{
FTA(int v)
: E(v)
{}
void print() const override{
std::println("FTA: {}", value);
}
};
struct Finance : public E{
Finance(int v)
: E(v)
{}
void print() const override{
std::println("Finance: {}", value);
}
};
std::vector<std::unique_ptr<E>> create_objects(){
std::vector<std::unique_ptr<E>> objects;
objects.push_back(std::make_unique<FTA>(42));
objects.push_back(std::make_unique<Finance>(100));
return objects;
}
std::map<std::string, std::unique_ptr<E>> create_object_map(){
std::map<std::string, std::unique_ptr<E>> object_map;
object_map["fta"] = std::make_unique<FTA>(42);
object_map["finance"] = std::make_unique<Finance>(100);
return object_map;
};
template <typename E>
std::shared_ptr<E> create_shared_object(){
return std::make_shared<FTA>(42);
}
template <typename E>
std::shared_ptr<E> create_shared_finance_object(){
return std::make_shared<Finance>(100);
}
template <typename T>
class Wrapper{
public:
Wrapper(T value)
: value_(value)
{}
T get() const{
return value_;
}
private:
T value_;
};
class IInterface{
public:
virtual ~IInterface() = default;
void doSomething();
};
void IInterface::doSomething(){
std::println("IInterface::doSomething()");
}
void processInterface(IInterface& obj){
obj.doSomething();
}
int binary_search(const std::vector<int>& arr, int target){
int left = 0;
int right = static_cast<int>(arr.size()) - 1;
while (left <= right){
int mid = left + (right - left) / 2;
if (arr[mid] == target){
return mid;
// Target found
}
else if (arr[mid] < target){
left = mid + 1;
// Search in the right half
}
else{
right = mid - 1;
// Search in the left half
}
}
return -1;
// Target not found
}
void print_vector(const std::vector<int>& vec){
std::println("[");
for (const auto& elem : vec){
std::println(" {}", elem);
}
std::println("]");
}
template <typename T>
class CircularBuffer{
public:
CircularBuffer(size_t capacity)
: capacity_(capacity)
, buffer_(capacity)
, head_(0)
, size_(0)
{}
void push(const T& item){
buffer_[head_] = item;
head_ = (head_ + 1) % capacity_;
if (size_ < capacity_){
++size_;
}
}
std::vector<T> getBuffer() const{
std::vector<T> result;
for (size_t i = 0; i < size_; ++i){
size_t index = (head_ + capacity_ - size_ + i) % capacity_;
result.push_back(buffer_[index]);
}
return result;
}
private:
size_t capacity_;
std::vector<T> buffer_;
size_t head_;
size_t size_;
};
CircularBuffer<int> create_circular_buffer(size_t capacity){
return CircularBuffer<int>(capacity);
}
namespace math{
template <typename T>
T clamp(T value, T min, T max){
return std::max(min, std::min(value, max));
}
void test_clamp(){
int a = 5;
int b = math::clamp(a, 1, 10);
// b should be 5
int c = math::clamp(a, 6, 10);
// c should be 6
int d = math::clamp(a, 1, 4);
// d should be 4
std::println("Clamp results: b={}, c={}, d={}", b, c, d);
}
template <typename T>
T square(T value){
return value * value;
}
}
// namespace math
// namespace math
void check_if_stmt(){
if (true){
std::println("This is true");
}
else{
std::println("This is false");
}
}
void check_if_else_stmt(){
int x = 10;
if (x > 5){
std::println("x is greater than 5");
}
else{
std::println("x is not greater than 5");
}
}
void check_for_loop(){
for (int i = 0; i < 5; ++i){
std::println("Loop iteration: {}", i);
}
}
void check_while_loop(){
int count = 0;
while (count < 3){
std::println("Count is: {}", count);
++count;
}
}
void check_switch_case(int value){
switch (value){
case 1:
std::println("Value is 1");
break;
case 2:
std::println("Value is 2");
break;
case 3:
std::println("Value is 3");
break;
case 4:
std::println("Value is 4");
break;
default:
std::println("Value is something else");
break;
}
}
using namespace rllm;
using KMap = std::map<std::string, std::unique_ptr<E>>;
using KMapShared = std::map<std::string, std::shared_ptr<E>>;
using my_int = int;
using my_float = float;
using my_shared_int = std::shared_ptr<int>;
int main(){
C c;
printf("sizeof(A) = %zu\n", sizeof(A));
printf("sizeof(B) = %zu\n", sizeof(B));
c.foo(A{});
c.foo(B{});
if (c){
std::println("C is true");
}
else{
std::println("C is false");
}
while (true){
std::println("Enter a line of text (or 'exit' to quit): ");
std::print("> ");
std::string line;
if (!std::getline(std::cin, line) || line == "exit"){
std::println("Exiting...");
break;
}
std::println("You entered: '{}'", line);
}
auto thr = std::thread([]() {
std::println("Thread started");
std::this_thread::sleep_for(std::chrono::seconds(1));
std::println("Thread finished");
});
thr.join();
auto q = std::async(std::launch::async, [&]() {
std::println("Async task started");
std::this_thread::sleep_for(std::chrono::seconds(1));
std::println("Async task finished");
return 123;
});
q.wait();
auto v = q.get();
const auto local = []() {
std::println("Local lambda called");
};
local();
}
