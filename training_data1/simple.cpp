#pragma once
#include "Corpus.hpp"
#include "NeuralNetwork.hpp"
#include <algorithm>
#define LOG_TRAINING_PROGRESS(...) \
if (verbose) {                   \
std::println(__VA_ARGS__);     \
}
enum class A {
};
enum class B {
};
class C {
public:
void foo(A a) {}
void foo(B b) {}
};
int main() {
C c;
c.foo(A{});
c.foo(B{});
if (c) {
std::println("C is true");
} else {
std::println("C is false");
}
while (true) {
std::println("Enter a line of text (or 'exit' to quit): ");
std::print("> ");
std::string line;
if (!std::getline(std::cin, line) || line == "exit") {
std::println("Exiting...");
break;
}
std::println("You entered: '{}'", line);
}
}
