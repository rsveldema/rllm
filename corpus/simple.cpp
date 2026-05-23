#pragma once
#include "Corpus.hpp"
#include "NeuralNetwork.hpp"
#include <algorithm>

#define LOG_TRAINING_PROGRESS(...) \
    if (verbose) \
    { \
        std::println(__VA_ARGS__); \
    }

enum class A {};
enum class B {};

class C
{
  public:
    void foo(A a) {}
    void foo(B b) {}
};

int main()
{
    C c;
    c.foo(A{});
    c.foo(B{});
}