// Syntax curriculum: preprocessing happens before ordinary C++ parsing.
// These examples cover object macros, function macros, token joining, and conditions.
#include <print>
#include <string_view>
#define SAMPLE7_VERSION_MAJOR 1
#define SAMPLE7_VERSION_MINOR 4
#define SAMPLE7_FEATURE_LOGGING 1
#define SAMPLE7_FEATURE_CACHE 1
#define SAMPLE7_DEFAULT_NAME "preprocessor"
#define SAMPLE7_JOIN_IMPL(left, right) left##right
#define SAMPLE7_JOIN(left, right) SAMPLE7_JOIN_IMPL(left, right)
#define SAMPLE7_STRINGIFY_IMPL(value) #value
#define SAMPLE7_STRINGIFY(value) SAMPLE7_STRINGIFY_IMPL(value)
#define SAMPLE7_MAXIMUM(left, right) ((left) > (right) ? (left) : (right))
#ifndef SAMPLE7_HEADER_GUARD
#define SAMPLE7_HEADER_GUARD
namespace sample7 {
#if defined(SAMPLE7_FEATURE_LOGGING)
constexpr bool logging_enabled = true;
#else
constexpr bool logging_enabled = false;
#endif
#if defined(SAMPLE7_FEATURE_CACHE)
constexpr bool cache_enabled = true;
#else
constexpr bool cache_enabled = false;
#endif
#ifdef SAMPLE7_DEFAULT_NAME
constexpr std::string_view default_name = SAMPLE7_DEFAULT_NAME;
#else
constexpr std::string_view default_name = "unnamed";
#endif
#ifndef SAMPLE7_BUFFER_SIZE
#define SAMPLE7_BUFFER_SIZE 256
#endif
#ifndef SAMPLE7_RETRY_COUNT
#define SAMPLE7_RETRY_COUNT 3
#endif
#if !defined(SAMPLE7_DISABLE_VALIDATION)
constexpr bool validation_enabled = true;
#else
constexpr bool validation_enabled = false;
#endif
#if !defined(SAMPLE7_CUSTOM_TIMEOUT)
#define SAMPLE7_CUSTOM_TIMEOUT 30
#endif
#if defined(SAMPLE7_FEATURE_LOGGING) && defined(SAMPLE7_FEATURE_CACHE)
constexpr bool combined_features_enabled = true;
#else
constexpr bool combined_features_enabled = false;
#endif
#if defined(SAMPLE7_VERSION_MAJOR) && SAMPLE7_VERSION_MAJOR >= 1
constexpr bool supported_version = true;
#else
constexpr bool supported_version = false;
#endif
#ifdef SAMPLE7_EXPERIMENTAL
constexpr std::string_view release_channel = "experimental";
#elif defined(SAMPLE7_BETA)
constexpr std::string_view release_channel = "beta";
#else
constexpr std::string_view release_channel = "stable";
#endif
void print_configuration() {
std::println("name: {}", default_name);
std::println("version: {}.{}", SAMPLE7_VERSION_MAJOR, SAMPLE7_VERSION_MINOR);
std::println("logging: {}", logging_enabled);
std::println("cache: {}", cache_enabled);
std::println("validation: {}", validation_enabled);
std::println("combined features: {}", combined_features_enabled);
std::println("supported version: {}", supported_version);
std::println("buffer size: {}", SAMPLE7_BUFFER_SIZE);
std::println("retry count: {}", SAMPLE7_RETRY_COUNT);
std::println("timeout: {}", SAMPLE7_CUSTOM_TIMEOUT);
std::println("release channel: {}", release_channel);
std::println("maximum: {}", SAMPLE7_MAXIMUM(7, 11));
std::println("version macro: {}", SAMPLE7_STRINGIFY(SAMPLE7_VERSION_MAJOR));
}
}
// namespace sample7
#endif
#define SAMPLE7_MAIN_ENABLED 1
#if defined(SAMPLE7_MAIN_ENABLED)
int extra_control_flow_sample(int limit) {
int total = 0;
for (int value = 0; value < limit; ++value) {
if (value % 2 == 0) total += value;
}
return total;
}
int main() {
sample7::print_configuration();
}
#endif
