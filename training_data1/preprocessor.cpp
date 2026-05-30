// Training data: preprocessor directives
#pragma once

#ifndef PREPROCESSOR_EXAMPLE_HPP
#define PREPROCESSOR_EXAMPLE_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// Platform detection
#if defined(_WIN32) || defined(_WIN64)
#define PLATFORM_WINDOWS 1
#define PLATFORM_UNIX 0
#elif defined(__linux__)
#define PLATFORM_WINDOWS 0
#define PLATFORM_UNIX 1
#define PLATFORM_LINUX 1
#elif defined(__APPLE__)
#define PLATFORM_WINDOWS 0
#define PLATFORM_UNIX 1
#define PLATFORM_MACOS 1
#else
#error "Unsupported platform"
#endif

// Compiler detection
#if defined(__clang__)
#define COMPILER_CLANG 1
#define COMPILER_GCC 0
#define COMPILER_MSVC 0
#elif defined(__GNUC__)
#define COMPILER_CLANG 0
#define COMPILER_GCC 1
#define COMPILER_MSVC 0
#elif defined(_MSC_VER)
#define COMPILER_CLANG 0
#define COMPILER_GCC 0
#define COMPILER_MSVC 1
#endif

// Architecture detection
#if defined(__x86_64__) || defined(_M_X64)
#define ARCH_X64 1
#define ARCH_BITS 64
#elif defined(__i386__) || defined(_M_IX86)
#define ARCH_X86 1
#define ARCH_BITS 32
#elif defined(__aarch64__)
#define ARCH_ARM64 1
#define ARCH_BITS 64
#endif

// Debug vs Release
#ifdef NDEBUG
#define DEBUG_BUILD 0
#define ASSERT(expr) ((void)0)
#define DEBUG_ONLY(x)
#else
#define DEBUG_BUILD 1
#define ASSERT(expr) assert(expr)
#define DEBUG_ONLY(x) x
#endif

// Version macros
#define VERSION_MAJOR 2
#define VERSION_MINOR 5
#define VERSION_PATCH 1
#define MAKE_VERSION(major, minor, patch) \
  ((major) * 10000 + (minor) * 100 + (patch))
#define CURRENT_VERSION MAKE_VERSION(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)

#if CURRENT_VERSION >= MAKE_VERSION(2, 0, 0)
#define FEATURE_NEW_API 1
#else
#define FEATURE_NEW_API 0
#endif

// Attribute macros
#if COMPILER_GCC || COMPILER_CLANG
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define NOINLINE __attribute__((noinline))
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#define PACKED __attribute__((packed))
#define DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif COMPILER_MSVC
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define NOINLINE __declspec(noinline)
#define ALWAYS_INLINE __forceinline
#define PACKED
#define DEPRECATED(msg) __declspec(deprecated(msg))
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define NOINLINE
#define ALWAYS_INLINE inline
#define PACKED
#define DEPRECATED(msg)
#endif

// Stringify helpers
#define STRINGIFY_IMPL(x) #x
#define STRINGIFY(x) STRINGIFY_IMPL(x)
#define CONCAT_IMPL(a, b) a##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define UNIQUE_NAME(prefix) CONCAT(prefix, __LINE__)

// Feature flags
#ifndef FEATURE_LOGGING
#define FEATURE_LOGGING 1
#endif

#ifndef FEATURE_METRICS
#define FEATURE_METRICS 0
#endif

#ifndef FEATURE_THREADING
#define FEATURE_THREADING 1
#endif

#if FEATURE_LOGGING
#define LOG(fmt, ...) log_message(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) log_debug(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) log_warn(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) log_error(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#else
#define LOG(fmt, ...) ((void)0)
#define LOG_DEBUG(fmt, ...) ((void)0)
#define LOG_WARN(fmt, ...) ((void)0)
#define LOG_ERROR(fmt, ...) ((void)0)
#endif

#if FEATURE_METRICS
#define METRIC_INC(name) metrics_increment(name)
#define METRIC_TIMER(name) MetricTimer UNIQUE_NAME(_timer_)(name)
#else
#define METRIC_INC(name) ((void)0)
#define METRIC_TIMER(name)
#endif

// C++ standard version checks
#if __cplusplus >= 202002L
#define HAS_CPP20 1
#define CONSTEVAL consteval
#define NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
#define HAS_CPP20 0
#define CONSTEVAL constexpr
#define NO_UNIQUE_ADDRESS
#endif

#if __cplusplus >= 201703L
#define HAS_CPP17 1
#define NODISCARD [[nodiscard]]
#define MAYBE_UNUSED [[maybe_unused]]
#define FALLTHROUGH [[fallthrough]]
#else
#define HAS_CPP17 0
#define NODISCARD
#define MAYBE_UNUSED
#define FALLTHROUGH
#endif

// Integer type aliases via macros
#define DEFINE_INT_TYPES     \
  using i8 = std::int8_t;    \
  using i16 = std::int16_t;  \
  using i32 = std::int32_t;  \
  using i64 = std::int64_t;  \
  using u8 = std::uint8_t;   \
  using u16 = std::uint16_t; \
  using u32 = std::uint32_t; \
  using u64 = std::uint64_t; \
  using f32 = float;         \
  using f64 = double;

DEFINE_INT_TYPES

// Safe cast macro
#define SAFE_CAST(T, val)                                \
  ([&]() -> T {                                          \
    auto result = static_cast<T>(val);                   \
    ASSERT(static_cast<decltype(val)>(result) == (val)); \
    return result;                                       \
  }())

// Array size helper
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

// Offset of
#define FIELD_OFFSET(type, field) offsetof(type, field)

// Bit manipulation macros
#define BIT(n) (1u << (n))
#define BIT64(n) (1ull << (n))
#define SET_BIT(x, n) ((x) |= BIT(n))
#define CLR_BIT(x, n) ((x) &= ~BIT(n))
#define TST_BIT(x, n) (!!((x) & BIT(n)))
#define MASK(hi, lo) ((BIT((hi) - (lo) + 1) - 1) << (lo))

// Alignment macros
#define ALIGN_UP(val, align) (((val) + (align) - 1) & ~((align) - 1))
#define ALIGN_DOWN(val, align) ((val) & ~((align) - 1))
#define IS_ALIGNED(val, align) (((val) & ((align) - 1)) == 0)

// Min/max without std
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(val, lo, hi) MIN(MAX(val, lo), hi)

// Loop macros
#define REPEAT(n) for (std::size_t UNIQUE_NAME(_i_) = 0; UNIQUE_NAME(_i_) < (n); ++UNIQUE_NAME(_i_))
#define EACH(container) for (auto &UNIQUE_NAME(_e_) : (container))

// Scope guard
#define DEFER(code) auto UNIQUE_NAME(_defer_) = ::ScopeGuard([&]() { code; })

// Static assertions with message
#define STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#define STATIC_ASSERT_SIZE(T, sz) STATIC_ASSERT(sizeof(T) == (sz), \
                                                "sizeof(" #T ") != " STRINGIFY(sz))

STATIC_ASSERT(sizeof(i32) == 4, "i32 must be 4 bytes");
STATIC_ASSERT(sizeof(i64) == 8, "i64 must be 8 bytes");
STATIC_ASSERT(sizeof(f32) == 4, "f32 must be 4 bytes");
STATIC_ASSERT(sizeof(f64) == 8, "f64 must be 8 bytes");

// Conditional compilation blocks
#if PLATFORM_LINUX
#include <sys/types.h>
#include <unistd.h>
#define HAVE_POSIX_MEMALIGN 1
#define HAVE_MMAP 1
#endif

#if PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define HAVE_VIRTUALALLOC 1
#endif

// Token ID formatter -- included as needed
#ifdef RLLM_USE_TOKEN_ID_FORMATTER
#include "TokenIDFormatter.hpp"
#endif

// Numeric limits via macros
#define I8_MIN (-128)
#define I8_MAX (127)
#define U8_MAX (255u)
#define I16_MIN (-32768)
#define I16_MAX (32767)
#define U16_MAX (65535u)
#define I32_MIN (-2147483648)
#define I32_MAX (2147483647)
#define U32_MAX (4294967295u)

// Enum helper macros
#define ENUM_CLASS_OPERATORS(E)                                                                                                                  \
  inline E operator|(E a, E b) { return static_cast<E>(static_cast<std::underlying_type_t<E>>(a) | static_cast<std::underlying_type_t<E>>(b)); } \
  inline E operator&(E a, E b) { return static_cast<E>(static_cast<std::underlying_type_t<E>>(a) & static_cast<std::underlying_type_t<E>>(b)); } \
  inline E operator^(E a, E b) { return static_cast<E>(static_cast<std::underlying_type_t<E>>(a) ^ static_cast<std::underlying_type_t<E>>(b)); } \
  inline E operator~(E a) { return static_cast<E>(~static_cast<std::underlying_type_t<E>>(a)); }                                                 \
  inline E &operator|=(E &a, E b) {                                                                                                              \
    a = a | b;                                                                                                                                   \
    return a;                                                                                                                                    \
  }                                                                                                                                              \
  inline E &operator&=(E &a, E b) {                                                                                                              \
    a = a & b;                                                                                                                                   \
    return a;                                                                                                                                    \
  }                                                                                                                                              \
  inline E &operator^=(E &a, E b) {                                                                                                              \
    a = a ^ b;                                                                                                                                   \
    return a;                                                                                                                                    \
  }

enum class Flags : u32 {
  None = 0,
  ReadOnly = BIT(0),
  WriteOnly = BIT(1),
  ReadWrite = ReadOnly | WriteOnly,
  Append = BIT(2),
  Create = BIT(3),
  Truncate = BIT(4),
  Sync = BIT(5),
};
ENUM_CLASS_OPERATORS(Flags)

// Compile-time string hash
#define FNV1A_BASIS 2166136261u
#define FNV1A_PRIME 16777619u

constexpr u32 fnv1a(const char *s, u32 h = FNV1A_BASIS) {
  return (*s == '\0') ? h : fnv1a(s + 1, (h ^ static_cast<u32>(*s)) * FNV1A_PRIME);
}

#define HASH(str) fnv1a(str)

// Macro-based switch dispatch
#define DISPATCH_ON_TYPE(T, ...)                       \
  if constexpr (std::is_same_v<T, float>) {            \
    __VA_ARGS__                                        \
  } else if constexpr (std::is_same_v<T, double>) {    \
    __VA_ARGS__                                        \
  } else if constexpr (std::is_same_v<T, int>) {       \
    __VA_ARGS__                                        \
  } else {                                             \
    static_assert(sizeof(T) == 0, "unsupported type"); \
  }

// Include guard end
#endif // PREPROCESSOR_EXAMPLE_HPP
