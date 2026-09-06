// Generic least-significant-digit radix sort for unsigned integer values.
#include <array>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>
template<typename Iterator>
void radix_sort(Iterator first, Iterator last) {
using Value = typename std::iterator_traits<Iterator>::value_type;
static_assert(std::is_integral_v<Value>);
static_assert(std::is_unsigned_v<Value>);
constexpr std::size_t radix = 256;
std::vector<Value> buffer(first, last);
std::array<std::size_t, radix> counts{};
std::array<std::size_t, radix> offsets{};
for (std::size_t shift = 0; shift < sizeof(Value) * 8; shift += 8) {
counts.fill(0);
for (Value value : buffer) ++counts[(value >> shift) & 255];
for (std::size_t index = 1; index < radix; ++index)
offsets[index] = offsets[index - 1] + counts[index - 1];
std::vector<Value> ordered(buffer.size());
for (Value value : buffer)
ordered[offsets[(value >> shift) & 255]++] = value;
buffer.swap(ordered);
offsets.fill(0);
}
std::move(buffer.begin(), buffer.end(), first);
}
