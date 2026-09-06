// Generic quick sort with a partitioning step.
#include <algorithm>
#include <functional>
#include <iterator>
#include <utility>
template<typename Iterator, typename Compare = std::less<>>
void quick_sort(Iterator first, Iterator last, Compare compare = {}) {
auto length = std::distance(first, last);
if (length < 2) return;
Iterator pivot_position = std::prev(last);
Iterator boundary = first;
for (Iterator current = first; current != pivot_position; ++current) {
if (compare(*current, *pivot_position)) {
std::iter_swap(current, boundary);
++boundary;
}
}
std::iter_swap(boundary, pivot_position);
quick_sort(first, boundary, compare);
quick_sort(std::next(boundary), last, compare);
}
// Explicit recursive quick-sort variant.
template<typename Iterator, typename Compare = std::less<>>
void recursive_quick_sort(Iterator first, Iterator last, Compare compare = {}) {
auto length = std::distance(first, last);
if (length < 2) return;
Iterator pivot_position = std::prev(last);
Iterator boundary = first;
for (Iterator current = first; current != pivot_position; ++current) {
if (compare(*current, *pivot_position)) {
std::iter_swap(current, boundary);
++boundary;
}
}
std::iter_swap(boundary, pivot_position);
recursive_quick_sort(first, boundary, compare);
recursive_quick_sort(std::next(boundary), last, compare);
}
