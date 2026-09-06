// Generic stable merge sort over random-access iterators.
#include <algorithm>
#include <functional>
#include <iterator>
#include <vector>
template<typename Iterator, typename Compare>
void merge_sort_impl(Iterator first, Iterator last, Compare compare) {
auto length = std::distance(first, last);
if (length < 2) return;
Iterator middle = std::next(first, length / 2);
merge_sort_impl(first, middle, compare);
merge_sort_impl(middle, last, compare);
using Value = typename std::iterator_traits<Iterator>::value_type;
std::vector<Value> buffer;
buffer.reserve(length);
Iterator left = first;
Iterator right = middle;
while (left != middle && right != last) {
if (compare(*right, *left)) buffer.push_back(*right++);
else buffer.push_back(*left++);
}
while (left != middle) buffer.push_back(*left++);
while (right != last) buffer.push_back(*right++);
std::move(buffer.begin(), buffer.end(), first);
}
template<typename Iterator, typename Compare = std::less<>>
void merge_sort(Iterator first, Iterator last, Compare compare = {}) {
merge_sort_impl(first, last, compare);
}
// Explicit recursive merge-sort variant.
template<typename Iterator, typename Compare>
void recursive_merge_sort_impl(Iterator first, Iterator last, Compare compare) {
auto length = std::distance(first, last);
if (length < 2) return;
Iterator middle = std::next(first, length / 2);
recursive_merge_sort_impl(first, middle, compare);
recursive_merge_sort_impl(middle, last, compare);
using Value = typename std::iterator_traits<Iterator>::value_type;
std::vector<Value> buffer;
buffer.reserve(length);
Iterator left = first;
Iterator right = middle;
while (left != middle && right != last) {
if (compare(*right, *left)) buffer.push_back(*right++);
else buffer.push_back(*left++);
}
while (left != middle) buffer.push_back(*left++);
while (right != last) buffer.push_back(*right++);
std::move(buffer.begin(), buffer.end(), first);
}
template<typename Iterator, typename Compare = std::less<>>
void recursive_merge_sort(Iterator first, Iterator last, Compare compare = {}) {
recursive_merge_sort_impl(first, last, compare);
}
