// Generic binary search over a sorted random-access range.
#include <functional>
#include <iterator>
template<typename Iterator, typename Value, typename Compare = std::less<>>
Iterator binary_search(Iterator first, Iterator last, const Value& target, Compare compare = {}) {
while (first != last) {
auto length = std::distance(first, last);
Iterator middle = std::next(first, length / 2);
if (compare(*middle, target)) first = std::next(middle);
else if (compare(target, *middle)) last = middle;
else return middle;
}
return last;
}
// Recursive binary search preserving the original end iterator for failure.
template<typename Iterator, typename Value, typename Compare>
Iterator recursive_binary_search_impl(Iterator first, Iterator last, const Value& target,
Compare compare, Iterator not_found) {
if (first == last) return not_found;
auto length = std::distance(first, last);
Iterator middle = std::next(first, length / 2);
if (compare(*middle, target)) {
return recursive_binary_search_impl(std::next(middle), last, target, compare, not_found);
}
if (compare(target, *middle)) {
return recursive_binary_search_impl(first, middle, target, compare, not_found);
}
return middle;
}
template<typename Iterator, typename Value, typename Compare = std::less<>>
Iterator recursive_binary_search(Iterator first, Iterator last, const Value& target,
Compare compare = {}) {
return recursive_binary_search_impl(first, last, target, compare, last);
}
