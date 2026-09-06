// Generic search in a max heap with branches pruned by the heap property.
#include <functional>
#include <iterator>
#include <vector>
template<typename Iterator, typename Value, typename Compare = std::less<>>
Iterator heap_search(Iterator first, Iterator last, const Value& target, Compare compare = {}) {
using Difference = typename std::iterator_traits<Iterator>::difference_type;
const Difference length = std::distance(first, last);
if (length == 0) return last;
std::vector<Difference> pending{0};
while (!pending.empty()) {
const Difference index = pending.back();
pending.pop_back();
Iterator current = std::next(first, index);
if (compare(*current, target)) continue;
if (!compare(target, *current)) return current;
const Difference left = index * 2 + 1;
const Difference right = left + 1;
if (right < length) pending.push_back(right);
if (left < length) pending.push_back(left);
}
return last;
}
