// Generic bubble sort using adjacent iterator operations.
#include <functional>
#include <iterator>
#include <utility>
template<typename Iterator, typename Compare = std::less<>>
void bubble_sort(Iterator first, Iterator last, Compare compare = {}) {
if (first == last) return;
for (Iterator boundary = last; boundary != first; --boundary) {
bool changed = false;
Iterator left = first;
Iterator right = std::next(left);
while (right != boundary) {
if (compare(*right, *left)) {
std::iter_swap(left, right);
changed = true;
}
++left;
++right;
}
if (!changed) break;
}
}
