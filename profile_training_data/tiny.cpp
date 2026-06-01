#include <vector>
int add(int a, int b) { return a + b; }
int sub(int a, int b) { return a - b; }
int mul(int a, int b) { return a * b; }
int main() { std::vector<int> v{1, 2, 3}; return add(v[0], mul(v[1], v[2])); }
