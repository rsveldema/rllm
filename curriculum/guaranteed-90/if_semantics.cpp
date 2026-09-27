constexpr int choose(bool condition, int first, int second) {
if (condition) {
return first;
}
return second;
}
static_assert(choose(true, 3, 7) == 3);
static_assert(choose(false, 3, 7) == 7);
int main() {
return choose(true, 0, 1);
}
