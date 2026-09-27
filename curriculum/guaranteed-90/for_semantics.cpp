constexpr int sum_to(int limit) {
int total = 0;
for (int index = 0; index < limit; ++index) {
total = total + index;
}
return total;
}
static_assert(sum_to(0) == 0);
static_assert(sum_to(1) == 0);
static_assert(sum_to(4) == 6);
int main() {
return sum_to(4);
}
