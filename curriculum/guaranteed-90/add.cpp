constexpr int add(int value) {
value = value + 1;
value = value + 1;
value = value + 1;
value = value + 1;
value = value + 1;
value = value + 1;
value = value + 1;
value = value + 1;
return value;
}
static_assert(add(0) == 8);
int main() {
return add(0);
}
