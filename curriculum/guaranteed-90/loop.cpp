int repeat(int value) {
for (int index = 0; index < 8; ++index) {
value = value + 1;
value = value + 1;
value = value + 1;
value = value + 1;
}
return value;
}
int main() {
return repeat(0);
}
