class Multiplier {
public:
Multiplier(int initial) : value(initial) {}
void multiply() {
value = value * 2;
}
int get() const {
return value;
}
private:
int value;
};
int main() {
Multiplier multiplier(1);
for (int index = 0; index < 8; ++index) {
multiplier.multiply();
}
return multiplier.get();
}
