class Counter {
public:
Counter() : value(0) {}
void increment() {
if (value < 8) {
value = value + 1;
}
}
int get() const {
return value;
}
private:
int value;
};
int main() {
Counter counter;
counter.increment();
counter.increment();
counter.increment();
counter.increment();
counter.increment();
counter.increment();
counter.increment();
counter.increment();
return counter.get();
}
