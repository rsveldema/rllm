final class RadixSort {
static void radixSort(int[] values) {
int[] buffer = new int[values.length];
for (int shift = 0; shift < 32; shift += 8) {
int[] counts = new int[256];
for (int value : values) ++counts[(value >>> shift) & 255];
for (int index = 1; index < counts.length; ++index) counts[index] += counts[index - 1];
for (int index = values.length - 1; index >= 0; --index) {
int value = values[index];
int digit = (value >>> shift) & 255;
buffer[--counts[digit]] = value;
}
int[] temporary = values;
values = buffer;
buffer = temporary;
}
}
}
