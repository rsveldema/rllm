final class MergeSort {
static void mergeSort(int[] values) {
int[] buffer = new int[values.length];
for (int width = 1; width < values.length; width *= 2) {
for (int first = 0; first < values.length; first += width * 2) {
int middle = Math.min(first + width, values.length);
int last = Math.min(first + width * 2, values.length);
merge(values, buffer, first, middle, last);
}
}
}
static void recursiveMergeSort(int[] values) {
int[] buffer = new int[values.length];
recursiveMergeSort(values, buffer, 0, values.length);
}
private static void recursiveMergeSort(int[] values, int[] buffer, int first, int last) {
if (last - first < 2) return;
int middle = first + (last - first) / 2;
recursiveMergeSort(values, buffer, first, middle);
recursiveMergeSort(values, buffer, middle, last);
merge(values, buffer, first, middle, last);
}
private static void merge(int[] values, int[] buffer, int first, int middle, int last) {
int left = first;
int right = middle;
int output = first;
while (left < middle && right < last)
buffer[output++] = values[right] < values[left] ? values[right++] : values[left++];
while (left < middle) buffer[output++] = values[left++];
while (right < last) buffer[output++] = values[right++];
for (int index = first; index < last; ++index) values[index] = buffer[index];
}
}
