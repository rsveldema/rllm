final class QuickSort {
static void quickSort(int[] values) {
if (values.length < 2) return;
int[] firstStack = new int[values.length];
int[] lastStack = new int[values.length];
int size = 1;
firstStack[0] = 0;
lastStack[0] = values.length;
while (size > 0) {
int first = firstStack[--size];
int last = lastStack[size];
if (last - first < 2) continue;
int boundary = partition(values, first, last);
firstStack[size] = first;
lastStack[size++] = boundary;
firstStack[size] = boundary + 1;
lastStack[size++] = last;
}
}
static void recursiveQuickSort(int[] values) {
recursiveQuickSort(values, 0, values.length);
}
private static void recursiveQuickSort(int[] values, int first, int last) {
if (last - first < 2) return;
int boundary = partition(values, first, last);
recursiveQuickSort(values, first, boundary);
recursiveQuickSort(values, boundary + 1, last);
}
private static int partition(int[] values, int first, int last) {
int pivot = last - 1;
int boundary = first;
for (int current = first; current < pivot; ++current) {
if (values[current] < values[pivot]) {
int temporary = values[current];
values[current] = values[boundary];
values[boundary++] = temporary;
}
}
int temporary = values[boundary];
values[boundary] = values[pivot];
values[pivot] = temporary;
return boundary;
}
}
