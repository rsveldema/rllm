final class BubbleSort {
static void bubbleSort(int[] values) {
for (int boundary = values.length; boundary > 1; --boundary) {
boolean changed = false;
for (int index = 1; index < boundary; ++index) {
if (values[index] < values[index - 1]) {
int temporary = values[index - 1];
values[index - 1] = values[index];
values[index] = temporary;
changed = true;
}
}
if (!changed) break;
}
}
}
