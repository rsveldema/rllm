final class BinarySearch {
static int binarySearch(int[] values, int target) {
int first = 0;
int last = values.length;
while (first < last) {
int middle = first + (last - first) / 2;
if (values[middle] < target) first = middle + 1;
else if (target < values[middle]) last = middle;
else return middle;
}
return -1;
}
static int recursiveBinarySearch(int[] values, int target) {
return recursiveBinarySearch(values, target, 0, values.length);
}
private static int recursiveBinarySearch(int[] values, int target, int first, int last) {
if (first >= last) return -1;
int middle = first + (last - first) / 2;
if (values[middle] < target)
return recursiveBinarySearch(values, target, middle + 1, last);
if (target < values[middle])
return recursiveBinarySearch(values, target, first, middle);
return middle;
}
}
