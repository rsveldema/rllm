final class HeapSearch {
static int heapSearch(int[] values, int target) {
if (values.length == 0) return -1;
int[] pending = new int[values.length];
int pendingSize = 1;
pending[0] = 0;
while (pendingSize > 0) {
int index = pending[--pendingSize];
if (values[index] < target) continue;
if (values[index] == target) return index;
int left = index * 2 + 1;
int right = left + 1;
if (right < values.length) pending[pendingSize++] = right;
if (left < values.length) pending[pendingSize++] = left;
}
return -1;
}
}
