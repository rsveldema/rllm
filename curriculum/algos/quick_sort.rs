fn partition(values: &mut [i32], first: usize, last: usize) -> usize {
let pivot = last - 1;
let mut boundary = first;
for current in first..pivot {
if values[current] < values[pivot] {
values.swap(current, boundary);
boundary += 1;
}
}
values.swap(boundary, pivot);
boundary
}

pub fn quick_sort(values: &mut [i32]) {
let mut pending = vec![(0, values.len())];
while let Some((first, last)) = pending.pop() {
if last - first < 2 { continue; }
let boundary = partition(values, first, last);
pending.push((first, boundary));
pending.push((boundary + 1, last));
}
}

pub fn recursive_quick_sort(values: &mut [i32]) {
fn sort(values: &mut [i32], first: usize, last: usize) {
if last - first < 2 { return; }
let boundary = partition(values, first, last);
sort(values, first, boundary);
sort(values, boundary + 1, last);
}
sort(values, 0, values.len());
}
