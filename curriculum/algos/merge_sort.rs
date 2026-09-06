fn merge(values: &mut [i32], middle: usize) {
let left = values[..middle].to_vec();
let right = values[middle..].to_vec();
let mut left_index = 0;
let mut right_index = 0;
let mut output = 0;
while left_index < left.len() && right_index < right.len() {
if right[right_index] < left[left_index] {
values[output] = right[right_index];
right_index += 1;
} else {
values[output] = left[left_index];
left_index += 1;
}
output += 1;
}
while left_index < left.len() {
values[output] = left[left_index];
left_index += 1;
output += 1;
}
while right_index < right.len() {
values[output] = right[right_index];
right_index += 1;
output += 1;
}
}

pub fn merge_sort(values: &mut [i32]) {
let mut width = 1;
while width < values.len() {
let mut first = 0;
while first < values.len() {
let last = (first + width * 2).min(values.len());
let middle = (first + width).min(last) - first;
merge(&mut values[first..last], middle);
first += width * 2;
}
width *= 2;
}
}

pub fn recursive_merge_sort(values: &mut [i32]) {
if values.len() < 2 { return; }
let middle = values.len() / 2;
recursive_merge_sort(&mut values[..middle]);
recursive_merge_sort(&mut values[middle..]);
merge(values, middle);
}
