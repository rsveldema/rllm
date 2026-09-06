pub fn bubble_sort(values: &mut [i32]) {
let mut boundary = values.len();
while boundary > 1 {
let mut changed = false;
for index in 1..boundary {
if values[index] < values[index - 1] {
values.swap(index - 1, index);
changed = true;
}
}
if !changed { break; }
boundary -= 1;
}
}
