pub fn binary_search(values: &[i32], target: i32) -> Option<usize> {
let mut first = 0;
let mut last = values.len();
while first < last {
let middle = first + (last - first) / 2;
if values[middle] < target { first = middle + 1; }
else if target < values[middle] { last = middle; }
else { return Some(middle); }
}
None
}

pub fn recursive_binary_search(values: &[i32], target: i32) -> Option<usize> {
fn search(values: &[i32], target: i32, first: usize, last: usize) -> Option<usize> {
if first >= last { return None; }
let middle = first + (last - first) / 2;
if values[middle] < target { search(values, target, middle + 1, last) }
else if target < values[middle] { search(values, target, first, middle) }
else { Some(middle) }
}
search(values, target, 0, values.len())
}
