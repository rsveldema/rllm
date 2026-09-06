pub fn heap_search(values: &[i32], target: i32) -> Option<usize> {
if values.is_empty() { return None; }
let mut pending = vec![0];
while let Some(index) = pending.pop() {
if values[index] < target { continue; }
if values[index] == target { return Some(index); }
let left = index * 2 + 1;
let right = left + 1;
if right < values.len() { pending.push(right); }
if left < values.len() { pending.push(left); }
}
None
}
