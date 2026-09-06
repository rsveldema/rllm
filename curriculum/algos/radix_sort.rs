pub fn radix_sort(values: &mut [u32]) {
let mut buffer = values.to_vec();
for shift in (0..32).step_by(8) {
let mut counts = [0usize; 256];
for &value in values.iter() { counts[((value >> shift) & 255) as usize] += 1; }
let mut offsets = [0usize; 256];
for index in 1..256 { offsets[index] = offsets[index - 1] + counts[index - 1]; }
for &value in values.iter() {
let digit = ((value >> shift) & 255) as usize;
buffer[offsets[digit]] = value;
offsets[digit] += 1;
}
values.copy_from_slice(&buffer);
}
}
