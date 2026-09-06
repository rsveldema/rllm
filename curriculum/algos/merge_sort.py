def merge_sort(values):
	width = 1
	while width < len(values):
		for first in range(0, len(values), width * 2):
			middle = min(first + width, len(values))
			last = min(first + width * 2, len(values))
			values[first:last] = _merge(values[first:middle], values[middle:last])
		width *= 2

def _merge(left, right):
	result = []
	left_index = 0
	right_index = 0
	while left_index < len(left) and right_index < len(right):
		if right[right_index] < left[left_index]:
			result.append(right[right_index])
			right_index += 1
		else:
			result.append(left[left_index])
			left_index += 1
	result.extend(left[left_index:])
	result.extend(right[right_index:])
	return result

def recursive_merge_sort(values):
	if len(values) < 2:
		return values
	middle = len(values) // 2
	left = recursive_merge_sort(values[:middle])
	right = recursive_merge_sort(values[middle:])
	return _merge(left, right)
