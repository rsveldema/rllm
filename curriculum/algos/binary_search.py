def binary_search(values, target):
	first = 0
	last = len(values)
	while first < last:
		middle = first + (last - first) // 2
		if values[middle] < target:
			first = middle + 1
		elif target < values[middle]:
			last = middle
		else:
			return middle
	return -1

def recursive_binary_search(values, target, first=0, last=None):
	if last is None:
		last = len(values)
	if first >= last:
		return -1
	middle = first + (last - first) // 2
	if values[middle] < target:
		return recursive_binary_search(values, target, middle + 1, last)
	if target < values[middle]:
		return recursive_binary_search(values, target, first, middle)
	return middle
