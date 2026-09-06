def quick_sort(values):
	pending = [(0, len(values))]
	while pending:
		first, last = pending.pop()
		if last - first < 2:
			continue
		pivot = last - 1
		boundary = first
		for current in range(first, pivot):
			if values[current] < values[pivot]:
				values[current], values[boundary] = values[boundary], values[current]
				boundary += 1
		values[boundary], values[pivot] = values[pivot], values[boundary]
		pending.append((first, boundary))
		pending.append((boundary + 1, last))

def recursive_quick_sort(values, first=0, last=None):
	if last is None:
		last = len(values)
	if last - first < 2:
		return
	pivot = last - 1
	boundary = first
	for current in range(first, pivot):
		if values[current] < values[pivot]:
			values[current], values[boundary] = values[boundary], values[current]
			boundary += 1
	values[boundary], values[pivot] = values[pivot], values[boundary]
	recursive_quick_sort(values, first, boundary)
	recursive_quick_sort(values, boundary + 1, last)
