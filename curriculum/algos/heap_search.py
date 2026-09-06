def heap_search(values, target):
	pending = [0]
	while pending:
		index = pending.pop()
		if values[index] < target:
			continue
		if values[index] == target:
			return index
		left = index * 2 + 1
		right = left + 1
		if right < len(values):
			pending.append(right)
		if left < len(values):
			pending.append(left)
	return -1
