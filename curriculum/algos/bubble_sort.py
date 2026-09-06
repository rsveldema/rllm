def bubble_sort(values):
	boundary = len(values)
	while boundary > 1:
		changed = False
		for index in range(1, boundary):
			if values[index] < values[index - 1]:
				values[index - 1], values[index] = values[index], values[index - 1]
				changed = True
		if not changed:
			break
		boundary -= 1
