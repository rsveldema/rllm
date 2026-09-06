def radix_sort(values):
	shift = 0
	maximum = max(values, default=0)
	while maximum >> shift:
		counts = [0] * 256
		for value in values:
			counts[(value >> shift) & 255] += 1
		for index in range(1, 256):
			counts[index] += counts[index - 1]
		ordered = [0] * len(values)
		for value in reversed(values):
			digit = (value >> shift) & 255
			counts[digit] -= 1
			ordered[counts[digit]] = value
		values[:] = ordered
		shift += 8
