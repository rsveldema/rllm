import asyncio
import math as mathematics
from contextlib import asynccontextmanager, suppress
from typing import AsyncIterator, Iterator

type Number = int | float
type NameMap = dict[int, str]

DEFAULT_NAME: str | None = None
ENABLED = True
DISABLED = False
processed_count = 0


class ValidationError(Exception):
	pass


class Counter:
	def __init__(self, beginning: int = 0) -> None:
		self.value = beginning

	def increment(self) -> int:
		self.value += 1
		return self.value


def classify_number(value: Number) -> str:
	assert mathematics.isfinite(value), "value must be finite"
	if value < 0:
		return "negative"
	elif value == 0:
		return "zero"
	else:
		return "positive"


def normalize_name(name: str | None) -> str:
	if name is None or not name:
		return "anonymous"
	return name.strip()


def choose_value(primary: int | None, secondary: int | None) -> int:
	if primary is not None and primary > 0:
		return primary
	if secondary is not None and secondary > 0:
		return secondary
	return 0


def count_positive(values: list[int]) -> int:
	global processed_count
	count = 0
	for value in values:
		processed_count += 1
		if value <= 0:
			continue
		count += 1
	return count


def find_name(names: NameMap, target: str) -> int | None:
	for identifier, name in names.items():
		if name == target:
			break
	else:
		return None
	return identifier


def countdown(value: int) -> list[int]:
	result: list[int] = []
	while value > 0:
		result.append(value)
		value -= 1
	return result


def make_accumulator(start: int):
	total = start

	def add(value: int) -> int:
		nonlocal total
		total += value
		return total

	return add


def positive_values(values: list[int]) -> Iterator[int]:
	for value in values:
		if value > 0:
			yield value


def parse_integer(text: str) -> int:
	try:
		value = int(text)
		if value < 0:
			raise ValidationError("negative values are rejected")
	except ValueError as error:
		raise ValidationError("invalid integer") from error
	except ValidationError:
		return 0
	finally:
		text = text.strip()
	return value


def remove_temporary_value(values: dict[str, int]) -> None:
	if "temporary" in values:
		del values["temporary"]
	else:
		pass


def describe_command(command: tuple[str, object]) -> str:
	match command:
		case ("quit", _):
			return "stop"
		case ("load", str(filename)):
			return f"load {filename}"
		case ("scale", int(value)) if value > 0:
			return f"scale by {value}"
		case _:
			return "unknown"


def build_transform(multiplier: int):
	return lambda value: value * multiplier


@asynccontextmanager
async def opened_resource(name: str) -> AsyncIterator[str]:
	await asyncio.sleep(0)
	try:
		yield name
	finally:
		await asyncio.sleep(0)


async def generate_numbers(limit: int) -> AsyncIterator[int]:
	for value in range(limit):
		await asyncio.sleep(0)
		yield value


async def collect_numbers(limit: int) -> list[int]:
	result: list[int] = []
	async with opened_resource("numbers") as resource:
		assert resource == "numbers"
		async for value in generate_numbers(limit):
			result.append(value)
	return result


async def run_example() -> None:
	names: NameMap = {1: "primary", 2: "secondary"}
	values = [-2, 0, 3, 7]
	accumulator = make_accumulator(10)
	double = build_transform(2)

	with suppress(KeyError):
		del names[99]

	print(classify_number(3))
	print(normalize_name(DEFAULT_NAME))
	print(choose_value(None, 5))
	print(count_positive(values))
	print(find_name(names, "primary"))
	print(countdown(3))
	print(list(positive_values(values)))
	print(accumulator(4))
	print(double(6))
	print(parse_integer("12"))
	print(describe_command(("load", "model.bin")))
	print(await collect_numbers(4))


if __name__ == "__main__":
	asyncio.run(run_example())
