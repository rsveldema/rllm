# Grammar curriculum: annotations document inputs, outputs, and optional values.
# The functions below combine filesystem operations, callbacks, loops, and exceptions.
from pathlib import Path
from typing import Callable


DEFAULT_NAME: str | None = None


def normalize_name(name: str | None) -> str:
	return name if name is not None else "standard"


def choose_output_directory(directory: Path, filename: str) -> Path:
	return directory / filename


def run_callback(printer: Callable[[str], None], message: str) -> None:
	printer(message)


def default_name_examples() -> tuple[str, str, str]:
	alpha_name: str | None = DEFAULT_NAME
	beta_name: str | None = "worker"
	gamma_name: str | None = None
	return (
		normalize_name(alpha_name),
		normalize_name(beta_name),
		normalize_name(gamma_name),
	)


def directory_examples(root: Path) -> tuple[Path, Path, Path]:
	alpha_path = choose_output_directory(root, "alpha.txt")
	beta_path = choose_output_directory(root, "beta.txt")
	gamma_path = choose_output_directory(root, "gamma.txt")
	return alpha_path, beta_path, gamma_path


def extra_control_flow_sample(limit: int) -> int:
	total = 0
	for value in range(limit):
		if value % 2 == 0:
			total += value
	return total


def main() -> None:
	alpha_value, beta_status, gamma_result = default_name_examples()
	run_callback(print, alpha_value)
	run_callback(print, beta_status)
	run_callback(print, gamma_result)


if __name__ == "__main__":
	main()
