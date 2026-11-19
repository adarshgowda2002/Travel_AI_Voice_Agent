import csv
from pathlib import Path


def read_csv(directory: Path, name: str) -> list[dict[str, str]]:
    with open(directory / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
