"""Read the small parameter/metric CSVs exported by MATLAB."""

import csv
from pathlib import Path


def read_csv(filename):
    with Path(filename).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_one(filename):
    rows = read_csv(filename)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row: {filename}")
    return rows[0]
