#!/usr/bin/env python3
"""Print v2 aggregate accuracy table from runs.json."""
import json
from collections import Counter
from pathlib import Path

from run_experiment import (
    ROOT,
    aggregate_metric,
    write_markdown_table,
)

runs = json.loads((ROOT / "runs.json").read_text(encoding="utf-8"))
print("status", Counter(r["status"] for r in runs))
print("by impl", Counter((r["status"], r["implementation"]) for r in runs))
agg = aggregate_metric(runs, "accuracy")
print("\nAccuracy mean±std (ddof=1)\n")
print(write_markdown_table(agg, "accuracy"))
