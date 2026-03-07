#!/usr/bin/env python3
"""
Batch-aggregate per-file (input -> avg(score)) for many JSONL files.

Each input file produces:
    <out_dir>/step_<STEP>.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}:{lineno}: invalid JSON") from e


def coerce_int(x: Any) -> int:
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return int(float(x))


def aggregate_one_file(
    in_path: Path,
    out_dir: Path,
    strict_single_step: bool,
):
    sum_score: Dict[str, float] = defaultdict(float)
    count: Dict[str, int] = defaultdict(int)
    steps_seen = set()

    for rec in iter_jsonl(in_path):
        if not {"input", "score", "step"} <= rec.keys():
            continue

        try:
            inp = rec["input"]
            sc = float(rec["score"])
            st = coerce_int(rec["step"])
        except Exception:
            continue

        if not isinstance(inp, str):
            continue

        steps_seen.add(st)
        sum_score[inp] += sc
        count[inp] += 1

    if not count:
        raise RuntimeError(f"{in_path}: no usable records")

    if strict_single_step and len(steps_seen) != 1:
        raise RuntimeError(f"{in_path}: multiple step values {sorted(steps_seen)}")

    step_for_file = next(iter(steps_seen)) if len(steps_seen) == 1 else max(steps_seen)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ✅ NEW filename format
    out_path = out_dir / f"step_{step_for_file}.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for inp in sorted(count.keys()):
            avg = sum_score[inp] / count[inp]
            f.write(json.dumps({
                "step": step_for_file,
                "input": inp,
                "avg_score": avg,
                "n": count[inp],
            }, ensure_ascii=False) + "\n")

    return out_path


def iter_files(root: Path, pattern: str, recursive: bool):
    if root.is_file():
        yield root
    else:
        yield from (root.rglob(pattern) if recursive else root.glob(pattern))

# Usage for this experiment: python convert_records_to_training_data.py --in_dir=raw_train_records --out_dir=training_data --strict_single_step
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--pattern", default="*.jsonl")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--strict_single_step", action="store_true")
    args = ap.parse_args()

    files = sorted(iter_files(args.in_dir, args.pattern, args.recursive))
    if not files:
        raise RuntimeError("No JSONL files found.")

    for f in files:
        try:
            out = aggregate_one_file(
                f,
                args.out_dir,
                args.strict_single_step,
            )
            print(f"[OK] {f.name} → {out.name}")
        except Exception as e:
            print(f"[FAIL] {f.name}: {e}")


if __name__ == "__main__":
    main()