#!/usr/bin/env python3
"""
Batch convert JSONL files where:
  - input text is in field "question"
  - score is a list of booleans (True/False)
  - exactly one entry per unique question (optionally enforced)

For each record, compute:
  avg_score = mean(score_list) treating True=1, False=0
  n = len(score_list)

Output JSONL format:
  {"input": <question>, "avg_score": <float>, "n": <int>}

Output file naming:
  output filename matches the input filename (written under --out_dir).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}:{lineno}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise RuntimeError(f"{path}:{lineno}: expected JSON object per line, got {type(obj)}")
            yield obj


def bool_list_avg(xs: Any, *, path: Path, lineno: int) -> tuple[float, int]:
    if not isinstance(xs, list):
        raise RuntimeError(f"{path}:{lineno}: 'score' must be a list, got {type(xs)}")
    if len(xs) == 0:
        return 0.0, 0
    total = 0
    for i, v in enumerate(xs):
        if isinstance(v, bool):
            total += 1 if v else 0
        else:
            raise RuntimeError(f"{path}:{lineno}: score[{i}] must be bool, got {type(v)}")
    return total / len(xs), len(xs)


def process_one_file(in_path: Path, out_dir: Path, strict_unique_question: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / in_path.name  # ✅ match input filename

    seen_questions = set()

    with out_path.open("w", encoding="utf-8") as out_f:
        for lineno, rec in enumerate(iter_jsonl(in_path), start=1):
            q = rec.get("question")
            if not isinstance(q, str):
                continue

            if strict_unique_question:
                if q in seen_questions:
                    raise RuntimeError(f"{in_path}:{lineno}: duplicate question encountered")
                seen_questions.add(q)

            avg, n = bool_list_avg(rec.get("score"), path=in_path, lineno=lineno)

            out_f.write(
                json.dumps(
                    {
                        "input": q,
                        "avg_score": avg,
                        "n": n,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return out_path


def iter_files(root: Path, pattern: str, recursive: bool) -> Iterable[Path]:
    if root.is_file():
        yield root
    else:
        yield from (root.rglob(pattern) if recursive else root.glob(pattern))

# Usage for this experiment: python convert_eval_outputs_to_val_data.py --in_dir=raw_eval_results --out_dir=validation_data --strict_unique_question
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, required=True, help="Directory containing JSONL files (or a single file)")
    ap.add_argument("--out_dir", type=Path, required=True, help="Directory to write outputs")
    ap.add_argument("--pattern", type=str, default="*.jsonl")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument(
        "--strict_unique_question",
        action="store_true",
        help="Error if a file contains the same question more than once",
    )
    args = ap.parse_args()

    files = sorted(iter_files(args.in_dir, args.pattern, args.recursive))
    if not files:
        raise RuntimeError("No JSONL files found.")

    for p in files:
        try:
            out = process_one_file(p, args.out_dir, args.strict_unique_question)
            print(f"[OK] {p} -> {out}")
        except Exception as e:
            print(f"[FAIL] {p}: {e}")


if __name__ == "__main__":
    main()