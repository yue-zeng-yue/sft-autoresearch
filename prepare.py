#!/usr/bin/env python3
"""Fixed preparation step for SFT AutoResearch.

This script prepares a small demo dataset if no dataset exists yet and verifies
that the JSONL files use the expected chat-style schema.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "val.jsonl"
EVAL_PATH = DATA_DIR / "eval.jsonl"


DEMO_TRAIN = [
    {
        "id": "train-001",
        "messages": [
            {"role": "user", "content": "Return the word ready."},
            {"role": "assistant", "content": "ready"},
        ],
    },
    {
        "id": "train-002",
        "messages": [
            {"role": "user", "content": "What is 2 + 2? Answer with only the number."},
            {"role": "assistant", "content": "4"},
        ],
    },
    {
        "id": "train-003",
        "messages": [
            {"role": "user", "content": "Translate 'hello' to Chinese."},
            {"role": "assistant", "content": "你好"},
        ],
    },
]

DEMO_VAL = [
    {
        "id": "val-001",
        "messages": [
            {"role": "user", "content": "Return the word valid."},
            {"role": "assistant", "content": "valid"},
        ],
    }
]

DEMO_EVAL = [
    {
        "id": "eval-001",
        "messages": [{"role": "user", "content": "Return the word ready."}],
        "expected": "ready",
        "expected_contains": ["ready"],
    },
    {
        "id": "eval-002",
        "messages": [{"role": "user", "content": "What is 2 + 2? Answer with only the number."}],
        "expected": "4",
        "expected_contains": ["4"],
    },
]


def write_jsonl(path: Path, rows: list[dict], force: bool = False) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_jsonl(path: Path, *, eval_file: bool = False) -> int:
    if not path.exists():
        raise FileNotFoundError(f"missing required data file: {path}")

    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_no}: expected non-empty messages list")
            for msg in messages:
                if msg.get("role") not in {"system", "user", "assistant"}:
                    raise ValueError(f"{path}:{line_no}: invalid message role: {msg.get('role')!r}")
                if not isinstance(msg.get("content"), str):
                    raise ValueError(f"{path}:{line_no}: message content must be a string")
            if eval_file and "expected" not in row and "expected_contains" not in row:
                raise ValueError(f"{path}:{line_no}: eval rows need expected or expected_contains")
            count += 1
    return count


def check_swift() -> None:
    executable = shutil.which("swift")
    if executable is None:
        raise RuntimeError("ms-swift CLI not found. Install it, then ensure `swift` is on PATH.")
    result = subprocess.run([executable, "--help"], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            "`swift --help` failed. A `swift` executable exists, but the ms-swift "
            "Python package is not importable from that executable's Python environment.\n\n"
            f"swift executable: {executable}\n\n"
            f"{detail}\n\n"
            "Fix by installing ms-swift into the same environment that provides `swift`, "
            "or by setting train_config.yaml -> swift.executable to a working ms-swift CLI path."
        )
    print(f"swift CLI: {executable}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare fixed data for SFT AutoResearch")
    parser.add_argument("--force-demo-data", action="store_true", help="overwrite data/*.jsonl with demo data")
    parser.add_argument("--check-swift", action="store_true", help="verify ms-swift CLI is installed")
    args = parser.parse_args()

    write_jsonl(TRAIN_PATH, DEMO_TRAIN, force=args.force_demo_data)
    write_jsonl(VAL_PATH, DEMO_VAL, force=args.force_demo_data)
    write_jsonl(EVAL_PATH, DEMO_EVAL, force=args.force_demo_data)

    train_count = validate_jsonl(TRAIN_PATH)
    val_count = validate_jsonl(VAL_PATH)
    eval_count = validate_jsonl(EVAL_PATH, eval_file=True)

    print(f"train: {TRAIN_PATH} ({train_count} rows)")
    print(f"val:   {VAL_PATH} ({val_count} rows)")
    print(f"eval:  {EVAL_PATH} ({eval_count} rows)")

    if args.check_swift:
        check_swift()

    print("Done. Ready to run experiments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
