#!/usr/bin/env python3
"""Run one complete SFT AutoResearch experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised when PyYAML is absent
    import mini_yaml as yaml


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "train_config.yaml"
RUNS_DIR = ROOT / "runs"
RESULTS_PATH = ROOT / "results.tsv"


RESULT_COLUMNS = [
    "run_id",
    "commit",
    "score",
    "validation_loss",
    "exact_match",
    "contains_rate",
    "peak_vram_gb",
    "training_seconds",
    "status",
    "description",
    "config_hash",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def git_commit_short() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return "nogit"
    return result.stdout.strip() or "nogit"


def config_hash(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:12]


def make_run_id(commit: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{commit}"


def kill_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_logged(cmd: list[str], log_path: Path, timeout: int | None) -> tuple[int, float]:
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("+ " + " ".join(cmd) + "\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {timeout}s\n")
            log.flush()
            kill_process_group(process)
            return 124, time.time() - start
    return int(process.returncode), time.time() - start


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def append_results(row: dict[str, Any]) -> None:
    needs_header = not RESULTS_PATH.exists() or RESULTS_PATH.stat().st_size == 0
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, delimiter="\t", extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in RESULT_COLUMNS})


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    lines = ["---"]
    for key in [
        "run_id",
        "status",
        "score",
        "validation_loss",
        "exact_match",
        "contains_rate",
        "peak_vram_mb",
        "training_seconds",
        "description",
    ]:
        lines.append(f"{key}: {summary.get(key, '')}")
    text = "\n".join(lines) + "\n"
    (run_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one full SFT AutoResearch experiment")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--no-results", action="store_true", help="do not append to results.tsv")
    args = parser.parse_args()

    source_config = args.config.resolve()
    config = load_yaml(source_config)
    commit = git_commit_short()
    run_id = make_run_id(commit)
    run_dir = args.runs_dir.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    run_config = run_dir / "train_config.yaml"
    shutil.copy2(source_config, run_config)

    cfg_hash = config_hash(run_config)
    description = config.get("experiment", {}).get("description", "")
    kill_after = int(config.get("experiment", {}).get("kill_after_seconds", 3600))

    train_log = run_dir / "train.log"
    train_cmd = [
        sys.executable,
        str(ROOT / "train_sft.py"),
        "--config",
        str(run_config),
        "--run-dir",
        str(run_dir),
        "--timeout",
        str(kill_after),
    ]
    train_code, wall_seconds = run_logged(train_cmd, train_log, timeout=kill_after + 60)

    train_metrics = read_json(run_dir / "train_metrics.json")
    status = "ok" if train_code == 0 and train_metrics.get("status", "ok") == "ok" else "failed"

    eval_payload: dict[str, Any] = {}
    eval_metrics: dict[str, Any] = {}
    if status == "ok":
        model_dir = Path(train_metrics.get("checkpoint_dir") or train_metrics.get("model_dir") or run_dir / "model")
        eval_cmd = [
            sys.executable,
            str(ROOT / "eval.py"),
            "--config",
            str(run_config),
            "--run-dir",
            str(run_dir),
            "--model-dir",
            str(model_dir),
            "--output",
            str(run_dir / "eval.json"),
        ]
        eval_log = run_dir / "eval.log"
        eval_code, _ = run_logged(eval_cmd, eval_log, timeout=kill_after)
        if eval_code == 0:
            eval_payload = read_json(run_dir / "eval.json")
            eval_metrics = eval_payload.get("metrics", {})
        else:
            status = "eval_failed"
    else:
        eval_metrics = {}

    peak_vram_mb = eval_metrics.get("peak_vram_mb", train_metrics.get("peak_vram_mb"))
    training_seconds = eval_metrics.get("training_seconds", train_metrics.get("training_seconds", wall_seconds))

    summary = {
        "run_id": run_id,
        "commit": commit,
        "status": status,
        "description": description,
        "config_hash": cfg_hash,
        "score": eval_metrics.get("score", 0.0),
        "validation_loss": eval_metrics.get("validation_loss", train_metrics.get("validation_loss", "")),
        "exact_match": eval_metrics.get("exact_match", ""),
        "contains_rate": eval_metrics.get("contains_rate", ""),
        "peak_vram_mb": peak_vram_mb if peak_vram_mb is not None else "",
        "training_seconds": training_seconds,
        "train_returncode": train_code,
    }

    metrics_payload = {
        "summary": summary,
        "train_metrics": train_metrics,
        "eval": eval_payload,
    }
    dump_json(run_dir / "metrics.json", metrics_payload)
    write_summary(run_dir, summary)

    row = {
        **summary,
        "peak_vram_gb": round(float(peak_vram_mb) / 1024, 3) if peak_vram_mb not in (None, "") else "",
    }
    if not args.no_results:
        append_results(row)

    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
