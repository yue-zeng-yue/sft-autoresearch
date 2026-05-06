#!/usr/bin/env python3
"""Fixed ms-swift training adapter.

`run_experiment.py` calls this script. The script either launches `swift sft`
for real training or creates deterministic mock artifacts for smoke tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised when PyYAML is absent
    import mini_yaml as yaml


ROOT = Path(__file__).resolve().parent


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def stable_config_noise(config: dict[str, Any]) -> float:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return (value - 0.5) * 0.01


def mock_train(config: dict[str, Any], run_dir: Path) -> int:
    start = time.time()
    model_dir = run_dir / "model"
    checkpoint_dir = model_dir / "checkpoint-mock"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training = config.get("training", {})
    lora = config.get("lora", {})

    lr = float(training.get("learning_rate", 2e-5))
    warmup = float(training.get("warmup_ratio", 0.03))
    max_steps = int(training.get("max_steps", 100))
    batch = int(training.get("per_device_train_batch_size", 1))
    grad_accum = int(training.get("gradient_accumulation_steps", 16))
    rank = int(lora.get("lora_rank", 16))
    dropout = float(lora.get("lora_dropout", 0.05))

    lr_penalty = abs(math.log10(lr) - math.log10(2e-5)) * 0.12
    warmup_penalty = abs(warmup - 0.03) * 0.9
    step_bonus = min(max_steps, 500) / 500 * 0.04
    batch_bonus = min(batch * grad_accum, 64) / 64 * 0.03
    rank_bonus = 0.03 if rank in {16, 32} else 0.0
    dropout_penalty = abs(dropout - 0.05) * 0.3
    noise = stable_config_noise(config)

    validation_loss = 1.25 + lr_penalty + warmup_penalty + dropout_penalty - step_bonus - rank_bonus - noise
    peak_vram_mb = 4500 + rank * 45 + batch * 600 + int(training.get("max_length", 2048)) * 0.8
    training_seconds = min(2.0, 0.2 + max_steps / 1000)

    print("mock training started")
    time.sleep(training_seconds)
    print("mock training finished")

    dump_json(
        checkpoint_dir / "adapter_config.json",
        {
            "backend": "mock",
            "base_model_name_or_path": config.get("model", {}).get("name"),
            "peft_type": "LORA",
        },
    )
    dump_json(
        run_dir / "train_metrics.json",
        {
            "status": "ok",
            "model_dir": str(model_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "validation_loss": round(validation_loss, 6),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "training_seconds": round(time.time() - start, 3),
        },
    )
    return 0


def resolve_path(path_value: str, base: Path = ROOT) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((base / path).resolve())


def build_swift_config(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    model = config.get("model", {})
    data = config.get("data", {})
    training = config.get("training", {})
    lora = config.get("lora", {})
    swift = config.get("swift", {})

    train_type_arg = swift.get("train_type_arg", "tuner_type")
    output_dir = run_dir / "model"

    swift_config: dict[str, Any] = {
        "model": model.get("name"),
        "torch_dtype": model.get("torch_dtype", "bfloat16"),
        "dataset": [resolve_path(data.get("train", "data/train.jsonl"))],
        "val_dataset": resolve_path(data.get("val", "data/val.jsonl")),
        "output_dir": str(output_dir),
        train_type_arg: training.get("train_type", "lora"),
        "num_train_epochs": training.get("num_train_epochs", 1),
        "max_steps": training.get("max_steps", -1),
        "learning_rate": training.get("learning_rate"),
        "lr_scheduler_type": training.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": training.get("warmup_ratio", 0.03),
        "weight_decay": training.get("weight_decay", 0.0),
        "max_grad_norm": training.get("max_grad_norm", 1.0),
        "per_device_train_batch_size": training.get("per_device_train_batch_size", 1),
        "per_device_eval_batch_size": training.get("per_device_eval_batch_size", 1),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps", 16),
        "max_length": training.get("max_length", 2048),
        "logging_steps": training.get("logging_steps", 5),
        "eval_steps": training.get("eval_steps", 50),
        "save_steps": training.get("save_steps", 50),
        "save_total_limit": training.get("save_total_limit", 2),
        "dataloader_num_workers": training.get("dataloader_num_workers", 2),
        "lora_rank": lora.get("lora_rank", 16),
        "lora_alpha": lora.get("lora_alpha", 32),
        "lora_dropout": lora.get("lora_dropout", 0.05),
        "target_modules": lora.get("target_modules", "all-linear"),
    }

    if model.get("use_hf") is not None:
        swift_config["use_hf"] = bool(model.get("use_hf"))

    extra_args = swift.get("extra_args", {})
    if isinstance(extra_args, dict):
        swift_config.update(extra_args)

    return {key: value for key, value in swift_config.items() if value is not None}


def run_process(cmd: list[str], timeout: int | None, cwd: Path) -> int:
    print("+ " + " ".join(cmd), flush=True)
    process = subprocess.Popen(cmd, cwd=str(cwd), start_new_session=True)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"training timed out after {timeout}s; killing process group", flush=True)
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return 124
    return int(process.returncode)


def swift_train(config: dict[str, Any], run_dir: Path, timeout: int | None) -> int:
    swift = config.get("swift", {})
    executable = swift.get("executable", "swift")
    command = swift.get("command", "sft")

    swift_config = build_swift_config(config, run_dir)
    swift_config_path = run_dir / "swift_config.yaml"
    with swift_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(swift_config, f, sort_keys=False, allow_unicode=True)
    cmd = [executable, command]
    for key, value in swift_config.items():
        if isinstance(value, bool):
            cmd.extend([f"--{key}", "true" if value else "false"])
        elif isinstance(value, list):
            for item in value:
                cmd.extend([f"--{key}", str(item)])
        else:
            cmd.extend([f"--{key}", str(value)])


    start = time.time()
    code = run_process(cmd, timeout=timeout, cwd=ROOT)
    training_seconds = time.time() - start

    model_dir = run_dir / "model"
    checkpoint_dirs = sorted(
        model_dir.glob("**/checkpoint-*"),
        key=lambda p: p.stat().st_mtime,
    ) if model_dir.exists() else []
    checkpoint_dir = checkpoint_dirs[-1] if checkpoint_dirs else model_dir


    dump_json(
        run_dir / "train_metrics.json",
        {
            "status": "ok" if code == 0 else "failed",
            "returncode": code,
            "model_dir": str(model_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "training_seconds": round(training_seconds, 3),
            "peak_vram_mb": None,
        },
    )
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SFT through ms-swift")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    backend = config.get("backend", "swift")
    if backend == "mock":
        return mock_train(config, args.run_dir)
    if backend == "swift":
        return swift_train(config, args.run_dir, timeout=args.timeout)

    raise ValueError(f"unknown backend: {backend!r}")


if __name__ == "__main__":
    raise SystemExit(main())
