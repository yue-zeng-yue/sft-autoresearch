#!/usr/bin/env python3
"""Fixed evaluation harness for SFT AutoResearch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def resolve_path(path_value: str, base: Path = ROOT) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def stable_config_noise(config: dict[str, Any]) -> float:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    value = int(digest[8:16], 16) / 0xFFFFFFFF
    return (value - 0.5) * 0.01


def score_from_parts(metrics: dict[str, float], config: dict[str, Any]) -> float:
    weights = config.get("scoring", {}).get("weights", {})
    exact_weight = float(weights.get("exact_match", 0.35))
    contains_weight = float(weights.get("contains_rate", 0.35))
    loss_weight = float(weights.get("loss_score", 0.20))
    speed_weight = float(weights.get("speed_score", 0.10))

    validation_loss = float(metrics.get("validation_loss", 2.0))
    training_seconds = float(metrics.get("training_seconds", 1.0))
    loss_score = max(0.0, min(1.0, 1.0 / max(validation_loss, 1e-6)))
    speed_score = max(0.0, min(1.0, 1.0 / max(math.log1p(training_seconds), 1.0)))

    return (
        exact_weight * float(metrics.get("exact_match", 0.0))
        + contains_weight * float(metrics.get("contains_rate", 0.0))
        + loss_weight * loss_score
        + speed_weight * speed_score
    )


def mock_eval(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    train_metrics_path = run_dir / "train_metrics.json"
    train_metrics = json.loads(train_metrics_path.read_text(encoding="utf-8")) if train_metrics_path.exists() else {}

    training = config.get("training", {})
    lora = config.get("lora", {})
    lr = float(training.get("learning_rate", 2e-5))
    warmup = float(training.get("warmup_ratio", 0.03))
    rank = int(lora.get("lora_rank", 16))
    dropout = float(lora.get("lora_dropout", 0.05))

    lr_quality = max(0.0, 1.0 - abs(math.log10(lr) - math.log10(2e-5)) * 0.55)
    warmup_quality = max(0.0, 1.0 - abs(warmup - 0.03) * 5.0)
    rank_quality = 1.0 if rank in {16, 32} else 0.92
    dropout_quality = max(0.0, 1.0 - abs(dropout - 0.05) * 2.0)
    quality = max(0.0, min(1.0, 0.55 * lr_quality + 0.15 * warmup_quality + 0.15 * rank_quality + 0.15 * dropout_quality))
    quality = max(0.0, min(1.0, quality + stable_config_noise(config)))

    metrics = {
        "exact_match": round(quality, 6),
        "contains_rate": round(min(1.0, quality + 0.05), 6),
        "validation_loss": float(train_metrics.get("validation_loss", 1.5)),
        "training_seconds": float(train_metrics.get("training_seconds", 0.0)),
        "peak_vram_mb": train_metrics.get("peak_vram_mb"),
    }
    metrics["score"] = round(score_from_parts(metrics, config), 6)
    return {
        "backend": "mock",
        "metrics": metrics,
        "predictions": [],
    }


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def evaluate_predictions(rows: list[dict[str, Any]], predictions: list[str]) -> dict[str, float]:
    exact_hits = 0
    contains_hits = 0
    for row, prediction in zip(rows, predictions):
        pred_norm = normalize_text(prediction)
        expected = row.get("expected")
        if isinstance(expected, str) and pred_norm == normalize_text(expected):
            exact_hits += 1
        contains_values = row.get("expected_contains", [])
        if isinstance(contains_values, str):
            contains_values = [contains_values]
        if contains_values and all(normalize_text(str(item)) in pred_norm for item in contains_values):
            contains_hits += 1

    denom = max(len(rows), 1)
    return {
        "exact_match": exact_hits / denom,
        "contains_rate": contains_hits / denom,
    }


def swift_infer_eval(config: dict[str, Any], run_dir: Path, model_dir: Path) -> dict[str, Any]:
    try:
        from swift import InferRequest, RequestConfig, TransformersEngine
    except Exception as exc:  # pragma: no cover - depends on optional ms-swift install
        raise RuntimeError("swift Python inference API is unavailable; install ms-swift or use eval.backend=external") from exc

    data_path = resolve_path(config.get("data", {}).get("eval", "data/eval.jsonl"))
    rows = load_jsonl(data_path)
    eval_cfg = config.get("eval", {})
    model_name = config.get("model", {}).get("name")

    engine = TransformersEngine(model_name, adapters=[str(model_dir)])
    request_config = RequestConfig(
        max_tokens=int(eval_cfg.get("max_new_tokens", 256)),
        temperature=float(eval_cfg.get("temperature", 0.0)),
    )

    predictions: list[str] = []
    for row in rows:
        request = InferRequest(messages=row["messages"])
        response_list = engine.infer([request], request_config)
        message = response_list[0].choices[0].message
        predictions.append(str(message.content))

    train_metrics_path = run_dir / "train_metrics.json"
    train_metrics = json.loads(train_metrics_path.read_text(encoding="utf-8")) if train_metrics_path.exists() else {}
    metrics = evaluate_predictions(rows, predictions)
    metrics["validation_loss"] = float(train_metrics.get("validation_loss") or 2.0)
    metrics["training_seconds"] = float(train_metrics.get("training_seconds") or 0.0)
    metrics["peak_vram_mb"] = train_metrics.get("peak_vram_mb")
    metrics["score"] = round(score_from_parts(metrics, config), 6)

    return {
        "backend": "swift_infer",
        "metrics": metrics,
        "predictions": [
            {"id": row.get("id"), "prediction": pred}
            for row, pred in zip(rows, predictions)
        ],
    }


def external_eval(config: dict[str, Any], run_dir: Path, model_dir: Path) -> dict[str, Any]:
    command = config.get("eval", {}).get("external_command", [])
    if not isinstance(command, list) or not command:
        raise ValueError("eval.external_command must be a non-empty list for external eval")

    replacements = {
        "{run_dir}": str(run_dir),
        "{model_dir}": str(model_dir),
        "{eval_path}": str(resolve_path(config.get("data", {}).get("eval", "data/eval.jsonl"))),
    }
    cmd = [replacements.get(part, part) for part in command]
    result = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"external eval failed with code {result.returncode}:\n{result.stderr}")
    payload = json.loads(result.stdout)
    if "metrics" not in payload:
        payload = {"backend": "external", "metrics": payload}
    metrics = payload["metrics"]
    if "score" not in metrics:
        metrics["score"] = round(score_from_parts(metrics, config), 6)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained SFT adapter")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    backend = config.get("eval", {}).get("backend", config.get("backend", "mock"))

    if backend == "mock":
        payload = mock_eval(config, args.run_dir)
    elif backend == "swift_infer":
        payload = swift_infer_eval(config, args.run_dir, args.model_dir)
    elif backend == "external":
        payload = external_eval(config, args.run_dir, args.model_dir)
    else:
        raise ValueError(f"unknown eval backend: {backend!r}")

    dump_json(args.output, payload)
    print(json.dumps(payload["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
