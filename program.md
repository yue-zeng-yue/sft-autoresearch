# SFT AutoResearch Program

This project lets an agent research SFT hyperparameters under a fixed training and evaluation harness.

The design follows the core AutoResearch idea from `karpathy/autoresearch`: keep the benchmark fixed, expose one small editable surface to the agent, run experiments repeatedly, and keep only changes that improve the metric.

## Files

- `train_config.yaml` - the only file the agent edits.
- `prepare.py` - fixed data/model/environment preparation.
- `train_sft.py` - fixed ms-swift training adapter.
- `eval.py` - fixed evaluation script.
- `run_experiment.py` - fixed experiment runner and recorder.
- `results.tsv` - experiment history.
- `runs/` - per-run artifacts.

## Setup

Run once:

```bash
python3 prepare.py
```

For real SFT, install ms-swift separately and set:

```yaml
backend: swift
```

in `train_config.yaml`.

The default backend is `mock` so the full experiment loop can be smoke-tested without a GPU or ms-swift installation.

## Experiment Command

Every experiment is launched with:

```bash
python3 run_experiment.py
```

Do not call `train_sft.py` directly during research. `run_experiment.py` is the fixed lifecycle manager.

## Editable Surface

The agent may tune SFT hyperparameters in `train_config.yaml`, especially:

- `training.learning_rate`
- `training.lr_scheduler_type`
- `training.warmup_ratio`
- `training.num_train_epochs`
- `training.max_steps`
- `training.per_device_train_batch_size`
- `training.gradient_accumulation_steps`
- `training.max_length`
- `training.weight_decay`
- `training.max_grad_norm`
- `lora.lora_rank`
- `lora.lora_alpha`
- `lora.lora_dropout`
- `lora.target_modules`

The agent should not change:

- model identity
- dataset paths
- evaluation backend
- scoring weights
- runner scripts
- data files

## Search Space

Reasonable first-pass values:

```text
learning_rate: 5e-6, 1e-5, 2e-5, 3e-5, 5e-5
warmup_ratio: 0.0, 0.01, 0.03, 0.05, 0.1
lr_scheduler_type: constant, linear, cosine
max_steps: 100, 300, 500, 1000
per_device_train_batch_size: 1, 2, 4
gradient_accumulation_steps: 8, 16, 32
max_length: 1024, 2048, 4096
lora_rank: 8, 16, 32, 64
lora_alpha: rank, 2*rank
lora_dropout: 0.0, 0.03, 0.05, 0.1
weight_decay: 0.0, 0.01, 0.05
max_grad_norm: 0.5, 1.0, 2.0
```

Prefer one or two coordinated changes per experiment. Avoid changing many knobs at once unless testing a specific interaction.

## Metric

The primary metric is:

```text
score
```

Higher is better.

`run_experiment.py` prints a stable summary:

```text
---
run_id: ...
status: ok
score: ...
validation_loss: ...
exact_match: ...
contains_rate: ...
peak_vram_mb: ...
training_seconds: ...
```

## Keep / Discard

- Keep a change if `score` clearly improves without unacceptable memory or runtime cost.
- Confirm tiny improvements before trusting them.
- Discard regressions.
- Treat crashes as failed experiments.
- Prefer simpler configurations when scores are similar.

## Logging

`run_experiment.py` appends to `results.tsv` automatically. Do not manually edit results during normal research.
