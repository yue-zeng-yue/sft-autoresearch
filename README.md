# SFT AutoResearch

AutoResearch-style SFT hyperparameter loop using ms-swift as the training backend.

The agent only edits `train_config.yaml`. The fixed runner calls:

```text
run_experiment.py -> train_sft.py -> swift sft
                  -> eval.py
                  -> results.tsv + runs/<run_id>/
```

## Smoke Test

The default config uses `backend: mock`, so you can validate the lifecycle without a GPU:

```bash
python3 prepare.py
python3 run_experiment.py
```

The scripts prefer PyYAML when installed and fall back to `mini_yaml.py` for this repository's simple config shape.

## Real ms-swift Training

Install ms-swift in your environment, then edit `train_config.yaml`:

```yaml
backend: swift
eval:
  backend: swift_infer
```

Then run:

```bash
python3 prepare.py --check-swift
python3 run_experiment.py
```

If your ms-swift version expects `train_type` instead of `tuner_type`, change:

```yaml
swift:
  train_type_arg: train_type
```

## Outputs

Each experiment creates:

```text
runs/<run_id>/
  train_config.yaml
  swift_config.yaml
  train.log
  eval.log
  eval.json
  metrics.json
  summary.txt
```

`results.tsv` receives one row per run.
