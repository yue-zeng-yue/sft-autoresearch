# AGENTS.md

This repository is an AutoResearch-style SFT hyperparameter lab.

## Core Rule

The agent may modify only:

- `train_config.yaml`

The agent must not modify:

- `prepare.py`
- `train_sft.py`
- `eval.py`
- `run_experiment.py`
- `mini_yaml.py`
- `data/`
- previous `runs/`
- `results.tsv`, except when a human explicitly asks for manual repair

## Workflow

1. Read `program.md`.
2. Inspect `results.tsv` and the current `train_config.yaml`.
3. Make one clear hyperparameter change in `train_config.yaml`.
4. Run one experiment with:

```bash
python3 run_experiment.py
```

5. Use the printed summary and `results.tsv` to decide whether the change improved the result.
6. Keep useful changes and revert bad hyperparameter changes.

The training backend is ms-swift. `train_sft.py` is only an adapter that launches `swift sft`; do not edit it while researching.
