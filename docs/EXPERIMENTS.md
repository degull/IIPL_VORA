# Experiment guide

## Basic restoration training

```bash
python train.py --data-root /path/to/data --dataset rain100h --method vora_v1 --backbone swinir_lite --steps 10
```

Outputs go to `outputs/logs/`; checkpoints go to `checkpoints/`. Both are ignored by Git.

## Paper reproduction

`scripts/research/` contains optional paper-reproduction utilities. They
are kept separate from normal VoRA use.
are not necessary for normal VoRA use.

Suggested order:

1. Run `scripts/research/runs/run_main_comparison.py` for the basic comparison.
2. Run numbered programs from `scripts/research/runs/` for individual tables.
3. Use matching programs from `scripts/research/summaries/` to render results.
