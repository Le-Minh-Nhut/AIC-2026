# Experiment Logging

Record one experiment per isolated change. Do not change encoder, retrieval
configuration, and refinement policy together when comparing results.

`scripts/evaluate.py --experiment-log ...` appends immutable JSONL records with
the experiment ID, UTC timestamp, Git commit, dataset snapshot SHA-256, task,
encoder, sources, model revisions, complete supplied config content and hashes,
`R@1/5/20/50/100`, Final Score, evaluation/pipeline latency, and notes.

```bash
python scripts/evaluate.py \
  --ground-truth data/dev_ground_truth.json \
  --submission outputs/submissions/kis_dev.json \
  --task kis \
  --experiment-log outputs/experiments/experiments.jsonl \
  --experiment-id fg-pe-r1 \
  --dataset-snapshot data/reports/analysis_report.json \
  --config configs/retrieval.yaml \
  --config configs/kis.yaml \
  --runtime-metadata outputs/retrieval_debug/kis_dev.json \
  --pipeline-latency-ms 1234.5 \
  --notes "FG+PE RRF baseline"
```

The logger rejects a duplicate `experiment_id` instead of overwriting history.
Use `scripts/profile_pipeline.py` to measure a command before recording its
pipeline latency. It reports wall latency, child peak RAM, optional peak VRAM
through `nvidia-smi`, and storage after the command.
