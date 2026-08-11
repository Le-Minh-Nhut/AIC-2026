# Submission and Evaluation Format

The attached AIC 2026 document specifies each ranked result tuple but does not
specify a batch-file envelope. This repository therefore uses a strict,
versioned JSON adapter; replace only the writer adapter when BTC publishes an
official template. The validator still enforces the official tuple rules.

## Submission JSON

```json
{
  "schema_version": "1.0",
  "queries": [
    {
      "query_id": "kis-001",
      "task": "kis",
      "results": [{"video_id": "L01_V001", "frame_id": 505}]
    },
    {
      "query_id": "qna-001",
      "task": "qna",
      "results": [{"video_id": "L05_V005", "frame_id": 888, "answer": "5"}]
    },
    {
      "query_id": "trake-001",
      "task": "trake",
      "results": [{"video_id": "L10_V010", "frame_ids": [101, 151, 203, 251]}]
    }
  ]
}
```

`kis` accepts exactly `<video_id, frame_id>`, `qna` exactly `<video_id,
frame_id, answer>`, and `trake` exactly `<video_id, frame_ids[]>`. There can be
at most 100 ranked results per query. The validator rejects duplicate KIS/Q&A
video-frame pairs, duplicate TRAKE sequences, unknown videos, frame IDs outside
the verified `videos_manifest.parquet` bounds, blank Q&A answers, and TRAKE
frames that are not strictly increasing. With local development GT it also
validates task type and TRAKE event count.

## Development Ground Truth JSON

```json
{
  "schema_version": "1.0",
  "queries": [
    {
      "query_id": "kis-001",
      "task": "kis",
      "video_id": "L01_V001",
      "frame_windows": [[500, 510]]
    },
    {
      "query_id": "qna-001",
      "task": "qna",
      "video_id": "L05_V005",
      "frame_windows": [[800, 900]],
      "answer": "năm"
    },
    {
      "query_id": "trake-001",
      "task": "trake",
      "video_id": "L10_V010",
      "frame_windows": [[95, 105], [145, 155], [195, 205], [245, 255]]
    }
  ]
}
```

`python scripts/evaluate.py` calculates the published per-rank R-Score:
KIS/Q&A are binary, while TRAKE scores the fraction of matching event windows
only when the video matches. It then calculates `R@1`, `R@5`, `R@20`, `R@50`,
`R@100`, and their arithmetic mean Final Score.

## Commands

```bash
python scripts/build_submission.py --task kis --query-id kis-001 \
  --debug-json outputs/retrieval_debug/kis_example.json \
  --output outputs/submissions/dev.json
python scripts/build_submission.py --task trake --query-id trake-001 \
  --debug-json outputs/retrieval_debug/trake_example.json \
  --output outputs/submissions/dev.json --append
python scripts/validate_submission.py --submission outputs/submissions/dev.json \
  --video-manifest data/manifests/videos_manifest.parquet \
  --ground-truth data/dev_ground_truth.json
python scripts/evaluate.py --ground-truth data/dev_ground_truth.json \
  --submission outputs/submissions/dev.json --task kis
```

The build command applies configured score ordering, temporal/video diversity,
and near-duplicate TRAKE sequence suppression before writing. Validation runs
before every real benchmark/submission.
