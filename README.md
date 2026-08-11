# AIC 2026

Data foundation and multi-encoder retrieval pipeline for the AIC 2026 system.
The repository currently implements Milestones 1–9: reproducible data
acquisition/audit, BTC CLIP KIS coarse retrieval, resumable FG-CLIP2 and
PE-Core embedding/FAISS pipelines, weighted RRF fusion, and two-pass original
video frame refinement, plus ordered-event TRAKE alignment with joint dense
frame refinement, plus multi-frame Q&A with local Qwen3-VL, plus optional
BM25 OCR, ASR, and discovered-metadata retrieval. Larger model downloads,
full-dataset encoding, and competition experiments are intentionally not run
on this laptop.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Commands

```bash
python scripts/download_data.py --list
python scripts/download_data.py --dry-run
python scripts/download_data.py --only support --workers 2
python scripts/extract_data.py
python scripts/build_manifests.py
python scripts/analyze_data.py
python scripts/run_kis.py --query "a person opens a door" --top-k 100
python scripts/encode_keyframes.py --encoder fgclip2_large
python scripts/build_indexes.py --encoder fgclip2_large
python scripts/run_kis.py --encoder fgclip2_large --query "a person opens a door" --top-k 100
python scripts/encode_keyframes.py --encoder pecore_g14_448 --checkpoint /models/PE-Core-G14-448.pt
python scripts/build_indexes.py --encoder pecore_g14_448
python scripts/run_kis.py --encoder pecore_g14_448 --pe-checkpoint /models/PE-Core-G14-448.pt --query "a person opens a door" --top-k 100
python scripts/run_kis.py --encoder fg_pe_fusion --pe-checkpoint /models/PE-Core-G14-448.pt --query "a person opens a door" --top-k 100
python scripts/run_kis.py --encoder fgclip2_large --query "a person opens a door" --top-k 100
python scripts/run_kis.py --encoder fgclip2_large --coarse-only --query "a person opens a door" --top-k 100
python scripts/run_trake.py --encoder fg_pe_fusion --pe-checkpoint /models/PE-Core-G14-448.pt --query "approach run -> takeoff -> clear the bar -> landing"
python scripts/run_trake.py --encoder fgclip2_large --coarse-only --query $'1. athlete approaches the high jump bar\n2. athlete takes off\n3. athlete clears the bar\n4. athlete lands on the mat'
python scripts/run_qna.py --encoder fg_pe_fusion --pe-checkpoint /models/PE-Core-G14-448.pt --vlm-checkpoint /models/Qwen3-VL-8B-Instruct --event-description "a person opens a door" --question "What color is the door?"
python scripts/build_submission.py --task kis --query-id dev-kis-001 --debug-json outputs/retrieval_debug/kis_example.json --output outputs/submissions/kis.json
python scripts/validate_submission.py --submission outputs/submissions/kis.json --ground-truth data/dev_ground_truth.json
python scripts/evaluate.py --ground-truth data/dev_ground_truth.json --submission outputs/submissions/kis.json --task kis
python scripts/profile_pipeline.py --storage-path data/processed -- python scripts/run_kis.py --coarse-only --query "a person opens a door"
python scripts/cleanup_storage.py --encoder pecore_g14_448
python -m unittest discover -s tests -v
```

`data/` and `outputs/` are generated locally and excluded from Git. The
downloader fetches the official source sheet on every invocation and preserves
the fetched CSV in `data/manifests/source_sheet_snapshot.csv`.

The KIS command requires a locally prepared keyframe manifest and locally
available model weights. BTC requires a verified external feature-order
manifest; FG-CLIP2 and PE-Core write stable UID-order manifests during
encoding. PE-Core additionally needs Meta's `perception_models` repository
installed on the target PC, plus a local checkpoint. See
`docs/RETRIEVAL_DESIGN.md`. The repository never downloads a dataset or model
implicitly. Dense refinement also needs `opencv-python-headless`, an existing
`videos_manifest.parquet`, and original video files under `data/raw/videos/`.
TRAKE uses the same prepared index/model prerequisites as KIS. Its default
configuration runs M5 refinement for the top coarse sequences; pass
`--coarse-only` for retrieval/DP debugging without original videos.
Q&A similarly reuses KIS and M5, samples chronological original-video frames,
and loads Qwen3-VL only from a supplied local checkpoint. Install optional
runtime dependencies with `python -m pip install -e ".[qwen3-vl,refinement]"`;
the repository never downloads the VLM implicitly.

OCR, ASR, and metadata branches are disabled by default in
`configs/retrieval.yaml`. An offline producer writes their validated JSONL
artifacts under `data/processed/`; then enable each desired source and, for
metadata, set only fields confirmed by `docs/AIC2026_DATA_ANALYSIS.md`.
`run_kis.py`, `run_qna.py`, and `run_trake.py` share this runtime configuration
and emit per-source score, rank, text evidence, and RRF contribution in debug
JSON. See `docs/RETRIEVAL_DESIGN.md` for the artifact schemas.

Competition commands use versioned JSON submission/ground-truth artifacts. The
attached BTC document defines task result tuples but not a batch-file envelope;
`docs/SUBMISSION_FORMAT.md` documents the strict adapter used here. Validator
checks max-100, duplicate results, known video/frame bounds, Q&A answers, and
strict TRAKE frame order. `docs/OPERATIONS.md` is the complete PC runbook;
`docs/EXPERIMENTS.md` documents reproducible metric logging. Cleanup is always
a dry run until an explicit `--delete` flag is supplied.

## Source layout

Runtime modules are direct packages under `src/`: `data/`, `domain/`,
`download/`, and (from Milestone 2) `encoders/`, `indexing/`, `retrieval/`,
`refinement/`, `query/`, `trake/`, `qna/`, `tasks/`, `evaluation/`,
`submission/`, `experiments/`, and `hardening/`. Scripts add
`src/` to `sys.path`; installation also exposes these packages directly. The
former `src/aic2026/` wrapper does not exist.

## Interactive competition UI

`src/api/` provides a lazy FastAPI facade over the existing KIS, Q&A, and
TRAKE services, while `web/frontend/` provides a keyboard-first React/Vite
competition desk. It does not add or change retrieval algorithms. See
`docs/WEB_UI.md` for routes, local model requirements, and launch commands.
