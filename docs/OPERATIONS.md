# PC Runbook and Definition of Done

This runbook is deliberately explicit. It does not download models implicitly,
delete artifacts, or start a full benchmark by itself.

## Clone and Environment

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL> AIC-2026
cd AIC-2026
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,btc-clip,fgclip2,pecore,faiss,refinement,qwen3-vl,web]"
git clone https://github.com/facebookresearch/perception_models.git ../perception_models
python -m pip install -e ../perception_models
```

Before retrieval, copy or prepare the licensed/local FG-CLIP2, PE-Core, BTC
CLIP, and Qwen3-VL checkpoints/caches on the PC. Set their local paths and
devices in `configs/models.yaml`; keep implicit downloading disabled.

## Data Foundation

```bash
python scripts/download_data.py --list
python scripts/download_data.py --only all --workers 2
python scripts/extract_data.py
python scripts/build_manifests.py
python scripts/analyze_data.py --strict
```

Stop and fix `BLOCKER`/`HIGH` findings before encoding. Preserve the source
sheet snapshot and generated reports as the dataset snapshot for experiments.

## Optional Text Artifacts

Run your chosen OCR/ASR jobs outside this repository, then write the validated
JSONL contracts in `docs/RETRIEVAL_DESIGN.md`. Enable only available branches
and confirmed metadata fields in `configs/retrieval.yaml`; no schema field or
OCR/ASR model is assumed by this codebase.

## Embeddings and Indexes

```bash
python scripts/encode_keyframes.py --encoder fgclip2_large --device cuda --storage-dtype float16
python scripts/build_indexes.py --encoder fgclip2_large
python scripts/encode_keyframes.py --encoder pecore_g14_448 --device cuda \
  --checkpoint /models/PE-Core-G14-448.pt --storage-dtype float16
python scripts/build_indexes.py --encoder pecore_g14_448
```

Encoding resumes only when the UID order, configuration hash, completed shard
checksums, and metadata remain valid. Do not use `--no-resume` unless a clean
rebuild is intentional.

## Bash Wrappers and Resume Audit

The `bash/` directory runs from the repository root and only wraps the existing
Python CLIs. Every wrapper accepts its documented environment variables plus
extra CLI arguments. Start with:

```bash
bash bash/setup.sh
DATA_ROOT=data DOWNLOAD_SCOPE=all bash bash/download_data.sh
DEVICE=cuda bash bash/encode_fgclip2.sh
DEVICE=cuda PE_CHECKPOINT=/models/PE-Core-G14-448.pt bash bash/encode_pecore.sh
RUN_ENCODE_PE=0 RUN_BUILD_PE_INDEX=0 DEVICE=cuda bash bash/run_pipeline.sh
```

Resume behavior is intentionally stage-specific:

- **Download — supported:** `download_data.py` keeps `.part` files, resumes
  with HTTP Range when the server supports it, otherwise restarts safely, and
  skips a verified completed archive.
- **Extraction — idempotent, not partial resume:** a rerun safely skips files
  with identical contents and fails on conflicting output; it does not claim
  to resume half-written archive extraction.
- **FG-CLIP2 / PE-Core embedding — supported:** wrappers leave resume enabled
  by default. Existing manifests, UID order, config hashes, shard checksums,
  dimensions, and metadata are validated before completed shards are skipped.
  `FG_NO_RESUME=1` or `PE_NO_RESUME=1` explicitly disables this safeguard.
- **FAISS index build — not resumable:** index build refuses existing output by
  default; it only replaces it when `FG_INDEX_OVERWRITE=1` or
  `PE_INDEX_OVERWRITE=1` is set explicitly.
- **Analyze, KIS, Q&A, TRAKE, evaluate, and profile — no job resume:** they are
  deterministic CLI runs with atomic report/debug output where implemented,
  but do not persist partially completed query batches as resumable jobs.
- **Cleanup — manual only:** `bash/cleanup.sh` is a dry run unless
  `CLEANUP_DELETE=1` is supplied. `run_pipeline.sh` never calls cleanup.

There is currently no training or fine-tuning pipeline and intentionally no
`train.sh`. If one is added, every checkpoint must resume the model, optimizer,
scheduler, scaler, epoch, global step, and complete RNG state.

## Task Runs and Submission

```bash
python scripts/run_kis.py --encoder fg_pe_fusion --seed 2026 \
  --pe-checkpoint /models/PE-Core-G14-448.pt --query "..." \
  --debug-output outputs/retrieval_debug/kis_q001.json
python scripts/run_qna.py --encoder fg_pe_fusion --seed 2026 \
  --pe-checkpoint /models/PE-Core-G14-448.pt \
  --vlm-checkpoint /models/Qwen3-VL-8B-Instruct \
  --event-description "..." --question "..." \
  --debug-output outputs/retrieval_debug/qna_q001.json
python scripts/run_trake.py --encoder fg_pe_fusion --seed 2026 \
  --pe-checkpoint /models/PE-Core-G14-448.pt --query "..." \
  --debug-output outputs/retrieval_debug/trake_q001.json
```

Build one record per query with `scripts/build_submission.py`; repeat with
`--append`. Then always validate and evaluate against held-out local GT:

```bash
python scripts/validate_submission.py --submission outputs/submissions/dev.json \
  --video-manifest data/manifests/videos_manifest.parquet \
  --ground-truth data/dev_ground_truth.json
python scripts/evaluate.py --ground-truth data/dev_ground_truth.json \
  --submission outputs/submissions/dev.json --task kis \
  --experiment-log outputs/experiments/experiments.jsonl \
  --experiment-id fg-pe-kis-r1 \
  --dataset-snapshot data/reports/analysis_report.json \
  --config configs/retrieval.yaml --config configs/kis.yaml
```

Run the equivalent evaluation command separately for `qna` and `trake`.

## Benchmark, Profile, and Cleanup

```bash
python scripts/profile_pipeline.py --data-root data --storage-path data/processed \
  --output outputs/profiles/kis_q001.json -- \
  python scripts/run_kis.py --encoder fg_pe_fusion --coarse-only --query "..."
python scripts/cleanup_storage.py --encoder pecore_g14_448
python scripts/cleanup_storage.py --encoder pecore_g14_448 --delete
```

The first cleanup command is a dry run. The second is intentionally destructive
and only removes paths listed under `storage_cleanup.targets` in
`configs/hardening.yaml`; it never runs as part of encoding, indexing,
evaluation, or profiling.

## Definition of Done

The implementation-level baseline is complete when all repository tests, lint,
and compilation pass; this includes M1–M9 contracts, resume checks, safe
extraction, mapping, visual/text retrieval, refinement, TRAKE, Q&A, evaluator,
Top-100 writer/validator, experiment logging, profiling, deterministic setup,
and explicit cleanup safeguards.

Operational completion remains a PC gate: real data audit has no blocking
issues, local checkpoints load, FG/PE embeddings and indexes are built,
OCR/ASR artifacts are validated if enabled, each task emits valid Top-100
submissions, held-out metrics are logged, and the selected configuration has
been profiled. This laptop run intentionally does not claim those full-data or
full-model checks.

## Interactive Web UI

The local competition desk adds no alternate retrieval path: its FastAPI
backend lazily wraps the existing KIS, Q&A, TRAKE, RRF, refinement, VLM, and
submission services. Install the `web` extra, then install the Vite packages
once and run:

```bash
cd web/frontend && npm install && cd ../..
DEVICE=cuda PE_CHECKPOINT=/models/PE-Core-G14-448.pt \
  VLM_CHECKPOINT=/models/Qwen3-VL-8B-Instruct bash bash/run_web.sh
cd web/frontend && npm run dev
```

For one command that starts both local processes, use
`WEB_MODE=full bash bash/run_web.sh`. See `docs/WEB_UI.md` for routes, source
toggle constraints, keyboard workflow, and security boundaries. The UI has a
manual-description panel only; it intentionally does not capture, record, or
upload competition video.
