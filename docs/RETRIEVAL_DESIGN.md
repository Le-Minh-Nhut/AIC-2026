# BTC CLIP, FG-CLIP2, and PE-Core Retrieval Design

## Scope

Milestone 2 implements only KIS coarse retrieval:

```text
query → local BTC-compatible CLIP text encoder → exact cosine search
      → verified keyframe metadata → temporal NMS → KIS candidates
```

Milestone 3 additionally implements FG-CLIP2-Large image/text encoding,
resumable keyframe embedding shards, and exact FAISS `IndexFlatIP` search.
Milestone 4 adds PE-Core-G14-448 as an independent keyframe ranker and fuses
FG/PE rankings with weighted Reciprocal Rank Fusion (RRF). It does not download
data or weights and does not refine video frames.

## Direct Source Layout

Python modules are imported directly from `src/`, for example
`from indexing.feature_store import BtcClipFeatureStore`. There is no
`src/aic2026/` wrapper package.

## Feature-Order Contract

The BTC `.npy` feature files have no safe implicit relation to a Parquet
keyframe-manifest row. `BtcClipFeatureStore` therefore requires an explicit
JSON manifest before it will search any vector:

```json
{
  "feature_files": ["features_000.npy"],
  "keyframe_uids": ["L21_V001:000000", "L21_V001:000001"],
  "mapping_verified": true,
  "verification_method": "describe the checked official mapping source"
}
```

Feature paths are relative to this JSON file. The passed files must match the
listed files exactly; UID count and uniqueness, feature rows, metadata UID
coverage, frame IDs, and timestamps are all validated. The baseline rejects
unverified order, NaN/Inf, zero vectors, inconsistent dimensions, and missing
frame mappings instead of guessing.

## Model Contract

`OpenClipTextBackend` accepts only an existing local checkpoint and optional
`open_clip_torch`/`torch` installation. It never downloads a model implicitly.
The compatible checkpoint must be independently verified against the supplied
BTC image features; dimensional equality alone is insufficient proof.

## Command

```bash
python scripts/run_kis.py \
  --query "a person opens a door" \
  --feature-file data/raw/btc_clip_features/features_000.npy \
  --feature-order-manifest data/manifests/btc_clip_feature_order.json \
  --checkpoint /models/verified_btc_clip_vit_b32.pt \
  --top-k 100
```

The command writes `outputs/retrieval_debug/kis_<query-hash>.json`, including
query, stable keyframe UID, video ID, `frame_id`, timestamp, score, rank, image
path, validation audit, and aggregated video candidates.

## FG-CLIP2 Pipeline

`FGCLIP2Encoder` wraps the official `get_image_features` and
`get_text_features(..., walk_type="long")` methods. `local_files_only: true`
is the default in `configs/models.yaml`, so cache/model preparation happens on
the target PC rather than this laptop.

```bash
python scripts/encode_keyframes.py --encoder fgclip2_large
python scripts/build_indexes.py --encoder fgclip2_large
python scripts/run_kis.py --encoder fgclip2_large --query "a person opens a door" --top-k 100
```

The encoder writes `manifest.json`, `shard_*.npy`, and `shard_*.json` to
`data/processed/embeddings/fgclip2_large/`. Each shard records its row range,
stable `keyframe_uids`, runtime dimension, dtype, checksum, and normalized
status. Resume validates all completed shards and their UID mapping before
skipping them. The parent manifest records model ID/revision, preprocessing
configuration hash, source UID order/hash, runtime dimension, dtype, and Git
commit when available.

`scripts/build_indexes.py` builds `IndexFlatIP` incrementally from those
shards. Its FAISS manifest persists a checksummed `index_ids.jsonl`; KIS checks
that FAISS row order exactly equals the embedding store UID order before it
retrieves a frame.

The default storage dtype is FP32. FP16 is configurable but must only be used
after an accuracy comparison on representative real data; this laptop run does
not make that accuracy claim.

## PE-Core Pipeline and RRF

`PECoreEncoder` shares the `ImageTextEncoder` and batching contract with
`FGCLIP2Encoder`. Its backend loads a supplied local checkpoint through Meta's
official `perception_models` implementation; it does not call the package's
download path. Install that repository and its runtime dependencies on the
target PC before running PE-Core.

```bash
python scripts/encode_keyframes.py \
  --encoder pecore_g14_448 \
  --checkpoint /models/PE-Core-G14-448.pt
python scripts/build_indexes.py --encoder pecore_g14_448
python scripts/run_kis.py \
  --encoder pecore_g14_448 \
  --pe-checkpoint /models/PE-Core-G14-448.pt \
  --query "a person opens a door" --top-k 100
python scripts/run_kis.py \
  --encoder fg_pe_fusion \
  --pe-checkpoint /models/PE-Core-G14-448.pt \
  --query "a person opens a door" --top-k 100
```

PE-Core uses the same resumable `OfflineKeyframeEmbedder`, normalized shard
format, keyframe UID order checks, and FAISS `IndexFlatIP` builder as FG-CLIP2;
only model loading and preprocessing differ. PE never replaces FG: `fgclip2_large`
and `pecore_g14_448` run independently, while `fg_pe_fusion` retrieves both
rankings before temporal NMS.

Fusion is configured in `configs/retrieval.yaml` with `fusion.rrf_k` and
per-source `fusion.weights`. It uses
`sum(weight[source] / (rrf_k + source_rank))`, never an arithmetic sum of
cross-model cosine scores. Each debug JSON includes raw `source_rankings` and,
for every candidate, `source_scores` with original score, rank, RRF weight, and
RRF contribution. Ties are ordered by stable `keyframe_uid`.

## Dense Frame Refinement

Milestone 5 refines only the top `refinement.candidate_count` temporally diverse
coarse candidates. It resolves each `video_id` through
`videos_manifest.parquet`, validates the original-frame mapping, and decodes a
configurable coarse window around that frame.

1. The sparse pass samples `refinement.sparse_fps` inside
   `± refinement.coarse_window_sec`.
2. The dense pass decodes every frame inside
   `± refinement.dense_window_sec` around the sparse winner.

`FGCLIP2Encoder` and `PECoreEncoder` encode decoded frames through the same
image-text interface used for offline embeddings. Fusion mode uses the same
weighted RRF implementation as coarse retrieval; it does not sum raw model
scores. Temporal NMS remains a coarse stage and runs before refinement.

```bash
python scripts/run_kis.py \
  --encoder fg_pe_fusion \
  --pe-checkpoint /models/PE-Core-G14-448.pt \
  --query "a person opens a door" --top-k 100
```

The default requires `opencv-python-headless` and local source videos. Pass
`--coarse-only` to skip Stage 2. Debug JSON stores nested coarse candidates plus
final frames with `coarse_frame_id`, sparse frame/score, refined `frame_id`,
encoder source, source score/rank breakdown, and structured failures for
missing/corrupt videos or undecodable frames.

## TRAKE Ordered-event Retrieval

Milestone 6 composes the existing KIS coarse service once per ordered event.
`RuleBasedEventDecomposer` handles explicit numbered/bulleted lists, arrows,
and semicolon-delimited lists; it retains the complete sequence as context in
each event retrieval string, so short labels such as `landing` are not encoded
alone. The `EventDecomposer` contract permits a later LLM JSON decomposer
without changing retrieval or alignment.

```text
TRAKE query → event decomposition → FG / PE / RRF keyframe retrieval per event
            → union candidate videos → event × temporal-candidate matrix
            → k-best strict-monotonic DP → M5 dense alternatives per event
            → local strict-monotonic DP → ranked frame sequences
```

`TemporalAligner` accepts only `f1 < f2 < ... < fN`; it never selects an
independent argmax per event. Optional `min_temporal_gap_sec` and
`max_temporal_gap_sec` are hard constraints. `gap_penalty` subtracts a linear
cost from every transition beyond the configured minimum. K-best paths are
deduplicated within `sequence_dedup_window_sec`, preserving distinct temporal
basins and candidate videos.

Dense TRAKE refinement asks the M5 `DenseFrameRefiner` for several alternatives
per event, then reruns local DP on the refined matrix. Thus a high-scoring
individual dense frame cannot break event order. Missing source videos, decode
errors, and dense sequences without a monotonic path are emitted as structured
debug failures; failed sequences retain their coarse result with status
`failed` rather than disappearing silently.

```bash
python scripts/run_trake.py \
  --encoder fg_pe_fusion \
  --pe-checkpoint /models/PE-Core-G14-448.pt \
  --query "approach run -> takeoff -> clear the bar -> landing"
```

`configs/trake.yaml` controls per-event keyframes, candidate-video cap,
temporal gaps/penalty, k-best count, sequence deduplication, the number of
coarse sequences to refine, and dense options per event. Each debug candidate
contains ordered final frames, event scores, the complete coarse and refined
evidence (including model source scores/ranks), and total alignment score.

## Multi-frame Q&A

Milestone 7 reuses the exact KIS coarse service selected by `--encoder`
(`fgclip2_large`, `pecore_g14_448`, or `fg_pe_fusion`) and optionally wraps it
with the M5 dense refiner. Retrieval receives only `event_description`; the
question is deliberately not appended to retrieval text, preventing an answer
guess from biasing visual candidate selection.

```text
event description → KIS retrieval / M5 refined frame → original-video clip
                  → chronological multi-frame sample → Qwen3-VL → normalize answer
```

`CandidateClipSelector` resolves the mapped/refined `video_id` and frame in
`videos_manifest.parquet`, samples an evenly spaced, strictly chronological
window that always contains the anchor frame, and fails explicitly for missing,
unreadable, or incorrectly decoded videos. It does not pass a single keyframe
when the configured window contains more frames.

`Qwen3VLAnswerer` implements the injected `VisualAnswerer` boundary and only
loads a supplied local checkpoint with `local_files_only=True`; the Q&A service
can therefore use a fake answerer in tests. It builds a short-answer prompt
over all sampled frames, asks for no explanation, and stores both raw and
normalized output. Normalization is conservative: case/punctuation cleanup,
whitelisted number/count, yes/no, and Vietnamese/English color variants are
canonicalized, while names and other free text are retained apart from casefold.

```bash
python scripts/run_qna.py \
  --encoder fg_pe_fusion \
  --pe-checkpoint /models/PE-Core-G14-448.pt \
  --vlm-checkpoint /models/Qwen3-VL-8B-Instruct \
  --event-description "a person opens a door" \
  --question "What color is the door?"
```

`configs/qna.yaml` controls retrieval/answer candidate counts, multi-frame
count, clip window, and M5 refinement. Debug JSON contains the selected
`video_id`/`frame_id`, raw and normalized answer, coarse/refinement scores,
encoder source evidence, and chronological clip frame IDs/timestamps. Candidate
clip, VLM, and normalization failures are retained with a stage-specific error.

## OCR, ASR, and Metadata Retrieval

Milestone 8 adds optional query-text branches under the same
`QueryCandidateRetriever` contract as visual retrieval. Each branch returns
mapped `Candidate` objects, so the existing weighted RRF, temporal NMS, video
aggregation, M5 refinement, TRAKE, and Q&A orchestration are reused unchanged.
All text paths are disabled by default in `configs/retrieval.yaml`; a disabled
source neither loads nor requires its artifact.

Offline producers persist one validated JSON object per line. The repository
does not download or run an OCR/ASR model implicitly.

```json
{"record_id":"ocr-001","keyframe_uid":"L21_V001:000010","text":"Emergency exit","bbox":[0,0,100,40],"confidence":0.98}
{"segment_id":"asr-001","video_id":"L21_V001","start_sec":12.4,"end_sec":14.1,"text":"the door is open"}
{"video_id":"L21_V001","fields":{"title":"Door demo","tags":["indoor","entrance"]}}
```

OCR keeps the text, flattened bounding box, confidence, and stable
`keyframe_uid`; each matching OCR document maps directly to that verified
keyframe. ASR keeps a timestamped segment and maps its midpoint to the nearest
verified keyframe in the same video. Metadata is a video-level hit and uses the
first chronological mapped keyframe as its deterministic representative. This
is a coarse anchor, not a claim that the metadata describes that exact frame.

All three branches use deterministic in-memory BM25 first. The metadata loader
requires an explicit `metadata.fields` list and rejects a configured field that
does not occur in the artifact. Populate that list only from actual fields
reported by `docs/AIC2026_DATA_ANALYSIS.md`; no schema such as `title` or
`tags` is assumed by default.

Enable only available artifacts and tune their existing RRF weights:

```yaml
fusion:
  method: rrf
  rrf_k: 60
  weights:
    fgclip2: 1.0
    pecore: 1.0
    ocr: 0.7
    asr: 0.7
    metadata: 0.5
auxiliary_retrieval:
  ocr:
    enabled: true
    records_path: processed/ocr/ocr_records.jsonl
  asr:
    enabled: true
    records_path: processed/asr/transcript_segments.jsonl
  metadata:
    enabled: true
    records_path: processed/metadata/metadata_records.jsonl
    fields: [title, tags]
```

`scripts/run_kis.py` creates the shared coarse runtime used by
`scripts/run_qna.py` and `scripts/run_trake.py`; these commands therefore all
honor the same source enablement. Debug JSON retains each raw source ranking,
its original score/rank and text evidence, plus its weight and RRF contribution
on every fused candidate.
