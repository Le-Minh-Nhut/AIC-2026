# AIC 2026 — CODEX MASTER IMPLEMENTATION PLAN

> Mục tiêu của tài liệu này: đưa nguyên file này cho Codex để triển khai repository AIC 2026 theo đúng thứ tự, không bỏ sót data pipeline, retrieval, KIS, Q&A, TRAKE, frame refinement, evaluation và testing.

---

## 0. Mục tiêu hệ thống

Xây một hệ thống retrieval cho 3 loại truy vấn của vòng sơ tuyển AIC 2026:

1. **Textual KIS**
   - Input: mô tả sự kiện bằng text.
   - Output: `video_id, frame_id`.
   - Điều kiện đúng: đúng video và `frame_id` nằm trong đoạn ground truth.

2. **Q&A**
   - Input: mô tả sự kiện + câu hỏi.
   - Output: `video_id, frame_id, answer`.
   - Điều kiện đúng: đúng video + frame trong đoạn GT + answer đúng ngữ nghĩa.

3. **TRAKE**
   - Input: một query mô tả chuỗi nhiều event theo thứ tự.
   - Output: `video_id, frame_id_1, ..., frame_id_N`.
   - Sai `video_id` => toàn bộ R-Score bằng 0.
   - Đúng video => điểm bằng tỉ lệ event có frame nằm trong cửa sổ GT tương ứng.
   - Semantic window có thể rất ngắn, vì vậy keyframe chỉ dùng cho coarse retrieval; cuối pipeline phải quay lại video gốc để refine frame.

Hệ thống phải tối ưu cho cách chấm:
`R@1`, `R@5`, `R@20`, `R@50`, `R@100`, sau đó lấy trung bình.

---

# 1. Nguyên tắc kiến trúc

## 1.1. Không xây 3 hệ thống tách rời

Ba task dùng chung retrieval backbone:

```text
                         ┌─ FG-CLIP2-Large
Query ─ preprocessing ──┼─ PE-Core-G14-448
                         ├─ OCR text retrieval
                         ├─ ASR/transcript retrieval
                         └─ metadata retrieval
                                  ↓
                           Candidate Fusion
                                  ↓
                           Candidate Reranking
                          /          |          \
                        KIS         Q&A        TRAKE
                         |            |           |
                  Frame refine   VLM answer   Temporal DP
                         |            |           |
                     output       output     Frame refine
```

## 1.2. Keyframe chỉ là coarse search

Không coi keyframe là kết quả cuối cùng.

```text
Query
  ↓
retrieve keyframe
  ↓
map keyframe → frame_id thật
  ↓
decode video gốc quanh candidate
  ↓
dense frame scoring
  ↓
semantic frame cuối cùng
```

## 1.3. Encoder strategy

### Primary image-text encoder
**FG-CLIP2-Large**

Vai trò:
- retrieval query ↔ keyframe;
- fine-grained visual matching;
- thuộc tính, object composition, relation;
- coarse frame candidate generation.

Model:
`qihoo360/fg-clip2-large`

### Secondary encoder / ensemble branch
**PE-Core-G14-448**

Vai trò:
- retrieval query ↔ keyframe như một encoder độc lập;
- tạo một rank list khác để ensemble;
- có thể thêm video/clip-level retrieval ở phase sau.

Checkpoint:
`PE-Core-G14-448`

### Cách fusion mặc định
Không cộng trực tiếp cosine score của hai model vì scale có thể khác nhau.

Dùng **Weighted Reciprocal Rank Fusion (RRF)** trước:

\[
RRF(x) = \sum_m \frac{w_m}{k + rank_m(x)}
\]

Default:
- `k = 60`
- `w_fgclip2 = 1.0`
- `w_pecore = 1.0`

Sau khi có validation/dev GT mới tune weight.

## 1.4. VLM cho Q&A và reranking

Default local VLM:
**Qwen3-VL-8B-Instruct**

Vai trò:
- Q&A trên clip / multi-frame;
- rerank top candidate khó;
- optional event verification trong TRAKE;
- OCR/visual reasoning bổ sung khi cần.

Không dùng VLM để encode toàn bộ database.

---

# 2. Cây thư mục repository

Codex phải tạo layout này:

```text
aic2026/
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
│
├── configs/
│   ├── data.yaml
│   ├── models.yaml
│   ├── retrieval.yaml
│   ├── kis.yaml
│   ├── qna.yaml
│   └── trake.yaml
│
├── data/                              # gitignored
│   ├── manifests/
│   │   ├── source_sheet_snapshot.csv
│   │   ├── archives_manifest.jsonl
│   │   ├── videos_manifest.parquet
│   │   └── keyframes_manifest.parquet
│   │
│   ├── raw/
│   │   ├── archives/
│   │   ├── videos/
│   │   ├── keyframes/
│   │   ├── btc_clip_features/
│   │   ├── map_keyframes/
│   │   ├── objects/
│   │   └── media_info/
│   │
│   ├── processed/
│   │   ├── embeddings/
│   │   │   ├── btc_clip/
│   │   │   ├── fgclip2_large/
│   │   │   └── pecore_g14_448/
│   │   │
│   │   ├── indexes/
│   │   │   ├── fgclip2_large/
│   │   │   └── pecore_g14_448/
│   │   │
│   │   ├── ocr/
│   │   ├── asr/
│   │   ├── metadata_text/
│   │   └── dense_frame_cache/
│   │
│   └── reports/
│       ├── data_analysis.json
│       └── data_analysis_tables/
│
├── docs/
│   ├── DATA_ANALYSIS.md
│   ├── RETRIEVAL_DESIGN.md
│   ├── EXPERIMENTS.md
│   └── SUBMISSION_FORMAT.md
│
├── src/                               # direct import root; no aic2026 wrapper
│   ├── config.py
│   ├── domain/
│   │   ├── models.py
│   │   └── protocols.py
│   ├── data/
│   │   ├── source_sheet.py
│   │   ├── archive_manifest.py
│   │   ├── video_manifest.py
│   │   ├── keyframe_manifest.py
│   │   ├── keyframe_mapping.py
│   │   └── repositories.py
│   ├── download/
│   │   ├── downloader.py
│   │   ├── extractor.py
│   │   └── integrity.py
│   ├── encoders/
│   │   ├── base.py
│   │   ├── btc_clip.py
│   │   ├── fgclip2.py
│   │   ├── multimodal.py
│   │   └── pecore.py
│   ├── indexing/
│   │   ├── base.py
│   │   ├── embedding_pipeline.py
│   │   ├── exact_index.py
│   │   ├── faiss_index.py
│   │   ├── metadata_store.py
│   │   └── sharded_feature_store.py
│   ├── query/
│   │   ├── normalizer.py
│   │   ├── translator.py
│   │   ├── query_variants.py
│   │   └── event_decomposer.py
│   ├── retrieval/
│   │   ├── candidate.py
│   │   ├── visual_retriever.py
│   │   ├── text_retriever.py
│   │   ├── fusion.py
│   │   ├── temporal_nms.py
│   │   └── video_aggregation.py
│   ├── refinement/
│   │   ├── video_decoder.py
│   │   ├── frame_sampler.py
│   │   ├── dense_frame_refiner.py
│   │   └── vlm_reranker.py
│   ├── qna/
│   │   ├── answerer.py
│   │   ├── frame_selector.py
│   │   └── answer_normalizer.py
│   ├── trake/
│   │   ├── event_candidates.py
│   │   ├── video_selector.py
│   │   ├── temporal_aligner.py
│   │   ├── kbest_alignment.py
│   │   └── event_refiner.py
│   ├── tasks/
│   │   ├── kis_service.py
│   │   ├── qna_service.py
│   │   └── trake_service.py
│   ├── evaluation/
│   │   ├── rscore.py
│   │   ├── final_score.py
│   │   └── evaluator.py
│   └── submission/
│       ├── ranker.py
│       └── writer.py
│
├── scripts/
│   ├── download_data.py
│   ├── extract_data.py
│   ├── analyze_data.py
│   ├── build_manifests.py
│   ├── encode_keyframes.py
│   ├── build_indexes.py
│   ├── run_kis.py
│   ├── run_qna.py
│   ├── run_trake.py
│   └── evaluate.py
│
├── tests/
│   ├── unit/
│   │   ├── test_source_sheet.py
│   │   ├── test_keyframe_mapping.py
│   │   ├── test_rrf.py
│   │   ├── test_temporal_nms.py
│   │   ├── test_temporal_aligner.py
│   │   ├── test_rscore.py
│   │   └── test_answer_normalizer.py
│   │
│   └── integration/
│       ├── test_download_dry_run.py
│       ├── test_manifest_pipeline.py
│       └── test_small_retrieval_pipeline.py
│
└── outputs/
    ├── logs/
    ├── experiments/
    ├── retrieval_debug/
    └── submissions/
```

---

# 3. SOLID contract

Codex phải giữ model/task phụ thuộc qua interface thay vì hardcode.

## 3.1. Core protocols

```python
class ImageTextEncoder(Protocol):
    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...

class VectorIndex(Protocol):
    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None: ...
    def search(self, query: np.ndarray, top_k: int) -> Sequence[SearchHit]: ...

class Retriever(Protocol):
    def retrieve(self, query: RetrievalQuery, top_k: int) -> Sequence[Candidate]: ...

class FrameRefiner(Protocol):
    def refine(self, candidate: Candidate, query: str) -> RefinedCandidate: ...

class TemporalAligner(Protocol):
    def align(self, events: Sequence[EventQuery], candidates: Sequence[EventCandidate]) -> Alignment: ...

class VisualAnswerer(Protocol):
    def answer(self, frames: Sequence[Image.Image], question: str) -> AnswerResult: ...
```

Task services chỉ orchestration, không chứa model loading.

---

# 4. DATA DOWNLOAD — bắt buộc làm đầu tiên

Nguồn spreadsheet:

`https://docs.google.com/spreadsheets/d/1rfn1fieTThS_Ki3SIoJ6uXOx2AhMq7wGCak6W4jZyZM/edit?gid=0#gid=0`

Sheet hiện có cột:
- `Filenames`
- `Download link`

## 4.1. `scripts/download_data.py`

### Nhiệm vụ

1. Fetch CSV từ Google Sheet.
2. Parse `Filenames`, `Download link`.
3. Loại row rỗng.
4. Validate URL.
5. Phân loại archive.
6. Save snapshot vào:
   `data/manifests/source_sheet_snapshot.csv`
7. Download vào:
   `data/raw/archives/`
8. Hỗ trợ resume.
9. Hỗ trợ retry.
10. Hỗ trợ skip existing.
11. Hỗ trợ dry-run.
12. Có progress bar.
13. Ghi JSONL manifest sau download.
14. Không tự xóa archive sau extract trừ khi user bật flag.

### CLI

```bash
python scripts/download_data.py --list
python scripts/download_data.py --dry-run
python scripts/download_data.py --only keyframes
python scripts/download_data.py --only videos
python scripts/download_data.py --only support
python scripts/download_data.py --only all
python scripts/download_data.py --only all --workers 2
python scripts/download_data.py --only all --extract
```

### Categories

```text
Keyframes_*.zip       → keyframes
Videos_*.zip          → videos
clip-features-*.zip   → btc_clip_features
map-keyframes-*.zip   → map_keyframes
media-info-*.zip      → media_info
objects-*.zip         → objects
```

### Download behavior

- HEAD request trước nếu server hỗ trợ.
- Ghi `Content-Length` nếu có.
- File đang tải dùng suffix `.part`.
- Nếu server hỗ trợ Range:
  - resume từ byte hiện có.
- Nếu server không hỗ trợ Range:
  - restart an toàn.
- Retry exponential backoff.
- Timeout rõ ràng.
- Không tải HTML error page rồi rename thành ZIP.
- Sau download:
  - kiểm tra ZIP signature;
  - test archive bằng `zipfile.ZipFile.testzip()`;
  - compute SHA256 local;
  - ghi size + hash vào manifest.
- Không được nói "verified against official checksum" vì BTC không cung cấp checksum trong sheet.

### Disk safety

Trước khi download:
- lấy free space;
- nếu biết Content-Length thì estimate tổng;
- cảnh báo nếu free space thấp;
- không tự đoán extracted size.

### Safe extraction

Reject:
- absolute paths;
- `../`;
- path traversal;
- symlink bất thường nếu có.

---

# 5. Snapshot hiện tại của spreadsheet

Tại thời điểm viết spec, sheet có **32 archive links**:
- 14 Keyframes archives
- 14 Videos archives
- 4 support archives

## 5.1. Keyframes — 14 archives

| # | Filename | URL |
|---|---|---|
| 1 | Keyframes_L21.zip | https://aic-data.ledo.io.vn/Keyframes_L21.zip |
| 2 | Keyframes_L22.zip | https://aic-data.ledo.io.vn/Keyframes_L22.zip |
| 3 | Keyframes_L23.zip | https://aic-data.ledo.io.vn/Keyframes_L23.zip |
| 4 | Keyframes_L24.zip | https://aic-data.ledo.io.vn/Keyframes_L24.zip |
| 5 | Keyframes_L25.zip | https://aic-data.ledo.io.vn/Keyframes_L25.zip |
| 6 | Keyframes_L26_a.zip | https://aic-data.ledo.io.vn/Keyframes_L26_a.zip |
| 7 | Keyframes_L26_b.zip | https://aic-data.ledo.io.vn/Keyframes_L26_b.zip |
| 8 | Keyframes_L26_c.zip | https://aic-data.ledo.io.vn/Keyframes_L26_c.zip |
| 9 | Keyframes_L26_d.zip | https://aic-data.ledo.io.vn/Keyframes_L26_d.zip |
| 10 | Keyframes_L26_e.zip | https://aic-data.ledo.io.vn/Keyframes_L26_e.zip |
| 11 | Keyframes_L27.zip | https://aic-data.ledo.io.vn/Keyframes_L27.zip |
| 12 | Keyframes_L28.zip | https://aic-data.ledo.io.vn/Keyframes_L28.zip |
| 13 | Keyframes_L29.zip | https://aic-data.ledo.io.vn/Keyframes_L29.zip |
| 14 | Keyframes_L30.zip | https://aic-data.ledo.io.vn/Keyframes_L30.zip |

## 5.2. Videos — 14 archives

| # | Filename | URL |
|---|---|---|
| 1 | Videos_L21_a.zip | https://aic-data.ledo.io.vn/Videos_L21_a.zip |
| 2 | Videos_L22_a.zip | https://aic-data.ledo.io.vn/Videos_L22_a.zip |
| 3 | Videos_L23_a.zip | https://aic-data.ledo.io.vn/Videos_L23_a.zip |
| 4 | Videos_L24_a.zip | https://aic-data.ledo.io.vn/Videos_L24_a.zip |
| 5 | Videos_L25_a.zip | https://aic-data.ledo.io.vn/Videos_L25_a.zip |
| 6 | Videos_L26_a.zip | https://aic-data.ledo.io.vn/Videos_L26_a.zip |
| 7 | Videos_L26_b.zip | https://aic-data.ledo.io.vn/Videos_L26_b.zip |
| 8 | Videos_L26_c.zip | https://aic-data.ledo.io.vn/Videos_L26_c.zip |
| 9 | Videos_L26_d.zip | https://aic-data.ledo.io.vn/Videos_L26_d.zip |
| 10 | Videos_L26_e.zip | https://aic-data.ledo.io.vn/Videos_L26_e.zip |
| 11 | Videos_L27_a.zip | https://aic-data.ledo.io.vn/Videos_L27_a.zip |
| 12 | Videos_L28_a.zip | https://aic-data.ledo.io.vn/Videos_L28_a.zip |
| 13 | Videos_L29_a.zip | https://aic-data.ledo.io.vn/Videos_L29_a.zip |
| 14 | Videos_L30_a.zip | https://aic-data.ledo.io.vn/Videos_L30_a.zip |

## 5.3. Support data — 4 archives

| Filename | URL |
|---|---|
| clip-features-32-aic25-b1.zip | https://aic-data.ledo.io.vn/clip-features-32-aic25-b1.zip |
| map-keyframes-aic25-b1.zip | https://aic-data.ledo.io.vn/map-keyframes-aic25-b1.zip |
| media-info-aic25-b1.zip | https://aic-data.ledo.io.vn/media-info-aic25-b1.zip |
| objects-aic25-b1.zip | https://aic-data.ledo.io.vn/objects-aic25-b1.zip |

**Important:** downloader phải fetch sheet mỗi lần chạy để batch mới được phát hiện tự động; snapshot bên trên chỉ là fallback/audit record.

---

# 6. Extraction layout

Sau extract, chuẩn hóa về logical structure, không phụ thuộc ZIP internal path.

```text
data/raw/
├── videos/
│   ├── L21/
│   │   ├── L21_V001.mp4
│   │   └── ...
│   ├── L22/
│   └── ...
│
├── keyframes/
│   ├── L21_V001/
│   │   ├── 0000.jpg
│   │   ├── 0001.jpg
│   │   └── ...
│   └── ...
│
├── btc_clip_features/
├── map_keyframes/
├── objects/
└── media_info/
```

Không rename video/keyframe ID nếu chưa có lý do.

Nếu archive có layout khác, extractor phải detect và normalize qua manifest thay vì move tùy tiện.

---

# 7. Data manifest layer

## 7.1. `videos_manifest.parquet`

Một row / video:

```text
video_id
video_path
group_id
fps
frame_count
duration_sec
width
height
video_codec
audio_codec
audio_sample_rate
audio_channels
has_audio
container
file_size_bytes
is_readable
```

## 7.2. `keyframes_manifest.parquet`

Một row / keyframe:

```text
keyframe_uid
video_id
keyframe_index
keyframe_path
original_frame_id
timestamp_sec
width
height
file_size_bytes
is_readable
has_mapping
```

`keyframe_uid` phải stable, ví dụ:

```text
L21_V001:000017
```

Không dùng array row position như identity duy nhất.

## 7.3. Mapping invariants

Mỗi keyframe phải satisfy nếu mapping tồn tại:

```text
0 <= original_frame_id < video.frame_count
timestamp_sec ≈ original_frame_id / fps
```

Trong cùng video:
- `keyframe_index` tăng dần;
- `original_frame_id` phải non-decreasing / ideally strictly increasing;
- duplicate mapping phải report.

---

# 8. Data analysis — phải chạy trước model mới

Implement:

```bash
python scripts/analyze_data.py
```

Output:
- `docs/DATA_ANALYSIS.md`
- `data/reports/data_analysis.json`
- optional CSV tables trong `data/reports/data_analysis_tables/`

Không hardcode số liệu. Báo cáo phải sinh trực tiếp từ data đã download.

Chi tiết đầy đủ nằm trong file companion:
`AIC2026_DATA_ANALYSIS.md`.

---

# 9. Phase A — BTC CLIP baseline

Mục đích:
- test mapping;
- test evaluator;
- test retrieval pipeline;
- không cần encode ảnh lại.

BTC cung cấp CLIP ViT-B/32 features cho keyframes.

Pipeline:

```text
Query
  ↓ BTC-compatible CLIP text encoder
normalized text vector
  ↓ inner product
BTC keyframe feature matrix
  ↓
Top-K keyframe
```

Điều kiện:
- xác minh feature dimension runtime;
- xác minh dtype;
- xác minh mỗi vector map đúng keyframe;
- kiểm tra norm;
- normalize nếu cần trước cosine;
- không đoán order nếu mapping file chưa verify.

Output debug:

```text
outputs/retrieval_debug/<query_id>.json
```

bao gồm:
- query;
- keyframe_uid;
- video_id;
- frame_id;
- score;
- rank;
- image path.

---

# 10. Phase B — FG-CLIP2-Large indexing

## 10.1. Encode toàn bộ keyframe offline

Command:

```bash
python scripts/encode_keyframes.py \
  --encoder fgclip2_large \
  --batch-size AUTO \
  --output data/processed/embeddings/fgclip2_large
```

Rules:
- inference mode;
- autocast nếu model hỗ trợ;
- batch size adaptive;
- normalize embedding L2;
- save FP16 nếu accuracy check không bị degrade đáng kể;
- shard output;
- không giữ toàn bộ ảnh/vector trong RAM;
- mỗi shard phải có metadata row range hoặc UID list;
- support resume;
- write model revision/config hash.

Store:

```text
embeddings/
├── shard_00000.npy
├── shard_00001.npy
├── ...
└── manifest.json
```

`manifest.json`:
- model name;
- model revision;
- preprocessing config;
- embedding dimension discovered runtime;
- dtype;
- normalized yes/no;
- count;
- keyframe UID order;
- creation timestamp;
- git commit nếu có.

## 10.2. Query language

FG-CLIP2 chủ yếu English/Chinese.

AIC query có thể tiếng Việt.

Do đó query layer phải giữ:
- `original_text_vi`
- `translated_text_en`
- optional `compact_visual_text_en`

Default retrieval:
- English translation làm primary.
- Có thể test Vietnamese trực tiếp nhưng không assume mạnh.

Không rewrite quá sáng tạo vì có thể làm mất điều kiện query.

---

# 11. Phase C — PE-Core-G14-448 indexing

Tương tự FG-CLIP2.

Command:

```bash
python scripts/encode_keyframes.py \
  --encoder pecore_g14_448 \
  --batch-size AUTO
```

Output riêng.

PE không thay FG-CLIP2. Nó là second independent ranker.

Sau đó retrieval:

```text
query
 ├── FG-CLIP2 → rank_fg
 └── PE-Core  → rank_pe
        ↓
       RRF
        ↓
 fused candidates
```

Không được ghi code theo kiểu “PE chỉ dùng video”; cả hai phải implement cùng `ImageTextEncoder` interface và đều có thể search keyframe.

---

# 12. Indexing

## 12.1. Baseline

Vì vector đã L2-normalized:

\[
cos(q, x) = q^T x
\]

FAISS:
- `IndexFlatIP` làm exact baseline;
- chỉ chuyển ANN index nếu latency/memory thực sự cần.

Không optimize sớm.

## 12.2. Metadata store

FAISS index row phải map tới:
- `keyframe_uid`
- `video_id`
- `original_frame_id`
- `timestamp_sec`
- `keyframe_path`

Không để mapping bằng “row number” rải rác ở nhiều file.

---

# 13. Candidate post-processing

## 13.1. Temporal NMS

Problem:

```text
rank 1: video A frame 1000
rank 2: video A frame 1005
rank 3: video A frame 1010
...
```

Lãng phí top ranking.

Temporal NMS:
- cùng video;
- nếu timestamp quá gần candidate cao hơn;
- suppress / demote.

Config:
```yaml
temporal_nms:
  enabled: true
  window_sec: 2.0
```

Không cố định 2.0 trong code.

## 13.2. Video aggregation

Từ keyframe score → video score.

Baseline:
```text
video_score = max keyframe score
```

Alternative:
```text
video_score = mean(top_m keyframe scores)
```

TRAKE sẽ dùng sequence-aware video score riêng, không dùng max đơn giản.

---

# 14. KIS pipeline

## Stage 1 — coarse retrieval

```text
query
 ↓ normalize/translate
FG-CLIP2 + PE-Core
 ↓
RRF
 ↓
top keyframes
 ↓
temporal NMS
 ↓
top candidate windows
```

## Stage 2 — dense frame refinement

Với mỗi high-priority candidate:

```text
keyframe original frame = F
fps = r
```

Window coarse:

```text
[F - 3*r, F + 3*r]
```

Configurable.

Hai pass:

### Pass A
- sample 3–5 FPS trong cửa sổ;
- encode;
- lấy điểm tốt nhất.

### Pass B
- quanh best timestamp ±0.5–1.0 sec;
- decode dense/full FPS;
- encode tất cả frame;
- lấy best frame.

Optional top candidate:
- VLM rerank 4–8 frames gần nhau.

Final KIS candidate:

```text
video_id
frame_id
score
source_keyframe_uid
coarse_score
refine_score
```

## Top-100 strategy

Không trả 100 frame gần nhau cùng video.

Phải balance:
- top confidence;
- video diversity;
- temporal diversity.

---

# 15. Q&A pipeline

Input logical model:

```python
QnAQuery(
    query_id=...,
    event_description=...,
    question=...,
)
```

Không nhập cả câu lẫn answer guess vào retrieval một cách tùy tiện.

## Stage 1 — retrieve event

Primary retrieval text:
`event_description`

Optional retrieval variants:
- `event_description + relevant visual constraints from question`
- nhưng không thêm thông tin suy đoán answer.

## Stage 2 — candidate clip

Với mỗi candidate:
- lấy frame quanh timestamp;
- sample multi-frame sequence;
- giữ temporal order;
- optional ASR text cùng đoạn.

## Stage 3 — Qwen3-VL answer

Prompt bắt model:
- chỉ trả lời dựa trên frames/clip;
- short answer;
- không giải thích dài nếu submission chỉ cần answer.

Store:
- raw answer;
- normalized answer;
- candidate score;
- answer consistency score.

## Stage 4 — answer normalization

Normalize:
- lowercase/casefold;
- Unicode;
- trim punctuation;
- số chữ ↔ digit khi phù hợp;
- color synonyms;
- yes/no variants;
- Vietnamese/English equivalents có whitelist rõ ràng;
- không aggressively rewrite named entities.

Examples:

```text
"5"
"five"
"Năm"
"5 people"
```

có thể canonicalize `5` cho count question.

---

# 16. TRAKE pipeline — phần ưu tiên cao nhất

TRAKE không phải chỉ “retrieve mỗi event rồi argmax”.

## 16.1. Event decomposition

Input:

```text
Tìm các khoảnh khắc:
1. chạy đà
2. giậm nhảy
3. bay qua xà
4. tiếp đất
```

Output structured:

```python
[
  EventQuery(index=0, text="athlete approaching the high jump bar"),
  EventQuery(index=1, text="athlete taking off from the ground for a high jump"),
  EventQuery(index=2, text="athlete clearing the high jump bar"),
  EventQuery(index=3, text="athlete landing on the mat after the high jump"),
]
```

Important:
- mỗi event giữ context của activity;
- không encode từ quá ngắn như `"landing"` nếu có thể ambiguous;
- preserve order.

Event decomposition có thể:
1. rule-based nếu query đã đánh số;
2. LLM structured JSON nếu query tự nhiên phức tạp;
3. validate event count.

## 16.2. Retrieve mỗi event

Mỗi event:
- FG-CLIP2 search;
- PE-Core search;
- RRF;
- lấy top candidates.

Không chọn frame final ở bước này.

## 16.3. Candidate video generation

Lấy union video xuất hiện trong top event candidates.

Một video mạnh phải:
- có evidence cho nhiều event;
- evidence có thể sắp theo temporal order.

Không score đơn giản:
`sum(max score từng event)` nếu bỏ qua thời gian.

## 16.4. Sequence-aware DP

Với video `v`, event `j`, candidate time `t`:

\[
S_{j,t} = fused\_similarity(event_j, frame_t)
\]

Tìm:

\[
t_1 < t_2 < ... < t_N
\]

maximize:

\[
\sum_{j=1}^{N} S_{j,t_j} - P(t_1,\ldots,t_N)
\]

Baseline DP:
- hard monotonic order;
- no gap penalty trước.

Phase 2:
- optional reasonable-gap penalty;
- optional min separation;
- optional max separation;
- tất cả configurable;
- chỉ bật khi dev data chứng minh tốt hơn.

Không encode prior về duration quá mạnh vì activity khác nhau.

## 16.5. K-best sequences

BTC chấm top 100.

TRAKE service phải có khả năng tạo nhiều candidate sequence:
- nhiều video;
- nhiều alignment trong cùng video nếu ambiguity cao.

Không duplicate sequences gần như giống nhau.

Sequence diversity:
- different video;
- different temporal basin;
- different event frame set.

## 16.6. Dense refinement từng event

Sau coarse DP:

```text
E1 coarse = frame 1000
E2 coarse = frame 1500
E3 coarse = frame 2000
E4 coarse = frame 2500
```

Mỗi event:
- decode window xung quanh;
- score dense frames bằng event query;
- refine frame.

Sau refine phải kiểm tra lại:

```text
f1 < f2 < ... < fN
```

Nếu refinement phá order:
- joint local DP trên dense candidate frames;
- không sửa thủ công bằng cách sort frame, vì sort làm mất event identity.

## 16.7. Semantic keyframe issue

CLIP cosine có thể tìm “đúng trạng thái” nhưng không nhất thiết tìm:
- first contact;
- first lift-off;
- exact highest point.

Do đó optional advanced refinement:
- optical flow / motion cue;
- VLM local frame comparison;
- pairwise before/after reasoning;
- event-specific predicate.

Nhưng chỉ triển khai sau khi coarse TRAKE + dense cosine baseline chạy ổn.

---

# 17. OCR / ASR / Metadata branches

Không làm trước visual baseline.

## 17.1. OCR

Offline OCR trên keyframes:
- text;
- bbox;
- confidence;
- `keyframe_uid`.

Index OCR text bằng:
- BM25 first;
- multilingual dense retriever later.

OCR hữu ích khi query chứa:
- biển hiệu;
- tên riêng;
- số;
- subtitle;
- text trên màn hình.

## 17.2. ASR

Extract audio từ video.

ASR output:
```text
video_id
start_sec
end_sec
text
```

Dùng timestamped segments.

Index transcript.

Khi ASR hit:
- map segment timestamp → nearest keyframes/video frames;
- fuse với visual retrieval.

## 17.3. Metadata

Index:
- title;
- description;
- other textual fields thật sự có trong media info.

Không assume field không tồn tại.
Schema phải discover từ data analysis.

---

# 18. Multimodal fusion

Candidate sources:

```text
FG-CLIP2
PE-Core
OCR
ASR
metadata
```

Phase 1:
- RRF.

Phase 2:
- weighted RRF tuned trên dev.

Không train learned fusion nếu không có đủ GT.

Candidate object:

```python
Candidate(
    video_id=...,
    frame_id=...,
    timestamp_sec=...,
    fused_score=...,
    source_scores={
        "fgclip2": ...,
        "pecore": ...,
        "ocr": ...,
        "asr": ...,
        "metadata": ...,
    },
)
```

Luôn giữ score breakdown để debug.

---

# 19. Evaluator

Implement examples từ official spec thành unit test.

## 19.1. KIS

\[
R = I(video=GT_v \land frame \in [s,e])
\]

## 19.2. Q&A

\[
R = I(video=GT_v \land frame \in [s,e] \land answer=GT_a)
\]

Answer semantic matching phải tách khỏi evaluator core:
- evaluator nhận canonical answer matcher strategy.

## 19.3. TRAKE

Nếu sai video:
\[
R=0
\]

Nếu đúng video:
\[
R = \frac{1}{N}\sum_j I(frame_j \in [s_j,e_j])
\]

## 19.4. Final Score

Với:
`k ∈ {1,5,20,50,100}`

\[
R@k = \max_{i \le k} R_i
\]

\[
Final = \frac{R@1+R@5+R@20+R@50+R@100}{5}
\]

Unit test phải cover:
- correct top1;
- correct only rank3;
- correct rank15;
- TRAKE partial 3/4;
- wrong TRAKE video = 0.

---

# 20. Experiment tracking

Mỗi experiment phải ghi:

```text
experiment_id
date
git_commit
dataset_snapshot
encoder
model_revision
query_strategy
index_type
fusion
top_k
NMS config
refinement config
task
R@1
R@5
R@20
R@50
R@100
Final Score
latency
notes
```

Append vào:
`docs/EXPERIMENTS.md`

Không đổi 3 thứ cùng lúc rồi kết luận model tốt hơn.

---

# 21. Logging / debug UI artifacts

Mỗi query nên có debug JSON:

```json
{
  "query_id": "...",
  "query": "...",
  "variants": [],
  "candidates": [],
  "final_outputs": []
}
```

Optional contact sheet:
- top 20 keyframes;
- rank;
- video_id;
- frame_id;
- FG score;
- PE score;
- fused score.

Đây là công cụ cực quan trọng để nhìn failure mode.

---

# 22. Implementation order cho Codex

## Milestone 1 — Repository + data
1. project skeleton
2. configs
3. downloader
4. safe extractor
5. source snapshot
6. manifests
7. data analyzer

**Acceptance**
- `download_data.py --dry-run` parse đủ current sheet.
- `--list` hiển thị đúng category.
- có unit tests.

## Milestone 2 — BTC baseline
1. load mapping
2. load BTC CLIP
3. text encoder compatible
4. exact cosine search
5. top-K
6. mapping to video/frame
7. temporal NMS
8. KIS output

**Acceptance**
- một query text trả top candidates end-to-end.

## Milestone 3 — FG-CLIP2
1. encoder wrapper
2. offline indexing
3. resume
4. FAISS
5. query search
6. KIS coarse retrieval

**Acceptance**
- toàn bộ keyframe có stable embedding + metadata mapping.
- search deterministic.

## Milestone 4 — PE-Core + fusion
1. PE wrapper
2. PE embeddings
3. RRF
4. score breakdown

**Acceptance**
- có thể chạy FG only / PE only / fusion bằng config.

## Milestone 5 — Dense frame refinement
1. decoder
2. coarse window
3. sparse pass
4. dense pass
5. refined frame

**Acceptance**
- candidate keyframe map về frame video thật và refine được.

## Milestone 6 — TRAKE
1. event parser
2. per-event retrieval
3. candidate videos
4. monotonic DP
5. k-best
6. dense joint refine

**Acceptance**
- synthetic 4-event test trả đúng order.
- wrong temporal order không được chọn nếu có ordered sequence tốt hơn.
- event parser giữ full action context cho các item ngắn; DP k-best và dense
  joint refinement đều lưu debug evidence/failure theo từng event.

## Milestone 7 — Q&A
1. candidate clip sampler
2. Qwen3-VL wrapper
3. short answer prompt
4. answer normalization
5. ranked outputs

**Acceptance**
- end-to-end query → video/frame/answer.
- retrieve chỉ bằng event description; sample chronological multi-frame clip;
  raw/normalized answer, source score, frame debug và VLM failures phải traceable.

## Milestone 8 — OCR/ASR/metadata
1. offline extraction
2. indexes
3. RRF fusion
4. ablation

**Acceptance**
- JSONL artifacts validate OCR `text`/bbox/confidence/keyframe UID and ASR
  timestamped transcript segments before retrieval.
- OCR, ASR, and metadata rank mapped keyframe candidates through a shared
  query-ranker contract; ASR maps segment midpoint to the nearest mapped
  keyframe and metadata uses only fields discovered by data analysis.
- KIS, Q&A, and TRAKE share source enablement and weighted-RRF configuration;
  debug output retains source rank/score, OCR/ASR/metadata evidence, and RRF
  contribution.

## Milestone 9 — competition hardening
1. profiling
2. caching
3. robust resume
4. corruption handling
5. top100 diversity
6. submission validation
7. reproducibility

**Acceptance**
- evaluator implements the official KIS/Q&A/TRAKE R-Score, `R@1/5/20/50/100`,
  and Final Score formulas with synthetic boundary tests.
- versioned writer/validator enforces task result shapes, 100-result limit,
  duplicate/video/frame/answer checks, and strict TRAKE order before submission.
- frame and sequence rankers enforce deterministic temporal/video diversity;
  experiment JSONL records provenance, metric, latency, and model/config data.
- profiling, explicit cache, deterministic setup, safe resume from prior
  milestones, and deletion-disabled-by-default storage cleanup are available as
  independent operational utilities.

---

# 23. Config defaults

## `configs/models.yaml`

```yaml
encoders:
  primary:
    name: fgclip2_large
    model_id: qihoo360/fg-clip2-large

  secondary:
    name: pecore_g14_448
    model_id: facebook/PE-Core-G14-448
    model_config: PE-Core-G14-448
    checkpoint: null

vlm:
  name: qwen3_vl_8b_instruct
  model_id: Qwen/Qwen3-VL-8B-Instruct
  quantization: configurable
```

## `configs/retrieval.yaml`

```yaml
search:
  top_k_per_encoder: 1000
  exact_index_first: true

fusion:
  method: rrf
  rrf_k: 60
  weights:
    fgclip2: 1.0
    pecore: 1.0

temporal_nms:
  enabled: true
  window_sec: 2.0

refinement:
  enabled: true
  coarse_window_sec: 3.0
  sparse_fps: 4
  dense_window_sec: 1.0
  candidate_count: 10
```

Các con số này chỉ là starting point, không phải truth. Tune bằng evaluator.

---

# 24. Performance / memory rules

- encode keyframes offline một lần;
- query inference chỉ encode text;
- embeddings dùng shard/memmap;
- image batch size adaptive;
- không load tất cả JPEG vào RAM;
- cache dense frames có TTL/LRU hoặc disk quota;
- VLM chỉ chạy top candidate;
- FAISS exact baseline trước;
- benchmark latency trước khi chuyển ANN;
- model loading singleton theo process nhưng dependency injection vẫn giữ interface sạch.

---

# 25. Không được làm sớm

Trước khi baseline end-to-end có score:

- không fine-tune FG-CLIP2;
- không fine-tune PE;
- không patch shuffling loss;
- không tự thiết kế custom loss;
- không train video transformer;
- không learned fusion phức tạp.

Thứ tự ROI:

```text
Data correctness
→ mapping correctness
→ retrieval
→ ranking/diversity
→ dense frame refinement
→ TRAKE temporal alignment
→ Q&A
→ multimodal support
→ fine-tuning
```

---

# 26. Failure modes phải audit

## Data
- ZIP hỏng;
- missing video;
- missing keyframe dir;
- mapping thiếu;
- frame id vượt video length;
- corrupt JPEG;
- CLIP array length lệch keyframe count;
- NaN/Inf embeddings;
- duplicate video IDs;
- duplicate keyframe IDs;
- wrong archive internal root.

## Retrieval
- query translation mất thuộc tính;
- 1 video chiếm hết top100;
- score scale khác giữa encoders;
- visually similar but wrong event;
- OCR exact text dominate sai context;
- ASR hit đúng từ nhưng sai cảnh.

## KIS
- coarse keyframe đúng video nhưng frame final chưa refine;
- refine window quá nhỏ.

## Q&A
- retrieve sai clip nhưng VLM hallucinate answer;
- answer đúng nghĩa nhưng formatting khác;
- count question bị frame sampling thiếu người.

## TRAKE
- event decomposition sai số lượng;
- event text mất global context;
- mỗi event retrieval đúng nhưng đến từ video khác;
- candidate video đủ events nhưng sai thứ tự;
- refine từng event độc lập phá monotonic order;
- semantic transition quá ngắn để cosine xác định exact frame.

---

# 27. Tests tối thiểu

## Unit
- sheet parser
- URL classifier
- resume logic mocked
- zip-slip protection
- mapping validation
- cosine normalization
- RRF
- temporal NMS
- video aggregation
- event decomposition validation
- DP monotonicity
- k-best dedup
- answer normalization
- KIS R-score
- Q&A R-score
- TRAKE R-score
- Final Score

## Integration
- tiny fake dataset 2 videos × 5 frames
- encode/search mocked vectors
- map keyframe → frame
- KIS end-to-end
- TRAKE 4-event end-to-end
- Q&A with mocked VLM

---

# 28. Definition of Done cho bản baseline thi được

Hệ thống chỉ được gọi là “baseline hoàn chỉnh” khi:

- [ ] download/extract reproducible
- [ ] data report generated
- [ ] keyframe ↔ video frame mapping verified
- [ ] BTC CLIP baseline chạy
- [ ] FG-CLIP2 index chạy
- [ ] PE-Core index chạy
- [ ] RRF fusion chạy
- [ ] temporal NMS chạy
- [ ] KIS coarse + refine chạy
- [ ] Q&A retrieval + VLM answer chạy
- [ ] TRAKE event retrieval + ordered DP + refine chạy
- [ ] evaluator đúng official formulas
- [ ] top100 writer chạy
- [ ] submission validator chạy
- [ ] debug output lưu đủ source scores
- [ ] unit tests pass
- [ ] có experiment log

---

# 29. First commands sau khi Codex hoàn thành Milestone 1

```bash
python scripts/download_data.py --list
python scripts/download_data.py --dry-run
python scripts/download_data.py --only support --workers 2
python scripts/download_data.py --only keyframes --workers 2
python scripts/download_data.py --only videos --workers 2
python scripts/extract_data.py
python scripts/build_manifests.py
python scripts/analyze_data.py
```

Sau khi data audit pass:

```bash
python scripts/encode_keyframes.py --encoder fgclip2_large
python scripts/build_indexes.py --encoder fgclip2_large

python scripts/encode_keyframes.py --encoder pecore_g14_448
python scripts/build_indexes.py --encoder pecore_g14_448
```

Sau đó mới:

```bash
python scripts/run_kis.py --query "..."
python scripts/run_qna.py --query-file ...
python scripts/run_trake.py --query-file ...
```

---

# 30. Source references

## Official AIC 2026
- Attached document: `Thong tin vong So tuyen AIC2026.pdf`
- Data sheet:
  https://docs.google.com/spreadsheets/d/1rfn1fieTThS_Ki3SIoJ6uXOx2AhMq7wGCak6W4jZyZM/edit?gid=0#gid=0

## FG-CLIP2
- Model:
  https://huggingface.co/qihoo360/fg-clip2-large
- Code:
  https://github.com/360CVGroup/FG-CLIP

## Perception Encoder
- Code / official docs:
  https://github.com/facebookresearch/perception_models
  https://github.com/facebookresearch/perception_models/blob/main/apps/pe/README.md

## Qwen3-VL
- Official repo:
  https://github.com/QwenLM/Qwen3-VL

---

# 31. Instruction block để đưa thẳng cho Codex

> Implement repository theo đúng tài liệu này theo từng milestone. Không code tất cả cùng một lúc. Sau mỗi milestone:
>
> 1. liệt kê file đã tạo/sửa;
> 2. giải thích architecture ngắn gọn;
> 3. chạy tests tương ứng;
> 4. báo command để tôi tự chạy;
> 5. không đi sang milestone tiếp theo nếu milestone hiện tại chưa pass;
> 6. không hardcode path máy cá nhân;
> 7. dùng `pathlib.Path`;
> 8. type hints đầy đủ;
> 9. dataclass/Pydantic cho data contracts khi phù hợp;
> 10. tuân thủ SOLID;
> 11. model-specific code chỉ nằm trong adapter/wrapper;
> 12. tất cả parameters phải config-driven;
> 13. không silently ignore corrupt/missing data — report rõ;
> 14. không invent schema nếu chưa inspect data thật;
> 15. không assume embedding dimension — discover runtime;
> 16. ưu tiên correctness trước optimization.
