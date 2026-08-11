# AIC 2026 — DATA ANALYSIS SPEC

> Đây là specification cho `scripts/analyze_data.py` và file report cuối cùng `docs/DATA_ANALYSIS.md`.
>
> Mục tiêu: trước khi train/index model, phải biết chính xác dataset có gì, thiếu gì, mapping có đúng không và các feature BTC có khớp keyframe hay không.

---

# 1. Source snapshot

Spreadsheet:
`https://docs.google.com/spreadsheets/d/1rfn1fieTThS_Ki3SIoJ6uXOx2AhMq7wGCak6W4jZyZM/edit?gid=0#gid=0`

Snapshot hiện tại:
- 32 archives
- 14 keyframe archives
- 14 video archives
- 4 support archives

Support:
- `clip-features-32-aic25-b1.zip`
- `map-keyframes-aic25-b1.zip`
- `media-info-aic25-b1.zip`
- `objects-aic25-b1.zip`

Official task document nói dữ liệu thi chính thức là video; keyframes, objects, CLIP features và metadata là dữ liệu hỗ trợ.

---

# 2. Report contract

`python scripts/analyze_data.py` phải tạo:

```text
docs/DATA_ANALYSIS.md
data/reports/data_analysis.json
data/reports/data_analysis_tables/
```

Report phải:
- deterministic;
- có timestamp;
- ghi dataset snapshot hash;
- không hardcode số liệu;
- mọi con số phải tính từ data thật;
- nếu field/schema không tồn tại thì ghi `NOT PRESENT`, không tự invent;
- nếu một check không chạy được thì ghi lý do.

---

# 3. Section A — Archive inventory

Cho mỗi archive:

```text
filename
category
download_url
downloaded
archive_path
file_size_bytes
sha256_local
zip_valid
entry_count
compressed_size
estimated_uncompressed_size từ ZIP metadata nếu có
extracted
```

Report:
- tổng số archive expected từ snapshot;
- số downloaded;
- số missing;
- số corrupt;
- tổng compressed bytes;
- tổng uncompressed bytes theo ZIP metadata;
- archive nào có path layout khác thường.

Audit:
- duplicate filenames;
- duplicate URLs;
- empty archives;
- suspicious HTML/error file masquerading as ZIP.

---

# 4. Section B — Extracted file inventory

Đếm:
- `.mp4`
- `.jpg/.jpeg/.png`
- `.npy`
- `.json`
- các extension khác.

Report:
- total files;
- total bytes;
- top 20 largest files;
- unexpected extensions;
- empty files;
- duplicate relative paths.

---

# 5. Section C — Video analysis

Cho mỗi video, probe bằng `ffprobe` hoặc PyAV/OpenCV fallback.

Fields:

```text
video_id
group_id
path
container
video_codec
audio_codec
width
height
fps
frame_count
duration_sec
bitrate
file_size_bytes
has_audio
audio_sample_rate
audio_channels
is_readable
probe_error
```

## 5.1. Summary

- total videos;
- videos/group L21...L30;
- readable vs unreadable;
- total duration;
- mean / median / p90 / p95 / max duration;
- fps distribution;
- resolution distribution;
- codec distribution;
- audio availability;
- audio codec distribution;
- sample rate distribution.

## 5.2. Consistency

Check:

\[
duration \approx frame\_count / fps
\]

Flag video nếu sai khác lớn hơn tolerance.

Check:
- frame count <= 0;
- fps <= 0;
- duration <= 0;
- width/height invalid;
- duplicate video ID;
- same video content hash optional.

---

# 6. Section D — Keyframe analysis

Cho mỗi keyframe:

```text
keyframe_uid
video_id
keyframe_index
path
width
height
mode
file_size_bytes
is_readable
original_frame_id
timestamp_sec
```

## 6.1. Counts

- total keyframes;
- keyframes/video;
- min;
- median;
- mean;
- p90/p95;
- max.

## 6.2. Sampling density

Nếu mapping + fps tồn tại:

```text
delta_frame = frame_id[i+1] - frame_id[i]
delta_sec = delta_frame / fps
```

Report:
- median interval per video;
- global distribution;
- p10/p50/p90/p95/max;
- videos sampled unusually sparse/dense.

Mục tiêu:
biết coarse keyframe retrieval cách video gốc bao xa.

## 6.3. Image properties

- resolution distribution;
- corrupt images;
- grayscale vs RGB nếu có;
- zero-byte;
- duplicate image hash optional.

---

# 7. Section E — Keyframe ↔ original video mapping

Đây là section critical.

Cho mỗi video:
- số keyframe files;
- số mapping rows;
- số matched;
- số missing mapping;
- số mapping không có image;
- số mapping trỏ video không tồn tại.

Invariant:

```text
0 <= frame_id < frame_count
```

Order:
- keyframe index increasing;
- original frame id non-decreasing.

Flag:
- negative frame;
- frame beyond video;
- duplicate keyframe index;
- duplicate frame mapping;
- non-monotonic frame IDs;
- malformed video IDs;
- video mismatch.

Nếu FPS valid:
\[
timestamp = frame\_id / fps
\]

Check timestamp <= duration + tolerance.

Summary table:
```text
video_id | keyframes | mappings | matched | missing | invalid | monotonic
```

---

# 8. Section F — BTC CLIP feature analysis

Không assume feature dimension.

Cho mỗi `.npy`:
- shape;
- dtype;
- byte size;
- NaN count;
- Inf count;
- zero-vector count;
- norm statistics;
- row count.

Critical check:
- row count phải map được với expected keyframes theo official ordering/mapping.
- nếu mapping chưa đủ rõ, report `UNRESOLVED` thay vì đoán.

Compute norm:
\[
\|x_i\|_2
\]

Report:
- min/mean/median/max norm;
- đã normalized gần 1 chưa;
- duplicate vectors optional;
- NaN/Inf.

Nếu feature order map được:
- random sample N rows;
- show `row → keyframe_uid`;
- verify reproducible.

---

# 9. Section G — Objects JSON analysis

Đầu tiên discover schema.

Report schema examples:
- top-level type;
- keys;
- nested keys;
- bbox representation;
- confidence field;
- class/label field.

Không hardcode Faster R-CNN schema trước khi inspect.

Sau khi adapter resolve schema:

Statistics:
- JSON files count;
- matched keyframes;
- missing JSON;
- corrupt JSON;
- detections/image;
- no-object image percentage;
- class count;
- top object classes;
- confidence distribution;
- bbox invalid count;
- bbox out-of-bounds count;
- duplicate detections optional.

Cross-check:
- object JSON filename ↔ keyframe filename.

---

# 10. Section H — Media metadata analysis

Discover actual JSON schema.

Report:
- metadata files count;
- videos with metadata;
- videos without metadata;
- coverage percentage;
- schema variants.

Cho từng field có thật:
- presence rate;
- type consistency;
- null/empty rate.

Potential textual fields:
- title;
- description;
- channel;
- tags;
- publish info.

**Chỉ report field nào thực sự tồn tại.**

---

# 11. Section I — Audio readiness

Từ video probe:

- videos with audio;
- no-audio videos;
- audio duration;
- sample rate;
- channels;
- codec.

Estimate ASR workload:
- total audio hours;
- duration/group.

Không cần chạy ASR trong `analyze_data.py`.

---

# 12. Section J — OCR readiness

Không cần OCR toàn dataset ở analysis phase.

Có thể sample deterministic:
- N keyframes/group;
- chỉ report image resolution và OCR feasibility sample nếu OCR dependency đã cài.

Nếu OCR chưa cài:
`SKIPPED — optional dependency`.

Không làm report phụ thuộc OCR để pass.

---

# 13. Section K — Storage analysis

Report:

```text
archives bytes
videos bytes
keyframes bytes
clip feature bytes
objects bytes
metadata bytes
processed bytes
free disk bytes
```

Estimate additional embedding storage chỉ sau khi runtime embedding dimension biết.

Formula:

\[
bytes \approx N_{keyframes} \times d \times bytes\_per\_value
\]

Report FP32 / FP16 estimate cho:
- FG-CLIP2;
- PE-Core.

Không hardcode `d` nếu encoder chưa load; nếu biết từ generated embedding manifest thì dùng số thật.

---

# 14. Section L — Index readiness

Checklist:

- keyframe IDs unique?
- mapping complete?
- image readable?
- embeddings available?
- embedding row order verified?
- normalized?
- metadata store buildable?

Output:

```text
BTC_CLIP_READY = true/false
FGCLIP2_ENCODING_READY = true/false
PECORE_ENCODING_READY = true/false
VIDEO_REFINEMENT_READY = true/false
OCR_READY = true/false
ASR_READY = true/false
```

Mỗi false phải có reason.

---

# 15. Section M — Cross-modal consistency

Audit các tập ID:

```text
V = video IDs
K = video IDs inferred from keyframes
M = video IDs from keyframe mapping
O = video IDs from objects
I = video IDs from media info
```

Report set differences:

```text
K - V
V - K
M - V
V - M
O - K
K - O
I - V
V - I
```

Đây là cách bắt data missing rất nhanh.

---

# 16. Section N — Corruption sampling

Deterministic random seed.

Sample:
- 20 videos/group;
- 100 keyframes/group hoặc configurable.

Video:
- decode first/middle/last frames.

Image:
- PIL load + RGB convert.

Nếu dataset nhỏ hơn sample target thì dùng toàn bộ.

Report all failures.

---

# 17. Section O — Temporal refinement feasibility

Từ keyframe interval distribution và FPS:

For each keyframe:
- distance to next keyframe in frames;
- interval seconds.

Report:
- nếu median keyframe gap lớn, dense refinement càng quan trọng;
- không kết luận accuracy, chỉ report geometry.

Compute approximate number of frames trong:
- ±1 sec;
- ±2 sec;
- ±3 sec

theo FPS distribution để ước lượng refinement compute.

---

# 18. Section P — Query task implications

Đây là phần phân tích, nhưng chỉ dựa trên data stats thật.

## KIS
Report:
- keyframe density;
- video availability;
- mapping coverage;
- implication cho coarse→dense refine.

## Q&A
Report:
- audio coverage;
- metadata coverage;
- candidate frames;
- whether multi-frame VLM feasible từ duration/keyframe stats.

## TRAKE
Report:
- keyframe gaps;
- mapping completeness;
- FPS distribution;
- reason dense original-video refinement is necessary.

Không claim model performance khi chưa evaluate.

---

# 19. Section Q — Data quality severity

Mỗi issue có severity:

### BLOCKER
- video archive corrupt;
- mapping fundamentally unresolved;
- keyframe/feature ordering impossible to establish.

### HIGH
- many corrupt videos;
- feature row mismatch;
- mapping frame out of bounds.

### MEDIUM
- missing objects;
- missing metadata;
- sparse keyframes.

### LOW
- cosmetic schema inconsistency;
- optional metadata field missing.

Report:

```text
severity | issue | count | affected IDs | recommended action
```

---

# 20. Section R — Machine-readable JSON schema

`data_analysis.json` nên có:

```json
{
  "dataset": {
    "snapshot_hash": "...",
    "generated_at": "...",
    "archives": {}
  },
  "videos": {
    "count": 0,
    "duration_hours": 0.0,
    "fps_stats": {},
    "resolution_counts": {}
  },
  "keyframes": {
    "count": 0,
    "per_video_stats": {},
    "interval_sec_stats": {}
  },
  "mapping": {
    "coverage": 0.0,
    "invalid_count": 0,
    "non_monotonic_count": 0
  },
  "btc_clip": {
    "files": [],
    "ready": false
  },
  "objects": {},
  "media_info": {},
  "storage": {},
  "readiness": {},
  "issues": []
}
```

Không bắt buộc đúng y nguyên nếu schema thật cần mở rộng, nhưng phải stable/versioned.

---

# 21. `docs/DATA_ANALYSIS.md` template

Codex phải generate format:

```markdown
# AIC 2026 Data Analysis

## Executive Summary
- Dataset snapshot:
- Archives:
- Videos:
- Keyframes:
- Mapping coverage:
- BTC CLIP status:
- Blocking issues:

## 1. Archives
...

## 2. Videos
...

## 3. Keyframes
...

## 4. Keyframe Mapping
...

## 5. BTC CLIP Features
...

## 6. Objects
...

## 7. Media Info
...

## 8. Audio
...

## 9. Storage
...

## 10. Cross-modal Consistency
...

## 11. Corruption Checks
...

## 12. Retrieval Readiness
...

## 13. Task Implications
### KIS
### Q&A
### TRAKE

## 14. Issues
...

## 15. Recommended Next Actions
...
```

---

# 22. Acceptance criteria

`analyze_data.py` chỉ pass khi:

- [ ] không crash vì missing optional component;
- [ ] corrupt file được report thay vì swallowed;
- [ ] video probe có fallback;
- [ ] mapping audit hoàn chỉnh;
- [ ] CLIP shape/dtype/norm audit;
- [ ] object schema discovery;
- [ ] media schema discovery;
- [ ] ID set-difference audit;
- [ ] storage report;
- [ ] readiness flags;
- [ ] Markdown + JSON cùng sinh trong một run;
- [ ] deterministic sampling;
- [ ] unit test bằng fake dataset.

---

# 23. Lệnh Codex phải support

```bash
python scripts/analyze_data.py

python scripts/analyze_data.py \
  --data-root data \
  --report docs/DATA_ANALYSIS.md

python scripts/analyze_data.py \
  --sample-decode-videos 20 \
  --sample-images 100

python scripts/analyze_data.py \
  --strict
```

`--strict`:
- exit non-zero nếu có BLOCKER/HIGH issue theo policy config.

---

# 24. Không được làm

- Không tự sửa data trong analyzer.
- Không delete corrupt file.
- Không rename silently.
- Không assume missing metadata là bug — official data có thể thiếu metadata.
- Không assume `.npy` order nếu chưa xác minh.
- Không assume every video has audio.
- Không assume every archive dùng cùng internal directory structure.
- Không bỏ exception rồi tiếp tục im lặng.
- Không ghi số liệu mẫu/placeholder vào final report mà không đánh dấu.
