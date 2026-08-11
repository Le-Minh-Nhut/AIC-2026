# AIC 2026 Interactive Competition UI

`web/frontend` is a local React + TypeScript + Vite client. `src/api` is a
FastAPI facade that lazily constructs and calls the existing M2--M8 KIS, Q&A,
and TRAKE services. It does not implement a second encoder, index, retriever,
RRF, dense-refinement, VLM, or temporal-alignment path.

## Start on the PC

Install the optional API dependencies and frontend dependencies once:

```bash
python -m pip install -e ".[web]"
cd web/frontend && npm install && cd ../..
DEVICE=cuda PE_CHECKPOINT=/models/PE-Core-G14-448.pt \
  VLM_CHECKPOINT=/models/Qwen3-VL-8B-Instruct bash bash/run_web.sh
```

The default command starts the API at `http://127.0.0.1:8000`. In a second
terminal, run `cd web/frontend && npm run dev` and open Vite's local URL.
Alternatively use `WEB_MODE=full bash bash/run_web.sh`; it starts Vite and the
API together. The browser never receives a model checkpoint, feature file, or
data path; it only receives result JSON and manifest-backed media URLs.

The API requires the same local FG/PE indexes/checkpoints as the corresponding
existing CLI command. Q&A additionally requires `VLM_CHECKPOINT`. No request
triggers a model or dataset download.

## Routes

| Route | Purpose |
| --- | --- |
| `GET /api/health` | API process and cache counts; no model load. |
| `GET /api/catalog` | Supported source names and cache status; no local paths. |
| `POST /api/kis/search` | Existing KIS coarse search and optional M5 refinement. |
| `POST /api/qna/answer` | Existing M7 retrieval, chronological clip selection, VLM, and normalizer. |
| `POST /api/trake/search` | Existing M6 decomposition, retrieval, DP alignment, and optional refinement. |
| `POST /api/submissions/prepare` | Existing M9 diversity/validation conversion, returned for clipboard preview. |
| `GET /api/media/videos/{video_id}` | Video resolved only through `videos_manifest.parquet`. |
| `GET /api/media/keyframes/{keyframe_uid}` | Keyframe resolved only through `keyframes_manifest.parquet`. |

`sources` supports `fgclip2`, `pecore`, `ocr`, `asr`, and `metadata`. At least
one visual source is required because the current M2--M8 runtime has no
text-only KIS mode. Auxiliary sources are enabled only for that request's
cached runtime and preserve their existing configured RRF weights.

## UI workflow

- **KIS:** large query box, `Ctrl+Enter`, compact Top-100 keyframe grid,
  manifest-served video playback at the candidate timestamp, a ±3-second frame
  scrubber, and per-source rank, score, evidence, and RRF contribution.
- **Final Mode:** stores recent textual KIS clues plus the top-10 ranking,
  candidate count, and top video in `sessionStorage`, preserving them while the
  browser tab stays open. It does not alter the retrieval algorithm.
- **Video KIS:** manual typed visual descriptions can populate the KIS query.
  The UI deliberately has no camera, screen capture, recording, or upload
  feature.
- **Q&A:** exposes chronological clip-sample seek controls in the video player,
  plus raw and normalized VLM answers.
- **TRAKE:** shows each returned DP sequence as an ordered event/frame timeline.
- **Submission:** prepares the current response with the existing M9 writer,
  then copies its JSON entry. Persist/append the entry with the normal
  `scripts/build_submission.py` flow before official validation.

## Backend environment

`bash/run_web.sh` maps common variables to the backend. Useful overrides are
`DATA_ROOT`, `DEVICE`, `PE_CHECKPOINT`, `VLM_CHECKPOINT`, `FG_MODEL_ID`,
`AIC_API_FG_EMBEDDING_MANIFEST`, `AIC_API_FG_INDEX_DIR`,
`AIC_API_PE_EMBEDDING_MANIFEST`, `AIC_API_PE_INDEX_DIR`, `WEB_HOST`, and
`WEB_PORT`. These are local server settings, never frontend configuration.
