import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { keyframeUrl, prepareSubmission, runKis, runQna, runTrake, videoUrl } from "./api";
import type { Candidate, ResultPayload, Source, Task, TrakeCandidate } from "./types";

const sourceLabels: Record<Source, string> = {
  fgclip2: "FG",
  pecore: "PE",
  ocr: "OCR",
  asr: "ASR",
  metadata: "Meta",
};

const defaultSources: Source[] = ["fgclip2", "pecore"];
const historyStorageKey = "aic2026-final-kis-history";

interface HistoryItem {
  query: string;
  candidateCount: number;
  topVideo?: string;
  ranking: Array<{ videoId: string; frameId: number; score: number }>;
}

function isTrakeCandidate(candidate: Candidate | TrakeCandidate): candidate is TrakeCandidate {
  return "ordered_frame_ids" in candidate;
}

function frameId(candidate: Candidate) {
  return candidate.frame_id ?? candidate.original_frame_id ?? 0;
}

function score(candidate: Candidate) {
  return candidate.refinement_score ?? candidate.score ?? candidate.retrieval_score ?? 0;
}

function formatTime(value?: number) {
  if (value === undefined || !Number.isFinite(value)) return "--:--";
  const minutes = Math.floor(value / 60);
  const seconds = value - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${seconds.toFixed(2).padStart(5, "0")}`;
}

export default function App() {
  const [task, setTask] = useState<Task>("kis");
  const [query, setQuery] = useState("");
  const [question, setQuestion] = useState("");
  const [sources, setSources] = useState<Source[]>(defaultSources);
  const [refine, setRefine] = useState(false);
  const [finalMode, setFinalMode] = useState(false);
  const [manualDescription, setManualDescription] = useState("");
  const [result, setResult] = useState<ResultPayload | null>(null);
  const [selected, setSelected] = useState<Candidate | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [copyStatus, setCopyStatus] = useState("");
  const [queryId, setQueryId] = useState("manual-query");
  const [history, setHistory] = useState<HistoryItem[]>(() => loadHistory());
  const queryBox = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    window.sessionStorage.setItem(historyStorageKey, JSON.stringify(history.slice(0, 20)));
  }, [history]);

  useEffect(() => {
    if (task === "kis") setRefine(false);
    if (task !== "kis") setFinalMode(false);
  }, [task]);

  const candidates = useMemo(
    () => (result?.candidates ?? []).filter((item): item is Candidate => !isTrakeCandidate(item)),
    [result],
  );
  const trakeCandidates = useMemo(
    () => (result?.candidates ?? []).filter((item): item is TrakeCandidate => isTrakeCandidate(item)),
    [result],
  );

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    const trimmedQuery = query.trim();
    if (!trimmedQuery || (task === "qna" && !question.trim())) {
      setError(task === "qna" ? "Enter event description and question." : "Enter a query.");
      return;
    }
    setLoading(true);
    setError("");
    setCopyStatus("");
    setSelected(null);
    try {
      const next =
        task === "kis"
          ? await runKis(trimmedQuery, sources, refine)
          : task === "qna"
            ? await runQna(trimmedQuery, question.trim(), sources, refine)
            : await runTrake(trimmedQuery, sources, refine);
      setResult(next);
      const first = next.candidates[0];
      if (first && !isTrakeCandidate(first)) setSelected(first);
      if (task === "kis" && finalMode) {
        const ranking = next.candidates
          .filter((candidate): candidate is Candidate => !isTrakeCandidate(candidate))
          .slice(0, 10)
          .map((candidate) => ({
            videoId: candidate.video_id,
            frameId: frameId(candidate),
            score: score(candidate),
          }));
        setHistory((items) => [
          {
            query: trimmedQuery,
            candidateCount: next.candidates.length,
            topVideo: first?.video_id,
            ranking,
          },
          ...items.filter((item) => item.query !== trimmedQuery),
        ]);
      }
    } catch (requestError) {
      setResult(null);
      setError(requestError instanceof Error ? requestError.message : "Search failed.");
    } finally {
      setLoading(false);
    }
  }

  function onQueryKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.ctrlKey && event.key === "Enter") {
      event.preventDefault();
      void submit();
    }
  }

  function toggleSource(source: Source) {
    setSources((current) => {
      if (!current.includes(source)) return [...current, source];
      const remaining = current.filter((value) => value !== source);
      if (
        ["fgclip2", "pecore"].includes(source) &&
        !remaining.some((value) => value === "fgclip2" || value === "pecore")
      ) {
        return current;
      }
      return remaining;
    });
  }

  function useManualDescription() {
    const description = manualDescription.trim();
    if (!description) return;
    setQuery(description);
    queryBox.current?.focus();
  }

  async function copySubmission() {
    if (!result) return;
    try {
      const prepared = await prepareSubmission(task, queryId.trim() || "manual-query", result);
      await navigator.clipboard.writeText(JSON.stringify(prepared.submission, null, 2));
      setCopyStatus(`Copied ${prepared.result_count} prepared result(s).`);
    } catch (copyError) {
      setCopyStatus(copyError instanceof Error ? copyError.message : "Could not prepare submission.");
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">AIC 2026 · LOCAL COMPETITION DESK</p>
          <h1>Retrieve. Inspect. Submit.</h1>
        </div>
        <div className="shortcut-hint">Ctrl + Enter <span>search</span></div>
      </header>

      <nav className="task-tabs" aria-label="Competition task">
        {(["kis", "qna", "trake"] as Task[]).map((value) => (
          <button
            className={task === value ? "tab active" : "tab"}
            key={value}
            onClick={() => setTask(value)}
            type="button"
          >
            {value.toUpperCase()}
          </button>
        ))}
      </nav>

      <section className="query-panel">
        <form onSubmit={submit}>
          <div className="field-heading">
            <label htmlFor="query">{task === "qna" ? "Event description" : "Query"}</label>
            <span>large input · keyboard first</span>
          </div>
          <textarea
            autoFocus
            id="query"
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={onQueryKeyDown}
            placeholder={
              task === "trake"
                ? "run-up → take-off → clear the bar → land"
                : "Describe the event, object, action, or clue…"
            }
            ref={queryBox}
            rows={3}
            value={query}
          />
          {task === "qna" && (
            <textarea
              aria-label="Question"
              className="question-box"
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={onQueryKeyDown}
              placeholder="Question for Qwen3-VL…"
              rows={2}
              value={question}
            />
          )}
          <div className="query-controls">
            <div className="source-toggles" aria-label="Retrieval sources">
              {(Object.keys(sourceLabels) as Source[]).map((source) => (
                <button
                  aria-pressed={sources.includes(source)}
                  className={sources.includes(source) ? "source-toggle on" : "source-toggle"}
                  key={source}
                  onClick={() => toggleSource(source)}
                  type="button"
                >
                  {sourceLabels[source]}
                </button>
              ))}
            </div>
            <label className="check-control">
              <input checked={refine} onChange={(event) => setRefine(event.target.checked)} type="checkbox" />
              Dense refine
            </label>
            <button className="search-button" disabled={loading} type="submit">
              {loading ? "Searching…" : "Search"}
            </button>
          </div>
        </form>
      </section>

      {task === "kis" && (
        <section className="workflow-row">
          <label className="final-mode">
            <input checked={finalMode} onChange={(event) => setFinalMode(event.target.checked)} type="checkbox" />
            <span>
              Final Mode
              <small>Keep textual KIS clue and ranking history in this browser session.</small>
            </span>
          </label>
          <div className="manual-description">
            <label htmlFor="manual-description">Video KIS manual description</label>
            <div>
              <input
                id="manual-description"
                onChange={(event) => setManualDescription(event.target.value)}
                placeholder="Type a visual clue from the video; no capture or recording."
                value={manualDescription}
              />
              <button onClick={useManualDescription} type="button">Use clue</button>
            </div>
          </div>
        </section>
      )}

      {error && <div className="notice error" role="alert">{error}</div>}
      {copyStatus && <div className="notice">{copyStatus}</div>}

      {task === "trake" ? (
        <TrakeResults candidates={trakeCandidates} events={result?.events ?? []} />
      ) : (
        <section className="result-layout">
          <CandidateGrid candidates={candidates} onSelect={setSelected} selected={selected} />
          <Inspector candidate={selected} task={task} />
        </section>
      )}

      {result && (
        <footer className="result-footer">
          <span>{result.candidates.length} candidate(s)</span>
          <span>{result.api?.latency_ms?.toFixed(0) ?? "--"} ms</span>
          <label>
            Submission ID
            <input onChange={(event) => setQueryId(event.target.value)} value={queryId} />
          </label>
          <button className="prepare-button" onClick={() => void copySubmission()} type="button">
            Prepare & copy submission
          </button>
        </footer>
      )}

      {finalMode && task === "kis" && (
        <aside className="history-panel">
          <div><strong>Final Mode history</strong><button onClick={() => setHistory([])} type="button">Clear</button></div>
          {history.length === 0 ? <p>No prior clues this session.</p> : history.map((item) => (
            <button className="history-item" key={item.query} onClick={() => setQuery(item.query)} type="button">
              <span>{item.query}</span>
              <small>{item.candidateCount} · {item.topVideo ?? "--"}</small>
              <em>{item.ranking.slice(0, 3).map((entry) => `${entry.videoId}/F${entry.frameId}=${entry.score.toFixed(3)}`).join(" · ")}</em>
            </button>
          ))}
        </aside>
      )}
    </main>
  );
}

function CandidateGrid({
  candidates,
  selected,
  onSelect,
}: {
  candidates: Candidate[];
  selected: Candidate | null;
  onSelect: (candidate: Candidate) => void;
}) {
  return (
    <section className="candidate-section">
      <div className="section-heading"><h2>Top-K keyframes</h2><span>Click to inspect video and source evidence.</span></div>
      {candidates.length === 0 ? (
        <div className="empty-state">Run a search to populate the keyframe grid.</div>
      ) : (
        <div className="candidate-grid">
          {candidates.map((candidate) => {
            const keyframe = candidate.keyframe_uid ?? candidate.source_keyframe_uid;
            return (
              <button
                className={selected?.video_id === candidate.video_id && frameId(selected) === frameId(candidate) ? "candidate-card selected" : "candidate-card"}
                key={`${candidate.rank}-${candidate.video_id}-${frameId(candidate)}`}
                onClick={() => onSelect(candidate)}
                type="button"
              >
                {keyframe ? <img alt="" loading="lazy" src={keyframeUrl(keyframe)} /> : <div className="thumbnail-placeholder">frame</div>}
                <span className="rank">#{candidate.rank}</span>
                <strong>{candidate.video_id}</strong>
                <small>F {frameId(candidate)} · {formatTime(candidate.timestamp_sec)}</small>
                <small className="score">{score(candidate).toFixed(4)}</small>
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}

function Inspector({ candidate, task }: { candidate: Candidate | null; task: Task }) {
  const video = useRef<HTMLVideoElement>(null);
  const center = candidate?.timestamp_sec ?? 0;
  const [scrubTime, setScrubTime] = useState(center);

  useEffect(() => setScrubTime(center), [center]);
  if (!candidate) return <aside className="inspector empty-state">Select a candidate to open its video timestamp.</aside>;

  const min = Math.max(0, center - 3);
  const max = Math.max(min + 0.2, center + 3);
  const clip = candidate.debug_candidate_frames;
  function scrub(value: number) {
    setScrubTime(value);
    if (video.current) video.current.currentTime = value;
  }

  return (
    <aside className="inspector">
      <div className="inspector-title"><h2>{candidate.video_id}</h2><span>frame {frameId(candidate)}</span></div>
      <video controls onLoadedMetadata={() => scrub(center)} ref={video} src={`${videoUrl(candidate.video_id)}#t=${center}`} />
      <label className="scrubber-label">Frame scrubber <strong>{formatTime(scrubTime)}</strong></label>
      <input aria-label="Frame scrubber" className="scrubber" max={max} min={min} onChange={(event) => scrub(Number(event.target.value))} step="0.04" type="range" value={scrubTime} />
      <div className="candidate-meta">
        <span>retrieval {candidate.retrieval_score?.toFixed(4) ?? "--"}</span>
        <span>refine {candidate.refinement_score?.toFixed(4) ?? "--"}</span>
        <span>final {score(candidate).toFixed(4)}</span>
      </div>
      <SourceEvidence candidate={candidate} />
      {task === "qna" && (
        <div className="qna-answer">
          <h3>Q&A answer</h3>
          <p><small>raw</small>{candidate.raw_answer ?? "--"}</p>
          <p><small>normalized</small><strong>{candidate.normalized_answer ?? "--"}</strong></p>
          {clip && (
            <div className="clip-frames">
              <span>Chronological clip</span>
              {clip.timestamps_sec.map((timestamp, index) => (
                <button key={`${clip.frame_ids[index]}-${timestamp}`} onClick={() => scrub(timestamp)} type="button">
                  F {clip.frame_ids[index]} · {formatTime(timestamp)}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </aside>
  );
}

function SourceEvidence({ candidate }: { candidate: Candidate }) {
  const scores = candidate.source_scores ?? [];
  return (
    <section className="source-evidence">
      <h3>Source ranks & RRF</h3>
      {scores.length === 0 ? <p>No per-source evidence returned.</p> : scores.map((source) => (
        <div className="source-row" key={`${source.source}-${source.rank}`}>
          <strong>{sourceLabels[source.source as Source] ?? source.source}</strong>
          <span>r{source.rank}</span><span>s {source.score.toFixed(4)}</span>
          <span>rrf {source.rrf_contribution?.toFixed(5) ?? "--"}</span>
          {source.evidence_text && <small title={source.evidence_text}>{source.evidence_text}</small>}
        </div>
      ))}
    </section>
  );
}

function TrakeResults({ candidates, events }: { candidates: TrakeCandidate[]; events: { index?: number; text?: string; retrieval_text?: string }[] }) {
  return (
    <section className="trake-results">
      <div className="section-heading"><h2>TRAKE timeline</h2><span>Ordered DP sequences; frame order is preserved.</span></div>
      {candidates.length === 0 ? <div className="empty-state">Run an ordered event query to see aligned sequences.</div> : candidates.map((candidate) => (
        <article className="trake-card" key={`${candidate.rank}-${candidate.video_id}-${candidate.ordered_frame_ids.join("-")}`}>
          <header><strong>#{candidate.rank} · {candidate.video_id}</strong><span>{candidate.total_alignment_score.toFixed(4)}</span></header>
          <div className="timeline">
            {candidate.ordered_frame_ids.map((frame, index) => (
              <div className="timeline-event" key={`${frame}-${index}`}>
                <small>{events[index]?.text ?? candidate.events[index]?.event?.text ?? `event ${index + 1}`}</small>
                <strong>F {frame}</strong>
                <span>{candidate.event_scores[index]?.toFixed(4) ?? "--"}</span>
              </div>
            ))}
          </div>
        </article>
      ))}
    </section>
  );
}

function loadHistory(): HistoryItem[] {
  try {
    const raw = window.sessionStorage.getItem(historyStorageKey);
    const value: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(value) ? value.filter(isHistoryItem) : [];
  } catch {
    return [];
  }
}

function isHistoryItem(value: unknown): value is HistoryItem {
  return Boolean(
    value && typeof value === "object" && "query" in value && "candidateCount" in value && "ranking" in value,
  );
}
