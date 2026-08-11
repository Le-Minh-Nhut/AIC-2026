export type Task = "kis" | "qna" | "trake";
export type Source = "fgclip2" | "pecore" | "ocr" | "asr" | "metadata";

export interface SourceScore {
  source: Source | string;
  rank: number;
  score: number;
  weight?: number | null;
  rrf_contribution?: number | null;
  evidence_id?: string | null;
  evidence_text?: string | null;
}

export interface Candidate {
  rank: number;
  video_id: string;
  frame_id?: number;
  original_frame_id?: number;
  timestamp_sec?: number;
  score?: number;
  retrieval_score?: number;
  refinement_score?: number | null;
  keyframe_uid?: string;
  source_keyframe_uid?: string;
  source?: string;
  source_scores?: SourceScore[];
  raw_answer?: string;
  normalized_answer?: string;
  debug_candidate_frames?: {
    anchor_frame_id: number;
    anchor_timestamp_sec: number;
    frame_ids: number[];
    timestamps_sec: number[];
  };
}

export interface TrakeEvent {
  event?: { index?: number; text?: string; retrieval_text?: string };
  coarse?: { frame_id?: number; timestamp_sec?: number; score?: number };
  refined?: { frame_id?: number; timestamp_sec?: number; score?: number } | null;
}

export interface TrakeCandidate {
  rank: number;
  video_id: string;
  ordered_frame_ids: number[];
  event_scores: number[];
  total_alignment_score: number;
  events: TrakeEvent[];
}

export interface ResultPayload {
  query?: string | { event_description: string; question: string; query_id?: string | null };
  candidates: Candidate[] | TrakeCandidate[];
  events?: { index?: number; text?: string; retrieval_text?: string }[];
  api?: { latency_ms?: number; selected_sources?: Source[]; refinement_enabled?: boolean };
}
