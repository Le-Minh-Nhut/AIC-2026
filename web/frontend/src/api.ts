import type { ResultPayload, Source, Task } from "./types";

const jsonHeaders = { "Content-Type": "application/json" };

export async function requestJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify(body),
  });
  const payload = (await response.json()) as T | { detail?: string };
  if (!response.ok) {
    throw new Error("detail" in payload ? payload.detail ?? "Request failed" : "Request failed");
  }
  return payload as T;
}

export function runKis(query: string, sources: Source[], refine: boolean): Promise<ResultPayload> {
  return requestJson("/api/kis/search", { query, top_k: 100, sources, refine });
}

export function runQna(
  eventDescription: string,
  question: string,
  sources: Source[],
  refine: boolean,
): Promise<ResultPayload> {
  return requestJson("/api/qna/answer", {
    event_description: eventDescription,
    question,
    sources,
    refine,
  });
}

export function runTrake(query: string, sources: Source[], refine: boolean): Promise<ResultPayload> {
  return requestJson("/api/trake/search", { query, sources, refine });
}

export function prepareSubmission(task: Task, queryId: string, result: ResultPayload) {
  return requestJson<{ submission: object; result_count: number }>("/api/submissions/prepare", {
    task,
    query_id: queryId,
    result,
  });
}

export const videoUrl = (videoId: string) => `/api/media/videos/${encodeURIComponent(videoId)}`;
export const keyframeUrl = (keyframeUid: string) =>
  `/api/media/keyframes/${encodeURIComponent(keyframeUid)}`;
