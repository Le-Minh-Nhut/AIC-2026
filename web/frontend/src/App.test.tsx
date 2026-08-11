import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

describe("competition UI", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ candidates: [], api: { latency_ms: 1 } }),
    }));
  });

  it("runs KIS from the keyboard-first Ctrl+Enter shortcut", async () => {
    render(<App />);
    const query = screen.getByLabelText("Query");
    fireEvent.change(query, { target: { value: "a runner clears a bar" } });
    fireEvent.keyDown(query, { key: "Enter", ctrlKey: true });
    expect(await screen.findByText("0 candidate(s)")).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledWith(
      "/api/kis/search",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("keeps the manual panel text-only", () => {
    render(<App />);
    expect(screen.getByText(/no capture or recording/i)).toBeInTheDocument();
  });
});
