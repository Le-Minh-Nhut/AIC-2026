from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Mapping

try:
    from fastapi.testclient import TestClient

    from api.app import create_app
except ModuleNotFoundError:
    TestClient = None


class _FakeFacade:
    def catalog(self) -> dict[str, object]:
        return {"runtime_cached": {"coarse": 0, "qna": 0, "trake": 0}, "sources": ["fgclip2"]}

    def search_kis(
        self, query: str, top_k: int, sources: tuple[str, ...], refine: bool
    ) -> dict[str, object]:
        if query != "person jumps" or top_k != 3 or sources != ("fgclip2", "ocr"):
            raise AssertionError("Unexpected KIS request")
        return {"query": query, "candidates": [], "api": {"selected_sources": list(sources)}}

    def answer_qna(
        self,
        event_description: str,
        question: str,
        query_id: str | None,
        sources: tuple[str, ...],
        refine: bool,
    ) -> dict[str, object]:
        return {
            "query": {"event_description": event_description, "question": question},
            "candidates": [],
        }

    def search_trake(self, query: str, sources: tuple[str, ...], refine: bool) -> dict[str, object]:
        return {"query": query, "candidates": []}

    def prepare_submission(
        self, task: str, query_id: str, result: Mapping[str, object]
    ) -> dict[str, object]:
        return {
            "submission": {"query_id": query_id, "task": task, "results": []},
            "result_count": 0,
        }


class _FakeMedia:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def video_path(self, video_id: str) -> Path:
        if video_id != "v1":
            raise ValueError("unknown video")
        return self.file_path

    def keyframe_path(self, keyframe_uid: str) -> Path:
        if keyframe_uid != "k1":
            raise ValueError("unknown keyframe")
        return self.file_path


@unittest.skipIf(TestClient is None, "FastAPI web optional dependencies are not installed")
class ApiRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.media_file = Path(self.temporary_directory.name) / "media.mp4"
        self.media_file.write_bytes(b"synthetic-media")
        self.client = TestClient(create_app(_FakeFacade(), _FakeMedia(self.media_file)))

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_health_and_kis_route_use_injected_services(self) -> None:
        self.assertEqual(self.client.get("/api/health").json()["status"], "ok")
        response = self.client.post(
            "/api/kis/search",
            json={"query": "person jumps", "top_k": 3, "sources": ["ocr", "fgclip2"]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["api"]["selected_sources"], ["fgclip2", "ocr"])

    def test_prepare_submission_and_manifest_media_routes(self) -> None:
        prepared = self.client.post(
            "/api/submissions/prepare",
            json={"task": "kis", "query_id": "q1", "result": {"candidates": []}},
        )
        self.assertEqual(prepared.status_code, 200)
        self.assertEqual(prepared.json()["submission"]["query_id"], "q1")
        self.assertEqual(self.client.get("/api/media/videos/v1").status_code, 200)
        self.assertEqual(self.client.get("/api/media/keyframes/k1").status_code, 200)

    def test_text_only_source_selection_returns_a_clear_error(self) -> None:
        response = self.client.post(
            "/api/kis/search", json={"query": "scoreboard", "sources": ["ocr"]}
        )
        self.assertEqual(response.status_code, 409)
        self.assertIn("FG-CLIP2 or PE-Core", response.json()["detail"])
