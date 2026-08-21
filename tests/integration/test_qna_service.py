from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

run_qna = importlib.import_module("run_qna")
domain_models = importlib.import_module("domain.models")
Candidate = domain_models.Candidate
VideoRecord = domain_models.VideoRecord
qna_normalizer = importlib.import_module("qna.answer_normalizer")
AnswerNormalizer = qna_normalizer.AnswerNormalizer
qna_answerer = importlib.import_module("qna.answerer")
VLMAnswererError = qna_answerer.VLMAnswererError
qna_frame_selector = importlib.import_module("qna.frame_selector")
CandidateClipSelector = qna_frame_selector.CandidateClipSelector
ClipSelectorConfig = qna_frame_selector.ClipSelectorConfig
frame_sampler = importlib.import_module("refinement.frame_sampler")
FrameSampler = frame_sampler.FrameSampler
video_decoder = importlib.import_module("refinement.video_decoder")
DecodedFrame = video_decoder.DecodedFrame
DecodedVideoInfo = video_decoder.DecodedVideoInfo
dense_refiner = importlib.import_module("refinement.dense_frame_refiner")
RefinedFrameCandidate = dense_refiner.RefinedFrameCandidate
kis_service = importlib.import_module("tasks.kis_service")
KisCoarseResult = kis_service.KisCoarseResult
KisRefinementResult = kis_service.KisRefinementResult
qna_service = importlib.import_module("tasks.qna_service")
QnAQuery = qna_service.QnAQuery
QnaService = qna_service.QnaService
QnaServiceConfig = qna_service.QnaServiceConfig
TaskType = importlib.import_module("domain.competition").TaskType
FrameDiversityConfig = importlib.import_module("submission.ranker").FrameDiversityConfig
submission_from_debug = importlib.import_module("submission.writer").submission_from_debug


def _video() -> VideoRecord:
    return VideoRecord(
        video_id="L21_V001",
        video_path="raw/videos/L21_V001.mp4",
        group_id="L21",
        fps=10.0,
        frame_count=41,
        duration_sec=4.1,
        width=2,
        height=2,
        video_codec="fixture",
        audio_codec=None,
        audio_sample_rate=None,
        audio_channels=None,
        has_audio=False,
        container="fixture",
        file_size_bytes=1,
        is_readable=True,
    )


def _candidate(frame_id: int = 20, score: float = 0.8) -> Candidate:
    return Candidate(
        keyframe_uid="L21_V001:000001",
        video_id="L21_V001",
        original_frame_id=frame_id,
        timestamp_sec=frame_id / 10.0,
        keyframe_path="raw/keyframes/L21_V001/000001.jpg",
        score=score,
        rank=1,
        source="fgclip2",
    )


class _SyntheticDecoder:
    def inspect(self, video_path: Path) -> DecodedVideoInfo:
        return DecodedVideoInfo(fps=10.0, frame_count=41)

    def decode_frames(self, video_path: Path, frame_ids: tuple[int, ...]) -> tuple[DecodedFrame, ...]:
        return tuple(
            DecodedFrame(
                frame_id=frame_id,
                timestamp_sec=frame_id / 10.0,
                image=Image.new("RGB", (2, 2), color=(frame_id, 0, 0)),
            )
            for frame_id in frame_ids
        )


class _FakeSearcher:
    def __init__(self, result: KisCoarseResult | KisRefinementResult) -> None:
        self.result = result
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> KisCoarseResult | KisRefinementResult:
        self.calls.append((query, top_k))
        return self.result


class _FakeAnswerer:
    def __init__(self, answer: str = "Five people") -> None:
        self.answer_value = answer
        self.calls: list[tuple[tuple[int, ...], str, str]] = []

    def answer(self, images, event_description: str, question: str) -> str:
        self.calls.append((tuple(image.getpixel((0, 0))[0] for image in images), event_description, question))
        return self.answer_value


class _FailingAnswerer:
    def answer(self, images, event_description: str, question: str) -> str:
        raise VLMAnswererError("fixture VLM unavailable")


def _clip_selector() -> CandidateClipSelector:
    return CandidateClipSelector(
        decoder=_SyntheticDecoder(),
        sampler=FrameSampler(),
        video_records=(_video(),),
        data_root=Path("/fixture"),
        config=ClipSelectorConfig(window_sec=1.0, multi_frame_count=5),
    )


def _coarse_result() -> KisCoarseResult:
    candidate = _candidate()
    return KisCoarseResult(
        query="person walks through a door",
        candidates=(candidate,),
        video_candidates=(),
        initial_candidate_count=1,
        temporal_nms_enabled=True,
    )


class QnaServiceIntegrationTests(unittest.TestCase):
    def test_answer_budget_preserves_recovery_candidates_through_rank_five(self) -> None:
        candidates = tuple(
            replace(
                _candidate(frame_id=rank * 5, score=1.0 - rank * 0.1),
                keyframe_uid=f"L21_V001:{rank:06d}",
                rank=rank,
            )
            for rank in range(1, 6)
        )
        retrieval = replace(
            _coarse_result(),
            candidates=candidates,
            initial_candidate_count=len(candidates),
        )
        searcher = _FakeSearcher(retrieval)
        answerer = _FakeAnswerer()
        service = QnaService(
            searcher=searcher,
            clip_selector=_clip_selector(),
            answerer=answerer,
            answer_normalizer=AnswerNormalizer(),
            config=QnaServiceConfig(retrieval_candidate_count=20, answer_candidate_count=5),
        )

        result = service.answer(
            QnAQuery("person walks through a door", "How many people are visible?")
        )

        self.assertEqual(searcher.calls, [("person walks through a door", 20)])
        self.assertEqual(len(result.candidates), 5)
        self.assertEqual(result.candidates[0].rank, 1)
        self.assertEqual(
            [candidate.frame_id for candidate in result.candidates],
            [5, 10, 15, 20, 25],
        )
        self.assertEqual(len(answerer.calls), 5)
        submission = submission_from_debug(
            TaskType.QNA,
            "qna-recovery",
            result.as_dict(),
            frame_config=FrameDiversityConfig(
                max_results=100,
                max_per_video=20,
                temporal_window_sec=0.0,
            ),
        )
        self.assertEqual(len(submission.candidates), 5)
        self.assertEqual(submission.candidates[-1].frame_ids, (25,))

    def test_service_retrieves_event_only_samples_clip_and_normalizes_answer(self) -> None:
        searcher = _FakeSearcher(_coarse_result())
        answerer = _FakeAnswerer()
        service = QnaService(
            searcher=searcher,
            clip_selector=_clip_selector(),
            answerer=answerer,
            answer_normalizer=AnswerNormalizer(),
            config=QnaServiceConfig(retrieval_candidate_count=3, answer_candidate_count=1),
        )

        result = service.answer(QnAQuery("person walks through a door", "How many people are visible?"))

        self.assertEqual(searcher.calls, [("person walks through a door", 3)])
        self.assertEqual(answerer.calls[0][0], (10, 15, 20, 25, 30))
        self.assertEqual(result.candidates[0].video_id, "L21_V001")
        self.assertEqual(result.candidates[0].frame_id, 20)
        self.assertEqual(result.candidates[0].raw_answer, "Five people")
        self.assertEqual(result.candidates[0].normalized_answer, "5")
        self.assertIsNone(result.candidates[0].refinement_score)
        self.assertEqual(result.candidates[0].as_dict()["debug_candidate_frames"]["frame_ids"], [10, 15, 20, 25, 30])

    def test_service_uses_m5_refined_frame_and_preserves_both_scores(self) -> None:
        refined = RefinedFrameCandidate(
            source_keyframe_uid="L21_V001:000001",
            video_id="L21_V001",
            coarse_original_frame_id=20,
            coarse_timestamp_sec=2.0,
            coarse_score=0.4,
            sparse_original_frame_id=14,
            sparse_timestamp_sec=1.4,
            sparse_score=0.7,
            original_frame_id=14,
            timestamp_sec=1.4,
            score=0.9,
            rank=1,
            source="fgclip2",
            source_scores=(),
        )
        coarse = _coarse_result()
        searcher = _FakeSearcher(KisRefinementResult(coarse, (refined,), ()))
        service = QnaService(
            searcher=searcher,
            clip_selector=_clip_selector(),
            answerer=_FakeAnswerer("blue"),
            answer_normalizer=AnswerNormalizer(),
            config=QnaServiceConfig(retrieval_candidate_count=1, answer_candidate_count=1),
        )

        result = service.answer(QnAQuery("person walks through a door", "What color is the bag?"))

        self.assertEqual(result.candidates[0].frame_id, 14)
        self.assertEqual(result.candidates[0].retrieval_score, 0.4)
        self.assertEqual(result.candidates[0].refinement_score, 0.9)
        self.assertEqual(result.candidates[0].normalized_answer, "blue")

    def test_vlm_failure_is_retained_as_structured_failure(self) -> None:
        service = QnaService(
            searcher=_FakeSearcher(_coarse_result()),
            clip_selector=_clip_selector(),
            answerer=_FailingAnswerer(),
            answer_normalizer=AnswerNormalizer(),
            config=QnaServiceConfig(retrieval_candidate_count=1, answer_candidate_count=1),
        )

        result = service.answer(QnAQuery("person walks through a door", "Is the door open?"))

        self.assertEqual(result.candidates, ())
        self.assertEqual(result.failures[0].stage, "vlm_answer")
        self.assertIn("fixture VLM unavailable", result.failures[0].error)


class QnaCliIntegrationTests(unittest.TestCase):
    def test_cli_runs_with_fake_retrieval_video_and_qwen_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "qna_debug.json"
            arguments = [
                "run_qna.py",
                "--coarse-only",
                "--event-description",
                "person walks through a door",
                "--question",
                "How many people are visible?",
                "--vlm-checkpoint",
                str(Path(temporary) / "qwen"),
                "--debug-output",
                str(output_path),
            ]
            runtime = SimpleNamespace(
                service=_FakeSearcher(_coarse_result()),
                runtime_metadata={"fixture": True},
                refinement_branches={},
                frame_fusion_config=None,
            )
            with patch.object(run_qna.run_kis, "load_keyframe_records_from_parquet", return_value=()), patch.object(
                run_qna.run_kis, "build_kis_coarse_runtime", return_value=runtime
            ), patch.object(run_qna, "load_video_records_from_parquet", return_value=(_video(),)), patch.object(
                run_qna, "OpenCVVideoDecoder", return_value=_SyntheticDecoder()
            ), patch.object(
                run_qna.Qwen3VLAnswerer,
                "from_local_checkpoint",
                return_value=_FakeAnswerer(),
            ), patch.object(sys, "argv", arguments):
                exit_code = run_qna.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["selected_encoder"], "fg_pe_fusion")
            self.assertFalse(payload["metadata"]["qna"]["refinement_enabled"])
            self.assertEqual(payload["candidates"][0]["normalized_answer"], "5")
