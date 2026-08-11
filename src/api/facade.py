"""Lazy, cached adapters around existing KIS, Q&A, and TRAKE services."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from config import configured_data_root, load_yaml_config, repository_root
from data.video_repository import load_video_records_from_parquet
from qna.answer_normalizer import AnswerNormalizer
from qna.answerer import Qwen3VLAnswerer
from qna.frame_selector import CandidateClipSelector, ClipSelectorConfig
from refinement.frame_sampler import FrameSampler
from refinement.video_decoder import OpenCVVideoDecoder
from submission.writer import submission_from_debug
from tasks.kis_service import KisDenseRefinementService
from tasks.qna_service import QnAQuery, QnaService, QnaServiceConfig
from tasks.trake_service import TrakeService, TrakeServiceConfig
from trake.event_refiner import TrakeDenseEventRefiner
from trake.temporal_aligner import TemporalAligner, TemporalAlignmentConfig
from trake.video_selector import CandidateVideoSelector
from query.event_decomposer import RuleBasedEventDecomposer

from api.sources import apply_source_selection, encoder_for_sources, normalize_sources


class CompetitionServiceFacade(Protocol):
    """The small API-facing surface, convenient to fake in route tests."""

    def catalog(self) -> dict[str, object]: ...

    def search_kis(
        self, query: str, top_k: int, sources: tuple[str, ...], refine: bool
    ) -> dict[str, object]: ...

    def answer_qna(
        self,
        event_description: str,
        question: str,
        query_id: str | None,
        sources: tuple[str, ...],
        refine: bool,
    ) -> dict[str, object]: ...

    def search_trake(
        self, query: str, sources: tuple[str, ...], refine: bool
    ) -> dict[str, object]: ...

    def prepare_submission(
        self, task: str, query_id: str, result: Mapping[str, object]
    ) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class ApiSettings:
    root: Path
    data_root: Path
    keyframe_manifest: Path
    video_manifest: Path
    device: str | None
    fg_embedding_manifest: Path | None
    fg_index_dir: Path | None
    fg_model_id: str | None
    fg_revision: str | None
    fg_batch_size: int | None
    pe_embedding_manifest: Path | None
    pe_index_dir: Path | None
    pe_checkpoint: Path | None
    pe_model_config: str | None
    pe_batch_size: int | None
    vlm_checkpoint: Path | None
    vlm_device_map: str | None
    vlm_dtype: str | None
    vlm_max_new_tokens: int | None

    @classmethod
    def from_environment(cls) -> "ApiSettings":
        root = Path(os.environ.get("AIC_API_REPO_ROOT", repository_root())).resolve()
        data_config = load_yaml_config(root / "configs" / "data.yaml")
        configured_root = Path(
            os.environ.get("AIC_API_DATA_ROOT", configured_data_root(data_config))
        )
        data_root = configured_root if configured_root.is_absolute() else (root / configured_root)
        data_root = data_root.resolve()

        def path_env(name: str) -> Path | None:
            raw = os.environ.get(name)
            if not raw:
                return None
            path = Path(raw)
            return path if path.is_absolute() else (root / path)

        def int_env(name: str) -> int | None:
            raw = os.environ.get(name)
            return int(raw) if raw else None

        return cls(
            root=root,
            data_root=data_root,
            keyframe_manifest=path_env("AIC_API_KEYFRAME_MANIFEST")
            or data_root / "manifests" / "keyframes_manifest.parquet",
            video_manifest=path_env("AIC_API_VIDEO_MANIFEST")
            or data_root / "manifests" / "videos_manifest.parquet",
            device=os.environ.get("AIC_API_DEVICE"),
            fg_embedding_manifest=path_env("AIC_API_FG_EMBEDDING_MANIFEST"),
            fg_index_dir=path_env("AIC_API_FG_INDEX_DIR"),
            fg_model_id=os.environ.get("AIC_API_FG_MODEL_ID"),
            fg_revision=os.environ.get("AIC_API_FG_REVISION"),
            fg_batch_size=int_env("AIC_API_FG_BATCH_SIZE"),
            pe_embedding_manifest=path_env("AIC_API_PE_EMBEDDING_MANIFEST"),
            pe_index_dir=path_env("AIC_API_PE_INDEX_DIR"),
            pe_checkpoint=path_env("AIC_API_PE_CHECKPOINT"),
            pe_model_config=os.environ.get("AIC_API_PE_MODEL_CONFIG"),
            pe_batch_size=int_env("AIC_API_PE_BATCH_SIZE"),
            vlm_checkpoint=path_env("AIC_API_VLM_CHECKPOINT"),
            vlm_device_map=os.environ.get("AIC_API_VLM_DEVICE_MAP"),
            vlm_dtype=os.environ.get("AIC_API_VLM_DTYPE"),
            vlm_max_new_tokens=int_env("AIC_API_VLM_MAX_NEW_TOKENS"),
        )


class LocalCompetitionServiceFacade:
    """Load local checkpoints/indexes once, then delegate to existing services unchanged."""

    def __init__(self, settings: ApiSettings | None = None) -> None:
        self.settings = settings or ApiSettings.from_environment()
        self._coarse_runtimes: dict[tuple[str, ...], Any] = {}
        self._frame_refiners: dict[tuple[str, ...], Any] = {}
        self._qna_services: dict[tuple[tuple[str, ...], bool], QnaService] = {}
        self._trake_services: dict[tuple[tuple[str, ...], bool], TrakeService] = {}
        self._lock = threading.RLock()
        self._search_lock = threading.Lock()
        self._modules: tuple[Any, Any] | None = None

    def catalog(self) -> dict[str, object]:
        return {
            "sources": ["fgclip2", "pecore", "ocr", "asr", "metadata"],
            "default_sources": ["fgclip2", "pecore"],
            "media": {"manifest_backed": True},
            "runtime_cached": {
                "coarse": len(self._coarse_runtimes),
                "qna": len(self._qna_services),
                "trake": len(self._trake_services),
            },
        }

    def search_kis(
        self, query: str, top_k: int, sources: tuple[str, ...], refine: bool
    ) -> dict[str, object]:
        started = time.perf_counter()
        normalized = normalize_sources(sources)
        with self._search_lock:
            runtime = self._coarse_runtime(normalized)
            service: Any = runtime.service
            if refine:
                service = KisDenseRefinementService(
                    service, self._frame_refiner(normalized, runtime)
                )
            payload = service.search(query, top_k).as_dict()
        return self._with_runtime(payload, normalized, refine, runtime.runtime_metadata, started)

    def answer_qna(
        self,
        event_description: str,
        question: str,
        query_id: str | None,
        sources: tuple[str, ...],
        refine: bool,
    ) -> dict[str, object]:
        started = time.perf_counter()
        normalized = normalize_sources(sources)
        with self._search_lock:
            service = self._qna_service(normalized, refine)
            payload = service.answer(QnAQuery(event_description, question, query_id)).as_dict()
            runtime = self._coarse_runtime(normalized)
        return self._with_runtime(payload, normalized, refine, runtime.runtime_metadata, started)

    def search_trake(
        self, query: str, sources: tuple[str, ...], refine: bool
    ) -> dict[str, object]:
        started = time.perf_counter()
        normalized = normalize_sources(sources)
        with self._search_lock:
            service = self._trake_service(normalized, refine)
            payload = service.search(query).as_dict()
            runtime = self._coarse_runtime(normalized)
        return self._with_runtime(payload, normalized, refine, runtime.runtime_metadata, started)

    def prepare_submission(
        self, task: str, query_id: str, result: Mapping[str, object]
    ) -> dict[str, object]:
        from domain.competition import TaskType
        from submission.writer import submission_query_to_dict

        try:
            task_type = TaskType(task)
        except ValueError as error:
            raise ValueError("task must be kis, qna, or trake") from error
        query = submission_from_debug(task_type, query_id, result)
        return {
            "submission": submission_query_to_dict(query),
            "result_count": len(query.candidates),
        }

    def _coarse_runtime(self, sources: tuple[str, ...]) -> Any:
        with self._lock:
            cached = self._coarse_runtimes.get(sources)
            if cached is not None:
                return cached
            run_kis, _ = self._script_modules()
            retrieval_config, models_config = self._configs()
            selected_config = apply_source_selection(retrieval_config, sources)
            keyframes = run_kis.load_keyframe_records_from_parquet(self.settings.keyframe_manifest)
            runtime = run_kis.build_kis_coarse_runtime(
                args=self._arguments(refine=False),
                root=self.settings.root,
                data_root=self.settings.data_root,
                retrieval_config=selected_config,
                models_config=models_config,
                keyframes=keyframes,
                encoder_name=encoder_for_sources(sources),
            )
            self._coarse_runtimes[sources] = runtime
            return runtime

    def _frame_refiner(self, sources: tuple[str, ...], runtime: Any) -> Any:
        with self._lock:
            cached = self._frame_refiners.get(sources)
            if cached is not None:
                return cached
            run_kis, _ = self._script_modules()
            retrieval_config, _ = self._configs()
            refiner = run_kis.build_dense_frame_refiner(
                data_root=self.settings.data_root,
                video_manifest=self.settings.video_manifest,
                retrieval_config=retrieval_config,
                refinement_branches=runtime.refinement_branches,
                frame_fusion_config=runtime.frame_fusion_config,
            )
            self._frame_refiners[sources] = refiner
            return refiner

    def _qna_service(self, sources: tuple[str, ...], refine: bool) -> QnaService:
        key = (sources, refine)
        with self._lock:
            cached = self._qna_services.get(key)
            if cached is not None:
                return cached
            _, run_qna = self._script_modules()
            _, models_config = self._configs()
            qna_config = load_yaml_config(self.settings.root / "configs" / "qna.yaml")
            runtime = self._coarse_runtime(sources)
            searcher: Any = runtime.service
            if refine:
                searcher = KisDenseRefinementService(
                    searcher, self._frame_refiner(sources, runtime)
                )
            vlm_config = models_config["vlm"]
            if vlm_config.get("local_files_only") is not True:
                raise ValueError(
                    "configs/models.yaml must keep Qwen3-VL implicit downloads disabled"
                )
            checkpoint = self.settings.vlm_checkpoint or run_qna._optional_path(
                vlm_config.get("checkpoint"), self.settings.root
            )
            if checkpoint is None:
                raise ValueError(
                    "No local Qwen3-VL checkpoint is configured; set AIC_API_VLM_CHECKPOINT"
                )
            answerer = Qwen3VLAnswerer.from_local_checkpoint(
                checkpoint=checkpoint,
                device_map=self.settings.vlm_device_map or str(vlm_config["device_map"]),
                dtype=self.settings.vlm_dtype or str(vlm_config["dtype"]),
                max_new_tokens=(
                    self.settings.vlm_max_new_tokens or int(vlm_config["max_new_tokens"])
                ),
            )
            videos = load_video_records_from_parquet(self.settings.video_manifest)
            service = QnaService(
                searcher=searcher,
                clip_selector=CandidateClipSelector(
                    decoder=OpenCVVideoDecoder(),
                    sampler=FrameSampler(),
                    video_records=videos,
                    data_root=self.settings.data_root,
                    config=ClipSelectorConfig(
                        window_sec=float(qna_config["clip_window_sec"]),
                        multi_frame_count=int(qna_config["multi_frame_count"]),
                    ),
                ),
                answerer=answerer,
                answer_normalizer=AnswerNormalizer(),
                config=QnaServiceConfig(
                    retrieval_candidate_count=int(qna_config["candidate_count"]),
                    answer_candidate_count=int(qna_config["answer_candidate_count"]),
                ),
            )
            self._qna_services[key] = service
            return service

    def _trake_service(self, sources: tuple[str, ...], refine: bool) -> TrakeService:
        key = (sources, refine)
        with self._lock:
            cached = self._trake_services.get(key)
            if cached is not None:
                return cached
            trake_config = load_yaml_config(self.settings.root / "configs" / "trake.yaml")
            runtime = self._coarse_runtime(sources)
            temporal = TemporalAlignmentConfig(
                min_temporal_gap_sec=float(trake_config["min_temporal_gap_sec"]),
                max_temporal_gap_sec=(
                    float(trake_config["max_temporal_gap_sec"])
                    if trake_config.get("max_temporal_gap_sec") is not None
                    else None
                ),
                gap_penalty=float(trake_config["gap_penalty"]),
                k_best_sequences=int(trake_config["k_best_sequences"]),
                sequence_dedup_window_sec=float(trake_config["sequence_dedup_window_sec"]),
            )
            event_refiner = None
            if refine:
                event_refiner = TrakeDenseEventRefiner(
                    frame_refiner=self._frame_refiner(sources, runtime),
                    temporal_aligner=TemporalAligner(temporal),
                    local_frame_candidates=int(trake_config["local_frame_candidates"]),
                )
            service = TrakeService(
                event_decomposer=RuleBasedEventDecomposer(),
                coarse_searcher=runtime.service,
                video_selector=CandidateVideoSelector(),
                temporal_aligner=TemporalAligner(temporal),
                config=TrakeServiceConfig(
                    event_top_k=int(trake_config["event_top_k"]),
                    candidate_videos=int(trake_config["candidate_videos"]),
                    k_best_sequences=temporal.k_best_sequences,
                    sequences_to_refine=int(trake_config["sequences_to_refine"]),
                ),
                event_refiner=event_refiner,
            )
            self._trake_services[key] = service
            return service

    def _configs(self) -> tuple[dict[str, object], dict[str, object]]:
        return (
            load_yaml_config(self.settings.root / "configs" / "retrieval.yaml"),
            load_yaml_config(self.settings.root / "configs" / "models.yaml"),
        )

    def _script_modules(self) -> tuple[Any, Any]:
        if self._modules is not None:
            return self._modules
        scripts_path = str(self.settings.root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self._modules = (
            importlib.import_module("run_kis"),
            importlib.import_module("run_qna"),
        )
        return self._modules

    def _arguments(self, refine: bool) -> argparse.Namespace:
        return argparse.Namespace(
            data_root=self.settings.data_root,
            seed=None,
            feature_file=None,
            feature_order_manifest=None,
            keyframe_manifest=self.settings.keyframe_manifest,
            checkpoint=None,
            model_name=None,
            device=self.settings.device,
            batch_size=None,
            fg_embedding_manifest=self.settings.fg_embedding_manifest,
            fg_index_dir=self.settings.fg_index_dir,
            fg_model_id=self.settings.fg_model_id,
            fg_revision=self.settings.fg_revision,
            fg_batch_size=self.settings.fg_batch_size,
            pe_embedding_manifest=self.settings.pe_embedding_manifest,
            pe_index_dir=self.settings.pe_index_dir,
            pe_checkpoint=self.settings.pe_checkpoint,
            pe_model_config=self.settings.pe_model_config,
            pe_batch_size=self.settings.pe_batch_size,
            video_manifest=self.settings.video_manifest,
            vlm_checkpoint=self.settings.vlm_checkpoint,
            vlm_device_map=self.settings.vlm_device_map,
            vlm_dtype=self.settings.vlm_dtype,
            max_new_tokens=self.settings.vlm_max_new_tokens,
            coarse_only=not refine,
            no_temporal_nms=False,
        )

    @staticmethod
    def _with_runtime(
        payload: dict[str, object],
        sources: tuple[str, ...],
        refine: bool,
        runtime_metadata: Mapping[str, object],
        started: float,
    ) -> dict[str, object]:
        return {
            **payload,
            "api": {
                "selected_sources": list(sources),
                "refinement_enabled": refine,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "runtime": runtime_metadata,
            },
        }
