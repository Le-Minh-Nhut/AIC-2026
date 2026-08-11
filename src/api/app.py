"""HTTP routes for the local competition UI. Models load lazily on first query."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.facade import ApiSettings, CompetitionServiceFacade, LocalCompetitionServiceFacade
from api.media import ManifestMediaRepository, MediaLookupError
from api.sources import normalize_sources


class SearchRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=2_000)]
    top_k: Annotated[int, Field(ge=1, le=100)] = 100
    sources: list[str] = Field(default_factory=lambda: ["fgclip2", "pecore"])
    refine: bool = False


class QnaRequest(BaseModel):
    event_description: Annotated[str, Field(min_length=1, max_length=2_000)]
    question: Annotated[str, Field(min_length=1, max_length=2_000)]
    query_id: str | None = Field(default=None, max_length=256)
    sources: list[str] = Field(default_factory=lambda: ["fgclip2", "pecore"])
    refine: bool = True


class TrakeRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=2_000)]
    sources: list[str] = Field(default_factory=lambda: ["fgclip2", "pecore"])
    refine: bool = True


class SubmissionPrepareRequest(BaseModel):
    task: Annotated[str, Field(pattern="^(kis|qna|trake)$")]
    query_id: Annotated[str, Field(min_length=1, max_length=256)]
    result: dict[str, Any]


def create_app(
    facade: CompetitionServiceFacade | None = None,
    media: ManifestMediaRepository | None = None,
) -> FastAPI:
    """Create a dependency-injectable API app; construction never loads a model."""

    settings = ApiSettings.from_environment()
    service_facade = facade or LocalCompetitionServiceFacade(settings)
    media_repository = media or ManifestMediaRepository(
        settings.data_root, settings.keyframe_manifest, settings.video_manifest
    )
    app = FastAPI(title="AIC 2026 Competition UI API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "models_loaded": service_facade.catalog()["runtime_cached"]}

    @app.get("/api/catalog")
    def catalog() -> dict[str, object]:
        return service_facade.catalog()

    @app.post("/api/kis/search")
    def search_kis(request: SearchRequest) -> dict[str, object]:
        return _run_service(
            lambda: service_facade.search_kis(
                request.query, request.top_k, normalize_sources(request.sources), request.refine
            )
        )

    @app.post("/api/qna/answer")
    def answer_qna(request: QnaRequest) -> dict[str, object]:
        return _run_service(
            lambda: service_facade.answer_qna(
                request.event_description,
                request.question,
                request.query_id,
                normalize_sources(request.sources),
                request.refine,
            )
        )

    @app.post("/api/trake/search")
    def search_trake(request: TrakeRequest) -> dict[str, object]:
        return _run_service(
            lambda: service_facade.search_trake(
                request.query, normalize_sources(request.sources), request.refine
            )
        )

    @app.post("/api/submissions/prepare")
    def prepare_submission(request: SubmissionPrepareRequest) -> dict[str, object]:
        return _run_service(
            lambda: service_facade.prepare_submission(
                request.task, request.query_id, request.result
            )
        )

    @app.get("/api/media/videos/{video_id}")
    def video(video_id: Annotated[str, Path(min_length=1, max_length=512)]) -> FileResponse:
        return _media_response(lambda: media_repository.video_path(video_id), "video/mp4")

    @app.get("/api/media/keyframes/{keyframe_uid}")
    def keyframe(keyframe_uid: Annotated[str, Path(min_length=1, max_length=512)]) -> FileResponse:
        return _media_response(lambda: media_repository.keyframe_path(keyframe_uid), None)

    return app


def _run_service(operation: Any) -> dict[str, object]:
    try:
        return operation()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Competition service failed: {error}"
        ) from error


def _media_response(operation: Any, media_type: str | None) -> FileResponse:
    try:
        return FileResponse(operation(), media_type=media_type)
    except (FileNotFoundError, MediaLookupError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


app = create_app()
