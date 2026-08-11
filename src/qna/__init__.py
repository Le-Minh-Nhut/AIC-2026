"""Multi-frame visual question answering primitives and orchestration."""

from qna.answer_normalizer import AnswerNormalizer, normalize_answer
from qna.answerer import Qwen3VLAnswerer, VLMAnswererError, VLMUnavailableError
from qna.frame_selector import CandidateClipSelector, ClipSelectorConfig, SampledClip

__all__ = [
    "AnswerNormalizer",
    "CandidateClipSelector",
    "ClipSelectorConfig",
    "Qwen3VLAnswerer",
    "SampledClip",
    "VLMAnswererError",
    "VLMUnavailableError",
    "normalize_answer",
]
