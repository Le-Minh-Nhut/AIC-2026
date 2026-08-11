"""Dependency-injectable Qwen3-VL adapter for concise multi-frame answers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from PIL import Image


class VLMAnswererError(RuntimeError):
    pass


class VLMUnavailableError(VLMAnswererError):
    pass


def build_short_answer_prompt(event_description: str, question: str) -> str:
    if not event_description.strip() or not question.strip():
        raise VLMAnswererError("Event description and question must be non-empty")
    return (
        "Answer the question using only the chronologically ordered video frames. "
        "Return one short answer only, without explanation.\n"
        f"Event description: {event_description.strip()}\n"
        f"Question: {question.strip()}"
    )


class Qwen3VLAnswerer:
    """Local Qwen3-VL adapter; callers can inject any test double via the protocol."""

    def __init__(self, model: Any, processor: Any, max_new_tokens: int = 32) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        self._model = model
        self._processor = processor
        self._max_new_tokens = max_new_tokens

    @classmethod
    def from_local_checkpoint(
        cls,
        checkpoint: Path,
        device_map: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 32,
    ) -> "Qwen3VLAnswerer":
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Local Qwen3-VL checkpoint does not exist: {checkpoint}")
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise VLMUnavailableError(
                "Qwen3-VL requires optional transformers>=4.57 and torch dependencies"
            ) from error
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                str(checkpoint),
                dtype=dtype,
                device_map=device_map,
                local_files_only=True,
            )
            processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
        except Exception as error:
            raise VLMUnavailableError(f"Unable to load local Qwen3-VL checkpoint {checkpoint}: {error}") from error
        return cls(model=model, processor=processor, max_new_tokens=max_new_tokens)

    def answer(
        self,
        images: Sequence[Image.Image],
        event_description: str,
        question: str,
    ) -> str:
        if not images:
            raise VLMAnswererError("Qwen3-VL requires at least one video frame")
        prompt = build_short_answer_prompt(event_description, question)
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": image} for image in images),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            if hasattr(inputs, "to") and hasattr(self._model, "device"):
                inputs = inputs.to(self._model.device)
            generated_ids = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
            input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
            generated_ids_trimmed = [
                output_ids[len(input_ids_row) :]
                for input_ids_row, output_ids in zip(input_ids, generated_ids, strict=True)
            ]
            answers = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except VLMAnswererError:
            raise
        except Exception as error:
            raise VLMAnswererError(f"Qwen3-VL generation failed: {error}") from error
        if len(answers) != 1 or not isinstance(answers[0], str) or not answers[0].strip():
            raise VLMAnswererError("Qwen3-VL returned an empty or invalid answer")
        return answers[0].strip()
