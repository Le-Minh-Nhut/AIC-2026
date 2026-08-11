"""FG-CLIP2-Large adapter with local/cache-only Hugging Face loading."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image

from encoders.base import EncoderUnavailableError
from encoders.multimodal import BatchedImageTextEncoder, ImageTextEmbeddingBackend


class FGCLIP2Backend(ImageTextEmbeddingBackend, Protocol):
    """Structural contract implemented by FG-CLIP2 model backends."""


class FGCLIP2Encoder(BatchedImageTextEncoder):
    """Batched, normalized image/text encoder independent of model-loading details."""

    def __init__(self, backend: FGCLIP2Backend, batch_size: int) -> None:
        super().__init__(backend, batch_size, encoder_label="FG-CLIP2")


class HuggingFaceFGCLIP2Backend:
    """Direct implementation of the official FG-CLIP2 retrieval calls."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        image_processor: Any,
        torch_module: Any,
        device: str,
        max_num_patches: int,
        text_max_length: int,
        text_walk_type: str,
        use_autocast: bool,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._image_processor = image_processor
        self._torch = torch_module
        self._device = device
        self._max_num_patches = max_num_patches
        self._text_max_length = text_max_length
        self._text_walk_type = text_walk_type
        self._use_autocast = use_autocast
        self._embedding_dimension: int | None = None

    @property
    def embedding_dimension(self) -> int | None:
        return self._embedding_dimension

    @property
    def preprocessing_config(self) -> dict[str, object]:
        return {
            "max_num_patches": self._max_num_patches,
            "text_max_length": self._text_max_length,
            "text_walk_type": self._text_walk_type,
            "use_autocast": self._use_autocast,
        }

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        revision: str | None,
        device: str,
        local_files_only: bool,
        max_num_patches: int,
        text_max_length: int,
        text_walk_type: str,
        use_autocast: bool,
    ) -> "HuggingFaceFGCLIP2Backend":
        if max_num_patches < 1 or text_max_length < 1:
            raise ValueError("FG-CLIP2 preprocessing limits must be positive")
        if text_walk_type not in {"short", "box", "long"}:
            raise ValueError("FG-CLIP2 text_walk_type must be short, box, or long")
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise EncoderUnavailableError(
                "FG-CLIP2 requires optional dependencies torch and transformers"
            ) from error
        load_options = {"trust_remote_code": True, "local_files_only": local_files_only}
        if revision:
            load_options["revision"] = revision
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_options).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_id, **load_options)
            image_processor = AutoImageProcessor.from_pretrained(model_id, **load_options)
        except (OSError, ValueError) as error:
            location = "local cache" if local_files_only else "configured model source"
            raise EncoderUnavailableError(
                f"Unable to load FG-CLIP2 from {location}: {model_id}. "
                "No model is downloaded by this repository unless local_files_only is disabled explicitly."
            ) from error
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            torch_module=torch,
            device=device,
            max_num_patches=max_num_patches,
            text_max_length=text_max_length,
            text_walk_type=text_walk_type,
            use_autocast=use_autocast,
        )

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        rgb_images = [image.convert("RGB") for image in images]
        inputs = self._image_processor(
            images=rgb_images,
            max_num_patches=self._max_num_patches,
            return_tensors="pt",
        )
        vectors = self._run_feature_method("get_image_features", inputs)
        return self._to_numpy(vectors)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self._tokenizer(
            list(texts),
            padding="max_length",
            max_length=self._text_max_length,
            truncation=True,
            return_tensors="pt",
        )
        vectors = self._run_feature_method(
            "get_text_features",
            inputs,
            walk_type=self._text_walk_type,
        )
        return self._to_numpy(vectors)

    def _run_feature_method(self, method_name: str, inputs: Any, **kwargs: object) -> Any:
        moved_inputs = {key: value.to(self._device) for key, value in inputs.items()}
        method = getattr(self._model, method_name, None)
        if method is None:
            raise EncoderUnavailableError(f"Loaded FG-CLIP2 model does not expose {method_name}")
        with self._torch.inference_mode(), self._autocast_context():
            return method(**moved_inputs, **kwargs)

    def _autocast_context(self) -> Any:
        if self._use_autocast and str(self._device).startswith("cuda"):
            return self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
        return nullcontext()

    def _to_numpy(self, vectors: Any) -> np.ndarray:
        result = vectors.detach().float().cpu().numpy()
        if result.ndim != 2:
            raise EncoderUnavailableError(
                f"FG-CLIP2 returned unexpected embedding shape {result.shape}; expected [batch, dimension]"
            )
        self._embedding_dimension = int(result.shape[1])
        return result
