"""PE-Core-G14-448 adapter with explicit local checkpoint loading."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image

from encoders.base import EncoderUnavailableError
from encoders.multimodal import BatchedImageTextEncoder, ImageTextEmbeddingBackend


class PECoreBackend(ImageTextEmbeddingBackend, Protocol):
    """Structural contract implemented by PE-Core model backends."""


class PECoreEncoder(BatchedImageTextEncoder):
    """Batched, L2-normalized PE-Core image/text encoder."""

    def __init__(self, backend: PECoreBackend, batch_size: int) -> None:
        super().__init__(backend, batch_size, encoder_label="PE-Core")


class PerceptionCoreBackend:
    """Adapter for Meta's official ``perception_models`` PE-Core implementation."""

    def __init__(
        self,
        model: Any,
        image_transform: Any,
        tokenizer: Any,
        torch_module: Any,
        device: str,
        model_config: str,
        use_autocast: bool,
    ) -> None:
        self._model = model
        self._image_transform = image_transform
        self._tokenizer = tokenizer
        self._torch = torch_module
        self._device = device
        self._model_config = model_config
        self._use_autocast = use_autocast
        self._embedding_dimension: int | None = None

    @property
    def embedding_dimension(self) -> int | None:
        return self._embedding_dimension

    @property
    def preprocessing_config(self) -> dict[str, object]:
        return {
            "image_size": int(self._model.image_size),
            "text_context_length": int(self._model.context_length),
            "image_transform": "perception_models.get_image_transform",
            "text_tokenizer": "perception_models.get_text_tokenizer",
            "model_config": self._model_config,
            "use_autocast": self._use_autocast,
        }

    @classmethod
    def from_local_checkpoint(
        cls,
        checkpoint: Path,
        model_config: str,
        device: str,
        use_autocast: bool,
    ) -> "PerceptionCoreBackend":
        if not checkpoint.is_file():
            raise EncoderUnavailableError(
                f"PE-Core checkpoint does not exist: {checkpoint}. "
                "Prepare a local checkpoint; this repository never downloads one implicitly."
            )
        try:
            import torch
            import core.vision_encoder.pe as pe
            import core.vision_encoder.transforms as transforms
        except ImportError as error:
            raise EncoderUnavailableError(
                "PE-Core requires torch and Meta's official perception_models repository on PYTHONPATH"
            ) from error
        try:
            model = pe.CLIP.from_config(
                model_config,
                pretrained=True,
                checkpoint_path=str(checkpoint),
            ).to(device)
        except (OSError, RuntimeError, ValueError) as error:
            raise EncoderUnavailableError(
                f"Unable to load PE-Core model config {model_config} from {checkpoint}"
            ) from error
        model.eval()
        return cls(
            model=model,
            image_transform=transforms.get_image_transform(model.image_size),
            tokenizer=transforms.get_text_tokenizer(model.context_length),
            torch_module=torch,
            device=device,
            model_config=model_config,
            use_autocast=use_autocast,
        )

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        tensor = self._torch.stack(
            [self._image_transform(image.convert("RGB")) for image in images]
        ).to(self._device)
        with self._torch.inference_mode(), self._autocast_context():
            vectors = self._model.encode_image(tensor, normalize=False)
        return self._to_numpy(vectors)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        tokens = self._tokenizer(list(texts)).to(self._device)
        with self._torch.inference_mode(), self._autocast_context():
            vectors = self._model.encode_text(tokens, normalize=False)
        return self._to_numpy(vectors)

    def _autocast_context(self) -> Any:
        if self._use_autocast and str(self._device).startswith("cuda"):
            return self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
        return nullcontext()

    def _to_numpy(self, vectors: Any) -> np.ndarray:
        result = vectors.detach().float().cpu().numpy()
        if result.ndim != 2:
            raise EncoderUnavailableError(
                f"PE-Core returned unexpected embedding shape {result.shape}; expected [batch, dimension]"
            )
        self._embedding_dimension = int(result.shape[1])
        return result
