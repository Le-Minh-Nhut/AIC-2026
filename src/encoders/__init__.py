"""Text and image encoder adapters."""

from encoders.base import EncoderUnavailableError, l2_normalize
from encoders.btc_clip import BtcClipTextEncoder, OpenClipTextBackend
from encoders.fgclip2 import FGCLIP2Encoder, HuggingFaceFGCLIP2Backend
from encoders.pecore import PECoreEncoder, PerceptionCoreBackend

__all__ = [
    "BtcClipTextEncoder",
    "EncoderUnavailableError",
    "FGCLIP2Encoder",
    "HuggingFaceFGCLIP2Backend",
    "OpenClipTextBackend",
    "PECoreEncoder",
    "PerceptionCoreBackend",
    "l2_normalize",
]
