"""Small explicit JSON cache with atomic writes and corruption detection."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, TypeVar, cast


Value = TypeVar("Value")


class CacheError(ValueError):
    pass


class CacheCorruptionError(CacheError):
    pass


class JsonResultCache:
    """A caller-owned persistent cache; eviction happens only through explicit clear."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def get(self, key: str) -> object | None:
        path = self._path_for_key(key)
        if not path.exists():
            return None
        if not path.is_file():
            raise CacheCorruptionError(f"Cache entry is not a file: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CacheCorruptionError(f"Cache entry cannot be decoded: {path}") from error
        if not isinstance(payload, dict) or payload.get("key") != key or "value" not in payload:
            raise CacheCorruptionError(f"Cache entry has an invalid schema: {path}")
        return payload["value"]

    def put(self, key: str, value: object) -> Path:
        path = self._path_for_key(key)
        try:
            encoded = json.dumps({"schema_version": "1.0", "key": key, "value": value}, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise CacheError("Cache value must be JSON serializable") from error
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
        return path

    def get_or_compute(self, key: str, factory: Callable[[], Value]) -> Value:
        cached = self.get(key)
        if cached is not None:
            return cast(Value, cached)
        value = factory()
        self.put(key, value)
        return value

    def clear(self, key: str) -> bool:
        path = self._path_for_key(key)
        if not path.exists():
            return False
        if not path.is_file():
            raise CacheCorruptionError(f"Cache entry is not a file: {path}")
        path.unlink()
        return True

    def _path_for_key(self, key: str) -> Path:
        if not isinstance(key, str) or not key.strip():
            raise CacheError("Cache key must be non-empty text")
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._root / f"{digest}.json"
