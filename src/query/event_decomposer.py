"""Ordered TRAKE event decomposition with a replaceable rule-based baseline."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Protocol, Sequence


class EventDecompositionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EventQuery:
    """One ordered event plus the complete TRAKE context for visual retrieval."""

    index: int
    text: str
    context: str

    def __post_init__(self) -> None:
        if self.index < 0:
            raise EventDecompositionError("Event index must be non-negative")
        if not self.text.strip():
            raise EventDecompositionError("Event text must be non-empty")
        if not self.context.strip():
            raise EventDecompositionError("Event context must be non-empty")

    @property
    def retrieval_text(self) -> str:
        """Keep action context when an individual listed event is terse."""

        return f"Focus event {self.index + 1}: {self.text}\nSequence context: {self.context}"

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["retrieval_text"] = self.retrieval_text
        return value


class EventDecomposer(Protocol):
    def decompose(self, query: str) -> Sequence[EventQuery]: ...


class RuleBasedEventDecomposer:
    """Extract explicit numbered, bulleted, or arrow-delimited event lists.

    Unstructured wording is intentionally kept as one event.  A later LLM
    implementation can satisfy :class:`EventDecomposer` without changing
    TRAKE retrieval or temporal alignment.
    """

    _LIST_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+(.+?)\s*$")
    _PAREN_LIST_MARKER = re.compile(r"(?<!\w)\((\d+)\)\s*")
    _ARROW = re.compile(r"\s*(?:→|->|=>)\s*")

    def decompose(self, query: str) -> tuple[EventQuery, ...]:
        context = _normalize(query)
        if not context:
            raise EventDecompositionError("TRAKE query must be non-empty")
        parts = self._parenthesized_parts(query)
        if len(parts) < 2:
            parts = self._numbered_or_bulleted_parts(query)
        if len(parts) < 2:
            parts = self._delimiter_parts(context)
        if not parts:
            parts = (context,)
        return tuple(EventQuery(index=index, text=part, context=context) for index, part in enumerate(parts))

    def _numbered_or_bulleted_parts(self, query: str) -> tuple[str, ...]:
        parts = tuple(
            _strip_item_separators(normalized)
            for line in query.splitlines()
            if (match := self._LIST_ITEM.match(line)) and (normalized := _normalize(match.group(1)))
        )
        return parts

    def _parenthesized_parts(self, query: str) -> tuple[str, ...]:
        markers = tuple(self._PAREN_LIST_MARKER.finditer(query))
        if len(markers) < 2:
            return ()
        numbers = tuple(int(marker.group(1)) for marker in markers)
        if numbers[0] != 1 or numbers != tuple(range(1, len(numbers) + 1)):
            return ()
        prefix = query[: markers[0].start()].rstrip()
        if prefix and not prefix.endswith((":", "-", "–", "—")):
            return ()
        parts: list[str] = []
        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(query)
            part = _strip_item_separators(_normalize(query[marker.end() : end]))
            if not part:
                return ()
            parts.append(part)
        return tuple(parts)

    def _delimiter_parts(self, context: str) -> tuple[str, ...]:
        arrow_parts = tuple(
            _strip_item_separators(_normalize(part)) for part in self._ARROW.split(context)
        )
        if len(arrow_parts) >= 2 and all(arrow_parts):
            return arrow_parts
        semicolon_parts = tuple(
            _strip_item_separators(_normalize(part)) for part in context.split(";")
        )
        if len(semicolon_parts) >= 2 and all(semicolon_parts):
            return semicolon_parts
        return ()


def _normalize(value: str) -> str:
    return " ".join(value.split())


def _strip_item_separators(value: str) -> str:
    return value.strip(" \t\r\n,.;")
