"""Conservative canonicalization for short AIC Q&A answers."""

from __future__ import annotations

import re
import unicodedata


_EDGE_PUNCTUATION = " \t\n\r.,;:!?\\\"'`()[]{}<>"
_COUNT_UNITS = {
    "person",
    "people",
    "persons",
    "player",
    "players",
    "người",
    "nguoi",
    "người chơi",
    "nguoi choi",
}
_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "không": "0",
    "khong": "0",
    "một": "1",
    "mot": "1",
    "hai": "2",
    "ba": "3",
    "bốn": "4",
    "bon": "4",
    "năm": "5",
    "nam": "5",
    "sáu": "6",
    "sau": "6",
    "bảy": "7",
    "bay": "7",
    "tám": "8",
    "tam": "8",
    "chín": "9",
    "chin": "9",
    "mười": "10",
    "muoi": "10",
}
_YES = {"yes", "yeah", "yep", "true", "correct", "có", "co", "đúng", "dung", "phải", "phai"}
_NO = {"no", "nope", "false", "không", "khong", "ko", "không phải", "khong phai"}
_COLORS = {
    "red": "red",
    "dark red": "red",
    "light red": "red",
    "đỏ": "red",
    "do": "red",
    "màu đỏ": "red",
    "mau do": "red",
    "blue": "blue",
    "light blue": "blue",
    "dark blue": "blue",
    "xanh dương": "blue",
    "xanh duong": "blue",
    "màu xanh dương": "blue",
    "mau xanh duong": "blue",
    "green": "green",
    "light green": "green",
    "dark green": "green",
    "xanh lá": "green",
    "xanh la": "green",
    "màu xanh lá": "green",
    "mau xanh la": "green",
    "yellow": "yellow",
    "vàng": "yellow",
    "vang": "yellow",
    "màu vàng": "yellow",
    "mau vang": "yellow",
    "black": "black",
    "đen": "black",
    "den": "black",
    "white": "white",
    "trắng": "white",
    "trang": "white",
    "gray": "gray",
    "grey": "gray",
    "xám": "gray",
    "xam": "gray",
    "brown": "brown",
    "nâu": "brown",
    "nau": "brown",
    "orange": "orange",
    "cam": "orange",
    "purple": "purple",
    "tím": "purple",
    "tim": "purple",
    "pink": "pink",
    "hồng": "pink",
    "hong": "pink",
}
_COLOR_PREFIX = re.compile(
    r"^(?:(?:the )?colou?r(?: is)?|màu(?: sắc)?(?: là)?|mau(?: sac)?(?: la)?|"
    r"it is|it's|nó là|no la)\s+"
)


class AnswerNormalizationError(ValueError):
    pass


class AnswerNormalizer:
    def normalize(self, answer: str) -> str:
        return normalize_answer(answer)


def normalize_answer(answer: str) -> str:
    """Normalize known short-answer variants without rewriting entities or free text."""

    if not isinstance(answer, str):
        raise AnswerNormalizationError("Answer must be a string")
    normalized = unicodedata.normalize("NFKC", answer).casefold().strip(_EDGE_PUNCTUATION)
    normalized = " ".join(normalized.split())
    if not normalized:
        raise AnswerNormalizationError("Answer cannot be empty after normalization")
    if normalized in _YES:
        return "yes"
    if normalized in _NO:
        return "no"
    count = _normalize_count(normalized)
    if count is not None:
        return count
    color = _COLORS.get(normalized)
    if color is not None:
        return color
    color = _COLORS.get(_COLOR_PREFIX.sub("", normalized))
    return color if color is not None else normalized


def _normalize_count(value: str) -> str | None:
    words = value.split()
    if len(words) == 1:
        if words[0].isdigit():
            return str(int(words[0]))
        return _NUMBER_WORDS.get(words[0])
    if " ".join(words[1:]) in _COUNT_UNITS:
        if words[0].isdigit():
            return str(int(words[0]))
        return _NUMBER_WORDS.get(words[0])
    return None
