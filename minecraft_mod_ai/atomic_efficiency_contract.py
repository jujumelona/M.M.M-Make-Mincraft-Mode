from __future__ import annotations

import re
from functools import lru_cache
from typing import Any


_PREVIEW_BYTES = 2048
_CONJUNCTION = re.compile(
    r"(?<!\S)(?:and|then|plus|그리고|또한|및)(?=\s)",
    re.IGNORECASE,
)


def _preview(text: str) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= _PREVIEW_BYTES:
        return text
    return encoded[:_PREVIEW_BYTES].decode("utf-8", errors="ignore") + "…"


def install(atomic_module: Any) -> None:
    """Bound atomic coverage cost without weakening source binding."""

    if getattr(atomic_module, "_mmm_atomic_efficiency_contract", False):
        return

    def implementations(proposal: Any) -> dict[str, str]:
        result: dict[str, str] = {}
        for source_kind, values, id_name in (
            ("module", getattr(proposal, "modules", ()), "module_id"),
            ("asset", getattr(proposal, "assets", ()), "asset_id"),
            ("audio", getattr(proposal, "audio", ()), "sound_id"),
        ):
            for item in values:
                identity = str(getattr(item, id_name))
                raw = atomic_module._canonical(atomic_module._object(item))
                result[f"implementation:{source_kind}:{identity}"] = (
                    atomic_module._canonical(
                        {
                            "content_sha256": atomic_module._sha(raw),
                            "preview": _preview(raw),
                        }
                    )
                )
        return result

    def acceptances(proposal: Any) -> dict[str, str]:
        result: dict[str, str] = {}
        for index, value in enumerate(getattr(proposal, "acceptance_tests", ())):
            raw = str(value)
            result[f"acceptance:{index:08d}"] = atomic_module._canonical(
                {
                    "content_sha256": atomic_module._sha(raw),
                    "preview": _preview(raw),
                }
            )
        return result

    original_features = atomic_module._features
    cached_features = lru_cache(maxsize=2048)(original_features)
    cached_features._mmm_bounded_feature_cache = True

    def _segments(prompt: str, start: int, end: int) -> list[tuple[int, int]]:
        boundaries: list[tuple[int, int]] = []
        for index in range(start, end):
            if prompt[index] in {",", "，", "、"}:
                boundaries.append((index, index + 1))
        for match in _CONJUNCTION.finditer(prompt, start, end):
            boundaries.append((match.start(), match.start()))
        boundaries.sort()

        result: list[tuple[int, int]] = []
        cursor = start
        for cut, next_start in boundaries:
            if cut < cursor:
                continue
            left, right = cursor, cut
            while left < right and prompt[left].isspace():
                left += 1
            while right > left and prompt[right - 1].isspace():
                right -= 1
            if left < right:
                result.append((left, right))
            cursor = max(cursor, next_start)
        left, right = cursor, end
        while left < right and prompt[left].isspace():
            left += 1
        while right > left and prompt[right - 1].isspace():
            right -= 1
        if left < right:
            result.append((left, right))
        return result

    def atom_ranges(prompt: str) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for match in atomic_module._SENTENCE.finditer(prompt):
            start, end = match.span()
            while start < end and prompt[start].isspace():
                start += 1
            while end > start and prompt[end - 1].isspace():
                end -= 1
            if start >= end:
                continue

            cuts = _segments(prompt, start, end)
            if len(cuts) < 2:
                result.extend(atomic_module._split_range(prompt, start, end))
            else:
                for left, right in cuts:
                    result.extend(
                        atomic_module._split_range(prompt, left, right)
                    )

        if not result and prompt.strip():
            start = len(prompt) - len(prompt.lstrip())
            end = len(prompt.rstrip())
            result.extend(atomic_module._split_range(prompt, start, end))
        return result

    atom_ranges._mmm_enumeration_atomizer = True
    atom_ranges._mmm_conjunction_atomizer = True
    implementations._mmm_compact_catalog = True
    acceptances._mmm_compact_catalog = True
    atomic_module._features = cached_features
    atomic_module._implementations = implementations
    atomic_module._acceptances = acceptances
    atomic_module._atom_ranges = atom_ranges
    atomic_module._mmm_atomic_efficiency_contract = True
