from __future__ import annotations

"""Fact-level source reuse classifier.

Evaluates ImplementationFacts against existing project source code and assets
to determine whether an element should be REUSED (existing symbol matches exactly),
ADAPTED (strong structurally-compatible identity overlap), or created as NEW.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import Any

from .implementation_fact import ImplementationFact


class ReuseMode(str, Enum):
    REUSE = "REUSE"
    ADAPT = "ADAPT"
    NEW = "NEW"


@dataclass(frozen=True)
class FactReuseDecision:
    fact_id: str
    mode: ReuseMode
    matching_symbol: str = ""
    target_file: str = ""
    rationale: str = ""


_IDENTIFIER_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b|\b[a-z][a-z0-9_]{2,}\b")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_GENERIC_TOKENS = frozenset(
    {
        "mod",
        "mods",
        "registry",
        "registries",
        "register",
        "registered",
        "entry",
        "entries",
        "impl",
        "implementation",
        "handler",
        "manager",
    }
)


def _identity_tokens(value: str) -> tuple[str, ...]:
    """Normalize snake/kebab/camel-ish identifiers without substring matching."""
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value).lower()
    return tuple(
        token
        for token in _TOKEN_PATTERN.findall(normalized.replace("-", "_"))
        if token and token not in _GENERIC_TOKENS
    )


def _expected_path_markers(fact: ImplementationFact) -> tuple[str, ...]:
    name = str(getattr(fact.fact_type, "value", fact.fact_type)).upper()
    if name.startswith("ITEM_") or name == "ITEM_EXISTS":
        return ("item", "items")
    if name.startswith("BLOCK_") or name == "BLOCK_EXISTS":
        return ("block", "blocks")
    if name.startswith("ENTITY_") or name == "ENTITY_EXISTS":
        return ("entity", "entities")
    if name == "GUI_EXISTS":
        return ("gui", "screen", "menu")
    if name == "NETWORK_PACKET":
        return ("network", "packet")
    if name == "BLOCK_ENTITY_EXISTS":
        return ("blockentity", "block_entity", "machine")
    if name == "DATA_COMPONENT":
        return ("component",)
    if name == "WORLDGEN_FEATURE":
        return ("worldgen", "feature")
    if name == "DIMENSION":
        return ("dimension",)
    if name == "BIOME":
        return ("biome",)
    if name == "STATUS_EFFECT":
        return ("effect",)
    if name == "SOUND_EVENT":
        return ("sound",)
    if name == "PARTICLE_TYPE":
        return ("particle",)
    if name == "ADVANCEMENT":
        return ("advancement",)
    if name in {"CRAFTING_RECIPE", "SMELTING_RECIPE"}:
        return ("recipe", "recipes")
    if name == "REGISTRY_TAG":
        return ("tag", "tags")
    return ()


def _path_compatible(fact: ImplementationFact, file_path: str) -> bool:
    markers = _expected_path_markers(fact)
    if not markers or not file_path:
        return True
    normalized = file_path.lower().replace("-", "_")
    return any(marker in normalized for marker in markers)


def _strong_adapt_match(subject: str, candidate: str) -> bool:
    """Require token identity overlap; a bare substring is never sufficient."""
    subject_tokens = set(_identity_tokens(subject))
    candidate_tokens = set(_identity_tokens(candidate))
    if not subject_tokens or not candidate_tokens:
        return False
    overlap = subject_tokens & candidate_tokens
    if not overlap:
        return False
    union = subject_tokens | candidate_tokens
    # One-token identifiers must match exactly; multi-token identifiers need both
    # substantial overlap and at most one differing semantic token.
    if len(subject_tokens) == 1 or len(candidate_tokens) == 1:
        return subject_tokens == candidate_tokens
    return len(overlap) >= 2 and len(union - overlap) <= 1 and len(overlap) / len(union) >= 2 / 3


class FactReuseClassifier:
    """Classifies implementation facts against existing project files or pre-built index."""

    def __init__(
        self,
        project_root: Path | str | None = None,
        project_index: Mapping[str, Any] | None = None,
    ) -> None:
        self.project_root = Path(project_root) if project_root else None
        self.project_index = dict(project_index) if project_index is not None else None
        self._symbol_cache: dict[str, tuple[str, str]] | None = None

    def _build_cache(self) -> dict[str, tuple[str, str]]:
        if self._symbol_cache is not None:
            return self._symbol_cache

        cache: dict[str, tuple[str, str]] = {}
        if self.project_index:
            symbols = self.project_index.get("symbols", {})
            if isinstance(symbols, Mapping):
                for sym, info in symbols.items():
                    target_file = (
                        info.get("file", "") if isinstance(info, Mapping) else str(info)
                    )
                    cache[sym.lower()] = (sym, target_file)
            elif isinstance(symbols, list):
                for item in symbols:
                    if isinstance(item, str):
                        cache[item.lower()] = (item, "")
                    elif isinstance(item, Mapping):
                        sym = item.get("name") or item.get("symbol", "")
                        target_file = item.get("file", "")
                        if sym:
                            cache[sym.lower()] = (sym, target_file)

        if self.project_root and self.project_root.is_dir():
            for ext in ("*.java", "*.json", "*.png"):
                for path in self.project_root.rglob(ext):
                    rel = str(path.relative_to(self.project_root)).replace("\\", "/")
                    stem = path.stem.lower()
                    cache[stem] = (path.stem, rel)
                    if path.suffix == ".java":
                        try:
                            content = path.read_text(encoding="utf-8", errors="ignore")
                            for match in _IDENTIFIER_PATTERN.finditer(content):
                                sym = match.group(0)
                                if len(sym) >= 3:
                                    cache.setdefault(sym.lower(), (sym, rel))
                        except OSError:
                            continue

        self._symbol_cache = cache
        return cache

    def classify(self, fact: ImplementationFact) -> FactReuseDecision:
        cache = self._build_cache()
        subject = str(fact.subject or "").strip().lower()
        if not subject:
            return FactReuseDecision(
                fact_id=fact.fact_id,
                mode=ReuseMode.NEW,
                rationale="Empty subject requires new implementation",
            )

        # Exact normalized identity is the only REUSE condition.
        exact_keys = {subject, subject.replace("-", "_")}
        for exact_key in exact_keys:
            if exact_key in cache:
                sym, file_path = cache[exact_key]
                if _path_compatible(fact, file_path):
                    return FactReuseDecision(
                        fact_id=fact.fact_id,
                        mode=ReuseMode.REUSE,
                        matching_symbol=sym,
                        target_file=file_path,
                        rationale=f"Exact compatible symbol {sym!r} found in {file_path or 'project index'}",
                    )

        # ADAPT is deliberately conservative: compatible path plus strong token overlap.
        candidates: list[tuple[float, str, str]] = []
        subject_tokens = set(_identity_tokens(subject))
        for key, (sym, file_path) in cache.items():
            if not _path_compatible(fact, file_path) or not _strong_adapt_match(subject, key):
                continue
            candidate_tokens = set(_identity_tokens(key))
            union = subject_tokens | candidate_tokens
            score = len(subject_tokens & candidate_tokens) / len(union) if union else 0.0
            candidates.append((score, sym, file_path))
        if candidates:
            _, sym, file_path = max(candidates, key=lambda item: (item[0], item[1], item[2]))
            return FactReuseDecision(
                fact_id=fact.fact_id,
                mode=ReuseMode.ADAPT,
                matching_symbol=sym,
                target_file=file_path,
                rationale=f"Strong compatible identity overlap with {sym!r} in {file_path or 'project index'}",
            )

        return FactReuseDecision(
            fact_id=fact.fact_id,
            mode=ReuseMode.NEW,
            rationale=f"No exact or strongly compatible identity found for {subject!r}; generate new artifact",
        )

    def classify_all(
        self, facts: Iterable[ImplementationFact]
    ) -> list[FactReuseDecision]:
        return [self.classify(f) for f in facts]
