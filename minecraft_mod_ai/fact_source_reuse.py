from __future__ import annotations

"""Fact-level source reuse classifier.

Evaluates ImplementationFacts against existing project source code and assets
to determine whether an element should be REUSED (existing symbol matches exactly),
ADAPTED (partial/related symbol or base structure exists), or created as NEW.
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

        # 1. Exact match
        if subject in cache:
            sym, file_path = cache[subject]
            return FactReuseDecision(
                fact_id=fact.fact_id,
                mode=ReuseMode.REUSE,
                matching_symbol=sym,
                target_file=file_path,
                rationale=f"Exact matching symbol {sym!r} found in {file_path or 'project index'}",
            )

        # Also check constant form (e.g. RAW_MATERIAL -> raw_material)
        const_form = subject.replace("-", "_").upper().lower()
        if const_form in cache:
            sym, file_path = cache[const_form]
            return FactReuseDecision(
                fact_id=fact.fact_id,
                mode=ReuseMode.REUSE,
                matching_symbol=sym,
                target_file=file_path,
                rationale=f"Exact matching symbol {sym!r} found in {file_path or 'project index'}",
            )

        # 2. Adapt match: partial or prefix/suffix substring
        for key, (sym, file_path) in cache.items():
            if (subject in key or key in subject) and len(key) >= 3:
                return FactReuseDecision(
                    fact_id=fact.fact_id,
                    mode=ReuseMode.ADAPT,
                    matching_symbol=sym,
                    target_file=file_path,
                    rationale=f"Related symbol {sym!r} found in {file_path or 'project index'}; adapt existing logic",
                )

        # 3. New
        return FactReuseDecision(
            fact_id=fact.fact_id,
            mode=ReuseMode.NEW,
            rationale=f"No existing code or asset symbol found for {subject!r}; generate new artifact",
        )

    def classify_all(
        self, facts: Iterable[ImplementationFact]
    ) -> list[FactReuseDecision]:
        return [self.classify(f) for f in facts]
