from __future__ import annotations

"""Bounded source-excerpt selection from an exact version-specific mod pool."""

import os
import re
from dataclasses import dataclass

from .versioned_mod_reference_catalog import baseline_families, task_families
from .versioned_mod_reference_resolver import ExactReferenceResolver, ResolvedReference

_SOURCE_SUFFIXES = (".java", ".kt")
_ARCHITECTURE_PATH_TERMS = (
    "init", "registry", "register", "datagen", "data", "network", "packet", "payload",
    "config", "component", "blockentity", "block_entity", "screen", "menu", "event",
    "recipe", "resource", "client", "server",
)


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.environ.get(name, "").strip() or default)
    except ValueError:
        value = default
    return max(low, min(high, value))


def _tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{1,127}|[가-힣]{2,}", value or "")
    }


@dataclass(frozen=True)
class ReferenceExcerpt:
    repository: str
    role: str
    ref_name: str
    commit_sha: str
    license_spdx: str
    path: str
    source_sha: str
    text: str
    start_line: int
    end_line: int
    score: float
    metadata_sha256: str


class VersionedModReferenceRetriever:
    """Create a distinct baseline+task reference pool for one exact target."""

    def __init__(self, resolver: ExactReferenceResolver) -> None:
        self.resolver = resolver
        self.loader = resolver.loader
        self.baseline_family_limit = _env_int("MMM_REFERENCE_BASELINE_FAMILIES", 3, 1, 8)
        self.task_family_limit = _env_int("MMM_REFERENCE_TASK_FAMILIES", 2, 0, 8)
        self.file_fetch_limit = _env_int("MMM_REFERENCE_FILES_PER_FAMILY", 5, 1, 16)
        self.excerpt_lines = _env_int("MMM_REFERENCE_EXCERPT_LINES", 80, 32, 180)
        self.max_excerpts = _env_int("MMM_REFERENCE_MAX_EXCERPTS", 4, 1, 10)
        self.byte_budget = _env_int("MMM_REFERENCE_CONTEXT_BYTES", 12288, 4096, 32768)

    def retrieve(self, query: str, *, capability: str = "") -> tuple[ReferenceExcerpt, ...]:
        combined = " ".join(part for part in (query, capability) if str(part).strip())
        baseline = baseline_families(self.loader, limit=self.baseline_family_limit)
        baseline_repos = frozenset(family.repository for family in baseline)
        task = task_families(
            self.loader,
            combined,
            exclude=baseline_repos,
            limit=self.task_family_limit,
        )
        candidates: list[ReferenceExcerpt] = []
        requests = [(family, "baseline") for family in baseline]
        requests.extend((family, "task") for family in task)
        for family, role in requests:
            resolved = self.resolver.resolve(family)
            if resolved is not None:
                candidates.extend(self._source_excerpts(resolved, combined, role=role))

        candidates.sort(
            key=lambda item: (
                0 if item.role == "task" else 1,
                -item.score,
                item.repository,
                item.path,
            )
        )
        selected: list[ReferenceExcerpt] = []
        seen_source: set[tuple[str, str, str]] = set()
        seen_repo: set[str] = set()
        used = 0

        def size_of(item: ReferenceExcerpt) -> int:
            return len(item.text.encode("utf-8"))

        def admit(item: ReferenceExcerpt) -> bool:
            nonlocal used
            key = (item.repository, item.commit_sha, item.source_sha)
            if key in seen_source:
                return False
            size = size_of(item)
            if size <= 0 or used + size > self.byte_budget:
                return False
            seen_source.add(key)
            seen_repo.add(item.repository)
            selected.append(item)
            used += size
            return True

        # Preserve the intended two-part textbook whenever the byte budget permits it:
        # one task-specific donor plus one similarity-independent architecture baseline.
        if self.max_excerpts >= 2:
            task_heads = self._repo_heads(candidates, role="task")
            baseline_heads = self._repo_heads(candidates, role="baseline")
            feasible_pairs = [
                (task_item, baseline_item)
                for task_item in task_heads
                for baseline_item in baseline_heads
                if task_item.repository != baseline_item.repository
                and size_of(task_item) + size_of(baseline_item) <= self.byte_budget
            ]
            if feasible_pairs:
                task_item, baseline_item = max(
                    feasible_pairs,
                    key=lambda pair: (
                        pair[0].score + pair[1].score,
                        min(pair[0].score, pair[1].score),
                        -(size_of(pair[0]) + size_of(pair[1])),
                    ),
                )
                admit(task_item)
                admit(baseline_item)
                if len(selected) >= self.max_excerpts:
                    return tuple(selected)

        for role in ("task", "baseline"):
            for item in candidates:
                if item.role != role or item.repository in seen_repo:
                    continue
                admit(item)
                if len(selected) >= self.max_excerpts:
                    return tuple(selected)
        for item in candidates:
            admit(item)
            if len(selected) >= self.max_excerpts:
                break
        return tuple(selected)

    @staticmethod
    def _repo_heads(
        candidates: list[ReferenceExcerpt], *, role: str
    ) -> tuple[ReferenceExcerpt, ...]:
        result: list[ReferenceExcerpt] = []
        seen: set[str] = set()
        for item in candidates:
            if item.role != role or item.repository in seen:
                continue
            seen.add(item.repository)
            result.append(item)
        return tuple(result)

    def pool_receipt(self) -> dict[str, object]:
        baseline = baseline_families(self.loader, limit=self.baseline_family_limit)
        return {
            "schema_version": "mmm/versioned-mod-reference-pool-v1",
            "target": {
                "minecraft_version": self.resolver.minecraft_version,
                "loader": self.resolver.loader,
                "mappings": self.resolver.mappings,
            },
            "baseline_candidates": [family.repository for family in baseline],
            "baseline_is_similarity_independent": True,
            "task_donors_are_additive": True,
            "dual_role_pair_preferred_when_feasible": True,
            "compatibility_owner": "host",
            "model_may_not_admit_references": True,
            "immutable_commit_required": True,
            "license_fail_closed": True,
        }

    def _source_excerpts(
        self, resolved: ResolvedReference, query: str, *, role: str
    ) -> list[ReferenceExcerpt]:
        source_rows = [
            row
            for row in resolved.tree
            if str(row.get("type") or "") == "blob"
            and str(row.get("path") or "").endswith(_SOURCE_SUFFIXES)
            and "/src/" in ("/" + str(row.get("path") or ""))
        ]
        scored = [
            (self._path_score(str(row.get("path") or ""), query, resolved, role=role), row)
            for row in source_rows
        ]
        scored.sort(key=lambda item: (-item[0], str(item[1].get("path") or "")))
        result: list[ReferenceExcerpt] = []
        for path_score, row in scored[: self.file_fetch_limit]:
            blob_sha = str(row.get("sha") or "")
            path = str(row.get("path") or "")
            if not blob_sha or not path:
                continue
            text = self.resolver.blob(resolved.family.repository, blob_sha)
            if not text:
                continue
            excerpt, start, end, lexical = self._excerpt(text, query, role=role)
            if not excerpt:
                continue
            role_bonus = 0.15 if role == "task" else 0.08
            result.append(
                ReferenceExcerpt(
                    repository=resolved.family.repository,
                    role=role,
                    ref_name=resolved.ref_name,
                    commit_sha=resolved.commit_sha,
                    license_spdx=resolved.license_spdx,
                    path=path,
                    source_sha=blob_sha,
                    text=excerpt,
                    start_line=start,
                    end_line=end,
                    score=min(1.0, role_bonus + 0.55 * path_score + 0.30 * lexical),
                    metadata_sha256=resolved.metadata_sha256,
                )
            )
        return result

    def _path_score(
        self, path: str, query: str, resolved: ResolvedReference, *, role: str
    ) -> float:
        path_tokens = _tokens(path.replace("-", " ").replace("_", " "))
        query_tokens = _tokens(query)
        task_overlap = len(path_tokens & query_tokens) / max(1, min(8, len(query_tokens)))
        architecture = sum(
            1 for term in _ARCHITECTURE_PATH_TERMS if term in path.casefold()
        ) / len(_ARCHITECTURE_PATH_TERMS)
        family_hits = sum(
            1 for capability in resolved.family.capabilities
            if capability.casefold() in path.casefold()
        )
        family_score = min(1.0, family_hits / 3.0)
        if role == "baseline":
            return min(1.0, 0.55 * architecture + 0.25 * task_overlap + 0.20 * family_score)
        return min(1.0, 0.20 * architecture + 0.55 * task_overlap + 0.25 * family_score)

    def _excerpt(self, text: str, query: str, *, role: str) -> tuple[str, int, int, float]:
        lines = text.splitlines()
        if not lines:
            return "", 1, 1, 0.0
        needles = _tokens(query)
        if role == "baseline":
            needles |= set(_ARCHITECTURE_PATH_TERMS)
        best_line = 0
        best_hits = -1
        for index, line in enumerate(lines):
            hits = len(_tokens(line) & needles)
            if hits > best_hits:
                best_line = index
                best_hits = hits
        radius = max(8, self.excerpt_lines // 2)
        start = max(0, best_line - radius)
        end = min(len(lines), start + self.excerpt_lines)
        start = max(0, end - self.excerpt_lines)
        excerpt = "\n".join(lines[start:end]).strip()
        lexical = min(1.0, max(0, best_hits) / max(1, min(6, len(needles))))
        return excerpt, start + 1, end, lexical


__all__ = ["ReferenceExcerpt", "VersionedModReferenceRetriever"]
