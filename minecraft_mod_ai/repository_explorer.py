from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .project_index import ProjectIndex

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,127}")
_JAVA_TYPE = re.compile(
    r"(?m)^\s*(?:public|protected|private|abstract|final|static|sealed|non-sealed|\s)*"
    r"(?:class|interface|enum|record)\s+([A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\s+extends\s+([A-Za-z0-9_.$<>?, ]+))?"
    r"(?:\s+implements\s+([A-Za-z0-9_.$<>?, ]+))?"
)
_JAVA_METHOD = re.compile(
    r"(?m)^\s*(?:@[A-Za-z0-9_.$()\", =]+\s*)*"
    r"(?:public|protected|private|static|final|synchronized|abstract|native|default|\s)+"
    r"(?:<[^>{}]+>\s*)?(?:[A-Za-z0-9_.$<>\[\]?,]+\s+)+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*(?:throws\s+[^{]+)?\{"
)
_JAVA_IMPORT = re.compile(r"(?m)^\s*import\s+(?:static\s+)?([A-Za-z0-9_.$*]+)\s*;")
_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_CONTROL = frozenset({
    "if", "else", "for", "while", "switch", "case", "try", "catch", "finally",
    "return", "throw", "new", "synchronized", "stream", "map", "filter", "collect",
})
_API_HINTS = frozenset({
    "api", "method", "class", "interface", "symbol", "registry", "register", "event",
    "callback", "packet", "network", "codec", "serializer", "component", "world",
    "entity", "block", "item", "recipe", "datagen", "mixin",
})
_TRACE_HINTS = frozenset({
    "traceback", "exception", "error", "failed", "failure", "stack", "diagnostic",
    "cannot", "symbol", "compile", "gradle", "gametest",
})
_RIPPLE_HINTS = frozenset({
    "change", "modify", "replace", "rename", "refactor", "dependency", "depends",
    "callers", "usage", "uses", "impact", "ripple",
})
_PROCEDURAL_HINTS = frozenset({
    "create", "build", "generate", "implement", "load", "save", "sync", "spawn",
    "register", "validate", "serialize", "deserialize", "send", "receive", "tick",
    "open", "close", "update", "apply", "repair",
})


@dataclass(frozen=True)
class CodeRegion:
    path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    text: str
    lexical_score: float = 0.0
    graph_score: float = 0.0
    api_score: float = 0.0
    procedural_score: float = 0.0
    semantic_score: float = 0.0
    reranker_score: float = 0.0
    diagnostic_score: float = 0.0

    @property
    def score(self) -> float:
        return (
            self.lexical_score + self.graph_score + self.api_score + self.procedural_score
            + self.semantic_score + (2.0 * self.reranker_score) + self.diagnostic_score
        )

    @property
    def line_count(self) -> int:
        return max(1, self.end_line - self.start_line + 1)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["score"] = round(self.score, 6)
        value["line_count"] = self.line_count
        return value


@dataclass(frozen=True)
class ExplorationResult:
    query: str
    route: str
    regions: tuple[CodeRegion, ...]
    line_budget: int
    lines_selected: int
    candidate_count: int
    graph_edges_considered: int
    semantic_used: bool
    rerank_used: bool
    query_terms: tuple[str, ...]
    covered_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/repository-exploration-v1",
            "query": self.query,
            "route": self.route,
            "line_budget": self.line_budget,
            "lines_selected": self.lines_selected,
            "candidate_count": self.candidate_count,
            "graph_edges_considered": self.graph_edges_considered,
            "semantic_used": self.semantic_used,
            "rerank_used": self.rerank_used,
            "query_terms": list(self.query_terms),
            "covered_terms": list(self.covered_terms),
            "missing_terms": list(self.missing_terms),
            "regions": [region.to_dict() for region in self.regions],
        }


@dataclass(frozen=True)
class _ParsedRegion:
    path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    text: str
    tokens: frozenset[str]
    calls: frozenset[str]
    procedure: frozenset[str]
    api_tokens: frozenset[str]


class RepositoryExplorer:
    """Task-adaptive repository explorer with line-level, graph and procedural retrieval."""

    def __init__(self, index: ProjectIndex, *, router: Any | None = None) -> None:
        self.index = index
        self.router = router
        self._regions, self._imports, self._inheritance, self._symbol_paths = self._parse()

    def explore(
        self,
        query: str,
        *,
        diagnostic_paths: Iterable[str] = (),
        line_budget: int = 320,
        semantic: bool | None = None,
        rerank: bool | None = None,
    ) -> ExplorationResult:
        query = str(query or "").strip()
        if not query:
            raise ValueError("repository exploration query must be non-empty")
        if type(line_budget) is not int or line_budget < 1:
            raise ValueError("line_budget must be a positive integer")

        route = classify_exploration_route(query, diagnostic_paths=diagnostic_paths)
        query_terms = _meaningful_terms(query)
        query_set = set(query_terms)
        diagnostics = {
            self.index._normalize_path(str(path))
            for path in diagnostic_paths
            if str(path).strip()
        }
        diagnostics.discard("")

        weighted: list[CodeRegion] = []
        seed_symbols: set[str] = set()
        for region in self._regions:
            path_terms = set(_terms(region.path))
            symbol_terms = set(_terms(region.symbol))
            lexical = (
                3.0 * len(query_set & symbol_terms)
                + 1.5 * len(query_set & path_terms)
                + 0.35 * len(query_set & set(region.tokens))
            )
            diagnostic = 8.0 if region.path in diagnostics else 0.0
            api = 0.0
            if route in {"api", "trace", "global"}:
                api = 1.2 * len(query_set & set(region.api_tokens))
                if region.kind in {"class", "interface", "enum", "record", "method"}:
                    api += 0.15
            procedural = 0.0
            if route in {"procedural", "global"}:
                qproc = _query_procedure_terms(query_set)
                procedural = _jaccard(qproc, set(region.procedure)) * 4.0
                procedural += 0.15 * len(qproc & set(region.calls))

            if lexical > 0 or diagnostic > 0 or api > 0 or procedural > 0:
                seed_symbols.add(region.symbol)
                weighted.append(
                    CodeRegion(
                        path=region.path,
                        symbol=region.symbol,
                        kind=region.kind,
                        start_line=region.start_line,
                        end_line=region.end_line,
                        text=region.text,
                        lexical_score=lexical,
                        api_score=api,
                        procedural_score=procedural,
                        diagnostic_score=diagnostic,
                    )
                )

        weighted.sort(key=lambda item: (-item.score, item.line_count, item.path, item.start_line))
        candidate_map = {_region_key(item): item for item in weighted[:96]}
        graph_edges_considered = 0
        if route in {"trace", "ripple", "global", "api"}:
            neighbors, graph_edges_considered = self._graph_expand(
                seed_symbols=seed_symbols,
                seed_paths={item.path for item in candidate_map.values()} | diagnostics,
                max_hops=2,
                max_regions=64,
            )
            for parsed, graph_score in neighbors:
                key = _parsed_key(parsed)
                existing = candidate_map.get(key)
                if existing is None:
                    candidate_map[key] = CodeRegion(
                        path=parsed.path,
                        symbol=parsed.symbol,
                        kind=parsed.kind,
                        start_line=parsed.start_line,
                        end_line=parsed.end_line,
                        text=parsed.text,
                        graph_score=graph_score,
                    )
                elif graph_score > existing.graph_score:
                    candidate_map[key] = _replace_score(existing, graph_score=graph_score)

        candidates = list(candidate_map.values())
        candidates.sort(key=lambda item: (-item.score, item.line_count, item.path, item.start_line))

        use_semantic = (
            self.router is not None and route in {"api", "procedural", "global"}
            if semantic is None else bool(semantic and self.router is not None)
        )
        if use_semantic and candidates:
            candidates = self._semantic_score(query, candidates[:64]) + candidates[64:]
            candidates.sort(key=lambda item: (-item.score, item.line_count, item.path, item.start_line))

        use_rerank = (
            self.router is not None and route in {"api", "trace", "procedural", "global"}
            if rerank is None else bool(rerank and self.router is not None)
        )
        if use_rerank and candidates:
            candidates = self._rerank(query, candidates[:32]) + candidates[32:]
            candidates.sort(key=lambda item: (-item.score, item.line_count, item.path, item.start_line))

        selected: list[CodeRegion] = []
        consumed = 0
        selected_keys: set[tuple[str, int, int]] = set()
        for region in candidates:
            key = (region.path, region.start_line, region.end_line)
            if key in selected_keys:
                continue
            if consumed + region.line_count > line_budget:
                if selected:
                    continue
                region = _clip_region(region, line_budget)
            selected.append(region)
            selected_keys.add((region.path, region.start_line, region.end_line))
            consumed += region.line_count
            if consumed >= line_budget:
                break

        covered: set[str] = set()
        for region in selected:
            covered.update(query_set & set(_terms(region.text)))
            covered.update(query_set & set(_terms(region.path)))
            covered.update(query_set & set(_terms(region.symbol)))
        return ExplorationResult(
            query=query,
            route=route,
            regions=tuple(selected),
            line_budget=line_budget,
            lines_selected=consumed,
            candidate_count=len(candidates),
            graph_edges_considered=graph_edges_considered,
            semantic_used=use_semantic,
            rerank_used=use_rerank,
            query_terms=tuple(sorted(query_set)),
            covered_terms=tuple(sorted(covered)),
            missing_terms=tuple(sorted(query_set - covered)),
        )

    def _parse(self) -> tuple[
        tuple[_ParsedRegion, ...], dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]
    ]:
        regions: list[_ParsedRegion] = []
        imports: dict[str, set[str]] = {}
        inheritance: dict[str, set[str]] = {}
        symbol_paths: dict[str, set[str]] = {}
        for item in self.index.files:
            if item.size_bytes > self.index.policy.max_single_file_bytes:
                continue
            path = self.index.root / item.path
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if item.path.endswith(".java"):
                new_regions, file_imports, file_inheritance = _parse_java(item.path, text)
                imports[item.path] = file_imports
                inheritance[item.path] = file_inheritance
            else:
                new_regions = [
                    _make_region(
                        item.path, Path(item.path).name, "file", 1,
                        max(1, len(text.splitlines())), text,
                    )
                ]
            regions.extend(new_regions)
            for region in new_regions:
                symbol_paths.setdefault(region.symbol, set()).add(region.path)
        return tuple(regions), imports, inheritance, symbol_paths

    def _graph_expand(
        self,
        *,
        seed_symbols: set[str],
        seed_paths: set[str],
        max_hops: int,
        max_regions: int,
    ) -> tuple[list[tuple[_ParsedRegion, float]], int]:
        path_neighbors: dict[str, set[str]] = {}
        edges = 0
        for path, imports in self._imports.items():
            targets = {
                target
                for imported in imports
                for symbol, paths in self._symbol_paths.items()
                if symbol == imported.rsplit(".", 1)[-1]
                for target in paths
            }
            if targets:
                path_neighbors.setdefault(path, set()).update(targets)
                edges += len(targets)
        for path, parents in self._inheritance.items():
            targets = {
                target
                for parent in parents
                for target in self._symbol_paths.get(parent.rsplit(".", 1)[-1], set())
            }
            if targets:
                path_neighbors.setdefault(path, set()).update(targets)
                edges += len(targets)
        for region in self._regions:
            for call in region.calls:
                for target in self._symbol_paths.get(call, set()):
                    path_neighbors.setdefault(region.path, set()).add(target)
                    edges += 1

        frontier = set(seed_paths)
        for symbol in seed_symbols:
            frontier.update(self._symbol_paths.get(symbol, set()))
        seen = set(frontier)
        scored_paths: dict[str, float] = {}
        for hop in range(1, max_hops + 1):
            next_frontier: set[str] = set()
            hop_score = 1.5 / hop
            for path in sorted(frontier):
                for neighbor in sorted(path_neighbors.get(path, set())):
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    next_frontier.add(neighbor)
                    scored_paths[neighbor] = max(scored_paths.get(neighbor, 0.0), hop_score)
            frontier = next_frontier
            if not frontier:
                break

        result = [
            (region, scored_paths[region.path])
            for region in self._regions
            if region.path in scored_paths
        ]
        result.sort(key=lambda pair: (-pair[1], pair[0].path, pair[0].start_line))
        return result[:max_regions], edges

    def _semantic_score(self, query: str, candidates: Sequence[CodeRegion]) -> list[CodeRegion]:
        texts = [query, *[item.text for item in candidates]]
        vectors = self.router.embed(texts)
        if len(vectors) != len(texts) or not vectors or not vectors[0]:
            return list(candidates)
        q = vectors[0]
        result: list[CodeRegion] = []
        for item, vector in zip(candidates, vectors[1:], strict=True):
            result.append(_replace_score(item, semantic_score=max(0.0, _cosine(q, vector))))
        return result

    def _rerank(self, query: str, candidates: Sequence[CodeRegion]) -> list[CodeRegion]:
        scores = self.router.rerank(query, [item.text for item in candidates])
        if len(scores) != len(candidates):
            return list(candidates)
        return [
            _replace_score(item, reranker_score=float(score))
            for item, score in zip(candidates, scores, strict=True)
        ]


def classify_exploration_route(query: str, *, diagnostic_paths: Iterable[str] = ()) -> str:
    terms = set(_terms(query))
    if any(str(path).strip() for path in diagnostic_paths) or terms & _TRACE_HINTS:
        return "trace"
    if terms & _RIPPLE_HINTS:
        return "ripple"
    if terms & _API_HINTS:
        return "api"
    if terms & _PROCEDURAL_HINTS:
        return "procedural"
    if len(terms) > 18:
        return "global"
    return "lexical"


def _parse_java(path: str, text: str) -> tuple[list[_ParsedRegion], set[str], set[str]]:
    lines = text.splitlines()
    imports = {match.group(1) for match in _JAVA_IMPORT.finditer(text)}
    parents: set[str] = set()
    regions: list[_ParsedRegion] = []
    for match in _JAVA_TYPE.finditer(text):
        symbol = match.group(1)
        for group in match.groups()[1:]:
            if group:
                parents.update(token.strip() for token in re.split(r"[, ]+", group) if token.strip())
        start = text[: match.start()].count("\n") + 1
        end = _balanced_block_end(lines, start)
        kind_match = re.search(r"\b(class|interface|enum|record)\b", match.group(0))
        kind = kind_match.group(1) if kind_match else "class"
        regions.append(_make_region(path, symbol, kind, start, end, "\n".join(lines[start - 1 : end])))
    for match in _JAVA_METHOD.finditer(text):
        symbol = match.group(1)
        if symbol in {"if", "for", "while", "switch", "catch", "synchronized"}:
            continue
        start = text[: match.start()].count("\n") + 1
        end = _balanced_block_end(lines, start)
        regions.append(_make_region(path, symbol, "method", start, end, "\n".join(lines[start - 1 : end])))
    if not regions:
        regions.append(_make_region(path, Path(path).name, "file", 1, max(1, len(lines)), text))
    regions.sort(key=lambda r: (r.start_line, r.end_line, r.symbol))
    return regions, imports, parents


def _balanced_block_end(lines: Sequence[str], start_line: int) -> int:
    depth = 0
    opened = False
    for idx in range(max(0, start_line - 1), len(lines)):
        line = lines[idx]
        depth += line.count("{")
        if "{" in line:
            opened = True
        depth -= line.count("}")
        if opened and depth <= 0:
            return idx + 1
    return len(lines)


def _make_region(path: str, symbol: str, kind: str, start: int, end: int, text: str) -> _ParsedRegion:
    tokens = frozenset(_terms(text))
    calls = frozenset(
        name for name in _CALL.findall(text)
        if name not in {"if", "for", "while", "switch", "catch", "return", "new"}
    )
    procedure = frozenset(set(calls) | (set(tokens) & _CONTROL) | (set(tokens) & _PROCEDURAL_HINTS))
    api_tokens = frozenset(set(calls) | set(_terms(symbol)) | (set(tokens) & _API_HINTS))
    return _ParsedRegion(
        path=path,
        symbol=symbol,
        kind=kind,
        start_line=start,
        end_line=end,
        text=text,
        tokens=tokens,
        calls=calls,
        procedure=procedure,
        api_tokens=api_tokens,
    )


def _terms(value: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN.findall(str(value or "")))


def _meaningful_terms(value: str) -> tuple[str, ...]:
    stop = {"the", "and", "for", "with", "from", "this", "that", "into", "when", "where", "what"}
    return tuple(sorted({term for term in _terms(value) if term not in stop}))


def _query_procedure_terms(query_terms: set[str]) -> set[str]:
    return {
        term for term in query_terms
        if term in _PROCEDURAL_HINTS or term in _CONTROL or term in _API_HINTS
    } | query_terms


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    lnorm = math.sqrt(sum(float(a) * float(a) for a in left))
    rnorm = math.sqrt(sum(float(b) * float(b) for b in right))
    if lnorm <= 0 or rnorm <= 0:
        return 0.0
    return dot / (lnorm * rnorm)


def _region_key(region: CodeRegion) -> tuple[str, int, int]:
    return region.path, region.start_line, region.end_line


def _parsed_key(region: _ParsedRegion) -> tuple[str, int, int]:
    return region.path, region.start_line, region.end_line


def _replace_score(region: CodeRegion, **updates: float) -> CodeRegion:
    data = asdict(region)
    data.update(updates)
    return CodeRegion(**data)


def _clip_region(region: CodeRegion, line_budget: int) -> CodeRegion:
    lines = region.text.splitlines()
    clipped = "\n".join(lines[:line_budget])
    return CodeRegion(
        path=region.path,
        symbol=region.symbol,
        kind=region.kind,
        start_line=region.start_line,
        end_line=region.start_line + max(0, len(clipped.splitlines()) - 1),
        text=clipped,
        lexical_score=region.lexical_score,
        graph_score=region.graph_score,
        api_score=region.api_score,
        procedural_score=region.procedural_score,
        semantic_score=region.semantic_score,
        reranker_score=region.reranker_score,
        diagnostic_score=region.diagnostic_score,
    )


def exploration_fingerprint(result: ExplorationResult) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "CodeRegion",
    "ExplorationResult",
    "RepositoryExplorer",
    "classify_exploration_route",
    "exploration_fingerprint",
]
