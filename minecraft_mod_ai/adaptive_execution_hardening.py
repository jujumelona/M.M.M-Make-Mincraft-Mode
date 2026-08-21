from __future__ import annotations

import re
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Sequence

_EXPLORER_MARKER = "__mmm_capability_aware_explorer_v1__"
_SCALING_MARKER = "__mmm_budget_aware_test_time_scaling_v1__"
_RESEARCH_SEED_MARKER = "__mmm_positive_research_graph_seed_v2__"
_RESEARCH_METRIC_MARKER = "__mmm_complete_research_metric_vector_v1__"
_RESEARCH_IO_MARKER = "__mmm_repository_research_io_cache_v1__"
_DEPENDENCY_MARKER = "__mmm_dependency_byte_budget_v1__"
_CAMEL_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[0-9]+")


def harden_adaptive_execution() -> None:
    """Make retrieval and test-time scaling adaptive without adding duplicate owners."""
    _harden_repository_explorer()
    _harden_research_code_context()
    _harden_dependency_retrieval()
    _harden_test_time_scaling()


def _harden_repository_explorer() -> None:
    from .repository_explorer import RepositoryExplorer

    current = RepositoryExplorer.explore
    if getattr(current, _EXPLORER_MARKER, False):
        return

    @wraps(current)
    def explore(self: Any, query: str, *args: Any, **kwargs: Any):
        router = getattr(self, "router", None)
        if not _has_callable(router, "embed"):
            kwargs["semantic"] = False
        if not _has_callable(router, "rerank"):
            kwargs["rerank"] = False
        return current(self, query, *args, **kwargs)

    setattr(explore, _EXPLORER_MARKER, True)
    RepositoryExplorer.explore = explore


def _index_item(context: Any, path: str) -> Any | None:
    index = getattr(context, "_mmm_index_by_path", None)
    if not isinstance(index, dict):
        index = {
            str(item.path): item
            for item in getattr(getattr(context, "index", None), "files", ())
            if getattr(item, "path", None)
        }
        context._mmm_index_by_path = index
    return index.get(path)


def _source_lines(context: Any, path: str) -> tuple[str, ...]:
    cache = getattr(context, "_mmm_source_lines_by_path", None)
    if not isinstance(cache, dict):
        cache = {}
        context._mmm_source_lines_by_path = cache
    cached = cache.get(path)
    if isinstance(cached, tuple):
        return cached
    file_path = Path(context.root) / path
    lines = tuple(file_path.read_text(encoding="utf-8", errors="replace").splitlines())
    cache[path] = lines
    return lines


def _harden_research_code_context() -> None:
    from . import research_code_context as research

    current_filter = research.ResearchCodeContext._semantic_symbol_filter
    if not getattr(current_filter, _RESEARCH_SEED_MARKER, False):

        @wraps(current_filter)
        def semantic_symbol_filter(
            self: Any,
            query: str,
            symbols: Sequence[Any],
            *,
            limit: int,
        ) -> list[Any]:
            if not symbols:
                return []
            candidates = list(symbols)
            rerank = getattr(getattr(self, "router", None), "rerank", None)
            if callable(rerank):
                texts = [
                    research._join_query(
                        symbol.signature,
                        symbol.path,
                        self._symbol_text(symbol)[:3000],
                    )
                    for symbol in candidates
                ]
                try:
                    raw_scores = rerank(query, texts)
                    if len(raw_scores) == len(candidates):
                        scored = [
                            (float(score), symbol)
                            for score, symbol in zip(raw_scores, candidates, strict=True)
                        ]
                        positive = [item for item in scored if item[0] > 0.0]
                        if positive:
                            positive.sort(
                                key=lambda item: (-item[0], item[1].path, item[1].start_line)
                            )
                            return [symbol for _score, symbol in positive[:limit]]
                        query_tokens = research._tokens(query)
                        exact = [
                            symbol
                            for symbol in candidates
                            if query_tokens
                            & research._tokens(
                                research._join_query(symbol.name, symbol.signature, symbol.path)
                            )
                        ]
                        if exact:
                            return exact[:limit]
                        # No lexical seed exists: these are already structurally discovered
                        # graph neighbors, so preserve the real call/import edge instead of
                        # letting an all-zero reranker invent a global ordering.
                        return candidates[:limit]
                except Exception:
                    pass
            return current_filter(self, query, symbols, limit=limit)

        setattr(semantic_symbol_filter, _RESEARCH_SEED_MARKER, True)
        research.ResearchCodeContext._semantic_symbol_filter = semantic_symbol_filter

    current_symbol_text = research.ResearchCodeContext._symbol_text
    if not getattr(current_symbol_text, _RESEARCH_IO_MARKER, False):

        @wraps(current_symbol_text)
        def symbol_text(self: Any, symbol: Any) -> str:
            lines = _source_lines(self, symbol.path)
            start = max(0, int(symbol.start_line) - 1)
            end = min(len(lines), max(int(symbol.end_line), int(symbol.start_line)))
            return "\n".join(lines[start:end])

        setattr(symbol_text, _RESEARCH_IO_MARKER, True)
        research.ResearchCodeContext._symbol_text = symbol_text

    current_symbol_evidence = research.ResearchCodeContext._symbol_evidence
    if not getattr(current_symbol_evidence, _RESEARCH_IO_MARKER, False):

        @wraps(current_symbol_evidence)
        def symbol_evidence(
            self: Any,
            symbol: Any,
            *,
            query: str,
            graph_hop: int,
            target_plan: str = "",
        ):
            text = self._symbol_text(symbol)
            if not text.strip():
                return None
            indexed = _index_item(self, symbol.path)
            quality = research._quality(text, path=symbol.path)
            example_plan = research._code_plan(text)
            metrics = research._retrieval_metrics(
                query,
                text,
                path=symbol.path,
                symbols=(symbol.name,),
                graph_hop=graph_hop,
                quality=quality,
                target_plan=target_plan,
                example_plan=example_plan,
            )
            return research.Evidence(
                evidence_id=f"repo:{symbol.symbol_id}",
                source_type="repository_symbol",
                path=symbol.path,
                text=text,
                sha256=indexed.sha256 if indexed else research._sha(text),
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                symbols=(symbol.name,),
                metrics=metrics,
                quality=quality,
                bestfit_score=research._adaptive_score(query, metrics, quality),
                graph_hop=graph_hop,
                algorithmic_plan=example_plan,
            )

        setattr(symbol_evidence, _RESEARCH_IO_MARKER, True)
        research.ResearchCodeContext._symbol_evidence = symbol_evidence

    current_file_evidence = research.ResearchCodeContext._file_evidence
    if not getattr(current_file_evidence, _RESEARCH_IO_MARKER, False):

        @wraps(current_file_evidence)
        def file_evidence(
            self: Any,
            path: str,
            text: str,
            *,
            query: str,
            target_plan: str,
        ):
            indexed = _index_item(self, path)
            symbols = tuple(
                sorted(
                    set(research._TYPE.findall(text))
                    | {value[0] for value in research._METHOD.findall(text)}
                )
            )[:20]
            quality = research._quality(text, path=path)
            example_plan = research._code_plan(text)
            metrics = research._retrieval_metrics(
                query,
                text,
                path=path,
                symbols=symbols,
                graph_hop=None,
                quality=quality,
                target_plan=target_plan,
                example_plan=example_plan,
            )
            return research.Evidence(
                evidence_id=f"file:{path}:{indexed.sha256 if indexed else research._sha(text)}",
                source_type="repository_file",
                path=path,
                text=text,
                sha256=indexed.sha256 if indexed else research._sha(text),
                end_line=text.count("\n") + 1,
                symbols=symbols,
                metrics=metrics,
                quality=quality,
                bestfit_score=research._adaptive_score(query, metrics, quality),
                algorithmic_plan=example_plan,
            )

        setattr(file_evidence, _RESEARCH_IO_MARKER, True)
        research.ResearchCodeContext._file_evidence = file_evidence

    current_metrics = research._retrieval_metrics
    if not getattr(current_metrics, _RESEARCH_METRIC_MARKER, False):

        @wraps(current_metrics)
        def retrieval_metrics(
            query: str,
            text: str,
            *,
            path: str,
            symbols: Sequence[str],
            graph_hop: int | None,
            quality: Any,
            target_plan: str,
            example_plan: str,
        ) -> dict[str, float]:
            metrics = dict(
                current_metrics(
                    query,
                    text,
                    path=path,
                    symbols=symbols,
                    graph_hop=graph_hop,
                    quality=quality,
                    target_plan=target_plan,
                    example_plan=example_plan,
                )
            )
            metrics.setdefault(
                "plan_alignment",
                research._semantic_similarity(target_plan or query, example_plan or text),
            )
            return metrics

        setattr(retrieval_metrics, _RESEARCH_METRIC_MARKER, True)
        research._retrieval_metrics = retrieval_metrics

    current_weights = research._adaptive_weights
    if not getattr(current_weights, _RESEARCH_METRIC_MARKER, False):

        @wraps(current_weights)
        def adaptive_weights(
            query: str,
            metrics: Mapping[str, float] | None = None,
        ) -> dict[str, float]:
            weights = dict(current_weights(query, metrics))
            if "plan_alignment" not in weights:
                weights["plan_alignment"] = max(weights.values(), default=0.1)
                total = sum(max(0.0, float(value)) for value in weights.values())
                if total > 0:
                    weights = {
                        key: max(0.0, float(value)) / total
                        for key, value in weights.items()
                    }
            return weights

        setattr(adaptive_weights, _RESEARCH_METRIC_MARKER, True)
        research._adaptive_weights = adaptive_weights


def _dependency_tokens(research: Any, value: Any) -> set[str]:
    tokens = set(research._tokens(value))
    text = value if isinstance(value, str) else " ".join(str(item) for item in value)
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", text):
        tokens.update(part.casefold() for part in _CAMEL_PART.findall(token) if len(part) >= 2)
    return tokens


def _dependency_index(research: Any, reuse: Any, context: Any) -> dict[str, Any]:
    cached = getattr(context, reuse._DEPENDENCY_INDEX_ATTR, None)
    if isinstance(cached, dict) and cached.get("tokenization") == "camel-v2":
        return cached

    by_fqcn: dict[str, str] = {}
    by_type: dict[str, list[str]] = {}
    by_package: dict[str, list[str]] = {}
    token_to_paths: dict[str, set[str]] = {}
    contract_token_to_paths: dict[str, set[str]] = {}
    reverse_exact: dict[str, set[str]] = {}
    reverse_wildcard: dict[str, set[str]] = {}

    for path, unit in context.units.items():
        by_package.setdefault(unit.package, []).append(path)
        tokens = (
            _dependency_tokens(research, path)
            | _dependency_tokens(research, unit.package)
            | _dependency_tokens(research, unit.types)
        )
        for token in tokens:
            token_to_paths.setdefault(token, set()).add(path)
        for type_name in unit.types:
            fqcn = f"{unit.package}.{type_name}" if unit.package else type_name
            by_fqcn[fqcn] = path
            by_type.setdefault(type_name, []).append(path)
        if any(term in path.casefold() for term in ("contract", "interface", "/api/", "\\api\\")):
            for token in tokens | {"contract", "interface", "api"}:
                contract_token_to_paths.setdefault(token, set()).add(path)

    for path, unit in context.units.items():
        for imported in unit.imports:
            if imported.endswith(".*"):
                reverse_wildcard.setdefault(imported[:-2], set()).add(path)
            else:
                reverse_exact.setdefault(imported, set()).add(path)

    index = {
        "tokenization": "camel-v2",
        "by_fqcn": by_fqcn,
        "by_type": {key: tuple(sorted(value)) for key, value in by_type.items()},
        "by_package": {key: tuple(sorted(value)) for key, value in by_package.items()},
        "token_to_paths": {key: tuple(sorted(value)) for key, value in token_to_paths.items()},
        "contract_token_to_paths": {
            key: tuple(sorted(value)) for key, value in contract_token_to_paths.items()
        },
        "reverse_exact": {key: tuple(sorted(value)) for key, value in reverse_exact.items()},
        "reverse_wildcard": {
            key: tuple(sorted(value)) for key, value in reverse_wildcard.items()
        },
        "build_paths": tuple(
            sorted(
                indexed.path
                for indexed in context.index.files
                if Path(indexed.path).name in reuse._BUILD_NAMES
            )
        ),
    }
    setattr(context, reuse._DEPENDENCY_INDEX_ATTR, index)
    return index


def _bounded_paths(prefix: str, paths: Sequence[str], byte_budget: int) -> str:
    budget = max(512, int(byte_budget))
    selected: list[str] = []
    size = len(prefix.encode("utf-8"))
    for path in paths:
        encoded = len(path.encode("utf-8")) + 1
        if selected and size + encoded > budget:
            break
        if not selected and size + encoded > budget:
            return prefix
        selected.append(path)
        size += encoded
    return " ".join([prefix, *selected]).strip()


def _harden_dependency_retrieval() -> None:
    from . import research_code_context as research
    from . import research_coder_repair_reuse as reuse

    current_dependency = reuse._dependency_neighborhood_query
    if not getattr(current_dependency, _DEPENDENCY_MARKER, False):

        def dependency_neighborhood_query(
            research_module: Any,
            context: Any,
            query: str,
            plan_step: Any | None,
        ) -> str:
            query_tokens = _dependency_tokens(research_module, query)
            if plan_step is not None:
                query_tokens |= _dependency_tokens(
                    research_module, getattr(plan_step, "required_symbols", ())
                )
                query_tokens |= _dependency_tokens(
                    research_module, getattr(plan_step, "capability", "")
                )
            if not query_tokens:
                return ""

            index = _dependency_index(research_module, reuse, context)
            match_counts: dict[str, int] = {}
            for token in query_tokens:
                for path in index["token_to_paths"].get(token, ()):
                    match_counts[path] = match_counts.get(path, 0) + 1
            ranked_seeds = [
                path
                for path, _count in sorted(
                    match_counts.items(), key=lambda item: (-item[1], item[0])
                )
            ]
            if not ranked_seeds:
                return ""

            total_budget = max(1024, int(getattr(context, "byte_budget", 8192)) // 2)
            seed_budget = max(256, total_budget // 4)
            seed_text = _bounded_paths("", ranked_seeds, seed_budget)
            seeds = [value for value in seed_text.split() if value]
            if not seeds:
                return ""

            direct: set[str] = set()
            reverse: set[str] = set()
            seed_fqcns: set[str] = set()
            seed_packages: set[str] = set()
            for seed in seeds:
                unit = context.units.get(seed)
                if unit is None:
                    continue
                seed_packages.add(unit.package)
                for type_name in unit.types:
                    seed_fqcns.add(f"{unit.package}.{type_name}" if unit.package else type_name)
                for imported in unit.imports:
                    if imported.endswith(".*"):
                        direct.update(index["by_package"].get(imported[:-2], ()))
                        continue
                    target = index["by_fqcn"].get(imported)
                    if target:
                        direct.add(target)
                    else:
                        direct.update(index["by_type"].get(imported.rsplit(".", 1)[-1], ()))

            for fqcn in seed_fqcns:
                reverse.update(index["reverse_exact"].get(fqcn, ()))
            for package in seed_packages:
                reverse.update(index["reverse_wildcard"].get(package, ()))
            direct.difference_update(seeds)
            reverse.difference_update(seeds)

            contracts: set[str] = set()
            for token in query_tokens | {"contract", "interface", "api"}:
                contracts.update(index["contract_token_to_paths"].get(token, ()))
            contracts.difference_update(seeds)
            contracts.difference_update(direct)
            contracts.difference_update(reverse)

            wants_build = bool(
                query_tokens
                & {
                    "dependency",
                    "dependencies",
                    "import",
                    "api",
                    "registry",
                    "build",
                    "gradle",
                    "fabric",
                }
            )
            ordered = list(
                dict.fromkeys(
                    [
                        *seeds,
                        *sorted(direct),
                        *sorted(reverse),
                        *sorted(contracts),
                        *(index["build_paths"] if wants_build else ()),
                    ]
                )
            )
            if len(ordered) <= len(seeds):
                return ""
            prefix = "repository dependency neighborhood direct reverse shared contracts"
            return _bounded_paths(prefix, ordered, total_budget)

        setattr(dependency_neighborhood_query, _DEPENDENCY_MARKER, True)
        dependency_neighborhood_query.__wrapped__ = current_dependency  # type: ignore[attr-defined]
        reuse._dependency_neighborhood_query = dependency_neighborhood_query

    current_paths = research.ResearchCodeContext._query_paths
    if not getattr(current_paths, _DEPENDENCY_MARKER, False):

        @wraps(current_paths)
        def query_paths(self: Any, query: str, plan_step: Any | None):
            paths = list(dict.fromkeys(current_paths(self, query, plan_step)))
            dependency = next(
                (path for path in paths if "repository dependency neighborhood" in path),
                "",
            )
            if not dependency:
                dependency = reuse._dependency_neighborhood_query(
                    research, self, query, plan_step
                )
            if dependency:
                paths = [
                    path
                    for path in paths
                    if path != dependency and "known repository vocabulary" not in path
                ]
                paths.insert(1 if paths else 0, dependency)
            return tuple(dict.fromkeys(paths))

        setattr(query_paths, _DEPENDENCY_MARKER, True)
        query_paths.__wrapped__ = current_paths  # type: ignore[attr-defined]
        research.ResearchCodeContext._query_paths = query_paths


def _harden_test_time_scaling() -> None:
    from . import inference_time_scaling

    current = inference_time_scaling._scaling_mode
    if getattr(current, _SCALING_MARKER, False):
        return

    @wraps(current)
    def scaling_mode() -> str:
        mode = str(current())
        if mode != "auto":
            return mode
        try:
            from .llama_parallel_runtime_contract import _active_parallelism

            slots = int(_active_parallelism())
        except Exception:
            slots = 1
        return "auto" if slots > 1 else "off"

    setattr(scaling_mode, _SCALING_MARKER, True)
    inference_time_scaling._scaling_mode = scaling_mode


def _has_callable(owner: Any, name: str) -> bool:
    if owner is None:
        return False
    try:
        value = getattr(owner, name)
    except (AttributeError, RuntimeError, TypeError):
        return False
    return callable(value)


__all__ = ["harden_adaptive_execution"]
