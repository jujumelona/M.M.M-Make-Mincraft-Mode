from __future__ import annotations

"""Repository-aware research orchestration for production code generation.

This module adapts eight complementary retrieval/code-generation research directions
into one host-owned runtime:

* CAPIR / PERC: decompose a coarse module into implementation subtasks and retrieve
  against algorithmic plan steps rather than only the raw request.
* RepoCoder / EvoR: evolve queries and the evidence set from each generated draft
  until the evidence state reaches a fixed point.
* CoRet / DyRetriever: combine code semantics, repository hierarchy and an on-demand
  partial dependency/call graph rather than maintaining a global call graph.
* CodeRAG / AIRCoder: multi-path retrieval with query-adaptive metric fusion and a
  final BestFit reranking stage aligned with the configured coder stack.
* RAR / DocPrompting: retrieve target-scoped documentation and repository examples
  in a coupled two-step loop.
* CoQuIR: score correctness, efficiency, security and maintainability explicitly.
* Example Quality Matters: prefer semantically correct, readable, moderate-complexity
  stepwise examples and combine complementary examples across plan steps.
* PackMonitor: admit Gradle/Maven package names only from finite host-owned
  authoritative sets; generated patches cannot introduce unknown package names.

All retrieved source remains data, never execution authority. Source excerpts are
bound to path/SHA receipts supplied by ProjectIndex.
"""

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .platform_catalog import adapter_for_target
from .project_index import ProjectIndex

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.$:+/-]{1,127}|[가-힣]{2,}")
_IDENTIFIER = re.compile(r"\b[A-Za-z_$][A-Za-z0-9_$]*\b")
_PACKAGE = re.compile(r"^\s*package\s+([A-Za-z0-9_.]+)\s*;", re.MULTILINE)
_IMPORT = re.compile(r"^\s*import\s+(?:static\s+)?([A-Za-z0-9_.*]+)\s*;", re.MULTILINE)
_TYPE = re.compile(r"\b(?:class|interface|record|enum)\s+([A-Za-z_$][A-Za-z0-9_$]*)")
_METHOD = re.compile(
    r"(?m)^[ \t]*(?:public|protected|private|static|final|abstract|synchronized|native|default|\s)+"
    r"(?:<[^>{};]+>\s*)?(?:[A-Za-z_$][A-Za-z0-9_.$<>\[\], ?]*\s+)"
    r"([A-Za-z_$][A-Za-z0-9_$]*)\s*\(([^;{}()]*)\)\s*(?:throws\s+[^{]+)?\{"
)
_CALL = re.compile(r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
_MAVEN_COORD = re.compile(
    r"(?<![A-Za-z0-9_.-])([A-Za-z0-9_.-]+):([A-Za-z0-9_.-]+):([^'\"\s)]+)"
)
_PLUGIN_ID = re.compile(r"\bid\s*(?:\(\s*)?['\"]([A-Za-z0-9_.-]+)['\"]\s*\)?")
_PROPERTY_ASSIGN = re.compile(r"(?m)^\s*([A-Za-z0-9_.-]+)\s*=\s*([^\s#]+)\s*$")
_DANGEROUS = (
    "runtime.getruntime().exec",
    "processbuilder(",
    "objectinputstream(",
    "setaccessible(true)",
    "scriptengine",
)
_KEYWORDS = frozenset(
    {
        "if", "for", "while", "switch", "catch", "return", "new", "throw", "super",
        "this", "synchronized", "assert", "try", "do", "else", "case", "default",
    }
)


def _tokens(value: Any) -> set[str]:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return {token.casefold() for token in _TOKEN.findall(text)}


def _sha(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )


def _env_int(name: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(low, min(high, value))


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    action: str
    algorithmic_plan: str
    query: str
    required_symbols: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "algorithmic_plan": self.algorithmic_plan,
            "query": self.query,
            "required_symbols": list(self.required_symbols),
        }


@dataclass(frozen=True)
class QualityVector:
    correctness: float
    efficiency: float
    security: float
    maintainability: float
    complexity_fit: float
    readability: float

    @property
    def coqu_ir(self) -> float:
        return (
            self.correctness
            + self.efficiency
            + self.security
            + self.maintainability
        ) / 4.0

    @property
    def example_quality(self) -> float:
        return (
            0.45 * self.correctness
            + 0.30 * self.readability
            + 0.25 * self.complexity_fit
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "correctness": round(self.correctness, 6),
            "efficiency": round(self.efficiency, 6),
            "security": round(self.security, 6),
            "maintainability": round(self.maintainability, 6),
            "complexity_fit": round(self.complexity_fit, 6),
            "readability": round(self.readability, 6),
            "coquir": round(self.coqu_ir, 6),
            "example_quality": round(self.example_quality, 6),
        }


@dataclass(frozen=True)
class SourceSymbol:
    name: str
    path: str
    package: str
    kind: str
    start_line: int
    end_line: int
    signature: str

    @property
    def symbol_id(self) -> str:
        return f"{self.path}:{self.start_line}:{self.name}"


@dataclass
class Evidence:
    evidence_id: str
    source_type: str
    path: str
    text: str
    sha256: str
    start_line: int = 1
    end_line: int = 1
    symbols: tuple[str, ...] = ()
    plan_steps: set[str] = field(default_factory=set)
    metrics: dict[str, float] = field(default_factory=dict)
    quality: QualityVector | None = None
    bestfit_score: float = 0.0
    graph_hop: int | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_type": self.source_type,
            "path": self.path,
            "sha256": self.sha256,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbols": list(self.symbols),
            "plan_steps": sorted(self.plan_steps),
            "metrics": {key: round(value, 6) for key, value in sorted(self.metrics.items())},
            "quality": self.quality.to_dict() if self.quality else None,
            "bestfit_score": round(self.bestfit_score, 6),
            "graph_hop": self.graph_hop,
            "text": self.text,
        }


@dataclass(frozen=True)
class DependencyViolation:
    kind: str
    value: str
    path: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "value": self.value, "path": self.path}


@dataclass
class _JavaUnit:
    path: str
    package: str
    imports: tuple[str, ...]
    types: tuple[str, ...]
    methods: tuple[SourceSymbol, ...]


class DependencyMonitor:
    """Finite authoritative package-name admission for generated Gradle patches."""

    def __init__(
        self,
        project_root: Path,
        *,
        minecraft_version: str,
        loader: str,
        mappings: str,
    ) -> None:
        self.root = project_root
        self.minecraft_version = str(minecraft_version).strip()
        self.loader = str(loader).strip().casefold()
        self.mappings = str(mappings).strip()
        self.allowed_packages: set[str] = set()
        self.allowed_plugins: set[str] = {"java", "java-library", "maven-publish"}
        self.allowed_repositories: set[str] = {
            "https://maven.fabricmc.net/",
            "https://maven.fabricmc.net",
            "https://maven.neoforged.net/releases",
            "https://maven.minecraftforge.net/",
            "https://repo.maven.apache.org/maven2",
        }
        self._seed_platform()
        self._seed_existing_files()

    def _seed_platform(self) -> None:
        adapter = adapter_for_target(self.minecraft_version, self.loader)
        if adapter.loader == "fabric":
            self.allowed_packages.update(
                {
                    "net.fabricmc:fabric-loader",
                    "net.fabricmc.fabric-api:fabric-api",
                    "net.fabricmc:yarn",
                }
            )
            self.allowed_plugins.add("fabric-loom")

    def _seed_existing_files(self) -> None:
        for name in (
            "build.gradle",
            "build.gradle.kts",
            "settings.gradle",
            "settings.gradle.kts",
            "gradle.properties",
        ):
            path = self.root / name
            if not path.is_file() or path.is_symlink():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            self.admit_text(text, code_owned=True)

    def admit_text(self, text: str, *, code_owned: bool) -> None:
        if not code_owned:
            return
        for group, artifact, _version in _MAVEN_COORD.findall(text):
            self.allowed_packages.add(f"{group}:{artifact}")
        for plugin in _PLUGIN_ID.findall(text):
            self.allowed_plugins.add(plugin)

    def admit_research_context(self, value: Any) -> None:
        """Admit only coordinates from code-owned research/grounding envelopes."""

        if not isinstance(value, Mapping):
            return
        for key in ("research_context", "host_grounding", "technology_radar"):
            payload = value.get(key)
            if payload is not None:
                self.admit_text(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
                    code_owned=True,
                )

    def _operation_text(self, operation: Mapping[str, Any]) -> str:
        pieces: list[str] = []
        content = operation.get("content")
        if isinstance(content, str):
            pieces.append(content)
        replacements = operation.get("replacements")
        if isinstance(replacements, list):
            for item in replacements:
                if isinstance(item, Mapping):
                    for key in ("old", "new", "replacement", "value"):
                        value = item.get(key)
                        if isinstance(value, str):
                            pieces.append(value)
        return "\n".join(pieces)

    def validate_model_output(self, text: str) -> tuple[DependencyViolation, ...]:
        payload = _extract_json_object(text)
        if not payload:
            return ()
        operations = payload.get("operations")
        if not isinstance(operations, list):
            operations = payload.get("patch_operations")
        if not isinstance(operations, list):
            operations = payload.get("patches")
        if not isinstance(operations, list):
            return ()

        violations: list[DependencyViolation] = []
        for raw in operations:
            if not isinstance(raw, Mapping):
                continue
            path = str(raw.get("path", "")).replace("\\", "/")
            leaf = Path(path).name.casefold()
            if leaf not in {
                "build.gradle",
                "build.gradle.kts",
                "settings.gradle",
                "settings.gradle.kts",
                "gradle.properties",
            }:
                continue
            body = self._operation_text(raw)
            for group, artifact, _version in _MAVEN_COORD.findall(body):
                package = f"{group}:{artifact}"
                if package not in self.allowed_packages:
                    violations.append(
                        DependencyViolation("package", package, path)
                    )
            for plugin in _PLUGIN_ID.findall(body):
                if plugin not in self.allowed_plugins:
                    violations.append(
                        DependencyViolation("plugin", plugin, path)
                    )
            if leaf == "gradle.properties":
                expected = self._expected_properties()
                for key, value in _PROPERTY_ASSIGN.findall(body):
                    if key in expected and value != expected[key]:
                        violations.append(
                            DependencyViolation(
                                "target_property",
                                f"{key}={value}",
                                path,
                            )
                        )
        unique = {
            (item.kind, item.value, item.path): item for item in violations
        }
        return tuple(unique[key] for key in sorted(unique))

    def _expected_properties(self) -> dict[str, str]:
        adapter = adapter_for_target(self.minecraft_version, self.loader)
        expected = {"minecraft_version": adapter.minecraft_version}
        if adapter.loader == "fabric":
            expected.update(
                {
                    "loader_version": adapter.fabric_loader,
                    "fabric_version": adapter.fabric_api,
                    "loom_version": adapter.fabric_loom,
                }
            )
            if adapter.yarn_mappings != "mojang":
                expected["yarn_mappings"] = adapter.yarn_mappings
        return expected

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/dependency-monitor-v1",
            "policy": "finite_authoritative_package_admission",
            "target": {
                "minecraft_version": self.minecraft_version,
                "loader": self.loader,
                "mappings": self.mappings,
            },
            "allowed_packages": sorted(self.allowed_packages),
            "allowed_plugins": sorted(self.allowed_plugins),
            "allowed_packages_sha256": _sha(sorted(self.allowed_packages)),
            "zero_unknown_packages_in_accepted_patch": True,
        }


class ResearchCodeContext:
    """Build and evolve a bounded research context for one custom-module generation."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        project_index: ProjectIndex,
        router: Any,
        module: Any,
        minecraft_version: str,
        loader: str,
        mappings: str,
        byte_budget: int = 16 * 1024,
    ) -> None:
        self.root = Path(project_root).expanduser().resolve()
        self.index = project_index
        self.router = router
        self.module = module
        self.minecraft_version = str(minecraft_version).strip()
        self.loader = str(loader).strip().casefold()
        self.mappings = str(mappings).strip()
        if not self.minecraft_version or not self.loader or not self.mappings:
            raise ValueError(
                "ResearchCodeContext requires host-selected version, loader and mappings."
            )
        adapter_for_target(self.minecraft_version, self.loader)
        self.byte_budget = max(4096, int(byte_budget))
        self.graph_budget = _env_int("MMM_CODE_RESEARCH_GRAPH_NODES", 64, 8, 512)
        self.plan = _build_plan(module)
        self.units, self.symbols_by_name = self._index_repository_structure()
        self.evidence: dict[str, Evidence] = {}
        self.query_history: list[str] = []
        self.query_seen: set[str] = set()
        self.rounds: list[dict[str, Any]] = []
        self.monitor = DependencyMonitor(
            self.root,
            minecraft_version=self.minecraft_version,
            loader=self.loader,
            mappings=self.mappings,
        )
        self._last_bundle_sha = ""
        self._initial_complete = False

    def ingest_code_owned_request(self, messages: Sequence[Mapping[str, Any]]) -> None:
        """Read only known host envelopes to extend the dependency admission set."""

        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.lstrip().startswith("{"):
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                self.monitor.admit_research_context(payload)

    def initial_bundle(self) -> dict[str, Any]:
        if not self._initial_complete:
            for step in self.plan:
                self._research_step(step, reason="initial_plan")
            self._initial_complete = True
        return self.bundle()

    def evolve_from_generation(self, text: str) -> tuple[dict[str, Any] | None, tuple[DependencyViolation, ...]]:
        violations = self.monitor.validate_model_output(text)
        draft = _extract_json_object(text)
        evolution_queries = _draft_queries(draft)
        added_before = len(self.evidence)
        executed: list[str] = []
        for query in evolution_queries:
            if self._run_query(query, plan_step=None, reason="draft_evolution"):
                executed.append(query)
        added = len(self.evidence) - added_before
        self.rounds.append(
            {
                "round_index": len(self.rounds),
                "trigger": "generated_draft",
                "queries": executed,
                "new_evidence": added,
                "dependency_violations": [item.to_dict() for item in violations],
            }
        )
        if violations or added:
            return self.bundle(delta_only=False), violations
        return None, ()

    def evolve_from_failure(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
        tail = " ".join(
            str(message.get("content", ""))
            for message in messages[-3:]
            if isinstance(message.get("content"), str)
        )
        queries = _failure_queries(tail)
        added_before = len(self.evidence)
        executed = [
            query
            for query in queries
            if self._run_query(query, plan_step=None, reason="validation_failure")
        ]
        added = len(self.evidence) - added_before
        if not added:
            return None
        self.rounds.append(
            {
                "round_index": len(self.rounds),
                "trigger": "validation_failure",
                "queries": executed,
                "new_evidence": added,
            }
        )
        return self.bundle(delta_only=False)

    def _research_step(self, step: PlanStep, *, reason: str) -> None:
        # RAR/DocPrompting pass 1: documentation from the plan query.
        docs = self._retrieve_docs(step.query, step_id=step.step_id)
        doc_terms = _salient_terms(
            " ".join(item.text for item in docs),
            exclude=_tokens(step.query),
            limit=12,
        )
        example_query = " ".join([step.query, *doc_terms]).strip()
        # CAPIR/PERC + CoRet/DyRetriever + CodeRAG/AIRCoder example pass.
        examples = self._retrieve_repo_examples(
            example_query,
            plan_step=step,
        )
        # RAR pass 2: selected example symbols refine the documentation query.
        example_symbols = sorted(
            {
                symbol
                for item in examples[:6]
                for symbol in item.symbols
                if symbol
            }
        )[:12]
        if example_symbols:
            self._retrieve_docs(
                " ".join([step.query, *example_symbols]),
                step_id=step.step_id,
            )
        self.query_history.append(step.query)
        self.query_seen.add(step.query.casefold())
        self.rounds.append(
            {
                "round_index": len(self.rounds),
                "trigger": reason,
                "plan_step": step.step_id,
                "query": step.query,
                "docs": len(docs),
                "examples": len(examples),
            }
        )

    def _run_query(
        self,
        query: str,
        *,
        plan_step: PlanStep | None,
        reason: str,
    ) -> bool:
        normalized = " ".join(str(query).split())
        key = normalized.casefold()
        if len(normalized) < 2 or key in self.query_seen:
            return False
        self.query_seen.add(key)
        self.query_history.append(normalized)
        before = len(self.evidence)
        docs = self._retrieve_docs(
            normalized,
            step_id=plan_step.step_id if plan_step else "",
        )
        doc_terms = _salient_terms(
            " ".join(item.text for item in docs),
            exclude=_tokens(normalized),
            limit=10,
        )
        self._retrieve_repo_examples(
            " ".join([normalized, *doc_terms]),
            plan_step=plan_step,
        )
        self.rounds.append(
            {
                "round_index": len(self.rounds),
                "trigger": reason,
                "query": normalized,
                "new_evidence": len(self.evidence) - before,
            }
        )
        return len(self.evidence) > before

    def _retrieve_docs(self, query: str, *, step_id: str) -> list[Evidence]:
        try:
            from .retrieval import retrieve_official_evidence

            receipt = retrieve_official_evidence(
                query,
                minecraft_version=self.minecraft_version,
                loader=self.loader,
                mappings=self.mappings,
                limit=6,
            )
            payload = receipt.to_dict()
        except Exception as exc:
            self.rounds.append(
                {
                    "round_index": len(self.rounds),
                    "trigger": "official_docs_unavailable",
                    "query": query,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return []

        result: list[Evidence] = []
        hits = payload.get("hits")
        if not isinstance(hits, list):
            return result
        for hit in hits:
            if not isinstance(hit, Mapping):
                continue
            document_id = str(hit.get("document_id", ""))
            text = str(hit.get("excerpt", ""))
            if not document_id or not text:
                continue
            evidence_id = str(hit.get("evidence_id") or f"doc:{document_id}")
            item = Evidence(
                evidence_id=evidence_id,
                source_type="official_documentation",
                path=document_id,
                text=text,
                sha256=str(hit.get("content_sha256", "")),
                symbols=tuple(_salient_terms(text, exclude=set(), limit=12)),
                metrics={
                    "retrieval_score": float(hit.get("score", 0.0) or 0.0),
                    "coverage": float(payload.get("coverage", 0.0) or 0.0),
                },
            )
            if step_id:
                item.plan_steps.add(step_id)
            self._merge_evidence(item)
            result.append(item)
        return result

    def _retrieve_repo_examples(
        self,
        query: str,
        *,
        plan_step: PlanStep | None,
    ) -> list[Evidence]:
        entries = self._entry_points(query)
        graph = self._expand_partial_graph(entries)
        candidates: list[Evidence] = []
        for symbol, hop in graph:
            item = self._symbol_evidence(symbol, query=query, graph_hop=hop)
            if item is None:
                continue
            if plan_step is not None:
                item.plan_steps.add(plan_step.step_id)
            candidates.append(item)

        # Add ProjectIndex's exact relevance lane so non-Java resources/build files
        # remain retrievable alongside method-level structural evidence.
        try:
            selected = self.index.select(
                query=query,
                byte_budget=min(self.byte_budget, 12 * 1024),
            )
        except Exception:
            selected = {"files": []}
        for raw in selected.get("files", []):
            if not isinstance(raw, Mapping):
                continue
            text = str(raw.get("content", ""))
            path = str(raw.get("path", ""))
            if not text or not path:
                continue
            item = self._file_evidence(path, text, query=query)
            if plan_step is not None:
                item.plan_steps.add(plan_step.step_id)
            candidates.append(item)

        candidates = _dedupe_evidence(candidates)
        self._apply_bestfit_rerank(query, candidates)
        candidates.sort(
            key=lambda item: (
                -item.bestfit_score,
                -(item.quality.example_quality if item.quality else 0.0),
                item.path,
                item.start_line,
            )
        )
        selected_examples = _combine_examples(candidates, limit=10)
        for item in selected_examples:
            self._merge_evidence(item)
        return selected_examples

    def _index_repository_structure(
        self,
    ) -> tuple[dict[str, _JavaUnit], dict[str, list[SourceSymbol]]]:
        units: dict[str, _JavaUnit] = {}
        by_name: dict[str, list[SourceSymbol]] = {}
        for indexed in self.index.files:
            if not indexed.path.endswith(".java"):
                continue
            path = self.root / indexed.path
            if not path.is_file() or path.is_symlink():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            package_match = _PACKAGE.search(text)
            package = package_match.group(1) if package_match else ""
            imports = tuple(sorted(set(_IMPORT.findall(text))))
            types = tuple(sorted(set(_TYPE.findall(text))))
            methods: list[SourceSymbol] = []
            line_starts = _line_offsets(text)
            for match in _METHOD.finditer(text):
                name = match.group(1)
                if name in _KEYWORDS:
                    continue
                open_brace = text.find("{", match.start())
                end_offset = _matching_brace(text, open_brace)
                start_line = _offset_line(line_starts, match.start())
                end_line = _offset_line(
                    line_starts,
                    end_offset if end_offset >= 0 else match.end(),
                )
                signature = " ".join(match.group(0).split())[:512]
                symbol = SourceSymbol(
                    name=name,
                    path=indexed.path,
                    package=package,
                    kind="method",
                    start_line=start_line,
                    end_line=max(start_line, end_line),
                    signature=signature,
                )
                methods.append(symbol)
                by_name.setdefault(name, []).append(symbol)
            for type_name in types:
                symbol = SourceSymbol(
                    name=type_name,
                    path=indexed.path,
                    package=package,
                    kind="type",
                    start_line=1,
                    end_line=1,
                    signature=f"{package}.{type_name}" if package else type_name,
                )
                by_name.setdefault(type_name, []).append(symbol)
            units[indexed.path] = _JavaUnit(
                path=indexed.path,
                package=package,
                imports=imports,
                types=types,
                methods=tuple(methods),
            )
        return units, by_name

    def _entry_points(self, query: str) -> list[SourceSymbol]:
        query_tokens = _tokens(query)
        ranked: list[tuple[float, SourceSymbol]] = []
        for unit in self.units.values():
            path_tokens = _tokens(unit.path)
            import_tokens = _tokens(unit.imports)
            package_tokens = _tokens(unit.package)
            for symbol in unit.methods:
                symbol_tokens = _tokens(symbol.name) | _tokens(symbol.signature)
                lexical = _overlap(query_tokens, symbol_tokens)
                path_score = _overlap(query_tokens, path_tokens)
                dependency = _overlap(query_tokens, import_tokens)
                structure = _overlap(query_tokens, package_tokens)
                score = (
                    0.42 * lexical
                    + 0.24 * path_score
                    + 0.20 * dependency
                    + 0.14 * structure
                )
                if score > 0:
                    ranked.append((score, symbol))
        ranked.sort(key=lambda item: (-item[0], item[1].path, item[1].start_line))
        width = min(self.graph_budget, max(4, len(self.plan) * 3))
        return [symbol for _score, symbol in ranked[:width]]

    def _expand_partial_graph(
        self,
        entries: Sequence[SourceSymbol],
    ) -> list[tuple[SourceSymbol, int]]:
        """Build only the dependency graph reachable from selected entry points."""

        queue: list[tuple[SourceSymbol, int]] = [(item, 0) for item in entries]
        seen: set[str] = set()
        result: list[tuple[SourceSymbol, int]] = []
        while queue and len(seen) < self.graph_budget:
            symbol, hop = queue.pop(0)
            if symbol.symbol_id in seen:
                continue
            seen.add(symbol.symbol_id)
            result.append((symbol, hop))
            if hop >= 2 or symbol.kind != "method":
                continue
            body = self._symbol_text(symbol)
            calls = [
                name
                for name in _CALL.findall(body)
                if name not in _KEYWORDS and name != symbol.name
            ]
            for name in calls:
                for target in self.symbols_by_name.get(name, ()):
                    if target.symbol_id not in seen:
                        queue.append((target, hop + 1))
            unit = self.units.get(symbol.path)
            if unit is not None:
                for imported in unit.imports:
                    leaf = imported.rsplit(".", 1)[-1]
                    for target in self.symbols_by_name.get(leaf, ()):
                        if target.symbol_id not in seen:
                            queue.append((target, hop + 1))
        return result

    def _symbol_text(self, symbol: SourceSymbol) -> str:
        path = self.root / symbol.path
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, symbol.start_line - 1)
        end = min(len(lines), max(symbol.end_line, symbol.start_line))
        return "\n".join(lines[start:end])

    def _symbol_evidence(
        self,
        symbol: SourceSymbol,
        *,
        query: str,
        graph_hop: int,
    ) -> Evidence | None:
        text = self._symbol_text(symbol)
        if not text.strip():
            return None
        indexed = next(
            (item for item in self.index.files if item.path == symbol.path),
            None,
        )
        sha256 = indexed.sha256 if indexed is not None else _sha(text)
        quality = _quality(text, path=symbol.path)
        metrics = _retrieval_metrics(
            query,
            text,
            path=symbol.path,
            symbols=(symbol.name,),
            graph_hop=graph_hop,
            quality=quality,
        )
        score = _adaptive_score(query, metrics, quality)
        return Evidence(
            evidence_id=f"repo:{symbol.symbol_id}",
            source_type="repository_symbol",
            path=symbol.path,
            text=text,
            sha256=sha256,
            start_line=symbol.start_line,
            end_line=symbol.end_line,
            symbols=(symbol.name,),
            metrics=metrics,
            quality=quality,
            bestfit_score=score,
            graph_hop=graph_hop,
        )

    def _file_evidence(self, path: str, text: str, *, query: str) -> Evidence:
        indexed = next((item for item in self.index.files if item.path == path), None)
        sha256 = indexed.sha256 if indexed is not None else _sha(text)
        type_names = set(_TYPE.findall(text))
        method_names = {value[0] for value in _METHOD.findall(text)}
        flattened_symbols = tuple(sorted(type_names | method_names))[:16]
        quality = _quality(text, path=path)
        metrics = _retrieval_metrics(
            query,
            text,
            path=path,
            symbols=flattened_symbols,
            graph_hop=None,
            quality=quality,
        )
        score = _adaptive_score(query, metrics, quality)
        lines = text.count("\n") + 1
        return Evidence(
            evidence_id=f"file:{path}:{sha256}",
            source_type="repository_file",
            path=path,
            text=text,
            sha256=sha256,
            start_line=1,
            end_line=lines,
            symbols=flattened_symbols,
            metrics=metrics,
            quality=quality,
            bestfit_score=score,
        )

    def _apply_bestfit_rerank(self, query: str, items: list[Evidence]) -> None:
        if not items:
            return
        candidates = sorted(
            items,
            key=lambda item: (-item.bestfit_score, item.path, item.start_line),
        )[:24]
        try:
            scores = self.router.rerank(
                query,
                [item.text[:6000] for item in candidates],
            )
        except Exception:
            return
        if len(scores) != len(candidates):
            return
        numeric = [float(value) for value in scores]
        low = min(numeric)
        high = max(numeric)
        span = high - low
        for item, value in zip(candidates, numeric, strict=True):
            aligned = 0.5 if span <= 1e-12 else (value - low) / span
            item.metrics["preference_reranker"] = aligned
            item.bestfit_score = 0.72 * item.bestfit_score + 0.28 * aligned

    def _merge_evidence(self, incoming: Evidence) -> None:
        current = self.evidence.get(incoming.evidence_id)
        if current is None:
            self.evidence[incoming.evidence_id] = incoming
            return
        current.plan_steps.update(incoming.plan_steps)
        current.metrics.update(incoming.metrics)
        current.bestfit_score = max(current.bestfit_score, incoming.bestfit_score)

    def bundle(self, *, delta_only: bool = False) -> dict[str, Any]:
        del delta_only
        examples = [
            item
            for item in self.evidence.values()
            if item.source_type.startswith("repository_")
        ]
        docs = [
            item
            for item in self.evidence.values()
            if item.source_type == "official_documentation"
        ]
        examples.sort(
            key=lambda item: (
                -item.bestfit_score,
                -(item.quality.example_quality if item.quality else 0.0),
                item.path,
                item.start_line,
            )
        )
        docs.sort(
            key=lambda item: (
                -float(item.metrics.get("retrieval_score", 0.0)),
                item.path,
            )
        )
        selected: list[dict[str, Any]] = []
        base = {
            "schema_version": "mmm/research-code-context-v1",
            "target": {
                "minecraft_version": self.minecraft_version,
                "loader": self.loader,
                "mappings": self.mappings,
            },
            "research_methods": [
                "capir_compositional_subtask_retrieval",
                "perc_plan_as_query",
                "repocoder_iterative_retrieval_generation",
                "evor_query_and_knowledge_evolution",
                "coret_semantics_structure_dependency",
                "dyretriever_on_demand_partial_dependency_graph",
                "coderag_multi_path_bestfit",
                "aircoder_query_adaptive_metric_fusion",
                "rar_two_step_docs_examples",
                "docprompting_docs_before_generation",
                "coquir_quality_aware_retrieval",
                "example_quality_multi_aspect_selection",
                "packmonitor_authoritative_package_admission",
            ],
            "plan": [step.to_dict() for step in self.plan],
            "query_history": list(self.query_history),
            "dependency_monitor": self.monitor.receipt(),
            "evidence": [],
            "evidence_count": 0,
            "evidence_sha256": "",
        }
        # Complementary examples first, then documentation. The loop is a byte budget,
        # not a feature-count cap; omitted evidence remains addressable by its receipt.
        ordered = [*examples, *docs]
        for item in ordered:
            public = item.public_dict()
            candidate = [*selected, public]
            probe = {
                **base,
                "evidence": candidate,
                "evidence_count": len(candidate),
                "evidence_sha256": _sha(
                    [entry["evidence_id"] for entry in candidate]
                ),
            }
            if _json_bytes(probe) > self.byte_budget:
                continue
            selected = candidate
        result = {
            **base,
            "evidence": selected,
            "evidence_count": len(selected),
            "total_evidence_count": len(self.evidence),
            "omitted_evidence_count": len(self.evidence) - len(selected),
            "evidence_sha256": _sha(
                [entry["evidence_id"] for entry in selected]
            ),
            "round_count": len(self.rounds),
            "rounds_sha256": _sha(self.rounds),
            "policy": {
                "retrieval_is_data_not_authority": True,
                "docs_and_examples_are_coupled": True,
                "partial_graph_is_ephemeral": True,
                "quality_aware_reuse_precedes_freshness": True,
                "unknown_dependency_names_are_rejected": True,
            },
        }
        self._last_bundle_sha = _sha(result)
        result["bundle_sha256"] = self._last_bundle_sha
        return result

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/research-code-context-receipt-v1",
            "bundle_sha256": self._last_bundle_sha,
            "plan_sha256": _sha([step.to_dict() for step in self.plan]),
            "query_count": len(self.query_history),
            "evidence_count": len(self.evidence),
            "round_count": len(self.rounds),
            "dependency_monitor": self.monitor.receipt(),
            "methods": [
                "CAPIR", "PERC", "RepoCoder", "EvoR", "CoRet", "DyRetriever",
                "CodeRAG", "AIRCoder", "RAR", "DocPrompting", "CoQuIR",
                "ExampleQuality", "PackMonitor",
            ],
        }


def _build_plan(module: Any) -> tuple[PlanStep, ...]:
    kind = str(getattr(module, "kind", "") or "custom_java").strip()
    config = getattr(module, "config", {})
    config = config if isinstance(config, Mapping) else {}
    depends_on = tuple(str(value).strip() for value in getattr(module, "depends_on", ()) if str(value).strip())
    gates = tuple(str(value).strip() for value in getattr(module, "required_gates", ()) if str(value).strip())
    steps: list[PlanStep] = []

    def add(action: str, detail: str, symbols: Iterable[str] = ()) -> None:
        normalized = " ".join(detail.split())
        key = f"{action}:{normalized}".casefold()
        if any(step.step_id == _sha(key)[:24] for step in steps):
            return
        algorithm = (
            f"locate existing contract -> identify reusable implementation -> {action} -> "
            "preserve repository conventions -> validate target compatibility and tests"
        )
        query = f"{kind} {action} {normalized} {algorithm}"
        steps.append(
            PlanStep(
                step_id=_sha(key)[:24],
                action=action,
                algorithmic_plan=algorithm,
                query=query,
                required_symbols=tuple(sorted(set(str(value) for value in symbols if str(value)))),
            )
        )

    add("integrate_module", kind, (kind,))
    for key, value in sorted(config.items(), key=lambda item: str(item[0])):
        if str(key).startswith("_"):
            continue
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        add(f"implement_{key}", f"{key} {rendered}", _tokens(value))
    for dependency in depends_on:
        add("bind_dependency", dependency, (dependency,))
    for gate in gates:
        add("satisfy_gate", gate, (gate,))
    return tuple(steps)


def _draft_queries(payload: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(payload, Mapping):
        return ()
    operations = payload.get("operations")
    if not isinstance(operations, list):
        operations = payload.get("patch_operations")
    if not isinstance(operations, list):
        operations = payload.get("patches")
    if not isinstance(operations, list):
        return ()
    queries: list[str] = []
    for operation in operations:
        if not isinstance(operation, Mapping):
            continue
        path = str(operation.get("path", "")).replace("\\", "/")
        content = str(operation.get("content", ""))
        replacements = operation.get("replacements")
        if isinstance(replacements, list):
            content += "\n" + json.dumps(replacements, ensure_ascii=False, default=str)
        imports = _IMPORT.findall(content)
        calls = [name for name in _CALL.findall(content) if name not in _KEYWORDS]
        symbols = list(_TYPE.findall(content))
        queries.append(
            " ".join(
                [
                    "verify generated draft against repository",
                    path,
                    *imports[:12],
                    *calls[:20],
                    *symbols[:12],
                ]
            ).strip()
        )
    unique: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = query.casefold()
        if query and key not in seen:
            seen.add(key)
            unique.append(query)
    return tuple(unique)


def _failure_queries(text: str) -> tuple[str, ...]:
    tokens = [
        token
        for token in _TOKEN.findall(text)
        if token.casefold() not in {"error", "failure", "failed", "validation"}
    ]
    if not tokens:
        return ()
    base = " ".join(tokens[-40:])
    return (
        f"repository implementation relevant to validation failure {base}",
        f"dependency API signature test repair {base}",
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(str(text)):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(str(text)[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _salient_terms(text: str, *, exclude: set[str], limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for token in _TOKEN.findall(text):
        value = token.casefold()
        if len(value) < 3 or value in exclude or value in _KEYWORDS:
            continue
        counts[value] = counts.get(value, 0) + 1
    return [
        token
        for token, _count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
    ]


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


def _semantic_similarity(query: str, text: str) -> float:
    q = " ".join(sorted(_tokens(query)))
    t = " ".join(sorted(_tokens(text)))
    if not q or not t:
        return 0.0
    return SequenceMatcher(None, q[:4096], t[:8192]).ratio()


def _retrieval_metrics(
    query: str,
    text: str,
    *,
    path: str,
    symbols: Sequence[str],
    graph_hop: int | None,
    quality: QualityVector,
) -> dict[str, float]:
    query_tokens = _tokens(query)
    text_tokens = _tokens(text)
    path_tokens = _tokens(path)
    symbol_tokens = _tokens(symbols)
    lexical = _overlap(query_tokens, text_tokens)
    path_score = _overlap(query_tokens, path_tokens)
    symbol = _overlap(query_tokens, symbol_tokens)
    semantic = _semantic_similarity(query, text)
    dependency = min(
        1.0,
        0.6 * _overlap(query_tokens, _tokens(_IMPORT.findall(text)))
        + 0.4 * _overlap(query_tokens, _tokens(_CALL.findall(text))),
    )
    depth = max(1, len(Path(path).parts))
    structure = min(1.0, path_score + (0.15 if depth >= 4 else 0.05))
    call_graph = 0.0 if graph_hop is None else 1.0 / (1.0 + graph_hop)
    return {
        "lexical": lexical,
        "semantic": semantic,
        "path": path_score,
        "symbol": symbol,
        "dependency": dependency,
        "structure": structure,
        "call_graph": call_graph,
        "quality": 0.55 * quality.coqu_ir + 0.45 * quality.example_quality,
    }


def _adaptive_weights(query: str) -> dict[str, float]:
    lowered = query.casefold()
    identifiers = _IDENTIFIER.findall(query)
    weights = {
        "lexical": 1.0,
        "semantic": 0.8,
        "path": 0.65,
        "symbol": 0.75,
        "dependency": 0.75,
        "structure": 0.65,
        "call_graph": 0.65,
        "quality": 1.1,
    }
    if any("/" in token or "." in token for token in _TOKEN.findall(query)):
        weights["path"] += 0.8
        weights["structure"] += 0.5
    if any(
        token not in _KEYWORDS and any(char.isupper() for char in token[1:])
        for token in identifiers
    ):
        weights["symbol"] += 0.9
        weights["dependency"] += 0.5
    if any(word in lowered for word in ("call", "dependency", "depends", "bind", "api", "register")):
        weights["dependency"] += 0.9
        weights["call_graph"] += 0.8
    if any(word in lowered for word in ("test", "security", "safe", "validate", "gate")):
        weights["quality"] += 0.9
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def _adaptive_score(
    query: str,
    metrics: Mapping[str, float],
    quality: QualityVector,
) -> float:
    weights = _adaptive_weights(query)
    score = sum(weights[key] * float(metrics.get(key, 0.0)) for key in weights)
    # BestFit never lets semantic relevance erase severe quality defects.
    floor = min(
        quality.correctness,
        quality.security,
        quality.maintainability,
    )
    return max(0.0, min(1.0, score * (0.55 + 0.45 * floor)))


def _quality(text: str, *, path: str) -> QualityVector:
    lowered = text.casefold()
    lines = text.splitlines() or [""]
    line_count = len(lines)
    nonempty = [line for line in lines if line.strip()]
    long_ratio = sum(len(line) > 120 for line in nonempty) / max(1, len(nonempty))
    todo_penalty = min(0.5, 0.12 * sum(marker in lowered for marker in ("todo", "fixme", "unsupportedoperationexception")))
    balance_penalty = 0.25 if text.count("{") != text.count("}") else 0.0
    correctness = max(0.0, 1.0 - todo_penalty - balance_penalty)
    if "/test/" in path.replace("\\", "/").casefold() or path.casefold().endswith("test.java"):
        correctness = min(1.0, correctness + 0.08)

    nested_loop = bool(re.search(r"\bfor\s*\([^)]*\)\s*\{[^{}]{0,1000}\bfor\s*\(", text, re.DOTALL))
    efficiency = 1.0
    if nested_loop:
        efficiency -= 0.18
    if "thread.sleep(" in lowered:
        efficiency -= 0.22
    if lowered.count(".readallbytes(") + lowered.count(".readstring(") >= 2:
        efficiency -= 0.12
    efficiency = max(0.0, efficiency)

    security = 1.0 - 0.22 * sum(marker in lowered for marker in _DANGEROUS)
    if re.search(r"files\.(?:write|string|delete|move)\s*\(\s*paths\.get\s*\(\s*[A-Za-z_$]", lowered):
        security -= 0.15
    security = max(0.0, security)

    comment_lines = sum(line.strip().startswith(("//", "/*", "*")) for line in lines)
    comment_ratio = comment_lines / max(1, line_count)
    maintainability = 1.0 - min(0.45, long_ratio * 0.7)
    if line_count > 240:
        maintainability -= min(0.25, (line_count - 240) / 1000)
    if 0.03 <= comment_ratio <= 0.35:
        maintainability += 0.05
    maintainability = max(0.0, min(1.0, maintainability))

    # The example-quality paper finds moderate complexity preferable to trivial or
    # overly complicated examples. Peak around 45 lines with a broad useful band.
    complexity_fit = math.exp(-((line_count - 45.0) / 65.0) ** 2)
    indentation_jumps = 0
    last_indent = 0
    for line in nonempty:
        indent = len(line) - len(line.lstrip(" "))
        if indent - last_indent > 8:
            indentation_jumps += 1
        last_indent = indent
    readable_names = sum(
        len(name) >= 3 and name.casefold() not in {"tmp", "foo", "bar", "obj"}
        for name in _IDENTIFIER.findall(text)
    )
    readability = 1.0
    readability -= min(0.35, long_ratio * 0.8)
    readability -= min(0.25, indentation_jumps / max(1, line_count))
    if readable_names < max(1, line_count // 20):
        readability -= 0.12
    readability = max(0.0, min(1.0, readability))

    return QualityVector(
        correctness=correctness,
        efficiency=efficiency,
        security=security,
        maintainability=maintainability,
        complexity_fit=complexity_fit,
        readability=readability,
    )


def _combine_examples(items: Sequence[Evidence], *, limit: int) -> list[Evidence]:
    """Quality-aware complementary selection rather than naive top-k duplicates."""

    selected: list[Evidence] = []
    covered_symbols: set[str] = set()
    covered_steps: set[str] = set()
    remaining = list(items)
    while remaining and len(selected) < limit:
        best: Evidence | None = None
        best_value = -1.0
        for item in remaining:
            novelty_symbols = len(set(item.symbols) - covered_symbols)
            novelty_steps = len(item.plan_steps - covered_steps)
            duplicate_path = any(existing.path == item.path for existing in selected)
            quality = item.quality.example_quality if item.quality else 0.5
            value = (
                item.bestfit_score
                + 0.08 * novelty_symbols
                + 0.12 * novelty_steps
                + 0.18 * quality
                - (0.18 if duplicate_path else 0.0)
            )
            if value > best_value:
                best_value = value
                best = item
        assert best is not None
        selected.append(best)
        covered_symbols.update(best.symbols)
        covered_steps.update(best.plan_steps)
        remaining.remove(best)
    return selected


def _dedupe_evidence(items: Sequence[Evidence]) -> list[Evidence]:
    result: dict[str, Evidence] = {}
    for item in items:
        existing = result.get(item.evidence_id)
        if existing is None or item.bestfit_score > existing.bestfit_score:
            result[item.evidence_id] = item
        elif existing is not None:
            existing.plan_steps.update(item.plan_steps)
    return list(result.values())


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for match in re.finditer("\n", text):
        offsets.append(match.end())
    return offsets


def _offset_line(offsets: Sequence[int], offset: int) -> int:
    low = 0
    high = len(offsets)
    while low < high:
        mid = (low + high) // 2
        if offsets[mid] <= offset:
            low = mid + 1
        else:
            high = mid
    return max(1, low)


def _matching_brace(text: str, start: int) -> int:
    if start < 0 or start >= len(text) or text[start] != "{":
        return -1
    depth = 0
    in_string = False
    quote = ""
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


__all__ = [
    "DependencyMonitor",
    "DependencyViolation",
    "PlanStep",
    "QualityVector",
    "ResearchCodeContext",
]
