from __future__ import annotations

"""Research-driven repository reuse for the production coder hot path.

The host owns this engine. It translates the research ideas used by MMM into one
bounded runtime instead of exposing paper names as prompt decoration:

* CAPIR: decompose a coarse module request into capability/API subtasks and reduce
  redundant evidence per subtask.
* PERC: build an algorithmic plan for every subtask and compare it with
  structure-preserving plans extracted from repository examples.
* RepoCoder + EvoR: alternate generation and retrieval; evolve both queries and the
  bounded evidence knowledge base until a host fixed point is reached.
* CoRet + DyRetriever: combine semantics, repository hierarchy and dependencies while
  constructing only an ephemeral, query-specific partial call graph.
* CodeRAG + AIRCoder: construct multiple query paths, collect multi-path candidates,
  compute eight complementary retrieval signals, adapt fusion weights per query and
  preference-rerank the best candidates when a reranker is available.
* RAR + DocPrompting: retrieve exact-target official documentation first, use it to
  retrieve examples, then use example symbols to retrieve a second documentation pass.
* CoQuIR + Example Quality: rank relevance jointly with correctness, efficiency,
  security, maintainability, moderate complexity, readability and stepwise clarity.
* PackMonitor: expose a finite authoritative dependency admission set consumed by the
  decode-time monitor; model output is never allowed to expand that authority.

Retrieved or generated material is evidence only. Exact target/dependency authority
remains host-owned.
"""

import copy
import hashlib
import json
import math
import os
import re
import threading
from collections import Counter, deque
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
_PLUGIN_WITH_VERSION = re.compile(
    r"\bid\s*(?:\(\s*)?['\"]([A-Za-z0-9_.-]+)['\"]\s*\)?\s*version\s*['\"]([^'\"]+)['\"]"
)
_VERSION_CATALOG_MODULE = re.compile(
    r"\bmodule\s*=\s*['\"]([A-Za-z0-9_.-]+):([A-Za-z0-9_.-]+)['\"]"
)
_PROPERTY_ASSIGN = re.compile(r"(?m)^\s*([A-Za-z0-9_.-]+)\s*=\s*([^\s#]+)\s*$")
_URL = re.compile(r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+")
_CONTROL = re.compile(r"\b(if|else|for|while|switch|case|try|catch|finally|return|throw|new)\b")
_REGISTRATION = re.compile(
    r"\b(register|registry|event|callback|listener|subscribe|initialize|bootstrap|codec|packet|payload|tick)\w*\b",
    re.IGNORECASE,
)
_ASSERTION = re.compile(r"\b(assert|assertEquals|assertTrue|assertFalse|expect|verify|check)\w*\b")

_KEYWORDS = frozenset(
    {
        "if", "for", "while", "switch", "catch", "return", "new", "throw", "super",
        "this", "synchronized", "assert", "try", "do", "else", "case", "default",
    }
)
_DANGEROUS = (
    "runtime.getruntime().exec",
    "processbuilder(",
    "objectinputstream(",
    "setaccessible(true)",
    "scriptengine",
    "urlclassloader(",
    "system.setsecuritymanager",
)

_CACHE_LOCK = threading.RLock()
_STRUCTURE_CACHE: dict[str, tuple[dict[str, "_JavaUnit"], dict[str, list["SourceSymbol"]]]] = {}
_INITIAL_RESEARCH_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_LIMIT = 8

_RESEARCH_METHODS = (
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
)


def _tokens(value: Any) -> set[str]:
    text = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=str
    )
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
    capability: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "capability": self.capability,
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
    stepwise_clarity: float = 0.5

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
            0.38 * self.correctness
            + 0.22 * self.readability
            + 0.20 * self.complexity_fit
            + 0.20 * self.stepwise_clarity
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "correctness": round(self.correctness, 6),
            "efficiency": round(self.efficiency, 6),
            "security": round(self.security, 6),
            "maintainability": round(self.maintainability, 6),
            "complexity_fit": round(self.complexity_fit, 6),
            "readability": round(self.readability, 6),
            "stepwise_clarity": round(self.stepwise_clarity, 6),
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
    algorithmic_plan: str = ""

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
            "metrics": {
                key: round(value, 6) for key, value in sorted(self.metrics.items())
            },
            "quality": self.quality.to_dict() if self.quality else None,
            "bestfit_score": round(self.bestfit_score, 6),
            "graph_hop": self.graph_hop,
            "algorithmic_plan": self.algorithmic_plan,
            "text": self.text,
        }


@dataclass(frozen=True)
class DependencyViolation:
    kind: str
    value: str
    path: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "value": self.value, "path": self.path}


@dataclass(frozen=True)
class _JavaUnit:
    path: str
    package: str
    imports: tuple[str, ...]
    types: tuple[str, ...]
    methods: tuple[SourceSymbol, ...]


class DependencyMonitor:
    """Finite host-owned dependency authority used by the decode-time PackMonitor."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        minecraft_version: str,
        loader: str,
        mappings: str,
    ) -> None:
        self.root = Path(project_root).expanduser().resolve()
        self.minecraft_version = str(minecraft_version).strip()
        self.loader = str(loader).strip().casefold()
        self.mappings = str(mappings).strip()
        self.allowed_packages: set[str] = set()
        self.allowed_coordinates: set[str] = set()
        self.allowed_plugins: set[str] = {"java", "java-library", "maven-publish"}
        self.allowed_plugin_versions: set[tuple[str, str]] = set()
        self.allowed_repositories: set[str] = set()
        self._seed_platform()
        self._seed_existing_files()

    def _seed_platform(self) -> None:
        adapter = adapter_for_target(self.minecraft_version, self.loader)
        if adapter.loader != "fabric":
            return
        exact = {
            f"net.fabricmc:fabric-loader:{adapter.fabric_loader}",
            f"net.fabricmc.fabric-api:fabric-api:{adapter.fabric_api}",
            f"net.fabricmc:yarn:{adapter.yarn_mappings}",
            f"net.fabricmc:yarn:{adapter.yarn_mappings}:v2",
        }
        self.allowed_coordinates.update(exact)
        self.allowed_packages.update(value.rsplit(":", 1)[0] for value in exact)
        self.allowed_packages.add("net.fabricmc:yarn")
        self.allowed_plugins.add("fabric-loom")
        self.allowed_plugin_versions.add(("fabric-loom", adapter.fabric_loom))
        self.allowed_repositories.add("https://maven.fabricmc.net")

    def _seed_existing_files(self) -> None:
        for name in (
            "build.gradle",
            "build.gradle.kts",
            "settings.gradle",
            "settings.gradle.kts",
            "gradle.properties",
            "gradle/libs.versions.toml",
        ):
            path = self.root / name
            if not path.is_file() or path.is_symlink():
                continue
            self.admit_text(path.read_text(encoding="utf-8", errors="replace"), code_owned=True)

    def admit_text(self, text: str, *, code_owned: bool) -> None:
        if not code_owned:
            return
        for group, artifact, version in _MAVEN_COORD.findall(text):
            package = f"{group}:{artifact}"
            self.allowed_packages.add(package)
            if _literal_version(version):
                self.allowed_coordinates.add(f"{package}:{version}")
        self.allowed_plugins.update(_PLUGIN_ID.findall(text))
        self.allowed_plugin_versions.update(_PLUGIN_WITH_VERSION.findall(text))
        for group, artifact in _VERSION_CATALOG_MODULE.findall(text):
            self.allowed_packages.add(f"{group}:{artifact}")
        self.allowed_repositories.update(_normalized_repo_urls(text))

    def admit_research_context(self, value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        for key in ("research_context", "host_grounding", "technology_radar"):
            payload = value.get(key)
            if payload is not None:
                self.admit_text(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
                    code_owned=True,
                )

    def validate_model_output(self, text: str) -> tuple[DependencyViolation, ...]:
        payload = _extract_json_object(text)
        if not payload:
            return ()
        operations = _operations(payload)
        violations: list[DependencyViolation] = []
        for raw in operations:
            path = str(raw.get("path", "")).replace("\\", "/")
            leaf = Path(path).name.casefold()
            if leaf not in {
                "build.gradle", "build.gradle.kts", "settings.gradle",
                "settings.gradle.kts", "gradle.properties", "libs.versions.toml",
            }:
                continue
            body = _operation_text(raw)
            for group, artifact, version in _MAVEN_COORD.findall(body):
                package = f"{group}:{artifact}"
                coordinate = f"{package}:{version}"
                if package not in self.allowed_packages:
                    violations.append(DependencyViolation("package", package, path))
                elif _literal_version(version) and coordinate not in self.allowed_coordinates:
                    violations.append(DependencyViolation("coordinate", coordinate, path))
            for group, artifact in _VERSION_CATALOG_MODULE.findall(body):
                package = f"{group}:{artifact}"
                if package not in self.allowed_packages:
                    violations.append(DependencyViolation("package", package, path))
            for plugin in _PLUGIN_ID.findall(body):
                if plugin not in self.allowed_plugins:
                    violations.append(DependencyViolation("plugin", plugin, path))
            for plugin, version in _PLUGIN_WITH_VERSION.findall(body):
                if _literal_version(version) and (plugin, version) not in self.allowed_plugin_versions:
                    violations.append(
                        DependencyViolation("plugin_version", f"{plugin}:{version}", path)
                    )
            for repository in _normalized_repo_urls(body):
                if repository not in self.allowed_repositories:
                    violations.append(DependencyViolation("repository", repository, path))
            if leaf == "gradle.properties":
                expected = self._expected_properties()
                for key, value in _PROPERTY_ASSIGN.findall(body):
                    if key in expected and value != expected[key]:
                        violations.append(
                            DependencyViolation("target_property", f"{key}={value}", path)
                        )
        unique = {(item.kind, item.value, item.path): item for item in violations}
        return tuple(unique[key] for key in sorted(unique))

    def _expected_properties(self) -> dict[str, str]:
        adapter = adapter_for_target(self.minecraft_version, self.loader)
        result = {"minecraft_version": adapter.minecraft_version}
        if adapter.loader == "fabric":
            result.update(
                {
                    "loader_version": adapter.fabric_loader,
                    "fabric_version": adapter.fabric_api,
                    "loom_version": adapter.fabric_loom,
                }
            )
            if adapter.yarn_mappings != "mojang":
                result["yarn_mappings"] = adapter.yarn_mappings
        return result

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/dependency-monitor-v2",
            "policy": "finite_authoritative_exact_coordinate_admission",
            "target": {
                "minecraft_version": self.minecraft_version,
                "loader": self.loader,
                "mappings": self.mappings,
            },
            "allowed_package_count": len(self.allowed_packages),
            "allowed_coordinate_count": len(self.allowed_coordinates),
            "allowed_repository_count": len(self.allowed_repositories),
            "allowed_plugin_version_count": len(self.allowed_plugin_versions),
            "allowed_packages_sha256": _sha(sorted(self.allowed_packages)),
            "allowed_coordinates_sha256": _sha(sorted(self.allowed_coordinates)),
            "allowed_plugin_versions_sha256": _sha(sorted(self.allowed_plugin_versions)),
            "allowed_repositories_sha256": _sha(sorted(self.allowed_repositories)),
            "zero_unknown_packages_in_accepted_patch": True,
            "zero_unknown_literal_coordinates_in_accepted_patch": True,
            "zero_unknown_repositories_in_accepted_patch": True,
        }


class ResearchCodeContext:
    """Build and evolve bounded research evidence for one production code generation."""

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
            raise ValueError("ResearchCodeContext requires a host-selected exact target.")
        adapter = adapter_for_target(self.minecraft_version, self.loader)
        if self.mappings != adapter.yarn_mappings:
            raise ValueError("Research mappings disagree with the executable provider.")
        self.byte_budget = max(4096, int(byte_budget))
        self.graph_budget = _env_int("MMM_CODE_RESEARCH_GRAPH_NODES", 64, 8, 512)
        self.query_budget = _env_int("MMM_CODE_RESEARCH_QUERY_BUDGET", 48, 8, 256)
        self.plan = _build_plan(module)
        self._manifest_sha = self._manifest_snapshot_sha()
        self.units, self.symbols_by_name = self._cached_repository_structure()
        self.evidence: dict[str, Evidence] = {}
        self.query_history: list[str] = []
        self.query_seen: set[str] = set()
        self.knowledge_terms: Counter[str] = Counter()
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
        """Ingest host evidence as a third KB lane without granting model authority."""

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
            if not isinstance(payload, Mapping):
                continue
            self.monitor.admit_research_context(payload)
            for key in ("research_context", "host_grounding", "technology_radar"):
                value = payload.get(key)
                if value is None:
                    continue
                text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
                if not text or text == "null":
                    continue
                item = Evidence(
                    evidence_id=f"host:{key}:{_sha(value)}",
                    source_type=f"host_{key}",
                    path=f"<host:{key}>",
                    text=text[: max(2048, self.byte_budget // 3)],
                    sha256=_sha(value),
                    symbols=tuple(_salient_terms(text, exclude=set(), limit=20)),
                    metrics={"host_verified": 1.0, "retrieval_score": 1.0},
                    bestfit_score=1.0,
                    algorithmic_plan=_code_plan(text),
                )
                self._merge_evidence(item)
                self._evolve_knowledge(item.text)

    def initial_bundle(self) -> dict[str, Any]:
        if self._initial_complete:
            return self.bundle()
        cache_key = self._initial_cache_key()
        cached = _initial_cache_get(cache_key)
        if cached is not None:
            self._restore_initial_state(cached)
            self._initial_complete = True
            return self.bundle()
        for step in self.plan:
            self._research_step(step, reason="initial_plan")
        self._initial_complete = True
        _initial_cache_put(cache_key, self._snapshot_initial_state())
        return self.bundle()

    def _manifest_snapshot_sha(self) -> str:
        try:
            receipt = self.index.manifest_receipt()
        except Exception:
            return _sha([item.path + ":" + item.sha256 for item in self.index.files])
        if isinstance(receipt, Mapping):
            return str(receipt.get("sha256") or receipt.get("snapshot_hash") or _sha(receipt))
        return _sha(receipt)

    def _cached_repository_structure(
        self,
    ) -> tuple[dict[str, _JavaUnit], dict[str, list[SourceSymbol]]]:
        with _CACHE_LOCK:
            cached = _STRUCTURE_CACHE.get(self._manifest_sha)
            if cached is not None:
                units, symbols = cached
                return dict(units), {key: list(value) for key, value in symbols.items()}
        built = self._index_repository_structure()
        with _CACHE_LOCK:
            _bounded_cache_put(_STRUCTURE_CACHE, self._manifest_sha, built)
        units, symbols = built
        return dict(units), {key: list(value) for key, value in symbols.items()}

    def _initial_cache_key(self) -> str:
        host_ids = sorted(
            item.evidence_id
            for item in self.evidence.values()
            if item.source_type.startswith("host_")
        )
        return _sha(
            {
                "manifest": self._manifest_sha,
                "target": [self.minecraft_version, self.loader, self.mappings],
                "plan": [step.to_dict() for step in self.plan],
                "host_evidence": host_ids,
                "byte_budget": self.byte_budget,
            }
        )

    def _snapshot_initial_state(self) -> dict[str, Any]:
        return {
            "evidence": {key: _clone_evidence(value) for key, value in self.evidence.items()},
            "query_history": list(self.query_history),
            "query_seen": set(self.query_seen),
            "knowledge_terms": Counter(self.knowledge_terms),
            "rounds": copy.deepcopy(self.rounds),
        }

    def _restore_initial_state(self, state: Mapping[str, Any]) -> None:
        raw_evidence = state.get("evidence", {})
        if isinstance(raw_evidence, Mapping):
            self.evidence = {
                str(key): _clone_evidence(value)
                for key, value in raw_evidence.items()
                if isinstance(value, Evidence)
            }
        self.query_history = list(state.get("query_history", ()))
        self.query_seen = set(state.get("query_seen", ()))
        self.knowledge_terms = Counter(state.get("knowledge_terms", {}))
        self.rounds = copy.deepcopy(list(state.get("rounds", ())))

    def evolve_from_generation(
        self, text: str
    ) -> tuple[dict[str, Any] | None, tuple[DependencyViolation, ...]]:
        """RepoCoder/EvoR loop: draft -> query evolution -> KB evolution -> regenerate."""

        violations = self.monitor.validate_model_output(text)
        before = len(self.evidence)
        before_terms = set(self.knowledge_terms)
        queries = list(_draft_queries(_extract_json_object(text)))
        queries.extend(self._knowledge_evolution_queries(text, limit=4))
        executed: list[str] = []
        for query in queries:
            if len(self.query_history) >= self.query_budget:
                break
            if self._run_query(query, plan_step=None, reason="draft_evolution"):
                executed.append(query)
        added = len(self.evidence) - before
        new_terms = len(set(self.knowledge_terms) - before_terms)
        self.rounds.append(
            {
                "trigger": "generated_draft",
                "queries_sha256": _sha(executed),
                "query_count": len(executed),
                "new_evidence": added,
                "new_knowledge_terms": new_terms,
                "dependency_violations": [item.to_dict() for item in violations],
            }
        )
        return (self.bundle(), violations) if violations or added or new_terms else (None, ())

    def evolve_from_failure(
        self, messages: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any] | None:
        tail = " ".join(
            str(message.get("content", ""))
            for message in messages[-4:]
            if isinstance(message.get("content"), str)
        )
        before = len(self.evidence)
        executed: list[str] = []
        for query in _failure_queries(tail):
            if len(self.query_history) >= self.query_budget:
                break
            if self._run_query(query, plan_step=None, reason="validation_failure"):
                executed.append(query)
        if len(self.evidence) == before:
            return None
        self.rounds.append(
            {
                "trigger": "validation_failure",
                "queries_sha256": _sha(executed),
                "query_count": len(executed),
                "new_evidence": len(self.evidence) - before,
            }
        )
        return self.bundle()

    def _research_step(self, step: PlanStep, *, reason: str) -> None:
        docs = self._retrieve_docs(step.query, step_id=step.step_id)
        doc_terms = _salient_terms(
            " ".join(item.text for item in docs), exclude=_tokens(step.query), limit=12
        )
        examples = self._retrieve_repo_examples(
            _join_query(step.query, step.algorithmic_plan, *doc_terms),
            plan_step=step,
        )
        symbols = sorted(
            {symbol for item in examples[:8] for symbol in item.symbols if symbol}
        )[:16]
        if symbols:
            self._retrieve_docs(_join_query(step.query, *symbols), step_id=step.step_id)
        self._remember_query(step.query)
        self.rounds.append(
            {
                "trigger": reason,
                "plan_step": step.step_id,
                "docs": len(docs),
                "examples": len(examples),
                "covered_required_symbols": len(
                    set(step.required_symbols) & {symbol for item in examples for symbol in item.symbols}
                ),
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
        if len(normalized) < 2 or normalized.casefold() in self.query_seen:
            return False
        if len(self.query_history) >= self.query_budget:
            return False
        self._remember_query(normalized)
        before = len(self.evidence)
        docs = self._retrieve_docs(
            normalized, step_id=plan_step.step_id if plan_step else ""
        )
        terms = _salient_terms(
            " ".join(item.text for item in docs), exclude=_tokens(normalized), limit=10
        )
        self._retrieve_repo_examples(
            _join_query(normalized, *terms), plan_step=plan_step
        )
        self.rounds.append(
            {
                "trigger": reason,
                "query_sha256": _sha(normalized),
                "new_evidence": len(self.evidence) - before,
            }
        )
        return len(self.evidence) > before

    def _remember_query(self, query: str) -> None:
        key = query.casefold()
        if key in self.query_seen:
            return
        self.query_seen.add(key)
        self.query_history.append(query)

    def _evolve_knowledge(self, text: str) -> None:
        for term in _salient_terms(text, exclude=set(), limit=32):
            self.knowledge_terms[term] += 1

    def _knowledge_evolution_queries(self, draft: str, *, limit: int) -> list[str]:
        draft_terms = _tokens(draft)
        candidates = [
            term for term, count in self.knowledge_terms.most_common(32)
            if count > 0 and term not in draft_terms
        ]
        result: list[str] = []
        for step in self.plan:
            if len(result) >= limit:
                break
            novel = [term for term in candidates if term not in _tokens(step.query)][:6]
            if novel:
                result.append(_join_query(step.query, "knowledge evolution", *novel))
        return result

    def _retrieve_docs(self, query: str, *, step_id: str) -> list[Evidence]:
        try:
            from .retrieval import retrieve_official_evidence

            receipt = retrieve_official_evidence(
                query,
                minecraft_version=self.minecraft_version,
                loader=self.loader,
                mappings=self.mappings,
                limit=8,
            )
            payload = receipt.to_dict()
        except Exception as exc:
            self.rounds.append(
                {
                    "trigger": "official_docs_unavailable",
                    "query_sha256": _sha(query),
                    "error": f"{type(exc).__name__}: {exc}"[:512],
                }
            )
            return []
        result: list[Evidence] = []
        for raw in payload.get("hits", []):
            if not isinstance(raw, Mapping):
                continue
            document_id = str(raw.get("document_id", ""))
            text = str(raw.get("excerpt", ""))
            if not document_id or not text:
                continue
            item = Evidence(
                evidence_id=str(raw.get("evidence_id") or f"doc:{document_id}"),
                source_type="official_documentation",
                path=document_id,
                text=text,
                sha256=str(raw.get("content_sha256", "")) or _sha(text),
                symbols=tuple(_salient_terms(text, exclude=set(), limit=16)),
                metrics={
                    "retrieval_score": float(raw.get("score", 0.0) or 0.0),
                    "coverage": float(payload.get("coverage", 0.0) or 0.0),
                },
                bestfit_score=float(raw.get("score", 0.0) or 0.0),
                algorithmic_plan=_code_plan(text),
            )
            if step_id:
                item.plan_steps.add(step_id)
            self._merge_evidence(item)
            self._evolve_knowledge(item.text)
            result.append(item)
        return result

    def _query_paths(self, query: str, plan_step: PlanStep | None) -> tuple[str, ...]:
        paths = [query]
        if plan_step is not None:
            paths.extend(
                [
                    _join_query(plan_step.capability, plan_step.algorithmic_plan),
                    _join_query(plan_step.action, *plan_step.required_symbols),
                    _join_query("repository API dependency", plan_step.capability, *plan_step.required_symbols),
                ]
            )
        salient = [term for term, _count in self.knowledge_terms.most_common(12)]
        if salient:
            paths.append(_join_query(query, "known repository vocabulary", *salient[:8]))
        return tuple(dict.fromkeys(value for value in paths if value.strip()))

    def _retrieve_repo_examples(
        self, query: str, *, plan_step: PlanStep | None
    ) -> list[Evidence]:
        candidates: list[Evidence] = []
        rank_by_path: dict[str, dict[str, int]] = {}
        query_paths = self._query_paths(query, plan_step)

        graph_entries = self._entry_points(query)
        for symbol, hop in self._expand_partial_graph(graph_entries, query=query):
            item = self._symbol_evidence(
                symbol,
                query=query,
                graph_hop=hop,
                target_plan=plan_step.algorithmic_plan if plan_step else "",
            )
            if item is not None:
                if plan_step is not None:
                    item.plan_steps.add(plan_step.step_id)
                candidates.append(item)

        for path_index, path_query in enumerate(query_paths):
            try:
                selected = self.index.select(
                    query=path_query,
                    byte_budget=min(self.byte_budget, 12 * 1024),
                )
            except Exception:
                continue
            path_key = f"path_{path_index}"
            for rank, raw in enumerate(selected.get("files", []), start=1):
                if not isinstance(raw, Mapping):
                    continue
                path = str(raw.get("path", ""))
                text = str(raw.get("content", ""))
                if not path or not text:
                    continue
                item = self._file_evidence(
                    path,
                    text,
                    query=query,
                    target_plan=plan_step.algorithmic_plan if plan_step else "",
                )
                if plan_step is not None:
                    item.plan_steps.add(plan_step.step_id)
                candidates.append(item)
                rank_by_path.setdefault(item.evidence_id, {})[path_key] = rank

        candidates = _dedupe_evidence(candidates)
        for item in candidates:
            ranks = rank_by_path.get(item.evidence_id, {})
            if ranks:
                item.metrics["multipath_rrf"] = sum(1.0 / (60.0 + rank) for rank in ranks.values())
                item.metrics["multipath_coverage"] = len(ranks) / max(1, len(query_paths))
                item.bestfit_score = _adaptive_score(query, item.metrics, item.quality)

        self._apply_bestfit_rerank(query, candidates)
        candidates.sort(
            key=lambda item: (
                -item.bestfit_score,
                -(item.quality.example_quality if item.quality else 0.0),
                item.path,
                item.start_line,
            )
        )
        selected_examples = _combine_examples(
            candidates,
            limit=10,
            required_steps={plan_step.step_id} if plan_step else set(),
            required_symbols=set(plan_step.required_symbols) if plan_step else set(),
        )
        for item in selected_examples:
            self._merge_evidence(item)
            self._evolve_knowledge(item.text)
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
            offsets = _line_offsets(text)
            methods: list[SourceSymbol] = []
            for match in _METHOD.finditer(text):
                name = match.group(1)
                if name in _KEYWORDS:
                    continue
                brace = text.find("{", match.start())
                end = _matching_brace(text, brace)
                symbol = SourceSymbol(
                    name=name,
                    path=indexed.path,
                    package=package,
                    kind="method",
                    start_line=_offset_line(offsets, match.start()),
                    end_line=_offset_line(offsets, end if end >= 0 else match.end()),
                    signature=" ".join(match.group(0).split())[:512],
                )
                methods.append(symbol)
                by_name.setdefault(name, []).append(symbol)
            for name in types:
                by_name.setdefault(name, []).append(
                    SourceSymbol(
                        name=name,
                        path=indexed.path,
                        package=package,
                        kind="type",
                        start_line=1,
                        end_line=1,
                        signature=f"{package}.{name}" if package else name,
                    )
                )
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
            for symbol in unit.methods:
                score = (
                    0.34 * _overlap(query_tokens, _tokens(symbol.name) | _tokens(symbol.signature))
                    + 0.20 * _overlap(query_tokens, _tokens(unit.path))
                    + 0.18 * _overlap(query_tokens, _tokens(unit.imports))
                    + 0.12 * _overlap(query_tokens, _tokens(unit.package))
                    + 0.16 * _semantic_similarity(query, _join_query(symbol.signature, unit.path))
                )
                if score > 0:
                    ranked.append((score, symbol))
        ranked.sort(key=lambda item: (-item[0], item[1].path, item[1].start_line))
        shortlist = [item[1] for item in ranked[: min(self.graph_budget, 32)]]
        return self._semantic_symbol_filter(query, shortlist, limit=min(12, self.graph_budget))

    def _semantic_symbol_filter(
        self,
        query: str,
        symbols: Sequence[SourceSymbol],
        *,
        limit: int,
    ) -> list[SourceSymbol]:
        if not symbols:
            return []
        candidates = list(symbols)
        texts = [
            _join_query(symbol.signature, symbol.path, self._symbol_text(symbol)[:3000])
            for symbol in candidates
        ]
        try:
            raw_scores = self.router.rerank(query, texts)
            if len(raw_scores) == len(candidates):
                scored = [
                    (float(score), symbol)
                    for score, symbol in zip(raw_scores, candidates, strict=True)
                ]
                scored.sort(key=lambda item: (-item[0], item[1].path, item[1].start_line))
                return [symbol for _score, symbol in scored[:limit]]
        except Exception:
            pass
        return candidates[:limit]

    def _expand_partial_graph(
        self,
        entries: Sequence[SourceSymbol],
        *,
        query: str = "",
    ) -> list[tuple[SourceSymbol, int]]:
        if not query:
            query = " ".join(item.name for item in entries)
        queue: deque[tuple[SourceSymbol, int]] = deque((item, 0) for item in entries)
        seen: set[str] = set()
        result: list[tuple[SourceSymbol, int]] = []
        while queue and len(seen) < self.graph_budget:
            symbol, hop = queue.popleft()
            if symbol.symbol_id in seen:
                continue
            seen.add(symbol.symbol_id)
            result.append((symbol, hop))
            if hop >= 3 or symbol.kind != "method":
                continue
            body = self._symbol_text(symbol)
            neighbors: list[SourceSymbol] = []
            for name in _CALL.findall(body):
                if name in _KEYWORDS or name == symbol.name:
                    continue
                neighbors.extend(self.symbols_by_name.get(name, ()))
            unit = self.units.get(symbol.path)
            if unit is not None:
                for imported in unit.imports:
                    leaf = imported.rsplit(".", 1)[-1].removesuffix("*")
                    if leaf:
                        neighbors.extend(self.symbols_by_name.get(leaf, ()))
            unique = {
                item.symbol_id: item
                for item in neighbors
                if item.symbol_id not in seen
            }
            validated = self._semantic_symbol_filter(
                query,
                list(unique.values()),
                limit=min(8, max(1, self.graph_budget - len(seen))),
            )
            for target in validated:
                queue.append((target, hop + 1))
        return result

    def _symbol_text(self, symbol: SourceSymbol) -> str:
        lines = (self.root / symbol.path).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        start = max(0, symbol.start_line - 1)
        end = min(len(lines), max(symbol.end_line, symbol.start_line))
        return "\n".join(lines[start:end])

    def _symbol_evidence(
        self,
        symbol: SourceSymbol,
        *,
        query: str,
        graph_hop: int,
        target_plan: str = "",
    ) -> Evidence | None:
        text = self._symbol_text(symbol)
        if not text.strip():
            return None
        indexed = next((item for item in self.index.files if item.path == symbol.path), None)
        quality = _quality(text, path=symbol.path)
        example_plan = _code_plan(text)
        metrics = _retrieval_metrics(
            query,
            text,
            path=symbol.path,
            symbols=(symbol.name,),
            graph_hop=graph_hop,
            quality=quality,
            target_plan=target_plan,
            example_plan=example_plan,
        )
        return Evidence(
            evidence_id=f"repo:{symbol.symbol_id}",
            source_type="repository_symbol",
            path=symbol.path,
            text=text,
            sha256=indexed.sha256 if indexed else _sha(text),
            start_line=symbol.start_line,
            end_line=symbol.end_line,
            symbols=(symbol.name,),
            metrics=metrics,
            quality=quality,
            bestfit_score=_adaptive_score(query, metrics, quality),
            graph_hop=graph_hop,
            algorithmic_plan=example_plan,
        )

    def _file_evidence(
        self,
        path: str,
        text: str,
        *,
        query: str,
        target_plan: str,
    ) -> Evidence:
        indexed = next((item for item in self.index.files if item.path == path), None)
        symbols = tuple(
            sorted(set(_TYPE.findall(text)) | {value[0] for value in _METHOD.findall(text)})
        )[:20]
        quality = _quality(text, path=path)
        example_plan = _code_plan(text)
        metrics = _retrieval_metrics(
            query,
            text,
            path=path,
            symbols=symbols,
            graph_hop=None,
            quality=quality,
            target_plan=target_plan,
            example_plan=example_plan,
        )
        return Evidence(
            evidence_id=f"file:{path}:{indexed.sha256 if indexed else _sha(text)}",
            source_type="repository_file",
            path=path,
            text=text,
            sha256=indexed.sha256 if indexed else _sha(text),
            end_line=text.count("\n") + 1,
            symbols=symbols,
            metrics=metrics,
            quality=quality,
            bestfit_score=_adaptive_score(query, metrics, quality),
            algorithmic_plan=example_plan,
        )

    def _apply_bestfit_rerank(self, query: str, items: list[Evidence]) -> None:
        candidates = sorted(
            items, key=lambda item: (-item.bestfit_score, item.path, item.start_line)
        )[:24]
        if not candidates:
            return
        try:
            scores = self.router.rerank(query, [item.text[:6000] for item in candidates])
        except Exception:
            return
        if len(scores) != len(candidates):
            return
        values = [float(value) for value in scores]
        low, high = min(values), max(values)
        span = high - low
        for item, value in zip(candidates, values, strict=True):
            aligned = 0.5 if span <= 1e-12 else (value - low) / span
            item.metrics["preference_reranker"] = aligned
            item.bestfit_score = max(
                0.0,
                min(1.0, 0.68 * item.bestfit_score + 0.32 * aligned),
            )

    def _merge_evidence(self, incoming: Evidence) -> None:
        current = self.evidence.get(incoming.evidence_id)
        if current is None:
            self.evidence[incoming.evidence_id] = incoming
            return
        current.plan_steps.update(incoming.plan_steps)
        current.metrics.update(incoming.metrics)
        current.bestfit_score = max(current.bestfit_score, incoming.bestfit_score)
        if not current.algorithmic_plan and incoming.algorithmic_plan:
            current.algorithmic_plan = incoming.algorithmic_plan

    def bundle(self, *, delta_only: bool = False) -> dict[str, Any]:
        del delta_only
        examples = sorted(
            (
                item for item in self.evidence.values()
                if item.source_type.startswith("repository_")
            ),
            key=lambda item: (
                -item.bestfit_score,
                -(item.quality.example_quality if item.quality else 0.0),
                item.path,
                item.start_line,
            ),
        )
        docs = sorted(
            (
                item for item in self.evidence.values()
                if item.source_type == "official_documentation"
            ),
            key=lambda item: (-float(item.metrics.get("retrieval_score", 0.0)), item.path),
        )
        host = sorted(
            (
                item for item in self.evidence.values()
                if item.source_type.startswith("host_")
            ),
            key=lambda item: (item.source_type, item.path),
        )
        plan_payload = _bounded_plan_payload(
            self.plan,
            max_bytes=max(1024, self.byte_budget // 5),
        )
        base = {
            "schema_version": "mmm/research-code-context-v3",
            "target": {
                "minecraft_version": self.minecraft_version,
                "loader": self.loader,
                "mappings": self.mappings,
            },
            "research_methods": list(_RESEARCH_METHODS),
            "plan": plan_payload,
            "plan_count": len(self.plan),
            "plan_sha256": _sha([step.to_dict() for step in self.plan]),
            "query_count": len(self.query_history),
            "query_history_tail": self.query_history[-8:],
            "query_history_sha256": _sha(self.query_history),
            "knowledge_term_count": len(self.knowledge_terms),
            "knowledge_terms_sha256": _sha(sorted(self.knowledge_terms.items())),
            "dependency_monitor": self.monitor.receipt(),
            "round_count": len(self.rounds),
            "rounds_sha256": _sha(self.rounds),
            "total_evidence_count": len(self.evidence),
            "policy": {
                "retrieval_is_data_not_authority": True,
                "capability_decomposition_precedes_retrieval": True,
                "plan_as_query_and_plan_alignment": True,
                "docs_then_examples_then_docs": True,
                "partial_graph_is_ephemeral_and_semantically_filtered": True,
                "eight_metric_query_adaptive_fusion": True,
                "quality_aware_reuse_precedes_freshness": True,
                "generated_draft_can_evolve_queries_not_authority": True,
                "unknown_dependency_names_are_rejected": True,
                "unknown_literal_coordinates_are_rejected": True,
                "unknown_repositories_are_rejected": True,
            },
        }
        selected = [item.public_dict() for item in [*examples, *docs, *host]]
        while True:
            result = {
                **base,
                "evidence": selected,
                "evidence_count": len(selected),
                "omitted_evidence_count": len(self.evidence) - len(selected),
                "evidence_sha256": _sha([item["evidence_id"] for item in selected]),
            }
            result["bundle_sha256"] = _sha(result)
            if _json_bytes(result) <= self.byte_budget:
                self._last_bundle_sha = result["bundle_sha256"]
                return result
            if selected:
                selected.pop()
                continue
            base["plan"] = []
            base["query_history_tail"] = []
            result = {
                **base,
                "evidence": [],
                "evidence_count": 0,
                "omitted_evidence_count": len(self.evidence),
                "evidence_sha256": _sha([]),
            }
            result["bundle_sha256"] = _sha(result)
            if _json_bytes(result) > self.byte_budget:
                raise RuntimeError("Research metadata exceeds the explicit context byte budget.")
            self._last_bundle_sha = result["bundle_sha256"]
            return result

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/research-code-context-receipt-v3",
            "bundle_sha256": self._last_bundle_sha,
            "plan_sha256": _sha([step.to_dict() for step in self.plan]),
            "query_count": len(self.query_history),
            "knowledge_term_count": len(self.knowledge_terms),
            "evidence_count": len(self.evidence),
            "round_count": len(self.rounds),
            "dependency_monitor": self.monitor.receipt(),
            "methods": [
                "CAPIR", "PERC", "RepoCoder", "EvoR", "CoRet", "DyRetriever",
                "CodeRAG", "AIRCoder", "RAR", "DocPrompting", "CoQuIR",
                "ExampleQuality", "PackMonitor",
            ],
        }


def _bounded_cache_put(cache: dict[str, Any], key: str, value: Any) -> None:
    cache[key] = value
    while len(cache) > _CACHE_LIMIT:
        cache.pop(next(iter(cache)))


def _initial_cache_get(key: str) -> dict[str, Any] | None:
    with _CACHE_LOCK:
        value = _INITIAL_RESEARCH_CACHE.get(key)
        return copy.deepcopy(value) if value is not None else None


def _initial_cache_put(key: str, value: dict[str, Any]) -> None:
    with _CACHE_LOCK:
        _bounded_cache_put(_INITIAL_RESEARCH_CACHE, key, copy.deepcopy(value))


def _clone_evidence(item: Evidence) -> Evidence:
    return Evidence(
        evidence_id=item.evidence_id,
        source_type=item.source_type,
        path=item.path,
        text=item.text,
        sha256=item.sha256,
        start_line=item.start_line,
        end_line=item.end_line,
        symbols=tuple(item.symbols),
        plan_steps=set(item.plan_steps),
        metrics=dict(item.metrics),
        quality=item.quality,
        bestfit_score=item.bestfit_score,
        graph_hop=item.graph_hop,
        algorithmic_plan=item.algorithmic_plan,
    )


def _operation_text(operation: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    content = operation.get("content")
    if isinstance(content, str):
        pieces.append(content)
    replacements = operation.get("replacements")
    if isinstance(replacements, list):
        for item in replacements:
            if not isinstance(item, Mapping):
                continue
            for key in ("old", "new", "replacement", "value"):
                value = item.get(key)
                if isinstance(value, str):
                    pieces.append(value)
    return "\n".join(pieces)


def _operations(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("operations", "patch_operations", "patches"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _literal_version(version: str) -> bool:
    value = str(version).strip()
    return bool(value) and "$" not in value and "{" not in value and not value.startswith("libs.")


def _normalized_repo_urls(text: str) -> set[str]:
    return {
        value.rstrip("/.,;)")
        for value in _URL.findall(text)
        if "maven" in value.casefold() or "repo" in value.casefold()
    }


def _compact_detail(key: str, value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(rendered.encode("utf-8")) <= 192:
        return f"{key} {rendered}"
    terms = _salient_terms(rendered, exclude=set(), limit=14)
    return f"{key} {' '.join(terms)} value_sha256={_sha(value)}"


def _capability_for(action: str, detail: str) -> str:
    tokens = _salient_terms(_join_query(action, detail), exclude=set(), limit=8)
    return _join_query(action.replace("_", " "), *tokens)


def _algorithmic_plan(action: str, capability: str, detail: str) -> str:
    verbs = ["locate existing contract", "identify reusable implementation", "adapt", "bind", "validate"]
    lowered = _join_query(action, capability, detail).casefold()
    if any(word in lowered for word in ("network", "packet", "payload")):
        verbs = ["locate existing contract", "locate packet contract", "preserve server authority", "encode/decode", "register", "validate round trip"]
    elif any(word in lowered for word in ("persist", "save", "state", "component")):
        verbs = ["locate existing contract", "locate persistence contract", "load existing state", "apply mutation", "save state", "validate reload"]
    elif any(word in lowered for word in ("register", "registry", "item", "block", "entity")):
        verbs = ["locate existing contract", "locate registry convention", "construct value", "register identifier", "bind resources", "validate lookup"]
    elif any(word in lowered for word in ("event", "tick", "callback", "listener")):
        verbs = ["locate existing contract", "locate lifecycle hook", "register callback", "guard side effects", "execute behavior", "validate observable result"]
    return " -> ".join(verbs)


def _build_plan(module: Any) -> tuple[PlanStep, ...]:
    kind = str(getattr(module, "kind", "") or "custom_java").strip()
    config = getattr(module, "config", {})
    config = config if isinstance(config, Mapping) else {}
    depends_on = tuple(
        str(value).strip() for value in getattr(module, "depends_on", ()) if str(value).strip()
    )
    gates = tuple(
        str(value).strip() for value in getattr(module, "required_gates", ()) if str(value).strip()
    )
    steps: list[PlanStep] = []

    def add(action: str, detail: str, symbols: Iterable[str] = ()) -> None:
        normalized = " ".join(detail.split())
        capability = _capability_for(action, normalized)
        plan = _algorithmic_plan(action, capability, normalized)
        key = f"{action}:{normalized}:{plan}".casefold()
        step_id = _sha(key)[:24]
        if any(step.step_id == step_id for step in steps):
            return
        required = tuple(sorted({str(value) for value in symbols if str(value)}))
        steps.append(
            PlanStep(
                step_id=step_id,
                action=action,
                capability=capability,
                algorithmic_plan=plan,
                query=_join_query(kind, capability, normalized, plan, *required),
                required_symbols=required,
            )
        )

    add("integrate_module", kind, (kind,))
    for key, value in sorted(config.items(), key=lambda item: str(item[0])):
        if not str(key).startswith("_"):
            add(f"implement_{key}", _compact_detail(str(key), value), _tokens(value))
    for dependency in depends_on:
        add("bind_dependency", dependency, (dependency,))
    for gate in gates:
        add("satisfy_gate", gate, (gate,))
    return tuple(steps)


def _bounded_plan_payload(plan: Sequence[PlanStep], *, max_bytes: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for step in plan:
        candidate = [*result, step.to_dict()]
        if _json_bytes(candidate) > max_bytes:
            break
        result = candidate
    return result


def _draft_queries(payload: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(payload, Mapping):
        return ()
    queries: list[str] = []
    for operation in _operations(payload):
        path = str(operation.get("path", "")).replace("\\", "/")
        content = _operation_text(operation)
        plan = _code_plan(content)
        unresolved = [
            name for name in _CALL.findall(content)
            if name not in _KEYWORDS
        ][:24]
        query = _join_query(
            "verify generated draft against repository",
            path,
            plan,
            *_IMPORT.findall(content)[:12],
            *unresolved,
            *_TYPE.findall(content)[:12],
        )
        if query:
            queries.append(query)
    return tuple(dict.fromkeys(queries))


def _failure_queries(text: str) -> tuple[str, ...]:
    values = [
        token for token in _TOKEN.findall(text)
        if token.casefold() not in {"error", "failure", "failed", "validation"}
    ]
    if not values:
        return ()
    base = " ".join(values[-48:])
    return (
        f"repository implementation relevant to validation failure {base}",
        f"official API signature dependency contract repair {base}",
        f"existing test or validation pattern for {base}",
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    value = str(text)
    for index, char in enumerate(value):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _salient_terms(text: str, *, exclude: set[str], limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for token in _TOKEN.findall(text):
        value = token.casefold()
        if len(value) >= 3 and value not in exclude and value not in _KEYWORDS:
            counts[value] = counts.get(value, 0) + 1
    return [
        token for token, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _join_query(*parts: Any) -> str:
    values: list[str] = []
    for part in parts:
        if isinstance(part, str):
            text = part
        elif isinstance(part, Iterable) and not isinstance(part, (bytes, bytearray, Mapping)):
            text = " ".join(str(value) for value in part)
        else:
            text = str(part)
        text = " ".join(text.split())
        if text:
            values.append(text)
    return " ".join(values)


def _overlap(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left)) if left and right else 0.0


def _semantic_similarity(query: str, text: str) -> float:
    left = " ".join(sorted(_tokens(query)))
    right = " ".join(sorted(_tokens(text)))
    return SequenceMatcher(None, left[:4096], right[:8192]).ratio() if left and right else 0.0


def _code_plan(text: str) -> str:
    if not text:
        return ""
    controls = [value.casefold() for value in _CONTROL.findall(text)][:24]
    registrations = [value.casefold() for value in _REGISTRATION.findall(text)][:16]
    calls = [name for name in _CALL.findall(text) if name not in _KEYWORDS][:24]
    imports = [value.rsplit(".", 1)[-1] for value in _IMPORT.findall(text)][:12]
    assertions = _ASSERTION.findall(text)[:8]
    stages: list[str] = []
    if imports:
        stages.append("resolve " + " ".join(dict.fromkeys(imports)))
    if registrations:
        stages.append("bind " + " ".join(dict.fromkeys(registrations)))
    if controls:
        stages.append("flow " + " ".join(controls))
    if calls:
        stages.append("call " + " ".join(dict.fromkeys(calls)))
    if assertions:
        stages.append("verify " + " ".join(dict.fromkeys(assertions)))
    if not stages:
        stages.append("transform " + " ".join(_salient_terms(text, exclude=set(), limit=12)))
    return " -> ".join(stages)[:2048]


def _retrieval_metrics(
    query: str,
    text: str,
    *,
    path: str,
    symbols: Sequence[str],
    graph_hop: int | None,
    quality: QualityVector,
    target_plan: str,
    example_plan: str,
) -> dict[str, float]:
    query_tokens = _tokens(query)
    path_score = _overlap(query_tokens, _tokens(path))
    dependency_score = min(
        1.0,
        0.55 * _overlap(query_tokens, _tokens(_IMPORT.findall(text)))
        + 0.45 * _overlap(query_tokens, _tokens(_CALL.findall(text))),
    )
    structure_score = min(
        1.0,
        path_score
        + (0.12 if len(Path(path).parts) >= 4 else 0.04)
        + (0.12 if _TYPE.search(text) else 0.0),
    )
    return {
        "lexical": _overlap(query_tokens, _tokens(text)),
        "semantic": _semantic_similarity(query, text),
        "path": path_score,
        "symbol": _overlap(query_tokens, _tokens(symbols)),
        "dependency": dependency_score,
        "structure": structure_score,
        "call_graph": 0.0 if graph_hop is None else 1.0 / (1.0 + graph_hop),
        "plan_alignment": _semantic_similarity(target_plan or query, example_plan or text),
        "quality": 0.55 * quality.coqu_ir + 0.45 * quality.example_quality,
    }


def _adaptive_weights(
    query: str, metrics: Mapping[str, float] | None = None
) -> dict[str, float]:
    metrics = metrics or {}
    lowered = query.casefold()
    identifiers = _IDENTIFIER.findall(query)
    weights = {
        "lexical": 0.95,
        "semantic": 1.05,
        "path": 0.55,
        "symbol": 0.75,
        "dependency": 0.80,
        "structure": 0.65,
        "call_graph": 0.70,
        "plan_alignment": 1.10,
    }
    if any("/" in token or "." in token for token in _TOKEN.findall(query)):
        weights["path"] += 0.75
        weights["structure"] += 0.40
    if any(
        token not in _KEYWORDS and any(char.isupper() for char in token[1:])
        for token in identifiers
    ):
        weights["symbol"] += 0.85
        weights["dependency"] += 0.45
    if any(word in lowered for word in ("call", "dependency", "depends", "bind", "api", "register")):
        weights["dependency"] += 0.85
        weights["call_graph"] += 0.75
    if any(word in lowered for word in ("plan", "flow", "implement", "behavior", "algorithm")):
        weights["plan_alignment"] += 0.80
    if float(metrics.get("multipath_coverage", 0.0)) > 0:
        agreement = min(1.0, float(metrics.get("multipath_coverage", 0.0)))
        weights["semantic"] += 0.25 * agreement
        weights["structure"] += 0.20 * agreement
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def _adaptive_score(
    query: str,
    metrics: Mapping[str, float],
    quality: QualityVector | None,
) -> float:
    quality = quality or QualityVector(0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7)
    weights = _adaptive_weights(query, metrics)
    score = sum(weights[key] * float(metrics.get(key, 0.0)) for key in weights)
    rrf = min(1.0, 20.0 * float(metrics.get("multipath_rrf", 0.0)))
    score = 0.90 * score + 0.10 * rrf
    quality_floor = min(
        quality.correctness,
        quality.security,
        quality.maintainability,
    )
    quality_mix = 0.58 * quality.coqu_ir + 0.42 * quality.example_quality
    return max(0.0, min(1.0, score * (0.50 + 0.30 * quality_floor + 0.20 * quality_mix)))


def _quality(text: str, *, path: str) -> QualityVector:
    lowered = text.casefold()
    lines = text.splitlines() or [""]
    nonempty = [line for line in lines if line.strip()]
    line_count = len(lines)
    long_ratio = sum(len(line) > 120 for line in nonempty) / max(1, len(nonempty))
    todo_count = sum(marker in lowered for marker in ("todo", "fixme", "unsupportedoperationexception"))
    correctness = 1.0 - min(0.55, 0.14 * todo_count)
    if text.count("{") != text.count("}"):
        correctness -= 0.30
    if text.count("(") != text.count(")"):
        correctness -= 0.18
    path_lower = path.replace("\\", "/").casefold()
    if "/test/" in path_lower or path_lower.endswith("test.java"):
        correctness += 0.08
    correctness = max(0.0, min(1.0, correctness))

    nested_loop = bool(
        re.search(r"\bfor\s*\([^)]*\)\s*\{[^{}]{0,1000}\bfor\s*\(", text, re.DOTALL)
    )
    efficiency = max(
        0.0,
        1.0
        - (0.18 if nested_loop else 0.0)
        - (0.24 if "thread.sleep(" in lowered else 0.0)
        - (0.12 if lowered.count(".readallbytes(") + lowered.count(".readstring(") >= 2 else 0.0)
        - (0.10 if lowered.count("new arraylist") >= 4 else 0.0),
    )

    dangerous_count = sum(marker in lowered for marker in _DANGEROUS)
    security = max(0.0, 1.0 - min(0.88, 0.24 * dangerous_count))

    comments = sum(line.strip().startswith(("//", "/*", "*")) for line in lines)
    maintainability = 1.0 - min(0.45, long_ratio * 0.7)
    if line_count > 240:
        maintainability -= min(0.28, (line_count - 240) / 900)
    if 0.03 <= comments / max(1, line_count) <= 0.35:
        maintainability += 0.05
    if len(set(_CALL.findall(text))) > 64:
        maintainability -= 0.08
    maintainability = max(0.0, min(1.0, maintainability))

    complexity_fit = math.exp(-((line_count - 55.0) / 70.0) ** 2)
    readable_names = sum(
        len(name) >= 3 and name.casefold() not in {"tmp", "foo", "bar", "obj", "val", "var"}
        for name in _IDENTIFIER.findall(text)
    )
    readability = 1.0 - min(0.38, long_ratio * 0.85)
    if readable_names < max(1, line_count // 20):
        readability -= 0.14
    readability = max(0.0, min(1.0, readability))

    plan = _code_plan(text)
    stage_count = plan.count("->") + (1 if plan else 0)
    stepwise_clarity = min(1.0, 0.20 + 0.16 * min(stage_count, 5))
    if _ASSERTION.search(text):
        stepwise_clarity = min(1.0, stepwise_clarity + 0.08)
    if line_count < 4:
        stepwise_clarity *= 0.5

    return QualityVector(
        correctness,
        efficiency,
        security,
        maintainability,
        complexity_fit,
        readability,
        stepwise_clarity,
    )


def _combine_examples(
    items: Sequence[Evidence],
    *,
    limit: int,
    required_steps: set[str],
    required_symbols: set[str],
) -> list[Evidence]:
    selected: list[Evidence] = []
    covered_symbols: set[str] = set()
    covered_steps: set[str] = set()
    covered_plans: set[str] = set()
    remaining = list(items)
    while remaining and len(selected) < limit:
        def marginal(item: Evidence) -> tuple[float, str]:
            quality = item.quality.example_quality if item.quality else 0.5
            symbol_gain = len((set(item.symbols) & required_symbols) - covered_symbols)
            generic_symbol_gain = len(set(item.symbols) - covered_symbols)
            step_gain = len((item.plan_steps & required_steps) - covered_steps)
            plan_key = item.algorithmic_plan.casefold()
            plan_gain = 1.0 if plan_key and plan_key not in covered_plans else 0.0
            duplicate_path = any(old.path == item.path for old in selected)
            return (
                item.bestfit_score
                + 0.20 * symbol_gain
                + 0.05 * min(3, generic_symbol_gain)
                + 0.18 * step_gain
                + 0.10 * plan_gain
                + 0.20 * quality
                - (0.16 if duplicate_path else 0.0),
                item.evidence_id,
            )

        best = max(remaining, key=marginal)
        selected.append(best)
        covered_symbols.update(best.symbols)
        covered_steps.update(best.plan_steps)
        if best.algorithmic_plan:
            covered_plans.add(best.algorithmic_plan.casefold())
        remaining.remove(best)
    return selected


def _dedupe_evidence(items: Sequence[Evidence]) -> list[Evidence]:
    result: dict[str, Evidence] = {}
    for item in items:
        current = result.get(item.evidence_id)
        if current is None or item.bestfit_score > current.bestfit_score:
            if current is not None:
                item.plan_steps.update(current.plan_steps)
                item.metrics.update(current.metrics)
            result[item.evidence_id] = item
        else:
            current.plan_steps.update(item.plan_steps)
            current.metrics.update(item.metrics)
    return list(result.values())


def _line_offsets(text: str) -> list[int]:
    return [0, *(match.end() for match in re.finditer("\n", text))]


def _offset_line(offsets: Sequence[int], offset: int) -> int:
    low, high = 0, len(offsets)
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
        if char in {'\"', "'"}:
            in_string, quote = True, char
        elif char == "{":
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
