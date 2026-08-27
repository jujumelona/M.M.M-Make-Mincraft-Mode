from __future__ import annotations

"""Multi-Layer Typed Directional Artifact Dependency Graph and SCC Closure Engine.

Builds a directed dependency graph A -> B (A requires B) across 5 typed layers:
1. Java/Kotlin symbols, imports, and type references
2. Minecraft registry identifiers (modid:path) linking Java to data/assets
3. JSON model, texture, blockstate, recipe, and loot table links
4. Mod metadata entrypoints and Mixin configuration targets/members
5. Access wideners and build script source sets

Computes Strongly Connected Components (SCC) via Tarjan's algorithm and extracts
exact directional transitive closures for atomic, isolated subgraph compilation proof.
"""

import json
import re
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

_IDENTIFIER_TOKEN = re.compile(r"\b[A-Za-z_$][A-Za-z0-9_$]*\b")


class ArtifactKind(str, Enum):
    JAVA_SOURCE = "JAVA_SOURCE"
    KOTLIN_SOURCE = "KOTLIN_SOURCE"
    REGISTRY_ENTRY = "REGISTRY_ENTRY"
    BLOCKSTATE_JSON = "BLOCKSTATE_JSON"
    MODEL_JSON = "MODEL_JSON"
    TEXTURE_PNG = "TEXTURE_PNG"
    LOOT_TABLE_JSON = "LOOT_TABLE_JSON"
    RECIPE_JSON = "RECIPE_JSON"
    TAG_JSON = "TAG_JSON"
    MIXIN_CONFIG = "MIXIN_CONFIG"
    ACCESS_WIDENER = "ACCESS_WIDENER"
    MOD_METADATA = "MOD_METADATA"
    BUILD_SCRIPT = "BUILD_SCRIPT"
    OTHER = "OTHER"


@dataclass(frozen=True)
class ArtifactNode:
    id: str
    kind: ArtifactKind
    namespace: str = "common"
    logical_id: str = ""
    environment: str = "common"
    source_set: str = "main"
    rel_path: str = ""
    symbols_defined: tuple[str, ...] = ()
    symbols_referenced: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "namespace": self.namespace,
            "logical_id": self.logical_id,
            "environment": self.environment,
            "source_set": self.source_set,
            "rel_path": self.rel_path,
            "symbols_defined": list(self.symbols_defined),
            "symbols_referenced": list(self.symbols_referenced),
        }


@dataclass(frozen=True)
class ArtifactEdge:
    source_id: str
    target_id: str
    dependency_type: str
    is_unresolved: bool = False
    is_mandatory: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "dependency_type": self.dependency_type,
            "is_unresolved": self.is_unresolved,
            "is_mandatory": self.is_mandatory,
        }


@dataclass(frozen=True)
class UnresolvedArtifactEdge:
    source_id: str
    requested_target: str
    relation: str
    reason: str = "TARGET_NODE_NOT_FOUND"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "requested_target": self.requested_target,
            "relation": self.relation,
            "reason": self.reason,
        }


def _classify_artifact_kind(path_str: str) -> ArtifactKind:
    p = path_str.lower().replace("\\", "/")
    if p.endswith(".java"):
        return ArtifactKind.JAVA_SOURCE
    if p.endswith(".kt"):
        return ArtifactKind.KOTLIN_SOURCE
    if "/textures/" in p and p.endswith(".png"):
        return ArtifactKind.TEXTURE_PNG
    if "/models/" in p and p.endswith(".json"):
        return ArtifactKind.MODEL_JSON
    if "/blockstates/" in p and p.endswith(".json"):
        return ArtifactKind.BLOCKSTATE_JSON
    if "/loot_tables/" in p and p.endswith(".json"):
        return ArtifactKind.LOOT_TABLE_JSON
    if "/recipes/" in p and p.endswith(".json"):
        return ArtifactKind.RECIPE_JSON
    if "/tags/" in p and p.endswith(".json"):
        return ArtifactKind.TAG_JSON
    if "mixin" in p and p.endswith(".json"):
        return ArtifactKind.MIXIN_CONFIG
    if p.endswith(".accesswidener"):
        return ArtifactKind.ACCESS_WIDENER
    if (
        p.endswith("fabric.mod.json")
        or p.endswith("mods.toml")
        or p.endswith("neoforge.mods.toml")
    ):
        return ArtifactKind.MOD_METADATA
    if (
        p.endswith(".gradle")
        or p.endswith(".gradle.kts")
        or p.endswith(".properties")
        or p.endswith(".toml")
    ):
        return ArtifactKind.BUILD_SCRIPT
    return ArtifactKind.OTHER


_ALLOWED_TARGET_KINDS: dict[str, set[ArtifactKind]] = {
    "model_parent": {ArtifactKind.MODEL_JSON},
    "texture_ref": {ArtifactKind.TEXTURE_PNG},
    "blockstate_model": {ArtifactKind.MODEL_JSON},
    "mixin_target": {ArtifactKind.JAVA_SOURCE, ArtifactKind.KOTLIN_SOURCE},
    "entrypoint": {ArtifactKind.JAVA_SOURCE, ArtifactKind.KOTLIN_SOURCE},
    "tag_ref": {ArtifactKind.TAG_JSON},
    "import": {ArtifactKind.JAVA_SOURCE, ArtifactKind.KOTLIN_SOURCE},
    "registry": {
        ArtifactKind.MODEL_JSON,
        ArtifactKind.TEXTURE_PNG,
        ArtifactKind.LOOT_TABLE_JSON,
        ArtifactKind.RECIPE_JSON,
        ArtifactKind.TAG_JSON,
        ArtifactKind.BLOCKSTATE_JSON,
    },
    "data_ref": {
        ArtifactKind.LOOT_TABLE_JSON,
        ArtifactKind.RECIPE_JSON,
        ArtifactKind.TAG_JSON,
        ArtifactKind.MODEL_JSON,
    },
}


def _infer_source_set_and_env(path_str: str) -> tuple[str, str]:
    p = path_str.replace("\\", "/")
    env = "common"
    source_set = "main"
    if "src/client/" in p or "client" in p.lower():
        env = "client"
        source_set = "main"
    elif "src/server/" in p or "server" in p.lower():
        env = "server"
        source_set = "main"
    elif "src/test/" in p:
        source_set = "test"
    elif "src/gametest/" in p:
        source_set = "gametest"
    elif "src/datagen/" in p:
        source_set = "datagen"
    return source_set, env


def _infer_namespace_and_logical_id(
    rel_path: str,
    kind: ArtifactKind,
) -> tuple[str, str, str]:
    """Infer namespace, logical ID and execution environment from standard paths."""

    norm = rel_path.replace("\\", "/").strip("/")
    low = norm.lower()

    env = "common"
    if "client" in low:
        env = "client"
    elif "server" in low or "dedicated" in low:
        env = "server"

    match = re.search(r"(?:assets|data)/([a-zA-Z0-9_.-]+)/(.+)", norm)
    if match:
        namespace = match.group(1)
        subpath = match.group(2)
        logical_id = re.sub(
            r"^(?:models|textures|blockstates|loot_tables|recipes|tags)/",
            "",
            subpath,
        )
        logical_id = re.sub(r"\.[a-zA-Z0-9]+$", "", logical_id)
        return namespace, logical_id, env

    if kind in {ArtifactKind.JAVA_SOURCE, ArtifactKind.KOTLIN_SOURCE}:
        cleaned = re.sub(
            r"^src/(?:main|test|datagen)/(?:java|kotlin)/",
            "",
            norm,
        )
        cleaned = re.sub(r"\.(?:java|kt)$", "", cleaned)
        parts = cleaned.split("/")
        namespace = parts[0] if len(parts) > 1 else "common"
        logical_id = ".".join(parts)
        return namespace, logical_id, env

    stem = Path(norm).stem
    return "common", stem, env


def _simple_symbol(value: str) -> str:
    return str(value or "").rsplit(".", 1)[-1].rsplit("$", 1)[-1]


class ArtifactDependencyGraph:
    """Directed multi-layer dependency graph for complete mod artifact linkages."""

    @staticmethod
    def kind_for_path(path_str: str) -> ArtifactKind:
        """Expose the canonical path classifier to persisted repository graph users."""

        return _classify_artifact_kind(path_str)

    def __init__(self, target_context: Mapping[str, Any] | None = None) -> None:
        self.nodes: dict[str, ArtifactNode] = {}
        self.adjacency: dict[str, set[str]] = {}
        self.unresolved_edges: list[UnresolvedArtifactEdge] = []
        self.ambiguous_edges: list[UnresolvedArtifactEdge] = []
        context = dict(target_context or {})
        self.target_modid = str(context.get("target_modid") or "").strip().casefold()
        self.target_package = str(context.get("target_package") or "").strip()
        self.owned_namespaces = frozenset(
            str(value).strip().casefold()
            for value in context.get("owned_namespaces", ())
            if str(value).strip()
        )
        self.owned_packages = tuple(
            dict.fromkeys(
                str(value).strip()
                for value in context.get("owned_packages", ())
                if str(value).strip()
            )
        )

    @property
    def edges(self) -> tuple[ArtifactEdge, ...]:
        """Expose resolved and classified unresolved edges for final assembly gates."""

        values: list[ArtifactEdge] = []
        for source_id, targets in self.adjacency.items():
            values.extend(
                ArtifactEdge(
                    source_id=source_id,
                    target_id=target_id,
                    dependency_type="reference",
                )
                for target_id in sorted(targets)
            )
        for edge in (*self.unresolved_edges, *self.ambiguous_edges):
            values.append(
                ArtifactEdge(
                    source_id=edge.source_id,
                    target_id=edge.requested_target,
                    dependency_type=edge.relation,
                    is_unresolved=True,
                    is_mandatory=self.is_mandatory_unresolved(edge),
                )
            )
        return tuple(values)

    def is_mandatory_unresolved(self, edge: UnresolvedArtifactEdge) -> bool:
        """Return True only for unresolved references owned by the assembled target.

        JDK, Minecraft, loader and third-party classes/resources are external build
        dependencies and must be validated by Gradle, not mistaken for missing local
        artifacts. Explicit references to the target package or target resource
        namespace are mandatory and therefore fail final assembly when unresolved.
        """

        target = str(edge.requested_target or "").strip()
        if not target:
            return False
        low = target.casefold()

        if edge.relation in {"import", "mixin_target", "entrypoint"}:
            owned_packages = tuple(
                dict.fromkeys(
                    (
                        *((self.target_package,) if self.target_package else ()),
                        *self.owned_packages,
                    )
                )
            )
            if any(
                target == package or target.startswith(package + ".")
                for package in owned_packages
            ):
                return True
            external_prefixes = (
                "java.",
                "javax.",
                "jdk.",
                "kotlin.",
                "org.",
                "com.google.",
                "com.mojang.",
                "net.minecraft.",
                "net.fabricmc.",
                "net.minecraftforge.",
                "net.neoforged.",
            )
            return not target.startswith(external_prefixes) and "." not in target

        if low.startswith("minecraft:"):
            return False
        if edge.relation == "model_parent" and ":" not in low and low.startswith(
            ("item/", "block/", "builtin/")
        ):
            return False
        if ":" in low:
            namespace = low.split(":", 1)[0]
            return namespace in (
                self.owned_namespaces
                | ({self.target_modid} if self.target_modid else set())
            )

        if edge.relation in {
            "model_parent",
            "texture_ref",
            "blockstate_model",
            "registry",
            "data_ref",
            "tag_ref",
        }:
            return bool(self.target_modid or self.owned_namespaces)
        return False

    def add_node(self, node: ArtifactNode) -> None:
        self.nodes[node.id] = node
        self.adjacency.setdefault(node.id, set())

    def add_edge(self, source_id: str, target_id: str) -> None:
        if source_id in self.nodes and target_id in self.nodes:
            self.adjacency.setdefault(source_id, set()).add(target_id)
        elif source_id in self.nodes:
            self.unresolved_edges.append(
                UnresolvedArtifactEdge(
                    source_id=source_id,
                    requested_target=target_id,
                    relation="reference",
                    reason="TARGET_NODE_NOT_FOUND",
                )
            )

    def is_closure_complete(self, node_ids: Sequence[str]) -> bool:
        """Check whether the closure has no unresolved or ambiguous required edges."""

        node_set = set(node_ids)
        for edge in self.unresolved_edges:
            if edge.source_id in node_set and self.is_mandatory_unresolved(edge):
                return False
        for edge in self.ambiguous_edges:
            if edge.source_id in node_set:
                return False
        return True

    @classmethod
    def build_from_files(
        cls,
        files: Mapping[str, Any],
        known_symbols: Mapping[str, Sequence[str]] | None = None,
        target_context: Mapping[str, Any] | None = None,
    ) -> ArtifactDependencyGraph:
        """Parse files with schema-aware extractors and build directional edges."""

        graph = cls(target_context=target_context)
        fqcn_to_node: dict[str, str] = {}
        symbol_to_nodes: dict[str, list[str]] = {}
        simple_symbol_to_nodes: dict[str, list[str]] = {}
        logical_res_to_nodes: dict[str, list[str]] = {}

        for rel_path, content in files.items():
            del content
            kind = _classify_artifact_kind(rel_path)
            namespace, logical_id, env = _infer_namespace_and_logical_id(
                rel_path,
                kind,
            )
            node_id = rel_path

            defined_syms: list[str] = []
            if known_symbols and rel_path in known_symbols:
                defined_syms.extend(known_symbols[rel_path])
            elif kind in {ArtifactKind.JAVA_SOURCE, ArtifactKind.KOTLIN_SOURCE}:
                defined_syms.append(Path(rel_path).stem)

            source_set, source_env = _infer_source_set_and_env(rel_path)
            node = ArtifactNode(
                id=node_id,
                kind=kind,
                namespace=namespace,
                logical_id=logical_id,
                environment=source_env if source_env != "common" else env,
                source_set=source_set,
                rel_path=rel_path,
                symbols_defined=tuple(dict.fromkeys(defined_syms)),
            )
            graph.add_node(node)

            if kind in {ArtifactKind.JAVA_SOURCE, ArtifactKind.KOTLIN_SOURCE}:
                fqcn_to_node[logical_id] = node_id
                for symbol in node.symbols_defined:
                    symbol_to_nodes.setdefault(symbol, []).append(node_id)
                    simple = _simple_symbol(symbol)
                    if simple:
                        simple_symbol_to_nodes.setdefault(simple, []).append(node_id)

            if namespace and logical_id:
                resource_key = f"{namespace}:{logical_id}".lower()
                logical_res_to_nodes.setdefault(resource_key, []).append(node_id)
                logical_res_to_nodes.setdefault(
                    logical_id.split("/")[-1].lower(),
                    [],
                ).append(node_id)

        def resolve_and_link(source_id: str, target_ref: str, relation: str) -> None:
            clean_ref = str(target_ref or "").strip()
            if not clean_ref:
                return

            if clean_ref in fqcn_to_node:
                graph.add_edge(source_id, fqcn_to_node[clean_ref])
                return

            ref_low = clean_ref.lower()
            ref_path = ref_low.split(":")[-1]
            ref_stem = ref_path.split("/")[-1]

            candidates = None
            if ref_low in logical_res_to_nodes:
                candidates = logical_res_to_nodes[ref_low]
            elif ref_path in logical_res_to_nodes:
                candidates = logical_res_to_nodes[ref_path]
            elif ref_stem in logical_res_to_nodes:
                candidates = logical_res_to_nodes[ref_stem]
            if candidates is not None:
                allowed_kinds = _ALLOWED_TARGET_KINDS.get(relation)
                valid_matches = [
                    match
                    for match in candidates
                    if match != source_id
                    and (
                        allowed_kinds is None
                        or graph.nodes[match].kind in allowed_kinds
                    )
                ]
                if len(valid_matches) == 1:
                    graph.add_edge(source_id, valid_matches[0])
                elif len(valid_matches) > 1:
                    kinds = {graph.nodes[match].kind for match in valid_matches}
                    if (
                        relation in {"registry", "data_ref"}
                        and len(kinds) == len(valid_matches)
                    ):
                        for match in valid_matches:
                            graph.add_edge(source_id, match)
                    else:
                        graph.ambiguous_edges.append(
                            UnresolvedArtifactEdge(
                                source_id=source_id,
                                requested_target=clean_ref,
                                relation=relation,
                                reason="AMBIGUOUS_REFERENCE_COLLISION",
                            )
                        )
                return

            symbol_key = clean_ref.split(".")[-1]
            exact_nodes = symbol_to_nodes.get(symbol_key)
            target_nodes = exact_nodes or simple_symbol_to_nodes.get(symbol_key)
            if target_nodes:
                valid_matches = [
                    match for match in target_nodes if match != source_id
                ]
                if len(valid_matches) == 1:
                    graph.add_edge(source_id, valid_matches[0])
                    return
                if len(valid_matches) > 1:
                    graph.ambiguous_edges.append(
                        UnresolvedArtifactEdge(
                            source_id=source_id,
                            requested_target=clean_ref,
                            relation=relation,
                            reason="AMBIGUOUS_SYMBOL_COLLISION",
                        )
                    )
                    return

            graph.unresolved_edges.append(
                UnresolvedArtifactEdge(
                    source_id=source_id,
                    requested_target=clean_ref,
                    relation=relation,
                    reason="TARGET_NODE_NOT_FOUND",
                )
            )

        for rel_path, content in files.items():
            text = (
                content
                if isinstance(content, str)
                else content.decode("utf-8", errors="ignore")
            )
            source_id = rel_path
            kind = graph.nodes[source_id].kind

            if kind in {ArtifactKind.JAVA_SOURCE, ArtifactKind.KOTLIN_SOURCE}:
                for imported in re.findall(
                    r"(?:import|import\s+static)\s+([a-zA-Z0-9_.*]+);?",
                    text,
                ):
                    resolve_and_link(source_id, imported.strip(), "import")

                identifier_pattern = (
                    r'(?:Identifier\.of|new\s+Identifier)\s*\(\s*'
                    r'["\']([a-zA-Z0-9_.-]+)["\']\s*,\s*'
                    r'["\']([a-zA-Z0-9_/.-]+)["\']\s*\)'
                )
                for mod, path in re.findall(identifier_pattern, text):
                    resolve_and_link(source_id, f"{mod}:{path}", "registry")
                for resource_id in re.findall(
                    r'["\']([a-z0-9_.-]+:[a-z0-9_/.-]+)["\']',
                    text,
                ):
                    if not resource_id.startswith(
                        ("minecraft:", "fabric:", "forge:", "neoforge:", "c:")
                    ):
                        resolve_and_link(source_id, resource_id, "registry")

                referenced_tokens = set(_IDENTIFIER_TOKEN.findall(text))
                for symbol in referenced_tokens.intersection(simple_symbol_to_nodes):
                    for target in simple_symbol_to_nodes[symbol]:
                        if target != source_id:
                            graph.add_edge(source_id, target)

            elif kind == ArtifactKind.MODEL_JSON:
                try:
                    data = json.loads(text)
                    if isinstance(data, dict):
                        if isinstance(data.get("parent"), str):
                            resolve_and_link(
                                source_id,
                                data["parent"],
                                "model_parent",
                            )
                        textures = data.get("textures")
                        if isinstance(textures, dict):
                            for texture in textures.values():
                                if isinstance(texture, str) and not texture.startswith("#"):
                                    resolve_and_link(
                                        source_id,
                                        texture,
                                        "texture_ref",
                                    )
                except Exception:
                    pass

            elif kind == ArtifactKind.BLOCKSTATE_JSON:
                try:
                    json.loads(text)
                    for model_ref in re.findall(r'"model":\s*"([^"]+)"', text):
                        resolve_and_link(
                            source_id,
                            model_ref,
                            "blockstate_model",
                        )
                except Exception:
                    pass

            elif kind in {
                ArtifactKind.LOOT_TABLE_JSON,
                ArtifactKind.RECIPE_JSON,
                ArtifactKind.TAG_JSON,
            }:
                for ref in re.findall(
                    r'"([a-zA-Z0-9_.-]+:[a-zA-Z0-9_/.-]+)"',
                    text,
                ):
                    if not ref.startswith("minecraft:"):
                        resolve_and_link(source_id, ref, "data_ref")

            elif kind in {ArtifactKind.MOD_METADATA, ArtifactKind.MIXIN_CONFIG}:
                referenced_tokens = set(_IDENTIFIER_TOKEN.findall(text))
                for symbol in referenced_tokens.intersection(simple_symbol_to_nodes):
                    for target in simple_symbol_to_nodes[symbol]:
                        if target != source_id:
                            graph.add_edge(source_id, target)

        return graph

    def compute_scc(self) -> list[list[str]]:
        """Compute strongly connected components with Tarjan's algorithm."""

        index = 0
        indices: dict[str, int] = {}
        lowlink: dict[str, int] = {}
        on_stack: set[str] = set()
        stack: list[str] = []
        sccs: list[list[str]] = []

        def strongconnect(vertex: str) -> None:
            nonlocal index
            indices[vertex] = index
            lowlink[vertex] = index
            index += 1
            stack.append(vertex)
            on_stack.add(vertex)

            for target in self.adjacency.get(vertex, ()):
                if target not in indices:
                    strongconnect(target)
                    lowlink[vertex] = min(lowlink[vertex], lowlink[target])
                elif target in on_stack:
                    lowlink[vertex] = min(lowlink[vertex], indices[target])

            if lowlink[vertex] == indices[vertex]:
                component: list[str] = []
                while True:
                    target = stack.pop()
                    on_stack.remove(target)
                    component.append(target)
                    if target == vertex:
                        break
                sccs.append(component)

        for node_id in self.nodes:
            if node_id not in indices:
                strongconnect(node_id)

        return sccs

    def compute_directional_closures(
        self,
        seed_nodes: Sequence[str] | None = None,
    ) -> list[list[str]]:
        """Compute directional transitive closures on the SCC condensation DAG.

        For the unseeded case only SCC roots can produce maximal directional
        closures, so non-root closure construction and pairwise subset scans are
        skipped entirely.
        """

        sccs = self.compute_scc()
        node_to_scc: dict[str, int] = {}
        for scc_index, scc_nodes in enumerate(sccs):
            for node in scc_nodes:
                node_to_scc[node] = scc_index

        scc_dag_adj: dict[int, set[int]] = {
            index: set() for index in range(len(sccs))
        }
        incoming = [0 for _ in sccs]
        for source, targets in self.adjacency.items():
            source_scc = node_to_scc.get(source)
            if source_scc is None:
                continue
            for target in targets:
                target_scc = node_to_scc.get(target)
                if (
                    target_scc is not None
                    and source_scc != target_scc
                    and target_scc not in scc_dag_adj[source_scc]
                ):
                    scc_dag_adj[source_scc].add(target_scc)
                    incoming[target_scc] += 1

        def reachable_sccs(start_scc: int) -> set[int]:
            visited: set[int] = {start_scc}
            queue: deque[int] = deque((start_scc,))
            while queue:
                current = queue.popleft()
                for neighbor in scc_dag_adj.get(current, ()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            return visited

        subgraphs: list[list[str]] = []
        seen_closures: set[tuple[str, ...]] = set()

        if seed_nodes is not None:
            start_sccs = [
                node_to_scc[seed]
                for seed in seed_nodes
                if seed in node_to_scc
            ]
        else:
            start_sccs = [
                index for index, count in enumerate(incoming) if count == 0
            ]

        for start_scc in dict.fromkeys(start_sccs):
            reachable = reachable_sccs(start_scc)
            closure_nodes: list[str] = []
            for scc_index in reachable:
                closure_nodes.extend(sccs[scc_index])
            sorted_closure = tuple(sorted(closure_nodes))
            if sorted_closure not in seen_closures:
                seen_closures.add(sorted_closure)
                subgraphs.append(list(sorted_closure))

        return subgraphs
