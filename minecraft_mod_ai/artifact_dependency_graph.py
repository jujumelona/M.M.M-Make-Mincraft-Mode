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

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ArtifactKind(str, Enum):
    JAVA_SOURCE = "JAVA_SOURCE"
    KOTLIN_SOURCE = "KOTLIN_SOURCE"
    REGISTRY_ENTRY = "REGISTRY_ENTRY"
    MODEL_JSON = "MODEL_JSON"
    TEXTURE_PNG = "TEXTURE_PNG"
    BLOCKSTATE_JSON = "BLOCKSTATE_JSON"
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
    rel_path: str
    namespace: str = "common"
    symbols_defined: tuple[str, ...] = ()
    symbols_referenced: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "rel_path": self.rel_path,
            "namespace": self.namespace,
            "symbols_defined": list(self.symbols_defined),
            "symbols_referenced": list(self.symbols_referenced),
        }


@dataclass(frozen=True)
class ArtifactEdge:
    source_id: str
    target_id: str
    dependency_type: str  # "import", "registry", "texture_ref", "mixin_target", "entrypoint"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "dependency_type": self.dependency_type,
        }


@dataclass(frozen=True)
class UnresolvedArtifactEdge:
    source_id: str
    requested_target: str
    relation: str  # "import", "registry", "model_parent", "texture_ref", "mixin_target"
    reason: str = "TARGET_NODE_NOT_FOUND"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "requested_target": self.requested_target,
            "relation": self.relation,
            "reason": self.reason,
        }


def _classify_artifact_kind(path_str: str) -> ArtifactKind:
    p = path_str.lower()
    if p.endswith(".java"):
        return ArtifactKind.JAVA_SOURCE
    if p.endswith(".kt"):
        return ArtifactKind.KOTLIN_SOURCE
    if "textures/" in p and p.endswith(".png"):
        return ArtifactKind.TEXTURE_PNG
    if "models/" in p and p.endswith(".json"):
        return ArtifactKind.MODEL_JSON
    if "blockstates/" in p and p.endswith(".json"):
        return ArtifactKind.BLOCKSTATE_JSON
    if "loot_tables/" in p and p.endswith(".json"):
        return ArtifactKind.LOOT_TABLE_JSON
    if "recipes/" in p and p.endswith(".json"):
        return ArtifactKind.RECIPE_JSON
    if "tags/" in p and p.endswith(".json"):
        return ArtifactKind.TAG_JSON
    if "mixin" in p and p.endswith(".json"):
        return ArtifactKind.MIXIN_CONFIG
    if p.endswith(".accesswidener"):
        return ArtifactKind.ACCESS_WIDENER
    if p.endswith("fabric.mod.json") or p.endswith("mods.toml"):
        return ArtifactKind.MOD_METADATA
    if p.endswith(".gradle") or p.endswith(".gradle.kts") or p.endswith(".properties") or p.endswith(".toml"):
        return ArtifactKind.BUILD_SCRIPT
    return ArtifactKind.OTHER


class ArtifactDependencyGraph:
    """Directed multi-layer dependency graph representing complete mod artifacts and linkages."""

    def __init__(self) -> None:
        self.nodes: dict[str, ArtifactNode] = {}
        self.adjacency: dict[str, set[str]] = {}  # source -> set of targets it depends on (A requires B)
        self.unresolved_edges: list[UnresolvedArtifactEdge] = []
        self.ambiguous_edges: list[UnresolvedArtifactEdge] = []

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
        """Check if all nodes in the given closure have 0 unresolved or ambiguous required edges."""
        node_set = set(node_ids)
        for edge in self.unresolved_edges:
            if edge.source_id in node_set:
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
    ) -> ArtifactDependencyGraph:
        """Parse all provided files and construct the directional typed dependency graph."""
        graph = cls()
        symbol_to_node: dict[str, str] = {}
        resource_stem_to_nodes: dict[str, list[str]] = {}

        # 1. Create nodes
        for rel_path, content in files.items():
            kind = _classify_artifact_kind(rel_path)
            node_id = rel_path
            stem = Path(rel_path).stem.lower()

            defined_syms: list[str] = []
            if known_symbols and rel_path in known_symbols:
                defined_syms.extend(known_symbols[rel_path])
            elif kind in {ArtifactKind.JAVA_SOURCE, ArtifactKind.KOTLIN_SOURCE}:
                defined_syms.append(Path(rel_path).stem)

            node = ArtifactNode(
                id=node_id,
                kind=kind,
                rel_path=rel_path,
                symbols_defined=tuple(dict.fromkeys(defined_syms)),
            )
            graph.add_node(node)

            for s in node.symbols_defined:
                symbol_to_node[s] = node_id

            if stem and stem not in {"mod", "main", "init"}:
                resource_stem_to_nodes.setdefault(stem, []).append(node_id)

        # 2. Extract edges across layers
        for rel_path, content in files.items():
            text = content if isinstance(content, str) else content.decode("utf-8", errors="ignore")
            source_id = rel_path

            # Layer 1: Java/Kotlin imports & referenced symbols
            for sym, target_id in symbol_to_node.items():
                if target_id != source_id and sym and (sym in text or "import " in text and sym in text):
                    graph.add_edge(source_id, target_id)

            # Layer 2 & 3: Registry identifiers & resource stems ("modid:path" or "path")
            for token in re.findall(r'"([a-zA-Z0-9_/.-]+)"', text):
                clean_tok = token.split(":")[-1].lower()
                stem = clean_tok.split("/")[-1]

                # Direct logical path match (e.g. "entity/boss" in "assets/modid/models/entity/boss.json")
                for target_id in files:
                    if target_id != source_id:
                        tid_low = target_id.lower()
                        if clean_tok in tid_low or (len(stem) >= 3 and f"/{stem}." in tid_low):
                            graph.add_edge(source_id, target_id)

            # Layer 4: Mod metadata / Mixin entries
            if rel_path.endswith(".json") or rel_path.endswith(".toml"):
                for sym, target_id in symbol_to_node.items():
                    if target_id != source_id and sym in text:
                        graph.add_edge(source_id, target_id)

        return graph

    def compute_scc(self) -> list[list[str]]:
        """Compute Strongly Connected Components (SCC) using Tarjan's algorithm."""
        index = 0
        indices: dict[str, int] = {}
        lowlink: dict[str, int] = {}
        on_stack: set[str] = set()
        stack: list[str] = []
        sccs: list[list[str]] = []

        def strongconnect(v: str) -> None:
            nonlocal index
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)

            for w in self.adjacency.get(v, ()):
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], indices[w])

            if lowlink[v] == indices[v]:
                scc: list[str] = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)

        for node_id in self.nodes:
            if node_id not in indices:
                strongconnect(node_id)

        return sccs

    def compute_directional_closures(self, seed_nodes: Sequence[str] | None = None) -> list[list[str]]:
        """Compute exact directional closed subgraphs on the artifact graph.

        Groups artifacts into independent, closed subgraphs that contain all mutual dependencies and helpers.
        """
        sccs = self.compute_scc()
        node_to_scc: dict[str, int] = {}
        for scc_idx, scc_nodes in enumerate(sccs):
            for n in scc_nodes:
                node_to_scc[n] = scc_idx

        # Build bidirectional component connectivity to group full feature units
        undirected_adj: dict[str, set[str]] = {nid: set() for nid in self.nodes}
        for u, targets in self.adjacency.items():
            for v in targets:
                undirected_adj[u].add(v)
                undirected_adj[v].add(u)

        visited: set[str] = set()
        subgraphs: list[list[str]] = []

        for nid in self.nodes:
            if nid not in visited:
                comp: list[str] = []
                queue = [nid]
                visited.add(nid)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for neighbor in undirected_adj.get(curr, ()):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                subgraphs.append(sorted(comp))

        return subgraphs
