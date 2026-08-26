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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


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
    environment: str = "common"  # "client", "server", "common"
    source_set: str = "main"      # "main", "test", "datagen"
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
    dependency_type: str  # "import", "registry", "model_parent", "texture_ref", "mixin_target", "entrypoint"

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
    if p.endswith("fabric.mod.json") or p.endswith("mods.toml") or p.endswith("neoforge.mods.toml"):
        return ArtifactKind.MOD_METADATA
    if p.endswith(".gradle") or p.endswith(".gradle.kts") or p.endswith(".properties") or p.endswith(".toml"):
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
        env = "common"
        source_set = "test"
    elif "src/gametest/" in p:
        env = "common"
        source_set = "gametest"
    elif "src/datagen/" in p:
        env = "common"
        source_set = "datagen"
    return source_set, env


def _infer_namespace_and_logical_id(rel_path: str, kind: ArtifactKind) -> tuple[str, str, str]:
    """Infer namespace, logical_id, and environment ('client' | 'server' | 'common') from standard paths."""
    norm = rel_path.replace("\\", "/").strip("/")
    low = norm.lower()

    env = "common"
    if "client" in low:
        env = "client"
    elif "server" in low or "dedicated" in low:
        env = "server"

    # Match assets/<namespace>/... or data/<namespace>/...
    m = re.search(r"(?:assets|data)/([a-zA-Z0-9_.-]+)/(.+)", norm)
    if m:
        namespace = m.group(1)
        subpath = m.group(2)
        # Strip model/texture/recipe prefix directory and extension
        logical_id = re.sub(r"^(?:models|textures|blockstates|loot_tables|recipes|tags)/", "", subpath)
        logical_id = re.sub(r"\.[a-zA-Z0-9]+$", "", logical_id)
        return namespace, logical_id, env

    # Java/Kotlin package & class name
    if kind in {ArtifactKind.JAVA_SOURCE, ArtifactKind.KOTLIN_SOURCE}:
        cleaned = re.sub(r"^src/(?:main|test|datagen)/(?:java|kotlin)/", "", norm)
        cleaned = re.sub(r"\.(?:java|kt)$", "", cleaned)
        parts = cleaned.split("/")
        namespace = parts[0] if len(parts) > 1 else "common"
        logical_id = ".".join(parts)
        return namespace, logical_id, env

    stem = Path(norm).stem
    return "common", stem, env


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
        """Parse all provided files with schema-aware extractors and construct the directional dependency graph."""
        graph = cls()

        # Indexes for fast resolution
        fqcn_to_node: dict[str, str] = {}
        symbol_to_nodes: dict[str, list[str]] = {}
        logical_res_to_nodes: dict[str, list[str]] = {}  # "namespace:path" -> [node_id]

        # 1. Create nodes and index them
        for rel_path, content in files.items():
            kind = _classify_artifact_kind(rel_path)
            namespace, logical_id, env = _infer_namespace_and_logical_id(rel_path, kind)
            node_id = rel_path

            defined_syms: list[str] = []
            if known_symbols and rel_path in known_symbols:
                defined_syms.extend(known_symbols[rel_path])
            elif kind in {ArtifactKind.JAVA_SOURCE, ArtifactKind.KOTLIN_SOURCE}:
                defined_syms.append(Path(rel_path).stem)

            node = ArtifactNode(
                id=node_id,
                kind=kind,
                namespace=namespace,
                logical_id=logical_id,
                environment=env,
                rel_path=rel_path,
                symbols_defined=tuple(dict.fromkeys(defined_syms)),
            )
            graph.add_node(node)

            if kind in {ArtifactKind.JAVA_SOURCE, ArtifactKind.KOTLIN_SOURCE}:
                fqcn_to_node[logical_id] = node_id
                for sym in node.symbols_defined:
                    symbol_to_nodes.setdefault(sym, []).append(node_id)

            if namespace and logical_id:
                res_key = f"{namespace}:{logical_id}".lower()
                logical_res_to_nodes.setdefault(res_key, []).append(node_id)
                # Also index short stem
                logical_res_to_nodes.setdefault(logical_id.split("/")[-1].lower(), []).append(node_id)

        def resolve_and_link(source_id: str, target_ref: str, relation: str) -> None:
            clean_ref = str(target_ref or "").strip()
            if not clean_ref:
                return

            # Check exact FQCN
            if clean_ref in fqcn_to_node:
                graph.add_edge(source_id, fqcn_to_node[clean_ref])
                return

            # Check exact resource key or stem match
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
                    m for m in candidates
                    if m != source_id and (allowed_kinds is None or graph.nodes[m].kind in allowed_kinds)
                ]
                if len(valid_matches) == 1:
                    graph.add_edge(source_id, valid_matches[0])
                elif len(valid_matches) > 1:
                    kinds = {graph.nodes[m].kind for m in valid_matches}
                    if relation in {"registry", "data_ref"} and len(kinds) == len(valid_matches):
                        for m in valid_matches:
                            graph.add_edge(source_id, m)
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

            # Check symbol table
            sym_key = clean_ref.split(".")[-1]
            if sym_key in symbol_to_nodes:
                matches = symbol_to_nodes[sym_key]
                valid_matches = [m for m in matches if m != source_id]
                if len(valid_matches) == 1:
                    graph.add_edge(source_id, valid_matches[0])
                    return
                elif len(valid_matches) > 1:
                    graph.ambiguous_edges.append(
                        UnresolvedArtifactEdge(
                            source_id=source_id,
                            requested_target=clean_ref,
                            relation=relation,
                            reason="AMBIGUOUS_SYMBOL_COLLISION",
                        )
                    )
                    return

            # Target not found in workspace -> record unresolved edge
            graph.unresolved_edges.append(
                UnresolvedArtifactEdge(
                    source_id=source_id,
                    requested_target=clean_ref,
                    relation=relation,
                    reason="TARGET_NODE_NOT_FOUND",
                )
            )

        # 2. Extract references with schema-specific parsers
        for rel_path, content in files.items():
            text = content if isinstance(content, str) else content.decode("utf-8", errors="ignore")
            source_id = rel_path
            kind = graph.nodes[source_id].kind

            # Layer 1: Java / Kotlin source code
            if kind in {ArtifactKind.JAVA_SOURCE, ArtifactKind.KOTLIN_SOURCE}:
                # Imports
                for imp in re.findall(r"(?:import|import\s+static)\s+([a-zA-Z0-9_.*]+);?", text):
                    resolve_and_link(source_id, imp.strip(), "import")

                # Registry calls e.g. Identifier.of("modid", "boss") or new Identifier("modid", "boss")
                for mod, path in re.findall(r'(?:Identifier\.of|new\s+Identifier)\s*\(\s*["\']([a-zA-Z0-9_.-]+)["\']\s*,\s*["\']([a-zA-Z0-9_/.-]+)["\']\s*\)', text):
                    resolve_and_link(source_id, f"{mod}:{path}", "registry")

                # Known class names referenced in text
                for sym, target_nodes in symbol_to_nodes.items():
                    if sym in text and any(t != source_id for t in target_nodes):
                        for t in target_nodes:
                            if t != source_id:
                                graph.add_edge(source_id, t)

            # Layer 2: Model JSON
            elif kind == ArtifactKind.MODEL_JSON:
                try:
                    data = json.loads(text)
                    if isinstance(data, dict):
                        if "parent" in data and isinstance(data["parent"], str):
                            resolve_and_link(source_id, data["parent"], "model_parent")
                        if "textures" in data and isinstance(data["textures"], dict):
                            for tex in data["textures"].values():
                                if isinstance(tex, str) and not tex.startswith("#"):
                                    resolve_and_link(source_id, tex, "texture_ref")
                except Exception:
                    pass

            # Layer 3: Blockstate JSON
            elif kind == ArtifactKind.BLOCKSTATE_JSON:
                try:
                    data = json.loads(text)
                    if isinstance(data, dict):
                        for model_ref in re.findall(r'"model":\s*"([^"]+)"', text):
                            resolve_and_link(source_id, model_ref, "blockstate_model")
                except Exception:
                    pass

            # Layer 4: Loot Table JSON & Recipe JSON & Tag JSON
            elif kind in {ArtifactKind.LOOT_TABLE_JSON, ArtifactKind.RECIPE_JSON, ArtifactKind.TAG_JSON}:
                for ref in re.findall(r'"([a-zA-Z0-9_.-]+:[a-zA-Z0-9_/.-]+)"', text):
                    if not ref.startswith("minecraft:"):
                        resolve_and_link(source_id, ref, "data_ref")

            # Layer 5: Mod Metadata & Mixin JSON
            elif kind in {ArtifactKind.MOD_METADATA, ArtifactKind.MIXIN_CONFIG}:
                for sym, target_nodes in symbol_to_nodes.items():
                    if sym in text:
                        for t in target_nodes:
                            if t != source_id:
                                graph.add_edge(source_id, t)

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
        """Compute exact directional transitive closures on the SCC Condensation DAG.

        For each seed node: closure = seed SCC + all reachable outgoing required dependency SCCs (A -> B).
        Reverse dependents (nodes that require the seed) are strictly excluded.
        """
        sccs = self.compute_scc()
        node_to_scc: dict[str, int] = {}
        for scc_idx, scc_nodes in enumerate(sccs):
            for n in scc_nodes:
                node_to_scc[n] = scc_idx

        # Build SCC Condensation DAG: directed edge scc_u -> scc_v if u -> v (u requires v)
        scc_dag_adj: dict[int, set[int]] = {i: set() for i in range(len(sccs))}
        for u, targets in self.adjacency.items():
            u_scc = node_to_scc.get(u)
            if u_scc is None:
                continue
            for v in targets:
                v_scc = node_to_scc.get(v)
                if v_scc is not None and u_scc != v_scc:
                    scc_dag_adj[u_scc].add(v_scc)

        def get_reachable_sccs(start_scc: int) -> set[int]:
            visited: set[int] = {start_scc}
            queue = [start_scc]
            while queue:
                curr = queue.pop(0)
                for neighbor in scc_dag_adj.get(curr, ()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            return visited

        subgraphs: list[list[str]] = []
        seen_closures: set[tuple[str, ...]] = set()

        if seed_nodes is not None:
            # Directional closure starting from each explicit seed node
            for seed in seed_nodes:
                seed_scc = node_to_scc.get(seed)
                if seed_scc is None:
                    continue
                reachable = get_reachable_sccs(seed_scc)
                closure_nodes: list[str] = []
                for s_idx in reachable:
                    closure_nodes.extend(sccs[s_idx])
                sorted_closure = tuple(sorted(closure_nodes))
                if sorted_closure not in seen_closures:
                    seen_closures.add(sorted_closure)
                    subgraphs.append(list(sorted_closure))
            return subgraphs
        else:
            # Directional closures for all root/maximal SCC closures
            all_closures: list[set[str]] = []
            for scc_idx in range(len(sccs)):
                reachable = get_reachable_sccs(scc_idx)
                closure_nodes_set: set[str] = set()
                for s_idx in reachable:
                    closure_nodes_set.update(sccs[s_idx])
                all_closures.append(closure_nodes_set)

            maximal_closures: list[list[str]] = []
            for i, c_set in enumerate(all_closures):
                is_subsumed = any(
                    i != j and c_set < other_set
                    for j, other_set in enumerate(all_closures)
                )
                if not is_subsumed:
                    sorted_comp = sorted(c_set)
                    if sorted_comp not in maximal_closures:
                        maximal_closures.append(sorted_comp)

            return maximal_closures
