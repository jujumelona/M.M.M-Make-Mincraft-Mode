from __future__ import annotations

import re
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"anchor drifted: {label}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# Capability locator: rank every evidence-bearing seed by default.  An explicit
# caller/operator may still impose a cap, but the reuse engine itself must not.
# ---------------------------------------------------------------------------
p = Path("minecraft_mod_ai/capability_implementation_locator.py")
s = p.read_text(encoding="utf-8")
s = replace_once(
    s,
    "        max_seeds: int = 8,\n",
    "        max_seeds: int | None = None,\n",
    "locator max_seeds signature",
)
s = replace_once(
    s,
    '        """Locate candidate implementation seeds using multi-evidence scoring."""\n',
    '''        """Locate all evidence-bearing implementation seeds by default.

        ``max_seeds`` is an explicit operator resource control only.  The reuse
        pipeline deliberately leaves it unset so ordinal rank cannot turn an
        existing implementation into a false negative.
        """
        if max_seeds is not None and max_seeds < 1:
            raise ValueError("max_seeds must be positive when explicitly configured.")
''',
    "locator docstring",
)
s = replace_once(
    s,
    "        ranked.sort(key=lambda s: (-s.score, s.node_id))\n        return tuple(ranked[:max_seeds])\n",
    '''        ranked.sort(key=lambda s: (-s.score, s.node_id))
        if max_seeds is None:
            return tuple(ranked)
        return tuple(ranked[:max_seeds])
''',
    "locator return",
)
p.write_text(s, encoding="utf-8")


# ---------------------------------------------------------------------------
# Source transplant: remove cardinality cutoffs; retain only configurable
# resource budgets, recover complete Git trees, and scope unreadable evidence to
# the selected dependency closure instead of rejecting the whole repository.
# ---------------------------------------------------------------------------
p = Path("minecraft_mod_ai/source_transplant.py")
s = p.read_text(encoding="utf-8")
s = s.replace("A repository hit is never a reusable implementation by itself. Reuse is admitted\nonly after an immutable commit, SPDX license, exact target metadata (or an explicit\nadaptation classification), a bounded Java source slice, and hashes for every source\nblob have been recorded.", "A repository hit is never a reusable implementation by itself. Reuse is admitted\nonly after an immutable commit, SPDX license, exact target metadata (or an explicit\nadaptation classification), a resource-budgeted dependency-complete source slice,\nand hashes for every source blob have been recorded.")

s = replace_once(
    s,
    "_MAX_TREE_FILES = 20_000\n_MAX_SEEDS = 8\n_MAX_CLOSURE_FILES = 64\n_MAX_SLICE_BYTES = 1024 * 1024\n",
    '''def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _slice_byte_budget() -> int:
    return _env_int(
        "MMM_SOURCE_TRANSPLANT_SLICE_BYTE_BUDGET",
        8 * 1024 * 1024,
        minimum=64 * 1024,
        maximum=512 * 1024 * 1024,
    )


def _single_blob_byte_budget() -> int:
    return _env_int(
        "MMM_SOURCE_TRANSPLANT_SINGLE_BLOB_BYTE_BUDGET",
        16 * 1024 * 1024,
        minimum=64 * 1024,
        maximum=512 * 1024 * 1024,
    )


def _response_byte_budget() -> int:
    return _env_int(
        "MMM_SOURCE_TRANSPLANT_RESPONSE_BYTE_BUDGET",
        16 * 1024 * 1024,
        minimum=256 * 1024,
        maximum=512 * 1024 * 1024,
    )


def _tree_request_budget() -> int:
    # Work budget only.  It never truncates a successfully enumerated tree.
    return _env_int(
        "MMM_SOURCE_TRANSPLANT_TREE_REQUEST_BUDGET",
        2048,
        minimum=8,
        maximum=100_000,
    )
''',
    "source transplant constants",
)

insert_anchor = '''def repository_from_candidate(candidate: Mapping[str, Any]) -> str:
'''
helpers = '''def _donor_test_paths(blobs: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            path
            for path in blobs
            if path.endswith((".java", ".kt"))
            and any(
                marker in f"/{path.casefold()}/"
                for marker in ("/test/", "/gametest/")
            )
        )
    )


def _closure_paths(
    graph: ArtifactDependencyGraph,
    seed_paths: Sequence[str],
) -> tuple[str, ...]:
    """Return the complete graph closure without an ordinal file-count cutoff."""

    selected: list[str] = []
    seen: set[str] = set()
    for path in seed_paths:
        if path not in seen:
            seen.add(path)
            selected.append(path)
    for closure in graph.compute_directional_closures(seed_paths):
        for path in closure:
            if path not in seen:
                seen.add(path)
                selected.append(path)
    return tuple(selected)


def _repository_tree_entries(
    client: httpx.Client,
    repository: str,
    commit_sha: str,
) -> tuple[Mapping[str, Any], ...]:
    """Resolve the complete immutable Git tree, recovering GitHub truncation.

    The recursive Git Trees endpoint is used as the fast path.  If GitHub marks it
    truncated (or the response exceeds the configured transport byte budget), the
    repository is walked subtree-by-subtree.  A repository is never discarded just
    because it crosses a local file-count threshold.
    """

    commit = _github_json(
        client,
        f"https://api.github.com/repos/{repository}/git/commits/{quote(commit_sha, safe='')}",
    )
    tree_meta = commit.get("tree") if isinstance(commit, Mapping) else None
    root_sha = str(tree_meta.get("sha") or "") if isinstance(tree_meta, Mapping) else ""
    if not re.fullmatch(r"[0-9a-f]{40,64}", root_sha):
        raise SourceTransplantError("Pinned donor commit did not expose an immutable tree SHA.")

    tree_url = f"https://api.github.com/repos/{repository}/git/trees/{quote(root_sha, safe='')}"
    try:
        recursive = _github_json(client, tree_url, params={"recursive": "1"})
    except SourceTransplantError:
        recursive = None
    if isinstance(recursive, Mapping) and recursive.get("truncated") is not True:
        entries = recursive.get("tree")
        if isinstance(entries, list):
            return tuple(item for item in entries if isinstance(item, Mapping))

    budget = _tree_request_budget()
    requests = 0
    queue: list[tuple[str, str]] = [(root_sha, "")]
    seen_trees: set[str] = set()
    resolved: list[Mapping[str, Any]] = []
    while queue:
        if requests >= budget:
            raise SourceTransplantError(
                "Complete donor tree traversal exhausted the configured request budget."
            )
        tree_sha, prefix = queue.pop(0)
        if tree_sha in seen_trees:
            continue
        seen_trees.add(tree_sha)
        requests += 1
        payload = _github_json(
            client,
            f"https://api.github.com/repos/{repository}/git/trees/{quote(tree_sha, safe='')}",
        )
        if not isinstance(payload, Mapping) or payload.get("truncated") is True:
            raise SourceTransplantError("Non-recursive donor subtree response was incomplete.")
        entries = payload.get("tree")
        if not isinstance(entries, list):
            raise SourceTransplantError("GitHub donor subtree response had no tree entries.")
        for raw in entries:
            if not isinstance(raw, Mapping):
                continue
            leaf = str(raw.get("path") or "").strip("/")
            if not leaf:
                continue
            full_path = f"{prefix}/{leaf}".strip("/")
            kind = str(raw.get("type") or "")
            sha = str(raw.get("sha") or "")
            if kind == "tree" and sha:
                queue.append((sha, full_path))
                continue
            item = dict(raw)
            item["path"] = full_path
            resolved.append(item)
    return tuple(resolved)


'''
if insert_anchor not in s:
    raise SystemExit("repository candidate anchor drifted")
s = s.replace(insert_anchor, helpers + insert_anchor, 1)

s = replace_once(
    s,
    '''        donor_tests = tuple(
            sorted(
                path for path in blobs
                if path.endswith(".java") and "/test/" in f"/{path.casefold()}/"
            )[:24]
        )
''',
    '''        donor_tests = _donor_test_paths(blobs)
''',
    "donor tests cap",
)

s = replace_once(
    s,
    '''        unreadable = tuple(index.metadata.get("unreadable_artifacts") or ())
        if unreadable:
            # A partial repository index cannot prove a complete dependency closure.
            return None

        seed_evidence = CapabilityImplementationLocator.locate_seeds(
            capability,
            index,
            max_seeds=_MAX_SEEDS,
        )
''',
    '''        unreadable = tuple(index.metadata.get("unreadable_artifacts") or ())

        seed_evidence = CapabilityImplementationLocator.locate_seeds(
            capability,
            index,
        )
''',
    "unreadable global rejection and seed cap",
)

closure_pattern = re.compile(
    r'''        selected: list\[str\] = \[\]\n        selected_set: set\[str\] = set\(\)\n        for seed_path in seed_paths:\n(?:.|\n)*?        truncation_reason = ""\n        if len\(selected\) > _MAX_CLOSURE_FILES:\n            truncation_reason = f"Exceeded MAX_CLOSURE_FILES \({_MAX_CLOSURE_FILES}\)"\n            selected = selected\[:_MAX_CLOSURE_FILES\]\n            selected_set = set\(selected\)\n'''
)
match = closure_pattern.search(s)
if match is None:
    raise SystemExit("closure cap block drifted")
s = s[: match.start()] + '''        selected = list(_closure_paths(graph, seed_paths))
        selected_set = set(selected)
        truncation_reason = ""
''' + s[match.end() :]

s = replace_once(
    s,
    "        contents: dict[str, bytes] = {}\n        total_bytes = 0\n",
    '''        contents: dict[str, bytes] = {}
        total_bytes = 0
        slice_byte_budget = _slice_byte_budget()
''',
    "slice byte budget init",
)
s = s.replace("if total_bytes + len(raw) > _MAX_SLICE_BYTES:", "if total_bytes + len(raw) > slice_byte_budget:")
s = s.replace(
    'truncation_reason = f"Exceeded MAX_SLICE_BYTES ({_MAX_SLICE_BYTES})"',
    'truncation_reason = f"Exceeded configured slice byte budget ({slice_byte_budget})"',
)
s = s.replace('reason="MAX_SLICE_BYTES_EXCEEDED",', 'reason="SOURCE_TRANSPLANT_SLICE_BYTE_BUDGET_EXCEEDED",')

# Record unreadable artifacts only when they intersect the selected closure.  A
# completely unrelated unreadable file is not evidence that this capability slice
# is incomplete.
insert_unresolved = '''        unresolved_edges: list[UnresolvedArtifactEdge] = [
            edge
            for edge in (*graph.unresolved_edges, *graph.ambiguous_edges)
            if edge.source_id in selected_set
            and (
                edge in graph.ambiguous_edges
                or graph.is_mandatory_unresolved(edge)
            )
        ]
'''
replacement_unresolved = insert_unresolved + '''        for path in unreadable:
            if path in selected_set:
                unresolved_edges.append(
                    UnresolvedArtifactEdge(
                        source_id="closure_root",
                        requested_target=path,
                        relation="materialization",
                        reason="SELECTED_ARTIFACT_UNREADABLE",
                    )
                )
'''
s = replace_once(s, insert_unresolved, replacement_unresolved, "selected unreadable tracking")

# Snapshot uses complete-tree resolver and has no local tree cardinality rejection.
old_snapshot = '''        token = str(getattr(discovery_client, "github_token", "") or "").strip()
        client = _github_client(token)
        try:
            tree = _github_json(
                client,
                f"https://api.github.com/repos/{repository}/git/trees/{commit_sha}",
                params={"recursive": "1"},
            )
        finally:
            client.close()
        entries = tree.get("tree") if isinstance(tree, Mapping) else None
        if not isinstance(entries, list) or len(entries) > _MAX_TREE_FILES:
            return None
        blobs = {
'''
new_snapshot = '''        token = str(getattr(discovery_client, "github_token", "") or "").strip()
        client = _github_client(token)
        try:
            entries = _repository_tree_entries(client, repository, commit_sha)
        finally:
            client.close()
        blobs = {
'''
s = replace_once(s, old_snapshot, new_snapshot, "repository tree snapshot")

s = replace_once(
    s,
    '''def _github_json(client: httpx.Client, url: str, *, params: Mapping[str, str] | None = None) -> Any:
    response = client.get(url, params=params)
    response.raise_for_status()
    if len(response.content) > 4 * 1024 * 1024:
        raise SourceTransplantError("GitHub response exceeded source-transplant byte policy.")
    return response.json()
''',
    '''def _github_json(client: httpx.Client, url: str, *, params: Mapping[str, str] | None = None) -> Any:
    response = client.get(url, params=params)
    response.raise_for_status()
    limit = _response_byte_budget()
    if len(response.content) > limit:
        raise SourceTransplantError(
            f"GitHub response exceeded configured source-transplant response budget ({limit} bytes)."
        )
    return response.json()
''',
    "github json response budget",
)
s = s.replace(
    "        if len(raw) > _MAX_SLICE_BYTES:\n            raise SourceTransplantError(\"Single donor blob exceeds source-transplant byte policy.\")\n",
    '''        single_blob_budget = _single_blob_byte_budget()
        if len(raw) > single_blob_budget:
            raise SourceTransplantError(
                f"Single donor blob exceeded configured byte budget ({single_blob_budget} bytes)."
            )
''',
)
s = s.replace(
    '''        if len(values) >= 32:
            break
''',
    "",
)

for forbidden in (
    "_MAX_TREE_FILES",
    "_MAX_SEEDS",
    "_MAX_CLOSURE_FILES",
    "_MAX_SLICE_BYTES",
    ")[:24]",
    ")[:24",
):
    if forbidden in s:
        raise SystemExit(f"legacy source-transplant cutoff remains: {forbidden}")
p.write_text(s, encoding="utf-8")


# Tighten the newly added test so it proves exact set preservation, not ordering.
p = Path("tests/test_source_transplant_adaptive_retrieval.py")
s = p.read_text(encoding="utf-8")
s = s.replace(
    '''    assert len(found) == 17
    assert found[-1].node_id.endswith("TradeHandler9.java") or found[-1].node_id.endswith("TradeHandler16.java")
''',
    '''    assert len(found) == 17
    assert {item.node_id for item in found} == set(_fake_locator_index(17).files_by_path)
''',
)
p.write_text(s, encoding="utf-8")
