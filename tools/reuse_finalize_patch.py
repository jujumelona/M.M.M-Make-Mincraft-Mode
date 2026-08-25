from __future__ import annotations

import json
from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one patch anchor in {path}, found {count}: {old[:120]!r}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


# Preserve byte-level mutation state without replaying every old tool receipt.
replace_once(
    "minecraft_mod_ai/source_mutation_contract.py",
    "def _has_semantic_failure(payload: Mapping[str, Any]) -> bool:\n",
    '''def mutation_receipt_deltas(payload: Mapping[str, Any]) -> tuple[tuple[str, str, str], ...]:
    """Return path/before/after deltas proven by applied source-patch receipts."""

    result: list[tuple[str, str, str]] = []
    for item in _walk_mappings(payload):
        if str(item.get("schema_version", "")) != "mmm/source-patch-receipt-v1":
            continue
        if str(item.get("status", "")).strip().upper() != "APPLIED":
            continue
        operations = item.get("operations")
        if not isinstance(operations, Sequence) or isinstance(
            operations, (str, bytes, bytearray)
        ):
            continue
        for operation in operations:
            if not isinstance(operation, Mapping) or not _receipt_operation_changed(operation):
                continue
            path = str(operation.get("path") or operation.get("target_path") or "").strip()
            if path:
                result.append(
                    (
                        path.replace("\\\\", "/"),
                        str(operation.get("before_sha256") or ""),
                        str(operation.get("after_sha256") or ""),
                    )
                )
    return tuple(result)


def _has_semantic_failure(payload: Mapping[str, Any]) -> bool:
''',
)
replace_once(
    "minecraft_mod_ai/source_mutation_contract.py",
    '    "mutation_payload_applied",\n    "tool_payload",\n',
    '    "mutation_payload_applied",\n    "mutation_receipt_deltas",\n    "tool_payload",\n',
)

# Materialized donors expose an opaque id and manifest-relative paths.
replace_once(
    "minecraft_mod_ai/source_transplant.py",
    '                written.append({"path": str(destination), "sha256": actual, "size_bytes": len(raw)})\n            manifest = {\n                "repository": repository,\n',
    '''                written.append(
                    {
                        "path": str(destination),
                        "relative_path": path,
                        "sha256": actual,
                        "size_bytes": len(raw),
                    }
                )
            manifest = {
                "donor_id": donor_key,
                "repository": repository,
''',
)

# Do not leak host paths to the coder; tell it how to request more approved source.
replace_once(
    "minecraft_mod_ai/reuse_asset_upgrade_contract.py",
    '''            context.append(
                {
                    "repository": donor.get("repository"),
                    "commit_sha": donor.get("commit_sha"),
                    "license_id": donor.get("license_id"),
                    "capability": donor.get("capability"),
                    "path": str(path),
                    "sha256": item.get("sha256"),
                    "content": excerpt.decode("utf-8", errors="replace"),
                    "truncated": len(raw) > len(excerpt),
                }
            )
''',
    '''            relative_path = str(item.get("relative_path") or "").replace("\\\\", "/")
            if not relative_path:
                continue
            context.append(
                {
                    "donor_id": donor.get("donor_id"),
                    "repository": donor.get("repository"),
                    "commit_sha": donor.get("commit_sha"),
                    "license_id": donor.get("license_id"),
                    "capability": donor.get("capability"),
                    "path": relative_path,
                    "sha256": item.get("sha256"),
                    "content": excerpt.decode("utf-8", errors="replace"),
                    "truncated": len(raw) > len(excerpt),
                    "read_more_with": "read_reuse_source",
                }
            )
''',
)

# Host-owned read-only donor source tool.
replace_once(
    "minecraft_mod_ai/agent_tool_runtime.py",
    "import asyncio\nimport json\n",
    "import asyncio\nimport hashlib\nimport json\n",
)
replace_once(
    "minecraft_mod_ai/agent_tool_runtime.py",
    '_SOURCE_EDIT_TOOL = "apply_source_edit"\n_SOURCE_EDIT_DESCRIPTION = (\n',
    '''_SOURCE_EDIT_TOOL = "apply_source_edit"
_REUSE_SOURCE_TOOL = "read_reuse_source"
_REUSE_DONOR_ID = re.compile(r"^[0-9a-f]{20}$")
_REUSE_SOURCE_DESCRIPTION = (
    "Read a bounded line range from one approved, materialized reuse donor. "
    "Use only donor_id and relative path values supplied in _source_transplant_context; "
    "the host verifies the immutable manifest and SHA-256 before returning source."
)
_REUSE_SOURCE_SCHEMA = {
    "type": "object",
    "properties": {
        "donor_id": {"type": "string", "pattern": "^[0-9a-f]{20}$"},
        "path": {"type": "string", "minLength": 1},
        "start_line": {"type": "integer", "minimum": 1},
        "max_lines": {"type": "integer", "minimum": 1, "maximum": 400},
    },
    "required": ["donor_id", "path"],
    "additionalProperties": False,
}
_SOURCE_EDIT_DESCRIPTION = (
''',
)
replace_once(
    "minecraft_mod_ai/agent_tool_runtime.py",
    '''            if selected == "generation":
                if _SOURCE_EDIT_TOOL in names:
''',
    '''            if selected == "generation":
                if _REUSE_SOURCE_TOOL in names:
                    raise AgentToolRuntimeError(
                        "read_reuse_source must have exactly one host-owned model schema"
                    )
                schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": _REUSE_SOURCE_TOOL,
                            "description": _REUSE_SOURCE_DESCRIPTION,
                            "parameters": _REUSE_SOURCE_SCHEMA,
                        },
                    }
                )
                names.add(_REUSE_SOURCE_TOOL)
                if _SOURCE_EDIT_TOOL in names:
''',
)
replace_once(
    "minecraft_mod_ai/agent_tool_runtime.py",
    '''        try:
            if selected == "generation" and tool_name == _SOURCE_EDIT_TOOL:
''',
    '''        try:
            if selected == "generation" and tool_name == _REUSE_SOURCE_TOOL:
                result = _read_verified_reuse_source(self.workspace_root, payload)
            elif selected == "generation" and tool_name == _SOURCE_EDIT_TOOL:
''',
)
replace_once(
    "minecraft_mod_ai/agent_tool_runtime.py",
    "def _jsonable(value: Any) -> Any:\n",
    '''def _read_verified_reuse_source(
    workspace_root: str | Path,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    donor_id = str(arguments.get("donor_id") or "").strip().casefold()
    relative_path = str(arguments.get("path") or "").strip().replace("\\\\", "/")
    start_line = arguments.get("start_line", 1)
    max_lines = arguments.get("max_lines", 200)
    if not _REUSE_DONOR_ID.fullmatch(donor_id):
        raise AgentToolRuntimeError("Invalid approved reuse donor id")
    if (
        not relative_path
        or relative_path.startswith("/")
        or ":" in relative_path.split("/", 1)[0]
        or ".." in relative_path.split("/")
    ):
        raise AgentToolRuntimeError("Unsafe reuse source path")
    if type(start_line) is not int or start_line < 1:
        raise AgentToolRuntimeError("start_line must be a positive integer")
    if type(max_lines) is not int or not 1 <= max_lines <= 400:
        raise AgentToolRuntimeError("max_lines must be between 1 and 400")

    project_root, _ = _discover_model_project_root(workspace_root)
    donor_base = (project_root / ".minecraft_ai" / "reuse" / "donors").resolve()
    donor_root = donor_base / donor_id
    if not donor_root.is_dir() or donor_root.is_symlink():
        raise AgentToolRuntimeError("Approved reuse donor is not materialized")
    donor_root = donor_root.resolve()
    try:
        donor_root.relative_to(donor_base)
    except ValueError as exc:
        raise AgentToolRuntimeError("Reuse donor escaped the project evidence root") from exc

    manifest_path = donor_root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise AgentToolRuntimeError("Approved reuse donor manifest is unavailable")
    raw_manifest = manifest_path.read_bytes()
    if len(raw_manifest) > 1024 * 1024:
        raise AgentToolRuntimeError("Approved reuse donor manifest exceeds the host byte policy")
    try:
        manifest = json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgentToolRuntimeError("Approved reuse donor manifest is invalid") from exc
    if not isinstance(manifest, Mapping):
        raise AgentToolRuntimeError("Approved reuse donor manifest is invalid")
    if str(manifest.get("donor_id") or donor_id).strip().casefold() != donor_id:
        raise AgentToolRuntimeError("Approved reuse donor manifest id does not match")

    matched: Mapping[str, Any] | None = None
    files = manifest.get("files")
    if isinstance(files, Sequence) and not isinstance(files, (str, bytes, bytearray)):
        for item in files:
            if not isinstance(item, Mapping):
                continue
            item_relative = str(item.get("relative_path") or "").replace("\\\\", "/")
            if not item_relative:
                stored_path = Path(str(item.get("path") or ""))
                try:
                    item_relative = stored_path.resolve().relative_to(donor_root).as_posix()
                except (ValueError, OSError):
                    continue
            if item_relative == relative_path:
                matched = item
                break
    if matched is None:
        raise AgentToolRuntimeError("Requested source is not present in the approved donor manifest")

    source_path = (donor_root / relative_path).resolve()
    try:
        source_path.relative_to(donor_root)
    except ValueError as exc:
        raise AgentToolRuntimeError("Reuse source escaped the approved donor root") from exc
    if not source_path.is_file() or source_path.is_symlink():
        raise AgentToolRuntimeError("Approved reuse source file is unavailable")
    raw = source_path.read_bytes()
    actual_sha = "sha256:" + hashlib.sha256(raw).hexdigest()
    expected_sha = str(matched.get("sha256") or "")
    if not expected_sha or actual_sha != expected_sha:
        raise AgentToolRuntimeError("Approved reuse source SHA-256 no longer matches its manifest")

    lines = raw.decode("utf-8", errors="replace").splitlines()
    start_index = min(len(lines), start_line - 1)
    end_index = min(len(lines), start_index + max_lines)
    content = "\n".join(lines[start_index:end_index])
    encoded = content.encode("utf-8")
    if len(encoded) > _MAX_TOOL_RESULT_BYTES:
        content = encoded[:_MAX_TOOL_RESULT_BYTES].decode("utf-8", errors="ignore")
    return {
        "schema_version": "mmm/reuse-source-read-v1",
        "donor_id": donor_id,
        "repository": str(manifest.get("repository") or ""),
        "commit_sha": str(manifest.get("commit_sha") or ""),
        "license_id": str(manifest.get("license_id") or ""),
        "capability": str(manifest.get("capability") or ""),
        "path": relative_path,
        "sha256": actual_sha,
        "start_line": start_index + 1 if lines else 1,
        "end_line": end_index if lines else 0,
        "total_lines": len(lines),
        "content": content,
        "truncated": end_index < len(lines),
    }


def _jsonable(value: Any) -> Any:
''',
)

# Large-mod tool loop: read approved donors, allow completion after a real diff,
# and compact accumulated mutation transcripts to a host-owned state ledger.
replace_once(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    "    mutation_payload_applied,\n)\n",
    "    mutation_payload_applied,\n    mutation_receipt_deltas,\n)\n",
)
replace_once(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '_LOCALIZATION_EVIDENCE_TOOLS = frozenset({\n    "search_code_rag",\n',
    '_REUSE_SOURCE_TOOL = "read_reuse_source"\n\n_LOCALIZATION_EVIDENCE_TOOLS = frozenset({\n    "read_reuse_source",\n    "search_code_rag",\n',
)
replace_once(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '_READ_OBSERVE_TOOLS = frozenset({\n    "search_code_rag",\n',
    '_READ_OBSERVE_TOOLS = frozenset({\n    "read_reuse_source",\n    "search_code_rag",\n',
)
replace_once(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '    applied_mutations: list[str] = field(default_factory=list)\n    workspace_changed: bool = False\n',
    '    applied_mutations: list[str] = field(default_factory=list)\n    mutation_files: dict[str, str] = field(default_factory=dict)\n    workspace_changed: bool = False\n',
)
replace_once(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '''    def record_evidence(self, value: Any, *, usable: bool) -> bool:
        if not usable:
            return False
        fp = evidence_fingerprint(value)
        if fp is None:
            return False
        with self._lock:
            extracted_ctx = _extract_mutation_context_from_payload(value)
            if extracted_ctx is not None:
''',
    '''    def record_evidence(
        self,
        value: Any,
        *,
        usable: bool,
        extract_mutation_context: bool = True,
    ) -> bool:
        if not usable:
            return False
        fp = evidence_fingerprint(value)
        if fp is None:
            return False
        with self._lock:
            extracted_ctx = (
                _extract_mutation_context_from_payload(value)
                if extract_mutation_context
                else None
            )
            if extracted_ctx is not None:
''',
)
replace_once(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '''        if applied:
            with self._lock:
                self.applied_mutations.append(tool_name)
                self.workspace_changed = True
            return True
''',
    '''        if applied:
            deltas = mutation_receipt_deltas(payload)
            with self._lock:
                self.applied_mutations.append(tool_name)
                for path, _before_sha, after_sha in deltas:
                    self.mutation_files[path] = after_sha
                self.workspace_changed = True
            return True
''',
)
replace_once(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '''    attempted_sources: Sequence[str] | set[str] | frozenset[str] = frozenset(),
) -> tuple[Mapping[str, Any], ...]:
''',
    '''    attempted_sources: Sequence[str] | set[str] | frozenset[str] = frozenset(),
    reuse_context_available: bool = False,
) -> tuple[Mapping[str, Any], ...]:
''',
)
replace_once(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '''    if phase == LoopPhase.OBSERVE:
        stage = (
''',
    '''    if phase == LoopPhase.OBSERVE:
        if (
            reuse_context_available
            and _REUSE_SOURCE_TOOL in by_name
            and _REUSE_SOURCE_TOOL not in attempted_sources
        ):
            return (by_name[_REUSE_SOURCE_TOOL],)
        stage = (
''',
)
replace_once(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '''    elif phase == LoopPhase.ACT:
        selected_names = [name for name in by_name if name in _MUTATION_ACT_TOOLS]
''',
    '''    elif phase == LoopPhase.ACT:
        selected_names = [name for name in by_name if name in _MUTATION_ACT_TOOLS]
        if reuse_context_available and _REUSE_SOURCE_TOOL in by_name:
            selected_names.insert(0, _REUSE_SOURCE_TOOL)
''',
)
replace_once(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    "def _replace_live_messages(\n",
    '''def _reuse_source_context_available(messages: Sequence[Mapping[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, Mapping):
            text = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
        else:
            text = str(content or "")
        if "_source_transplant_context" in text and "donor_id" in text:
            return True
    return False


def _compact_mutation_history(
    messages: list[dict[str, Any]],
    state: HostRunState,
    *,
    initial_message_count: int,
    recent_messages: int = 12,
) -> bool:
    """Replace old full mutation transcripts with a compact host-owned receipt ledger."""
    if len(messages) <= initial_message_count + recent_messages + 1:
        return False
    tail_start = max(initial_message_count, len(messages) - recent_messages)
    while tail_start > initial_message_count and messages[tail_start].get("role") == "tool":
        tail_start -= 1
    tail: list[dict[str, Any]] = []
    for message in messages[tail_start:]:
        content = str(message.get("content") or "")
        if message.get("role") == "system" and content.startswith("[MMM HOST MUTATION LEDGER]"):
            continue
        tail.append(dict(message))
    with state._lock:
        ledger = {
            "schema_version": "mmm/mutation-ledger-v1",
            "applied_mutation_count": len(state.applied_mutations),
            "touched_files": [
                {"path": path, "after_sha256": sha}
                for path, sha in sorted(state.mutation_files.items())
            ],
        }
    ledger_message = {
        "role": "system",
        "content": "[MMM HOST MUTATION LEDGER] "
        + json.dumps(ledger, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    }
    messages[:] = [
        *[dict(message) for message in messages[:initial_message_count]],
        ledger_message,
        *tail,
    ]
    return True


def _replace_live_messages(
''',
)
replace_once(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '''    messages: list[dict[str, Any]] = [dict(message) for message in request.messages]
    all_exposed_tools = tuple(request.tools)
''',
    '''    messages: list[dict[str, Any]] = [dict(message) for message in request.messages]
    initial_message_count = len(messages)
    reuse_context_available = _reuse_source_context_available(messages)
    all_exposed_tools = tuple(request.tools)
''',
)
replace_once(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '''    if require_rag and not state.has_fresh_evidence:
        state.phase = LoopPhase.OBSERVE
    elif implementation_requires_mutation and not mutation_history_applied(messages) and mutation_ready:
''',
    '''    if require_rag and not state.has_fresh_evidence:
        state.phase = LoopPhase.OBSERVE
    elif reuse_context_available and implementation_requires_mutation:
        state.phase = LoopPhase.OBSERVE
    elif implementation_requires_mutation and not mutation_history_applied(messages) and mutation_ready:
''',
)
replace_once(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '''            mutation_context=state.mutation_context,
            attempted_sources=state.attempted_sources,
        )
''',
    '''            mutation_context=state.mutation_context,
            attempted_sources=state.attempted_sources,
            reuse_context_available=reuse_context_available,
        )
''',
)
replace_once(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '''        elif state.phase == LoopPhase.ACT:
            mutation_names = [n for n in phase_tool_names if n in _MUTATION_ACT_TOOLS]
            if len(mutation_names) == 1:
                tool_choice = {"type": "function", "function": {"name": mutation_names[0]}}
                parallel_tool_calls = False
''',
    '''        elif state.phase == LoopPhase.ACT:
            mutation_names = [n for n in phase_tool_names if n in _MUTATION_ACT_TOOLS]
            if state.workspace_changed or mutation_history_applied(messages):
                # A real source diff is the completion threshold. From here the model may
                # make another useful edit, read an approved donor, or finish naturally.
                tool_choice = "auto"
                parallel_tool_calls = False
            elif len(mutation_names) == 1:
                tool_choice = {"type": "function", "function": {"name": mutation_names[0]}}
                parallel_tool_calls = False
''',
)
replace_once(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '                recorded = state.record_evidence(payload.get("result"), usable=usable)\n',
    '''                recorded = state.record_evidence(
                    payload.get("result"),
                    usable=usable,
                    extract_mutation_context=call.name != _REUSE_SOURCE_TOOL,
                )
''',
)
replace_once(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '''        if turn_made_progress:
            state.clear_failure()
''',
    '''        if state.workspace_changed:
            _compact_mutation_history(
                messages,
                state,
                initial_message_count=initial_message_count,
            )

        if turn_made_progress:
            state.clear_failure()
''',
)

# Grounding: metadata must not evict the exact anchors it describes.
replace_once(
    "minecraft_mod_ai/repository_grounding.py",
    "from typing import Any, Iterable\n",
    "from typing import Any, Iterable, Sequence\n",
)
replace_once(
    "minecraft_mod_ai/repository_grounding.py",
    '''    ledger = {
        "schema_version": "mmm/source-observation-ledger-v2",
        "receipt": receipt,
        "exploration": exploration.to_dict(),
        "procedural_retrieval": procedural_receipt,
        "records": records,
    }
    while records and _json_size(ledger) > byte_budget:
        records.pop()
    return ledger
''',
    '''    exploration_meta = exploration.to_dict()
    duplicate_regions = exploration_meta.pop("regions", ())
    exploration_meta["region_count"] = (
        len(duplicate_regions)
        if isinstance(duplicate_regions, Sequence) and not isinstance(duplicate_regions, (str, bytes))
        else 0
    )
    exploration_meta["region_bodies_carried_by_records"] = True
    ledger = {
        "schema_version": "mmm/source-observation-ledger-v2",
        "receipt": receipt,
        "exploration": exploration_meta,
        "procedural_retrieval": procedural_receipt,
        "records": records,
    }

    baseline_count = sum(item.get("kind") == "global_exact_source_anchor" for item in records)
    protected = min(2, baseline_count)
    while len(records) > baseline_count and _json_size(ledger) > byte_budget:
        records.pop()
    if _json_size(ledger) > byte_budget:
        ledger["procedural_retrieval"] = {
            "candidate_region_count": procedural_receipt["candidate_region_count"],
            "aligned_region_count": procedural_receipt["aligned_region_count"],
            "secondary_procedure_query_used": procedural_receipt["secondary_procedure_query_used"],
            "generic_semantic_similarity_is_not_procedural_authority": True,
            "plan_step_count": len(procedure_plan.steps),
        }
    while len(records) > protected and _json_size(ledger) > byte_budget:
        records.pop()
    if records and _json_size(ledger) > byte_budget:
        metadata_size = _json_size({**ledger, "records": []})
        available = max(64, byte_budget - metadata_size - 32)
        per_record = max(32, available // len(records))
        records[:] = [_clip_record(item, per_record) for item in records]
    while len(records) > 1 and _json_size(ledger) > byte_budget:
        records.pop()
    if records and _json_size(ledger) > byte_budget:
        metadata_size = _json_size({**ledger, "records": []})
        if metadata_size < byte_budget:
            records[0] = _clip_record(records[0], max(16, byte_budget - metadata_size - 16))

    final_digest = hashlib.sha256()
    for record in records:
        _update_digest(final_digest, record)
    receipt["observation_count"] = len(records)
    receipt["observations_sha256"] = "sha256:" + final_digest.hexdigest()
    receipt["baseline_anchor_count"] = sum(
        item.get("kind") == "global_exact_source_anchor" for item in records
    )
    receipt["lines_selected"] = sum(_record_line_count(item) for item in records)
    if _json_size(ledger) > byte_budget:
        raise ValueError(
            "repository grounding metadata leaves no room for bounded exact source evidence"
        )
    return ledger
''',
)
replace_once(
    "minecraft_mod_ai/repository_grounding.py",
    '''    ranked.sort()

    records: list[dict[str, Any]] = []
''',
    '''    ranked.sort()
    if len(ranked) > 1:
        strongest = ranked[0][0]
        tier = [item for item in ranked if item[0] == strongest]
        remainder = [item for item in ranked if item[0] != strongest]
        if len(tier) > 1:
            ranked = [tier[0], tier[-1], *tier[1:-1], *remainder]

    records: list[dict[str, Any]] = []
''',
)

# Tests that encoded the pre-reuse signatures/alias behavior are stale.
qwen_test = Path("tests/test_llama_cpp_adapter_request_contract.py")
qwen_text = qwen_test.read_text(encoding="utf-8")
stale_row = '''        (
            "apply_source_edit",
            True,
            "action",
            "omitted required parameters: operation",
        ),
'''
if qwen_text.count(stale_row) != 1:
    raise SystemExit("stale Qwen negative alias row not found exactly once")
qwen_test.write_text(qwen_text.replace(stale_row, "", 1), encoding="utf-8")

target_test = Path("tests/test_target_snapshot_hardening.py")
target_text = target_test.read_text(encoding="utf-8")
stale_signature = 'lambda queries, _client: {query: () for query in queries},'
if target_text.count(stale_signature) != 2:
    raise SystemExit(
        f"expected two stale donor discovery fixtures, found {target_text.count(stale_signature)}"
    )
target_test.write_text(
    target_text.replace(
        stale_signature,
        'lambda queries, _client, **_kwargs: {query: () for query in queries},',
    ),
    encoding="utf-8",
)

# Optional CurseForge key: Colab secret if present; otherwise the lane remains disabled.
notebook_path = Path("M.M.M_Make_Mincraft_Mode_Colab.ipynb")
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
config_cell = next(
    (cell for cell in notebook.get("cells", []) if cell.get("id") == "mmm-config"),
    None,
)
if config_cell is None:
    raise SystemExit("canonical notebook mmm-config cell not found")
config_source = "".join(config_cell.get("source", []))
if "MMM_CURSEFORGE_API_KEY" not in config_source:
    config_source += '''
# Optional CurseForge catalogue lane. Store CURSEFORGE_API_KEY in Colab Secrets.
# If it is absent, Modrinth + GitHub reuse discovery continue normally.
import os as _mmm_os
if not _mmm_os.environ.get("MMM_CURSEFORGE_API_KEY", "").strip():
    try:
        from google.colab import userdata as _mmm_userdata
        _mmm_curseforge_key = str(_mmm_userdata.get("CURSEFORGE_API_KEY") or "").strip()
    except Exception:
        _mmm_curseforge_key = ""
    if _mmm_curseforge_key:
        _mmm_os.environ["MMM_CURSEFORGE_API_KEY"] = _mmm_curseforge_key
del _mmm_curseforge_key
'''
    config_cell["source"] = config_source.splitlines(keepends=True)
    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )

# Focused regression contracts.
Path("tests/test_reuse_source_reader_contract.py").write_text(
    r'''from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from minecraft_mod_ai.agent_tool_runtime import (
    AgentToolRuntime,
    AgentToolRuntimeError,
    _read_verified_reuse_source,
)
from minecraft_mod_ai.progress_aware_tool_loop import (
    HostRunState,
    LoopPhase,
    _compact_mutation_history,
    _filter_tools_for_phase,
)
from minecraft_mod_ai.reuse_asset_upgrade_contract import _materialized_donor_context


def _project(tmp_path: Path) -> tuple[Path, str, str, bytes]:
    root = tmp_path / "mod"
    (root / "src").mkdir(parents=True)
    (root / "build.gradle").write_text("plugins {}\n", encoding="utf-8")
    donor_id = "a" * 20
    relative = "src/main/java/example/Donor.java"
    donor_root = root / ".minecraft_ai/reuse/donors" / donor_id
    source = donor_root / relative
    source.parent.mkdir(parents=True)
    raw = b"package example;\npublic final class Donor {\n  static int value() { return 7; }\n}\n"
    source.write_bytes(raw)
    sha = "sha256:" + hashlib.sha256(raw).hexdigest()
    manifest = {
        "donor_id": donor_id,
        "repository": "example/donor",
        "commit_sha": "1" * 40,
        "license_id": "MIT",
        "capability": "economy.transaction",
        "files": [{"path": str(source), "relative_path": relative, "sha256": sha}],
    }
    (donor_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, donor_id, relative, raw


def test_verified_donor_reader_is_relative_hash_bound_and_read_only(tmp_path: Path) -> None:
    root, donor_id, relative, raw = _project(tmp_path)
    result = _read_verified_reuse_source(
        root, {"donor_id": donor_id, "path": relative, "start_line": 2, "max_lines": 2}
    )
    assert result["donor_id"] == donor_id
    assert result["path"] == relative
    assert "public final class Donor" in result["content"]
    assert str(tmp_path) not in json.dumps(result)
    source = root / ".minecraft_ai/reuse/donors" / donor_id / relative
    source.write_bytes(raw + b"// drift\n")
    with pytest.raises(AgentToolRuntimeError, match="SHA-256"):
        _read_verified_reuse_source(root, {"donor_id": donor_id, "path": relative})


def test_verified_donor_reader_rejects_traversal(tmp_path: Path) -> None:
    root, donor_id, _relative, _raw = _project(tmp_path)
    with pytest.raises(AgentToolRuntimeError, match="Unsafe"):
        _read_verified_reuse_source(root, {"donor_id": donor_id, "path": "../secret"})


def test_model_donor_context_never_contains_host_materialization_path(tmp_path: Path) -> None:
    root, donor_id, relative, raw = _project(tmp_path)
    source = root / ".minecraft_ai/reuse/donors" / donor_id / relative
    receipt = {
        "donors": [{
            "donor_id": donor_id,
            "repository": "example/donor",
            "commit_sha": "1" * 40,
            "license_id": "MIT",
            "capability": "economy.transaction",
            "files": [{
                "path": str(source),
                "relative_path": relative,
                "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            }],
        }]
    }
    context = _materialized_donor_context(receipt)
    assert context[0]["donor_id"] == donor_id
    assert context[0]["path"] == relative
    assert context[0]["read_more_with"] == "read_reuse_source"
    assert str(tmp_path) not in json.dumps(context)


def _schema(name: str) -> dict[str, object]:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


def test_act_exposes_reuse_reader_with_mutation_after_reuse_context() -> None:
    selected = _filter_tools_for_phase(
        (_schema("read_reuse_source"), _schema("apply_source_edit")),
        LoopPhase.ACT,
        "coder",
        reuse_context_available=True,
    )
    names = {item["function"]["name"] for item in selected}
    assert names == {"read_reuse_source", "apply_source_edit"}


def test_mutation_history_compacts_to_host_ledger_without_losing_file_state() -> None:
    state = HostRunState(workspace_changed=True)
    state.applied_mutations.extend(["apply_source_edit"] * 20)
    state.mutation_files["src/main/java/example/A.java"] = "sha256:after"
    messages = [
        {"role": "system", "content": "base"},
        {"role": "user", "content": "build"},
    ]
    for index in range(20):
        messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": str(index)}]})
        messages.append({"role": "tool", "name": "apply_source_edit", "content": "x" * 2000})
    assert _compact_mutation_history(messages, state, initial_message_count=2)
    assert len(messages) < 20
    ledger = next(
        item
        for item in messages
        if str(item.get("content", "")).startswith("[MMM HOST MUTATION LEDGER]")
    )
    assert "src/main/java/example/A.java" in ledger["content"]
    assert "sha256:after" in ledger["content"]


def test_generation_runtime_exposes_one_host_owned_reuse_reader(monkeypatch, tmp_path: Path) -> None:
    runtime = AgentToolRuntime(profile="t4_local", workspace_root=tmp_path)
    monkeypatch.setattr(runtime, "_run_async", lambda *_args: [])
    monkeypatch.setattr(runtime._external_bridge, "tool_schemas", lambda _stage: ())
    names = [item["function"]["name"] for item in runtime.tool_schemas("generation")]
    assert names.count("read_reuse_source") == 1
''',
    encoding="utf-8",
)

Path("tests/test_reuse_discovery_optional_curseforge.py").write_text(
    r'''from __future__ import annotations

import json

from minecraft_mod_ai import reuse_discovery


class _Client:
    def __init__(self) -> None:
        self.providers: list[str] = []

    def search(self, provider, query, **_kwargs):
        self.providers.append(provider)
        return {"candidates": []}


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_curseforge_is_silently_skipped_without_optional_key(monkeypatch) -> None:
    monkeypatch.delenv("MMM_CURSEFORGE_API_KEY", raising=False)
    monkeypatch.delenv("CURSEFORGE_API_KEY", raising=False)
    client = _Client()
    result = reuse_discovery.discover_repositories_for_graph(("trade.transaction",), client)
    assert result == {"trade.transaction": ()}
    assert set(client.providers) == {"github", "modrinth"}


def test_curseforge_key_enables_catalog_lane_without_leaking_secret(monkeypatch) -> None:
    secret = "curseforge-secret-value"
    monkeypatch.setenv("MMM_CURSEFORGE_API_KEY", secret)
    captured = {}

    def get(url, **kwargs):
        captured["url"] = url
        captured["headers"] = dict(kwargs.get("headers") or {})
        return _Response({
            "data": [{"links": {"sourceUrl": "https://github.com/example/trade-mod"}}]
        })

    monkeypatch.setattr(reuse_discovery.httpx, "get", get)
    client = _Client()
    result = reuse_discovery.discover_repositories_for_graph(("trade.transaction",), client)
    assert captured["url"] == "https://api.curseforge.com/v1/mods/search"
    assert captured["headers"]["x-api-key"] == secret
    assert result["trade.transaction"] == ("example/trade-mod",)
    assert secret not in json.dumps(result)
    assert set(client.providers) == {"github", "modrinth"}
''',
    encoding="utf-8",
)

notebook_test = Path("tests/test_notebook_registry_policy.py")
notebook_test_text = notebook_test.read_text(encoding="utf-8")
if "test_notebook_curseforge_key_is_optional_host_secret_only" not in notebook_test_text:
    notebook_test.write_text(
        notebook_test_text
        + '''\n\ndef test_notebook_curseforge_key_is_optional_host_secret_only() -> None:\n    notebook = _load_notebook()\n    config = next(cell for cell in notebook["cells"] if cell.get("id") == "mmm-config")\n    source = "".join(config.get("source", []))\n    assert 'userdata.get("CURSEFORGE_API_KEY")' in source\n    assert 'MMM_CURSEFORGE_API_KEY' in source\n    assert 'print(_mmm_curseforge_key' not in source\n    assert 'CURSEFORGE_API_KEY =' not in source\n''',
        encoding="utf-8",
    )
