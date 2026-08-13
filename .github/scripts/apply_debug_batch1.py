from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(rel: str) -> tuple[Path, str]:
    path = ROOT / rel
    return path, path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, got {count}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# Project mutations: reuse the canonical per-project lock rather than one global
# lock. Independent projects must never block each other.
# ---------------------------------------------------------------------------
path, text = read("minecraft_mod_ai/performance_final_contract.py")
if "from .project_write_lock import project_write_lock\n" not in text:
    marker = "from typing import Any, Callable, Iterable\n"
    text = once(
        text,
        marker,
        marker + "\nfrom .project_write_lock import project_write_lock\n",
        "project lock import",
    )
text = once(
    text,
    "_PROJECT_MUTATION_LOCK = threading.RLock()\n",
    "_SHARED_WRITER_FALLBACK_LOCK = threading.RLock()\n",
    "global project mutation lock",
)
text = once(
    text,
    "        with _PROJECT_MUTATION_LOCK:\n            return original(self, operation_list)\n",
    "        with project_write_lock(root):\n            return original(self, operation_list)\n",
    "source patch per-project lock",
)
helper = '''\n\ndef _project_root_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Path | None:\n    candidates: list[Any] = []\n    for key in ("project_root", "root", "workspace_root"):\n        if key in kwargs:\n            candidates.append(kwargs[key])\n    candidates.extend(args)\n    for value in candidates:\n        if not isinstance(value, (str, Path)):\n            continue\n        try:\n            candidate = Path(value).expanduser().resolve()\n        except (OSError, RuntimeError, ValueError):\n            continue\n        if candidate.is_dir() and not candidate.is_symlink():\n            return candidate\n    return None\n'''
anchor = "\ndef _install_serial_shared_writers(orchestrator_module: Any) -> None:\n"
if helper.strip() not in text:
    text = once(text, anchor, helper + anchor, "shared writer project-root helper")
text = once(
    text,
    '''        @wraps(original)\n        def serialized(*args: Any, __original: Callable[..., Any] = original, **kwargs: Any):\n            with _PROJECT_MUTATION_LOCK:\n                return __original(*args, **kwargs)\n''',
    '''        @wraps(original)\n        def serialized(*args: Any, __original: Callable[..., Any] = original, **kwargs: Any):\n            project_root = _project_root_from_call(args, kwargs)\n            if project_root is None:\n                # Keep safety for an unusual writer signature we cannot bind to a\n                # project, without forcing normal independent projects through this lock.\n                with _SHARED_WRITER_FALLBACK_LOCK:\n                    return __original(*args, **kwargs)\n            with project_write_lock(project_root):\n                return __original(*args, **kwargs)\n''',
    "shared writer lock",
)
text = text.replace("        with _PROJECT_MUTATION_LOCK:\n            staging_root = _clone_source_snapshot(live_root)\n", "        with project_write_lock(live_root):\n            staging_root = _clone_source_snapshot(live_root)\n", 1)
text = text.replace("            with _PROJECT_MUTATION_LOCK:\n                commit_receipt = _commit_staged_operations(\n", "            with project_write_lock(live_root):\n                commit_receipt = _commit_staged_operations(\n", 1)
if "_PROJECT_MUTATION_LOCK" in text:
    raise SystemExit("performance_final_contract still uses global project mutation lock")
write(path, text)


# ---------------------------------------------------------------------------
# Work graph: all generators that read-modify-write shared project registries use
# the commit lane. Custom LLM staging remains in the LLM lane.
# ---------------------------------------------------------------------------
path, text = read("minecraft_mod_ai/work_graph.py")
text = once(
    text,
    '''    elif kind == "module-shard" and gen_stage == "custom":\n        res_class = "llm"\n''',
    '''    elif kind == "module-shard" and gen_stage in {\n        "content",\n        "system",\n        "entity",\n        "audio-binding",\n    }:\n        res_class = "commit"\n    elif kind == "module-shard" and gen_stage == "custom":\n        res_class = "llm"\n''',
    "module commit lane",
)
write(path, text)


# ---------------------------------------------------------------------------
# Claim fencing must not discard the earlier commit-lane project lock. The late
# fence owns durable success but keeps same-project action+index publication under
# the canonical project lock.
# ---------------------------------------------------------------------------
path, text = read("minecraft_mod_ai/scheduler_claim_fencing_contract.py")
if "from .project_write_lock import project_write_lock\n" not in text:
    text = once(
        text,
        "from typing import Any, Callable\n",
        "from typing import Any, Callable\n\nfrom .project_write_lock import project_write_lock\n",
        "claim project lock import",
    )
old_try = '''        try:\n            receipt = action()\n            if not isinstance(receipt, dict):\n                raise orchestrator_module.CompleteProductionError(\n                    f"Work node {node.node_id} returned a non-object receipt."\n                )\n            ledger.raise_if_cancelled()\n            _commit_success(\n                work_graph_module,\n                orchestrator_module,\n                ledger,\n                node.node_id,\n                receipt,\n                attempt=claim_attempt,\n                owner=claim_owner,\n                shared_index=shared_index,\n            )\n            return receipt\n'''
new_try = '''        try:\n            project_root = (\n                getattr(shared_index, "root", None)\n                if node.resource_class == "commit" and shared_index is not None\n                else None\n            )\n\n            def execute_and_commit() -> dict[str, Any]:\n                receipt = action()\n                if not isinstance(receipt, dict):\n                    raise orchestrator_module.CompleteProductionError(\n                        f"Work node {node.node_id} returned a non-object receipt."\n                    )\n                ledger.raise_if_cancelled()\n                _commit_success(\n                    work_graph_module,\n                    orchestrator_module,\n                    ledger,\n                    node.node_id,\n                    receipt,\n                    attempt=claim_attempt,\n                    owner=claim_owner,\n                    shared_index=shared_index,\n                )\n                return receipt\n\n            if project_root is not None:\n                with project_write_lock(project_root):\n                    return execute_and_commit()\n            return execute_and_commit()\n'''
text = once(text, old_try, new_try, "claim fence preserves commit lock")
write(path, text)


# ---------------------------------------------------------------------------
# Production item repair: no numeric LLM-attempt budget and no host rule saying
# "field patch once, then replace". Continue while validation/model state changes;
# exact repeated state/output is the only cycle termination.
# ---------------------------------------------------------------------------
path, text = read("minecraft_mod_ai/production_page_durable_contract.py")
text = text.replace("_DEFAULT_MODEL_REPAIR_ATTEMPTS = 2\n", "")
start = text.find("\ndef _repair_attempt_budget() -> int:\n")
if start >= 0:
    end = text.find("\ndef _safe_identifier", start)
    if end < 0:
        raise SystemExit("repair budget helper end not found")
    text = text[:start] + text[end:]
text = once(
    text,
    '''    seen_states: set[str] = set()\n    seen_patch_hashes: set[str] = set()\n    max_attempts = _repair_attempt_budget()\n    round_index = 0\n    last_patch_sha256 = ""\n\n    for attempt_index in range(1, max_attempts + 1):\n        round_index = attempt_index\n        replacement_mode = not isinstance(current, dict) or attempt_index > 1\n''',
    '''    seen_states: set[str] = set()\n    seen_patch_hashes: set[str] = set()\n    round_index = 0\n    last_patch_sha256 = ""\n\n    while True:\n        round_index += 1\n        # Mapping-shaped items can always be repaired field-by-field. Whole-object\n        # regeneration is reserved for non-object values that cannot be field patched.\n        replacement_mode = not isinstance(current, dict)\n''',
    "unbounded progressive item repair loop",
)
text = text.replace("                round_index=round_index - 1,\n", "                round_index=max(0, round_index - 1),\n", 1)
exhaustion = '''\n    _raise_repair_failure(\n        module,\n        kind=kind,\n        index=index,\n        state_path=state_path,\n        original_fingerprint=original_fingerprint,\n        current=current,\n        error=error,\n        round_index=round_index,\n        reason="repair_budget_exhausted",\n        last_patch_sha256=last_patch_sha256,\n    )\n    raise AssertionError("unreachable")\n'''
if exhaustion in text:
    text = text.replace(exhaustion, "\n", 1)
if "_repair_attempt_budget" in text or "repair_budget_exhausted" in text or "attempt_index > 1" in text:
    raise SystemExit("numeric/forced item repair ceiling still present")
text = text.replace(
    '    """Resolve one production item without retry cycles or avoidable LLM calls."""',
    '    """Resolve one production item with progress-driven semantic repair."""',
)
write(path, text)


# ---------------------------------------------------------------------------
# External MCP: search evidence can mention historical versions and is not itself
# an authoritative runtime-target report. Require/report target only for explicitly
# authoritative target/status tools or routes.
# ---------------------------------------------------------------------------
path, text = read("minecraft_mod_ai/external_mcp_router.py")
old = '''        if not target.minecraft_version:\n            return\n        reported = _collect_target_values(result, route.get("response_target_fields", []))\n        conflicts = sorted(\n            value for value in reported\n            if value and value != target.minecraft_version\n        )\n        if conflicts:\n            raise ExternalMCPError(\n                "External MCP reported a Minecraft target that conflicts with the approved "\n                f"PlatformLock: expected {target.minecraft_version!r}, got {conflicts!r}."\n            )\n'''
new = '''        if not target.minecraft_version:\n            return\n        explicit_fields = route.get("response_target_fields", [])\n        tool = str(route.get("tool", "")).strip()\n        authoritative = bool(explicit_fields) or bool(route.get("require_reported_target")) or tool in {\n            "server_get_status",\n            "client_get_status",\n        }\n        if not authoritative:\n            # Search/docs results routinely contain historical Minecraft versions.\n            # Those references are evidence content, not provider runtime authority.\n            return\n        reported = _collect_target_values(result, explicit_fields)\n        if not reported:\n            raise ExternalMCPError(\n                f"External MCP authoritative tool {tool!r} did not report a Minecraft target."\n            )\n        conflicts = sorted(\n            value for value in reported\n            if value and value != target.minecraft_version\n        )\n        if conflicts:\n            raise ExternalMCPError(\n                "External MCP reported a Minecraft target that conflicts with the approved "\n                f"PlatformLock: expected {target.minecraft_version!r}, got {conflicts!r}."\n            )\n'''
text = once(text, old, new, "authoritative MCP target validation")
write(path, text)


# ---------------------------------------------------------------------------
# Mineflayer wait_for: unsupported conditions are a programming/configuration error,
# not a silent full-timeout poll.
# ---------------------------------------------------------------------------
path, text = read("integrations/mineflayer-1201/bridge.mjs")
text = once(
    text,
    '''  const condition = String(params.condition || "");\n  const timeoutMs = boundedInteger(params.timeout_ms ?? 10000, "wait timeout", 1, 60000);\n''',
    '''  const condition = String(params.condition || "");\n  const supportedConditions = new Set(["spawned", "window_open", "window_closed", "healthy"]);\n  if (!supportedConditions.has(condition)) {\n    throw new Error(`Unsupported wait_for condition: ${condition || "<empty>"}`);\n  }\n  const timeoutMs = boundedInteger(params.timeout_ms ?? 10000, "wait timeout", 1, 60000);\n''',
    "Mineflayer wait_for fail-fast",
)
write(path, text)


# ---------------------------------------------------------------------------
# Tests that were exercising obsolete composition APIs/semantics are updated to
# the live owner contracts. This does not weaken production safety.
# ---------------------------------------------------------------------------
path, text = read("tests/test_production_page_durable_contract.py")
text = text.replace(
    "from minecraft_mod_ai.execution_efficiency_contract import install\n",
    "from minecraft_mod_ai.planner_production_page_contract import install\n",
)
text = text.replace(
    "    install(complete_planner_module=complete_planner, work_graph_module=work_graph)\n",
    "    install(complete_planner)\n",
)
# Make the semantic repair fixture require two distinct field-level fixes.
text = text.replace(
    '''        if request["repair_mode"] == "field_patch":\n            return json.dumps(\n                {\n                    "target_fingerprint": request["target_fingerprint"],\n                    "set_fields": {"kind": "still_not_a_real_kind"},\n                    "delete_fields": [],\n                }\n            )\n        return json.dumps(\n            {\n                "target_fingerprint": request["target_fingerprint"],\n                "replacement": {\n                    "module_id": "semantic_fixed",\n                    "kind": "item",\n                    "config": {},\n                    "depends_on": [],\n                    "required_gates": [],\n                    "implements_deliverables": ["d1"],\n                },\n            }\n        )\n''',
    '''        if len(self.calls) == 1:\n            set_fields = {"kind": "item"}\n        else:\n            set_fields = {"config": {}}\n        return json.dumps(\n            {\n                "target_fingerprint": request["target_fingerprint"],\n                "set_fields": set_fields,\n                "delete_fields": [],\n            }\n        )\n''',
)
text = text.replace(
    "def test_semantic_validation_uses_field_patch_then_single_item_regeneration(\n",
    "def test_semantic_validation_keeps_field_patching_while_state_changes(\n",
)
text = text.replace(
    '        "modules": [_module("semantic_bad", kind="not_a_real_kind")],\n',
    '        "modules": [dict(_module("semantic_bad", kind="not_a_real_kind"), config="invalid")],\n',
    1,
)
text = text.replace(
    '''    assert [call["request"]["repair_mode"] for call in router.calls] == [\n        "field_patch",\n        "replacement",\n    ]\n    assert [item.module_id for item in parts.modules] == ["semantic_fixed"]\n''',
    '''    assert [call["request"]["repair_mode"] for call in router.calls] == [\n        "field_patch",\n        "field_patch",\n    ]\n    assert [item.module_id for item in parts.modules] == ["semantic_bad"]\n''',
)
text = text.replace(
    "def test_repeated_invalid_state_is_cut_off_after_two_distinct_repair_modes(\n",
    "def test_repeated_invalid_model_output_stops_exact_cycle(\n",
)
text = text.replace('    monkeypatch.setenv("MMM_PLANNER_ITEM_REPAIR_ATTEMPTS", "4")\n', "")
write(path, text)

# The late runner wrapper supersedes the older verified-cache wrapper with a newer
# target-aware verified cache contract. Check the live outer owner rather than an
# intentionally unwrapped marker.
path, text = read("tests/test_validation_execution_contract.py")
text = text.replace(
    '    assert getattr(GradleRunner._ensure_gradle, "_mmm_verified_distribution_cache", False)\n',
    '    assert getattr(GradleRunner._ensure_gradle, "_mmm_target_parallel_distribution", False)\n',
)
write(path, text)

print("debug batch 1 prepared")
