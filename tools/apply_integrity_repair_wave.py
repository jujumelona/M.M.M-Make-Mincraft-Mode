from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, value: str) -> None:
    (ROOT / path).write_text(value, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    value = text(path)
    count = value.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}: {old[:120]!r}")
    write(path, value.replace(old, new, 1))


# Keep one tuning-stage graph owner. Final context authority is part of the last
# multimodal stage instead of appearing as an extra composition stage.
replace_once(
    "minecraft_mod_ai/llama_tuning_pipeline.py",
    """        def install_multimodal_stage() -> None:\n            install_multimodal(self.autotune, self.hardware_policy)\n            self._install_autotune_liveness_guard()\n""",
    """        def install_multimodal_stage() -> None:\n            install_multimodal(self.autotune, self.hardware_policy)\n            self._install_autotune_liveness_guard()\n            # Final launch-context ownership belongs to the last composed stage.\n            self._install_profile_context_authority()\n""",
)
replace_once(
    "minecraft_mod_ai/llama_tuning_pipeline.py",
    """            TuningStage(\"qwen-transport\", install_qwen_runtime_transport),\n            TuningStage(\"multimodal\", install_multimodal_stage),\n            TuningStage(\"context-authority\", self._install_profile_context_authority),\n""",
    """            TuningStage(\"qwen-transport\", install_qwen_runtime_transport),\n            TuningStage(\"multimodal\", install_multimodal_stage),\n""",
)

# The TargetContract already contains the authoritative HOST snapshot. A duplicate
# planner payload is optional redundancy: verify it when present, never require it.
replace_once(
    "minecraft_mod_ai/platform_resolver.py",
    """        if adapter.host_facts_json:\n            from .resolved_version_context import ResolvedVersionContext\n\n            supplied_context = ResolvedVersionContext.from_dict(raw.get(\"resolved_version_context\", {}))\n            adapter.version_context.assert_context(supplied_context.context_id)\n""",
    """        if adapter.host_facts_json:\n            from .resolved_version_context import ResolvedVersionContext\n\n            supplied_payload = raw.get(\"resolved_version_context\")\n            if supplied_payload is not None:\n                if not isinstance(supplied_payload, Mapping) or not supplied_payload:\n                    raise ValueError(\"resolved_version_context must be a non-empty mapping when supplied\")\n                supplied_context = ResolvedVersionContext.from_dict(supplied_payload)\n                adapter.version_context.assert_context(supplied_context.context_id)\n""",
)

# CompleteProposal uses the platform TargetContract as the authority. Optional copies
# are integrity checks, not prerequisites for semantic-only/custom-java plans.
replace_once(
    "minecraft_mod_ai/complete_spec.py",
    """        if self.base_proposal.spec.platform.host_facts_json:\n            from .resolved_version_context import ResolvedVersionContext, VersionContextError\n\n            resolved = self.base_proposal.spec.platform.version_context\n            stored = ResolvedVersionContext.from_dict(self.game_design.get(\"_resolved_version_context\", {}))\n            resolved.assert_context(stored.context_id)\n            bindings = self.game_design.get(\"_artifact_version_contexts\", {})\n            expected = {\"module:\" + module.module_id for module in self.modules}\n            expected.update(\"asset:\" + asset.asset_id for asset in self.assets)\n            if not isinstance(bindings, dict) or set(bindings) != expected:\n                raise VersionContextError(\"ARTIFACT_CONTEXT_BINDING_MISSING\")\n            for identifier in bindings.values():\n                resolved.assert_context(identifier)\n            for job in self.game_design.get(\"_artifact_jobs\", ()):\n                resolved.assert_context(job.get(\"context_id\"))\n""",
    """        if self.base_proposal.spec.platform.host_facts_json:\n            from .resolved_version_context import ResolvedVersionContext, VersionContextError\n\n            resolved = self.base_proposal.spec.platform.version_context\n            stored_payload = self.game_design.get(\"_resolved_version_context\")\n            if stored_payload is not None:\n                if not isinstance(stored_payload, dict) or not stored_payload:\n                    raise VersionContextError(\"INVALID_VERSION_CONTEXT\")\n                stored = ResolvedVersionContext.from_dict(stored_payload)\n                resolved.assert_context(stored.context_id)\n\n            bindings = self.game_design.get(\"_artifact_version_contexts\")\n            if bindings is not None:\n                expected = {\"module:\" + module.module_id for module in self.modules}\n                expected.update(\"asset:\" + asset.asset_id for asset in self.assets)\n                if not isinstance(bindings, dict) or set(bindings) != expected:\n                    raise VersionContextError(\"ARTIFACT_CONTEXT_BINDING_MISSING\")\n                for identifier in bindings.values():\n                    resolved.assert_context(identifier)\n            for job in self.game_design.get(\"_artifact_jobs\", ()):\n                resolved.assert_context(job.get(\"context_id\"))\n""",
)

# Initial source grounding is intentionally atomic. A larger model context may provide
# more continuation pages, but must not inflate the first source page.
old_budget = re.compile(
    r"def _coder_project_context_budget\(\n    router: ModelRouter,\n    policy: ScalePolicy,\n    \*,\n    fast_mode: bool,\n\) -> int:\n    \"\"\"Bound source grounding by the live request capacity, not model capability\.\"\"\"\n\n    hard_cap = max\(1024, int\(policy\.model_context_bytes\)\)\n    if fast_mode:\n        return min\(hard_cap, 4 \* 1024\)\n    fallback = min\(hard_cap, 12 \* 1024\)\n    registry = getattr\(router, \"registry\", None\)\n    resolve_role = getattr\(registry, \"role\", None\)\n    profile = str\(getattr\(router, \"profile\", \"\"\) or \"\"\)\.strip\(\)\n    if not callable\(resolve_role\) or not profile:\n        return fallback\n    try:\n        config = resolve_role\(profile, \"coder\"\)\n        live_request_bytes = int\(request_message_budget\(config, \(\)\)\)\n    except Exception:\n        return fallback\n    if live_request_bytes <= 0:\n        return fallback\n    return min\(hard_cap, max\(1024, live_request_bytes // 2\)\)\n"
)
path = "minecraft_mod_ai/custom_module_generator.py"
value = text(path)
new_budget = '''def _coder_project_context_budget(\n    router: ModelRouter,\n    policy: ScalePolicy,\n    *,\n    fast_mode: bool,\n) -> int:\n    """Bound the first exact-source page; continuation owns additional context."""\n\n    del fast_mode  # Atomic source-page size is mode-independent.\n    hard_cap = min(max(1024, int(policy.model_context_bytes)), 4 * 1024)\n    registry = getattr(router, "registry", None)\n    resolve_role = getattr(registry, "role", None)\n    profile = str(getattr(router, "profile", "") or "").strip()\n    if not callable(resolve_role) or not profile:\n        return hard_cap\n    try:\n        config = resolve_role(profile, "coder")\n        live_request_bytes = int(request_message_budget(config, ()))\n    except Exception:\n        return hard_cap\n    if live_request_bytes <= 0:\n        return hard_cap\n    return min(hard_cap, max(1024, live_request_bytes // 2))\n'''
value, count = old_budget.subn(new_budget, value, count=1)
if count != 1:
    raise RuntimeError(f"{path}: context budget function not found exactly once")
write(path, value)

# Selected logical batch is a hard upper bound for ubatch as well.
replace_once(
    "minecraft_mod_ai/llama_server_kernel_autotune.py",
    """            if explicit_batch is not None:\n                _replace_option(args, (\"--batch-size\", \"-b\"), str(explicit_batch))\n            elif active_batch:\n                _replace_option(args, (\"--batch-size\", \"-b\"), str(_int(active_batch, 2048)))\n\n            generic_kv = os.environ.get(\"MMM_KV_CACHE_QUANT\", \"\").strip().lower()\n""",
    """            effective_batch: int | None = None\n            if explicit_batch is not None:\n                effective_batch = explicit_batch\n                _replace_option(args, (\"--batch-size\", \"-b\"), str(explicit_batch))\n            elif active_batch:\n                effective_batch = _int(active_batch, 2048)\n                _replace_option(args, (\"--batch-size\", \"-b\"), str(effective_batch))\n            if effective_batch is not None:\n                for ubatch_name in (\"--ubatch-size\", \"-ub\"):\n                    if ubatch_name in args:\n                        index = args.index(ubatch_name)\n                        if index + 1 < len(args):\n                            args[index + 1] = str(min(_int(args[index + 1], effective_batch), effective_batch))\n                        break\n\n            generic_kv = os.environ.get(\"MMM_KV_CACHE_QUANT\", \"\").strip().lower()\n""",
)

# Let llama-server own GPU layer sizing unless the operator provides another explicit
# policy later in the wrapper stack; do not inherit the old literal "all" sentinel.
replace_once(
    "minecraft_mod_ai/llama_server_hardware_policy.py",
    """def _apply_hardware_launch_policy(args: list[str]) -> list[str]:\n    \"\"\"Apply managed llama-server launch policy without enabling unused endpoints.\"\"\"\n\n    if \"--parallel\" not in args and \"-np\" not in args:\n""",
    """def _apply_hardware_launch_policy(args: list[str]) -> list[str]:\n    \"\"\"Apply managed llama-server launch policy without enabling unused endpoints.\"\"\"\n\n    for option in (\"--gpu-layers\", \"-ngl\"):\n        if option in args:\n            index = args.index(option)\n            if index + 1 < len(args) and str(args[index + 1]).strip().casefold() == \"all\":\n                args[index + 1] = \"auto\"\n            break\n    if \"--parallel\" not in args and \"-np\" not in args:\n""",
)

# Three structural levels are needed by the existing atomic worksheet records
# (object -> bounded array -> tiny object). Width and array bounds remain strict.
replace_once(
    "minecraft_mod_ai/model_output_atomicity_contract.py",
    "MAX_SCHEMA_DEPTH = 2\n",
    "MAX_SCHEMA_DEPTH = 3\n",
)

# Target-specific registry implementation is not a canonical responsibility. Keep the
# semantic registry leaves stable and let HOST implementation profiles vary by target.
path = "minecraft_mod_ai/product_support_matrix.py"
value = text(path)
value, count1 = re.subn(
    r"# Version-specific requirements \(additions to core set\)\nVERSION_SPECIFIC_REQUIREMENTS: dict\[str, list\[str\]\] = \{.*?\n\}\n\n# Versions that explicitly don't support certain leaves\nVERSION_SPECIFIC_UNSUPPORTED: dict\[str, list\[str\]\] = \{.*?\n\}\n",
    "# Target-specific implementation details are HOST-owned, not canonical leaves.\nVERSION_SPECIFIC_REQUIREMENTS: dict[str, list[str]] = {}\nVERSION_SPECIFIC_UNSUPPORTED: dict[str, list[str]] = {}\n",
    value,
    count=1,
    flags=re.S,
)
if count1 != 1:
    raise RuntimeError(f"{path}: pseudo canonical version-specific requirements block not found")
write(path, value)

# Colab accepts an explicit form value first, then the protected Colab secret store.
nb_path = ROOT / "M.M.M_Make_Mincraft_Mode_Colab.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))
configuration = next(cell for cell in nb["cells"] if cell.get("id") == "configuration")
source = "".join(configuration["source"])
old = '''curseforge_api_key = str(CURSEFORGE_API_KEY).strip()\nif curseforge_api_key:\n    os.environ["MMM_CURSEFORGE_API_KEY"] = curseforge_api_key\nelse:\n    os.environ.pop("MMM_CURSEFORGE_API_KEY", None)\n'''
new = '''curseforge_api_key = str(CURSEFORGE_API_KEY).strip()\nif not curseforge_api_key:\n    try:\n        from google.colab import userdata as colab_userdata\n        curseforge_api_key = str(colab_userdata.get("CURSEFORGE_API_KEY") or "").strip()\n    except (ImportError, KeyError, RuntimeError):\n        curseforge_api_key = ""\nif curseforge_api_key:\n    os.environ["MMM_CURSEFORGE_API_KEY"] = curseforge_api_key\nelse:\n    os.environ.pop("MMM_CURSEFORGE_API_KEY", None)\n'''
if source.count(old) != 1:
    raise RuntimeError("Colab CurseForge key block not found")
source = source.replace(old, new, 1)
configuration["source"] = source.splitlines(keepends=True)
nb_path.write_text(json.dumps(nb, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

# Test fakes track the new host-owned entity cardinality and continuation contracts.
for test_path, class_name in (
    ("tests/test_atomic_design_pipeline.py", "GraphRouter"),
    ("tests/test_content_design_graph_expansion.py", "ExpansionGraphRouter"),
):
    value = text(test_path)
    marker = '        context = json.loads(messages[-1]["content"])\n        accepted = context["accepted_records"]\n'
    replacement = '''        context = json.loads(messages[-1]["content"])\n        if tool_name == "submit_one_design_content_entity_count":\n            return {"count": len(self.nodes)}\n        if tool_name == "submit_one_design_continue_record":\n            target = str(context.get("target_template") or "")\n            return {"required": target == "design/research_fact" and not context.get("accepted_record_ids")}\n        accepted = context["accepted_records"]\n'''
    if value.count(marker) != 1:
        raise RuntimeError(f"{test_path}: router context marker not found")
    write(test_path, value.replace(marker, replacement, 1))

replace_once(
    "tests/test_atomic_design_pipeline.py",
    '    assert router.calls.count("submit_design_content_entity") == 3\n',
    '    assert router.calls.count("submit_design_content_entity") == 2\n    assert router.calls.count("submit_one_design_content_entity_count") == 1\n',
)

# Unit tests for the legacy deterministic generator run against the reviewed synthetic
# adapter whenever the production AUTO target intentionally advertises no legacy kinds.
replace_once(
    "tests/conftest.py",
    """    def generate_with_explicit_test_target(self, spec, root):\n        if spec.platform.is_unresolved():\n            spec = replace(spec, platform=_platform_lock_from_adapter(synthetic_adapter))\n        return original_generate(self, spec, root)\n""",
    """    def generate_with_explicit_test_target(self, spec, root):\n        if spec.platform.is_unresolved() or not spec.platform.deterministic_module_kinds:\n            spec = replace(spec, platform=_platform_lock_from_adapter(synthetic_adapter))\n        return original_generate(self, spec, root)\n""",
)

# Dependency truthfulness is about exact predicates from the supplied lock, not one
# historical fixture literal.
replace_once(
    "tests/test_distribution_truthfulness.py",
    '    assert by_id["fabric-api"]["version_predicates"] == ["test-api"]\n',
    '    assert by_id["fabric-api"]["version_predicates"] == [_platform_lock().fabric_api]\n',
)

# Strict command-generation semantics now require an explicit permission level.
path = "tests/test_content_catalog_scaling.py"
value = text(path)
value = value.replace("'message': f'Catalog command {index:05d}'})", "'message': f'Catalog command {index:05d}', 'permission_level': 0})")
value = value.replace("config={'literal': 'first', 'message': 'First'}", "config={'literal': 'first', 'message': 'First', 'permission_level': 0}")
value = value.replace("config={'literal': 'second', 'message': 'Second'}", "config={'literal': 'second', 'message': 'Second', 'permission_level': 0}")
write(path, value)

# Deterministic-content tests now exercise strict semantic records rather than legacy
# presentation-only payloads. Invalid graph-shape cases carry otherwise-valid config so
# their intended structural rejection remains the first error.
path = "tests/test_deterministic_minecraft_content_contract.py"
value = text(path)
value = value.replace('{"display_name_en": "Copper Hammer"}', '{"display_name": "Copper Hammer", "main_color": "#B87333"}', 1)
value = value.replace('assert modules[0].config == {"display_name_en": "Copper Hammer"}', 'assert modules[0].config == {"display_name": "Copper Hammer", "main_color": "#B87333"}')
value = value.replace('{"id": "copper_hammer", "kind": "item"},\n                    {"id": "copper_hammer", "kind": "block"}', '{"id": "copper_hammer", "kind": "item", "config": {"display_name": "Copper Hammer", "main_color": "#B87333"}},\n                    {"id": "copper_hammer", "kind": "block", "config": {"display_name": "Copper Block", "main_color": "#B87333", "hardness": 3.0}}')
value = value.replace('{"id": "copper_hammer",\n                        "kind": "item",\n                        "depends_on": ["copper_hammer"],', '{"id": "copper_hammer",\n                        "kind": "item",\n                        "config": {"display_name": "Copper Hammer", "main_color": "#B87333"},\n                        "depends_on": ["copper_hammer"],')
value = value.replace('{"modules": [{"id": "copper_hammer", "kind": "item"}]}', '{"modules": [{"id": "copper_hammer", "kind": "item", "config": {"display_name": "Copper Hammer", "main_color": "#B87333"}}]}', 1)
write(path, value)

print("repair wave applied")
