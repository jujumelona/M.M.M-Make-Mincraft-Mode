from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "minecraft_mod_ai"


def _identifier_matches(name: str, markers: tuple[str, ...]) -> bool:
    low = name.lower()
    for marker in markers:
        marker = marker.lower()
        if marker in {"audio", "speech"}:
            if marker in low:
                return True
        elif low == marker or low.startswith(marker + "_") or low.endswith("_" + marker):
            return True
    return False


def _expr_contains_marker(node: ast.AST | None, markers: tuple[str, ...]) -> bool:
    if node is None:
        return False
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and _identifier_matches(child.id, markers):
            return True
        if isinstance(child, ast.Attribute) and _identifier_matches(child.attr, markers):
            return True
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if any(m.lower() in child.value.lower() for m in markers if m in {"audio", "speech"}):
                return True
    return False


def _target_contains_marker(node: ast.AST, markers: tuple[str, ...]) -> bool:
    if isinstance(node, ast.Name):
        return _identifier_matches(node.id, markers)
    if isinstance(node, ast.Attribute):
        return _identifier_matches(node.attr, markers)
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(_target_contains_marker(item, markers) for item in node.elts)
    return False


class CapabilityStripper(ast.NodeTransformer):
    def __init__(self, markers: Iterable[str]) -> None:
        self.markers = tuple(markers)

    def _clean_args(self, args: ast.arguments) -> ast.arguments:
        positional = [*args.posonlyargs, *args.args]
        defaults_start = len(positional) - len(args.defaults)
        pairs: list[tuple[ast.arg, ast.expr | None, bool]] = []
        for index, arg in enumerate(positional):
            default = args.defaults[index - defaults_start] if index >= defaults_start else None
            pairs.append((arg, default, index < len(args.posonlyargs)))
        pairs = [p for p in pairs if not _identifier_matches(p[0].arg, self.markers)]
        posonly_count = sum(1 for _, _, was_posonly in pairs if was_posonly)
        args.posonlyargs = [arg for arg, _, _ in pairs[:posonly_count]]
        args.args = [arg for arg, _, _ in pairs[posonly_count:]]
        default_pairs = [(arg, default) for arg, default, _ in pairs if default is not None]
        args.defaults = [default for _, default in default_pairs if default is not None]
        kept_kw: list[ast.arg] = []
        kept_kw_defaults: list[ast.expr | None] = []
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            if not _identifier_matches(arg.arg, self.markers):
                kept_kw.append(arg)
                kept_kw_defaults.append(default)
        args.kwonlyargs = kept_kw
        args.kw_defaults = kept_kw_defaults
        if args.vararg and _identifier_matches(args.vararg.arg, self.markers):
            args.vararg = None
        if args.kwarg and _identifier_matches(args.kwarg.arg, self.markers):
            args.kwarg = None
        return args

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module and _identifier_matches(node.module, self.markers):
            return None
        node.names = [
            alias for alias in node.names
            if not _identifier_matches(alias.name, self.markers)
            and not (alias.asname and _identifier_matches(alias.asname, self.markers))
        ]
        return node if node.names else None

    def visit_Import(self, node: ast.Import):
        node.names = [
            alias for alias in node.names
            if not _identifier_matches(alias.name, self.markers)
            and not (alias.asname and _identifier_matches(alias.asname, self.markers))
        ]
        return node if node.names else None

    def visit_ClassDef(self, node: ast.ClassDef):
        if _identifier_matches(node.name, self.markers):
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if _identifier_matches(node.name, self.markers):
            return None
        node.args = self._clean_args(node.args)
        node = self.generic_visit(node)
        if not node.body:
            node.body = [ast.Pass()]
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if _identifier_matches(node.name, self.markers):
            return None
        node.args = self._clean_args(node.args)
        node = self.generic_visit(node)
        if not node.body:
            node.body = [ast.Pass()]
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if _target_contains_marker(node.target, self.markers):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        if any(_target_contains_marker(target, self.markers) for target in node.targets):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], (ast.Tuple, ast.List))
                and isinstance(node.value, (ast.Tuple, ast.List))
                and len(node.targets[0].elts) == len(node.value.elts)
            ):
                kept = [
                    (target, value)
                    for target, value in zip(node.targets[0].elts, node.value.elts)
                    if not _target_contains_marker(target, self.markers)
                ]
                if not kept:
                    return None
                node.targets[0].elts = [target for target, _ in kept]
                node.value.elts = [value for _, value in kept]
                return self.generic_visit(node)
            return None
        return self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        if _target_contains_marker(node.target, self.markers):
            return None
        return self.generic_visit(node)

    def visit_For(self, node: ast.For):
        if _target_contains_marker(node.target, self.markers) or _expr_contains_marker(node.iter, self.markers):
            return [self.visit(item) for item in node.orelse if self.visit(item) is not None]
        node = self.generic_visit(node)
        if not node.body:
            node.body = [ast.Pass()]
        return node

    visit_AsyncFor = visit_For

    def visit_If(self, node: ast.If):
        if _expr_contains_marker(node.test, self.markers):
            replacement = []
            for item in node.orelse:
                visited = self.visit(item)
                if isinstance(visited, list):
                    replacement.extend(visited)
                elif visited is not None:
                    replacement.append(visited)
            return replacement
        node = self.generic_visit(node)
        if not node.body:
            node.body = [ast.Pass()]
        return node

    def visit_Expr(self, node: ast.Expr):
        if _expr_contains_marker(node.value, self.markers):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        node.keywords = [
            kw for kw in node.keywords
            if kw.arg is None or not _identifier_matches(kw.arg, self.markers)
        ]
        node.args = [arg for arg in node.args if not _expr_contains_marker(arg, self.markers)]
        return node

    def visit_Dict(self, node: ast.Dict):
        kept: list[tuple[ast.expr | None, ast.expr]] = []
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and _identifier_matches(key.value, self.markers):
                continue
            if key is not None and _expr_contains_marker(key, self.markers):
                continue
            kept.append((key, value))
        node.keys = [key for key, _ in kept]
        node.values = [value for _, value in kept]
        return self.generic_visit(node)

    def _clean_elts(self, node):
        node.elts = [elt for elt in node.elts if not _expr_contains_marker(elt, self.markers)]
        return self.generic_visit(node)

    visit_Tuple = _clean_elts
    visit_List = _clean_elts
    visit_Set = _clean_elts

    def visit_keyword(self, node: ast.keyword):
        if node.arg and _identifier_matches(node.arg, self.markers):
            return None
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and _identifier_matches(node.id, self.markers):
            return ast.copy_location(ast.Constant(value=None), node)
        return node

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.ctx, ast.Load) and _identifier_matches(node.attr, self.markers):
            return ast.copy_location(ast.Constant(value=None), node)
        return self.generic_visit(node)


def _strip_capability(path: Path, *markers: str) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    tree = CapabilityStripper(markers).visit(tree)
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + "\n"
    compile(source, str(path), "exec")
    path.write_text(source, encoding="utf-8")


def _replace_complete_validate(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    replacement = ast.parse(
        '''
def validate(self, *, policy: ScalePolicy | None = None) -> None:
    policy = policy or ScalePolicy.from_environment()
    policy.validate()
    if self.schema_version not in {"mmm/complete-proposal-v1", "mmm/complete-proposal-v2"}:
        raise SpecValidationError(f"Unsupported complete proposal schema: {self.schema_version}")
    if type(self.proposal_version) is not int or self.proposal_version < 1:
        raise SpecValidationError("proposal_version must be a positive integer.")
    if not isinstance(self.requested_prompt, str) or not self.requested_prompt.strip():
        raise SpecValidationError("requested_prompt must not be empty.")
    self.base_proposal.validate()
    if not isinstance(self.game_design, dict) or not self.game_design:
        raise SpecValidationError("game_design must be a non-empty object.")
    try:
        validate_canonical_json(self.game_design)
    except (CanonicalJsonError, RecursionError) as exc:
        raise SpecValidationError("game_design must contain finite JSON values.") from exc
    if not self.modules:
        raise SpecValidationError("A complete proposal must contain at least one production module.")
    module_ids: set[str] = set()
    for module in self.modules:
        module.validate(policy=policy)
        if module.module_id in module_ids:
            raise SpecValidationError(f"Duplicate production module id: {module.module_id}")
        module_ids.add(module.module_id)
    for module in self.modules:
        missing = sorted(set(module.depends_on) - module_ids)
        if missing:
            raise SpecValidationError(f"Module {module.module_id} references unknown dependencies: {missing[:20]}")
        if module.module_id in module.depends_on:
            raise SpecValidationError(f"Module {module.module_id} may not depend on itself.")
    self._validate_acyclic()
    asset_ids: set[str] = set()
    asset_paths: set[str] = set()
    for asset in self.assets:
        asset.validate(policy=policy)
        normalized_path = asset.target_path.replace("\\\\", "/")
        if asset.asset_id in asset_ids:
            raise SpecValidationError(f"Duplicate asset id: {asset.asset_id}")
        if normalized_path in asset_paths:
            raise SpecValidationError(f"Duplicate asset target path: {normalized_path}")
        asset_ids.add(asset.asset_id)
        asset_paths.add(normalized_path)
    if not self.acceptance_tests:
        raise SpecValidationError("acceptance_tests must contain at least one test.")
    if len(self.acceptance_tests) != len(set(self.acceptance_tests)):
        raise SpecValidationError("acceptance_tests must not contain duplicates.")
    for test in self.acceptance_tests:
        if not isinstance(test, str) or not test.strip():
            raise SpecValidationError("acceptance_tests must contain non-empty strings.")
    if self.schema_version == "mmm/complete-proposal-v2":
        contract = self.game_design.get("_production_contract")
        if not isinstance(contract, dict):
            raise SpecValidationError("Complete proposal v2 requires game_design._production_contract.")
        try:
            from .production_contract import validate_production_contract
            validate_production_contract(contract, [module.module_id for module in self.modules], self.acceptance_tests)
        except ValueError as exc:
            raise SpecValidationError(f"Invalid production contract: {exc}") from exc
    if type(self.external_runtime_required) is not bool:
        raise SpecValidationError("external_runtime_required must be boolean.")
    if self.existing_input_sha256 and not _SHA.fullmatch(self.existing_input_sha256):
        raise SpecValidationError("existing_input_sha256 must be empty or a lowercase SHA-256 digest.")
    if self.approval_hash:
        if not _SHA.fullmatch(self.approval_hash):
            raise SpecValidationError("approval_hash must be a lowercase SHA-256 digest.")
        if self.approval_hash != self.calculate_hash():
            raise SpecValidationError("Complete proposal approval_hash does not match its payload.")
'''
    ).body[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "CompleteProposal":
            for index, item in enumerate(node.body):
                if isinstance(item, ast.FunctionDef) and item.name == "validate":
                    node.body[index] = replacement
                    ast.fix_missing_locations(tree)
                    text = ast.unparse(tree) + "\n"
                    compile(text, str(path), "exec")
                    path.write_text(text, encoding="utf-8")
                    return
    raise RuntimeError("CompleteProposal.validate not found")


def _rewrite_planner_template() -> None:
    path = PKG / "planner_template_schema.py"
    path.write_text(
        '''"""Single host-owned schema for production-page planning.

The model fills values only. Structure is owned by the host and unknown keys are
intentionally discarded before typed parsing.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

PRODUCTION_PAGE_TEMPLATE: dict[str, Any] = {
    "modules": [{"module_id": "example_module", "kind": "custom_java", "config": {"summary": "feature implementation"}, "depends_on": [], "required_gates": []}],
    "assets": [],
    "acceptance_tests": ["test_example_registers"],
    "completed_deliverables": ["example_deliverable"],
    "complete": True,
    "next_cursor": "",
}

_MODULE_KEYS = frozenset({"module_id", "kind", "config", "depends_on", "required_gates"})
_ASSET_KEYS = frozenset({"asset_id", "kind", "prompt", "target_path", "width", "height"})

def _id(value: Any, fallback: str) -> str:
    text = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    if not text or not text[0].isalpha():
        text = f"{fallback}_{text}".rstrip("_")
    return text[:63]

def _strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if isinstance(item, str) and item.strip()))

def build_batch_skeleton(batch_id: str, scope: str, deliverables: Sequence[str], exports: Sequence[str], depends_on_batches: Sequence[str] = (), known_module_ids: Sequence[str] = ()) -> dict[str, Any]:
    batch = _id(batch_id, "batch")
    completed = _strings(deliverables) or [f"{batch}_feature"]
    module_ids = [_id(item, "module") for item in exports] or [batch]
    known = set(known_module_ids)
    dependencies = [item for item in _strings(depends_on_batches) if item in known]
    return {
        "modules": [{"module_id": module_id, "kind": "custom_java", "config": {"summary": f"Implementation for {module_id}", "batch_id": batch}, "depends_on": dependencies, "required_gates": []} for module_id in module_ids],
        "assets": [],
        "acceptance_tests": [f"test_{module_id}_registers" for module_id in module_ids],
        "completed_deliverables": completed,
        "complete": True,
        "next_cursor": "",
    }

def merge_model_output_into_skeleton(skeleton: Mapping[str, Any], model_output: Mapping[str, Any], valid_module_catalog: set[str]) -> dict[str, Any]:
    modules: list[dict[str, Any]] = []
    for raw in model_output.get("modules", []) if isinstance(model_output.get("modules"), list) else []:
        if not isinstance(raw, dict):
            continue
        item = {key: raw[key] for key in _MODULE_KEYS if key in raw}
        module_id = _id(item.get("module_id"), "module")
        config = item.get("config") if isinstance(item.get("config"), dict) else {"summary": str(item.get("config") or "")}
        dependencies = [dep for dep in _strings(item.get("depends_on")) if dep in valid_module_catalog and dep != module_id]
        modules.append({"module_id": module_id, "kind": str(item.get("kind") or "custom_java"), "config": config, "depends_on": dependencies, "required_gates": _strings(item.get("required_gates"))})
    if not modules:
        modules = [dict(item) for item in skeleton.get("modules", []) if isinstance(item, dict)]
    assets: list[dict[str, Any]] = []
    for raw in model_output.get("assets", []) if isinstance(model_output.get("assets"), list) else []:
        if not isinstance(raw, dict) or not raw.get("asset_id"):
            continue
        item = {key: raw[key] for key in _ASSET_KEYS if key in raw}
        item["asset_id"] = _id(item.get("asset_id"), "asset")
        assets.append(item)
    tests = _strings(model_output.get("acceptance_tests")) or _strings(skeleton.get("acceptance_tests"))
    completed = _strings(model_output.get("completed_deliverables")) or _strings(skeleton.get("completed_deliverables"))
    return {"modules": modules, "assets": assets, "acceptance_tests": tests, "completed_deliverables": completed, "complete": bool(model_output.get("complete", True)), "next_cursor": str(model_output.get("next_cursor") or "")}
''',
        encoding="utf-8",
    )


def _rewrite_parallel_determinism() -> None:
    (PKG / "parallel_result_determinism_contract.py").write_text(
        '''from __future__ import annotations

import json
from functools import wraps
from typing import Any

def _canonical_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

def _sort_receipts(items: Any) -> Any:
    return sorted(items, key=_canonical_key) if isinstance(items, list) else items

def _canonicalize_generation_result(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    for key in ("module_receipts", "blockbench_receipts"):
        if key in result:
            result[key] = _sort_receipts(result[key])
    unresolved = result.get("unresolved")
    if isinstance(unresolved, list):
        result["unresolved"] = sorted(str(item) for item in unresolved)
    receipt = result.get("asset_receipt")
    if isinstance(receipt, dict) and isinstance(receipt.get("shards"), list):
        receipt["shards"] = _sort_receipts(receipt["shards"])
    return result

def install(*, orchestrator_module: Any) -> None:
    current_execute = orchestrator_module.CompleteProductionOrchestrator._execute_generation_work
    if getattr(current_execute, "_mmm_parallel_result_determinism", False):
        return
    @wraps(current_execute)
    def deterministic_execute(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _canonicalize_generation_result(current_execute(self, *args, **kwargs))
    deterministic_execute._mmm_parallel_result_determinism = True
    orchestrator_module.CompleteProductionOrchestrator._execute_generation_work = deterministic_execute
''',
        encoding="utf-8",
    )


def _clean_runtime_bootstrap() -> None:
    path = PKG / "runtime_bootstrap.py"
    text = path.read_text(encoding="utf-8")
    drop_fragments = (
        "audio_generator,",
        "from .audio_resume_efficiency_contract",
        "install_audio_resume_efficiency(audio_generator)",
        "complete_orchestrator.synthesize_audio_files = audio_generator.synthesize_audio_files",
        "from .planner_production_page_contract import install as install_planner_production_page",
        "install_planner_production_page(complete_planner)",
        "audio_generator_module=audio_generator,",
    )
    text = "\n".join(line for line in text.splitlines() if not any(fragment in line for fragment in drop_fragments)) + "\n"
    text = re.sub(r"install_parallel_result_determinism\(\s*orchestrator_module=complete_orchestrator,\s*\)", "install_parallel_result_determinism(orchestrator_module=complete_orchestrator)", text, flags=re.S)
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8")


def _clean_model_registry_python() -> None:
    path = PKG / "model_registry.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(', "speech_recognition"', '').replace('"speech_recognition",\n', '').replace('        "speech",\n', '')
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8")


def _clean_adapter_init() -> None:
    path = PKG / "model_adapters" / "__init__.py"
    text = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines() if "SpeechAdapter" not in line and ".speech" not in line) + "\n"
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8")


def _clean_registry_yaml(path: Path) -> None:
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    profiles = data.get("profiles", {}) if isinstance(data, dict) else {}
    for profile in profiles.values():
        if isinstance(profile, dict) and isinstance(profile.get("roles"), dict):
            profile["roles"].pop("speech_recognition", None)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _clean_packaged_skills() -> None:
    path = PKG / "packaged_skills.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    def clean(value):
        if isinstance(value, list):
            result = []
            for item in value:
                probe = json.dumps(item, ensure_ascii=False).lower()
                if "generate-audio" in probe or "procedural-audio" in probe or "speech_recognition" in probe:
                    continue
                result.append(clean(item))
            return result
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if "audio" not in key.lower() and "speech_recognition" not in key.lower()}
        return value
    path.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _remove_obsolete_tests() -> None:
    obsolete = (
        "planner_production_page_contract",
        "production_page_durable_contract",
        "audio_generator",
        "audio_resume_efficiency_contract",
        "SpeechAdapter",
        "speech_recognition",
        "AudioRequest",
    )
    for path in (ROOT / "tests").glob("test_*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(token in text for token in obsolete):
            path.unlink()


def _write_regression_tests() -> None:
    (ROOT / "tests" / "test_clean_production_contract.py").write_text(
        '''from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai.planner_template_schema import build_batch_skeleton, merge_model_output_into_skeleton

ROOT = Path(__file__).resolve().parents[1]

def test_host_template_matches_typed_boundary() -> None:
    page = build_batch_skeleton("quality_packaging", "quality", ["smoke"], ["quality_packaging"])
    assert set(page) == {"modules", "assets", "acceptance_tests", "completed_deliverables", "complete", "next_cursor"}
    assert set(page["modules"][0]) == {"module_id", "kind", "config", "depends_on", "required_gates"}

def test_model_cannot_extend_host_schema() -> None:
    skeleton = build_batch_skeleton("custom_features", "custom", ["feature"], ["custom_features"])
    merged = merge_model_output_into_skeleton(skeleton, {"modules": [{"module_id": "custom_features", "kind": "custom_java", "config": {}, "depends_on": [], "required_gates": [], "implements_deliverables": ["bad"]}], "audio": [{"sound_id": "bad"}], "unknown": True}, set())
    assert "audio" not in merged
    assert "unknown" not in merged
    assert "implements_deliverables" not in merged["modules"][0]

def test_removed_dedicated_media_layers_do_not_exist() -> None:
    removed = [
        ROOT / "minecraft_mod_ai/audio_generator.py",
        ROOT / "minecraft_mod_ai/audio_resume_efficiency_contract.py",
        ROOT / "minecraft_mod_ai/model_adapters/speech.py",
        ROOT / "minecraft_mod_ai/planner_production_page_contract.py",
        ROOT / "minecraft_mod_ai/production_page_durable_contract.py",
        ROOT / "plugins/mmm-minecraft-mod-ai/skills/generate-audio/SKILL.md",
    ]
    assert not any(path.exists() for path in removed)

def test_bootstrap_has_no_removed_runtime_hooks() -> None:
    text = (ROOT / "minecraft_mod_ai/runtime_bootstrap.py").read_text(encoding="utf-8")
    for token in ("audio_generator", "audio_resume_efficiency_contract", "planner_production_page_contract"):
        assert token not in text
''',
        encoding="utf-8",
    )


def _delete_paths() -> None:
    targets = [
        PKG / "audio_generator.py",
        PKG / "audio_resume_efficiency_contract.py",
        PKG / "model_adapters" / "speech.py",
        PKG / "planner_production_page_contract.py",
        PKG / "production_page_durable_contract.py",
        ROOT / "plugins" / "mmm-minecraft-mod-ai" / "skills" / "generate-audio",
    ]
    for path in targets:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _assert_no_dedicated_references() -> None:
    banned = re.compile(r"AudioRequest|audio_generator|audio_resume_efficiency_contract|planner_production_page_contract|production_page_durable_contract|SpeechAdapter|speech_recognition|generate-audio")
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path == Path(__file__).resolve():
            continue
        if path.suffix.lower() not in {".py", ".json", ".yaml", ".yml", ".md", ".toml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if banned.search(text):
            offenders.append(str(path.relative_to(ROOT)))
    if offenders:
        raise RuntimeError("dedicated removed-capability references remain: " + ", ".join(offenders[:40]))


def main() -> None:
    _rewrite_planner_template()
    _strip_capability(PKG / "complete_spec.py", "audio", "implements_deliverables")
    _replace_complete_validate(PKG / "complete_spec.py")
    _strip_capability(PKG / "complete_planner.py", "audio", "implements_deliverables")
    _strip_capability(PKG / "complete_orchestrator.py", "audio")
    _strip_capability(PKG / "production_contract.py", "audio", "ai_voice", "speech", "tts", "asr")
    _clean_runtime_bootstrap()
    _rewrite_parallel_determinism()
    _clean_model_registry_python()
    _clean_adapter_init()
    for registry in (ROOT / "config/model_registry.yaml", PKG / "config/model_registry.yaml"):
        if registry.exists():
            _clean_registry_yaml(registry)
    _clean_packaged_skills()
    _delete_paths()
    _remove_obsolete_tests()
    _write_regression_tests()
    _assert_no_dedicated_references()


if __name__ == "__main__":
    main()
