from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import project_manifest_hash_efficiency_contract as contract

ROOT = Path(__file__).resolve().parents[1]


class _FakeOrchestrator:
    def __init__(self) -> None:
        self.calls = 0

    def _project_manifest_hash(self, project_root: Path) -> str:
        self.calls += 1
        payload = []
        root = Path(project_root)
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if any(part == ".minecraft_ai" for part in relative.parts):
                continue
            if path.suffix.lower() not in {".java", ".json", ".gradle", ".properties"} and path.name not in {
                "build.gradle",
                "settings.gradle",
                "gradle.properties",
                "fabric.mod.json",
            }:
                continue
            payload.append((relative.as_posix(), path.read_bytes()))
        return repr(payload)

    def execute(self, project_root: Path) -> tuple[str, str]:
        return (
            self._project_manifest_hash(project_root),
            self._project_manifest_hash(project_root),
        )


def _modules():
    orchestrator_module = SimpleNamespace(CompleteProductionOrchestrator=_FakeOrchestrator)
    project_index_module = SimpleNamespace(
        _IGNORED_PARTS={".minecraft_ai", ".git", ".gradle", "build", "run", ".cache", "node_modules"},
        _TEXT_SUFFIXES={".java", ".json", ".gradle", ".properties"},
    )
    return orchestrator_module, project_index_module


def _write_source(tmp_path: Path, text: str = "class Test {}\n") -> Path:
    source = tmp_path / "src/main/java/example/Test.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(text, encoding="utf-8")
    return source


def test_unchanged_tree_reuses_full_manifest_hash(tmp_path: Path) -> None:
    contract._CACHE.clear()
    orchestrator_module, project_index_module = _modules()
    contract.install(orchestrator_module, project_index_module)
    _write_source(tmp_path)

    orchestrator = orchestrator_module.CompleteProductionOrchestrator()
    first = orchestrator._project_manifest_hash(tmp_path)
    second = orchestrator._project_manifest_hash(tmp_path)

    assert second == first
    assert orchestrator.calls == 1


def test_execution_scope_skips_repeated_metadata_scans(monkeypatch, tmp_path: Path) -> None:
    contract._CACHE.clear()
    orchestrator_module, project_index_module = _modules()
    contract.install(orchestrator_module, project_index_module)
    _write_source(tmp_path)

    real_signature = contract._metadata_signature
    signature_calls = 0

    def counted_signature(module, root):
        nonlocal signature_calls
        signature_calls += 1
        return real_signature(module, root)

    monkeypatch.setattr(contract, "_metadata_signature", counted_signature)
    orchestrator = orchestrator_module.CompleteProductionOrchestrator()
    first, second = orchestrator.execute(tmp_path)

    assert second == first
    assert orchestrator.calls == 1
    # The first authoritative commitment is fenced before+after. The second use in
    # the same execute() call is O(1) and performs no recursive metadata scan.
    assert signature_calls == 2


def test_execution_scope_is_not_reused_across_runs(tmp_path: Path) -> None:
    contract._CACHE.clear()
    orchestrator_module, project_index_module = _modules()
    contract.install(orchestrator_module, project_index_module)
    source = _write_source(tmp_path)

    orchestrator = orchestrator_module.CompleteProductionOrchestrator()
    first, _ = orchestrator.execute(tmp_path)
    source.write_text("class Test { int value = 1; }\n", encoding="utf-8")
    second, _ = orchestrator.execute(tmp_path)

    assert second != first
    assert orchestrator.calls == 2


def test_source_change_invalidates_metadata_cache(tmp_path: Path) -> None:
    contract._CACHE.clear()
    orchestrator_module, project_index_module = _modules()
    contract.install(orchestrator_module, project_index_module)
    source = _write_source(tmp_path)

    orchestrator = orchestrator_module.CompleteProductionOrchestrator()
    first = orchestrator._project_manifest_hash(tmp_path)
    source.write_text("class Test { int value = 1; }\n", encoding="utf-8")
    second = orchestrator._project_manifest_hash(tmp_path)

    assert second != first
    assert orchestrator.calls == 2


def test_audit_metadata_does_not_invalidate_source_manifest_cache(tmp_path: Path) -> None:
    contract._CACHE.clear()
    orchestrator_module, project_index_module = _modules()
    contract.install(orchestrator_module, project_index_module)
    _write_source(tmp_path)

    orchestrator = orchestrator_module.CompleteProductionOrchestrator()
    first = orchestrator._project_manifest_hash(tmp_path)

    audit = tmp_path / ".minecraft_ai/project-index.json"
    audit.parent.mkdir(parents=True)
    audit.write_text('{"status":"updated"}\n', encoding="utf-8")
    second = orchestrator._project_manifest_hash(tmp_path)

    assert second == first
    assert orchestrator.calls == 1


def test_metadata_signature_prunes_ignored_subtrees(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_source(tmp_path)
    ignored = tmp_path / "build/generated"
    ignored.mkdir(parents=True)
    (ignored / "Ignored.java").write_text("class Ignored {}\n", encoding="utf-8")

    real_stat = Path.stat

    def guarded_stat(path: Path, *args, **kwargs):
        relative = path.relative_to(tmp_path)
        if "build" in relative.parts:
            raise AssertionError("ignored build subtree must not be stat-scanned")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    _orchestrator_module, project_index_module = _modules()

    signature = contract._metadata_signature(project_index_module, tmp_path)

    assert len(signature) == 64


def test_execution_manifest_reuse_stays_before_build_repair_mutation() -> None:
    source = (ROOT / "minecraft_mod_ai/complete_orchestrator.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CompleteProductionOrchestrator"
    )
    execute = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute"
    )
    build_repair = next(
        node
        for node in execute.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_with_repair"
    )
    build_bundle = next(
        node
        for node in execute.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "build_bundle" for target in node.targets)
    )

    def manifest_calls(node: ast.AST) -> list[ast.Call]:
        return [
            item
            for item in ast.walk(node)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "_project_manifest_hash"
        ]

    calls = manifest_calls(execute)
    assert len(calls) >= 2
    assert manifest_calls(build_repair) == []
    assert all(call.lineno <= build_bundle.end_lineno for call in calls)
    assert not any(call.lineno > build_bundle.end_lineno for call in calls)
