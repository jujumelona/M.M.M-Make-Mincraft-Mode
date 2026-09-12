from __future__ import annotations

from pathlib import Path


ORCHESTRATOR = Path("minecraft_mod_ai/complete_orchestrator.py")
INDEX_REUSE = Path("minecraft_mod_ai/project_index_execution_reuse_contract.py")
MANIFEST_CACHE = Path("minecraft_mod_ai/project_manifest_hash_efficiency_contract.py")
REGRESSION_TESTS = Path("tests/test_validation_repair_regressions.py")
MANIFEST_TESTS = Path("tests/test_project_manifest_hash_efficiency_contract.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def patch_index_reuse() -> None:
    text = INDEX_REUSE.read_text(encoding="utf-8")
    marker = "def reconcile_project_mutation("
    if marker not in text:
        anchor = "\ndef tune_gradle_resources(project_root: str | Path, *args: Any, **kwargs: Any):\n"
        addition = '''\ndef reconcile_project_mutation(project_root: str | Path, receipt: Any) -> str | None:\n    \"\"\"Incrementally reconcile execution-scoped indexes after an exact mutation receipt.\"\"\"\n\n    if not _POST_GENERATION.get():\n        return None\n    cache = _EXECUTION_INDEXES.get()\n    if cache is None:\n        return None\n    _update_from_receipt(project_root, receipt)\n    indexes = _cached_indexes_for_root(cache, project_root)\n    if not indexes:\n        return None\n    manifests = {str(index.manifest_receipt()[\"sha256\"]) for index in indexes}\n    if len(manifests) != 1:\n        raise RuntimeError(\"Execution ProjectIndex instances disagree after source mutation.\")\n    return manifests.pop()\n\n'''
        text = replace_once(text, anchor, addition + anchor, "index reconciliation helper")
        text = replace_once(
            text,
            '    "project_index",\n    "tune_gradle_resources",\n',
            '    "project_index",\n    "reconcile_project_mutation",\n    "tune_gradle_resources",\n',
            "index reconciliation export",
        )
        INDEX_REUSE.write_text(text, encoding="utf-8")


def patch_manifest_cache() -> None:
    text = MANIFEST_CACHE.read_text(encoding="utf-8")
    marker = "def replace_execution_manifest("
    if marker not in text:
        anchor = "\ndef install(orchestrator_module: Any, project_index_module: Any) -> None:\n"
        addition = '''\ndef replace_execution_manifest(project_root: str | Path, manifest_sha256: str) -> None:\n    \"\"\"Replace the run-local cached manifest after receipt-proven source mutation.\"\"\"\n\n    value = str(manifest_sha256).strip()\n    if not value.startswith(\"sha256:\"):\n        raise ValueError(\"manifest_sha256 must be a sha256: value\")\n    root_key = str(Path(project_root).expanduser().resolve())\n    with _CACHE_LOCK:\n        _CACHE.pop(root_key, None)\n    execution_cache = _EXECUTION_CACHE.get()\n    if execution_cache is not None:\n        execution_cache[root_key] = value\n\n'''
        text = replace_once(text, anchor, addition + anchor, "execution manifest replacement helper")
        text = replace_once(
            text,
            '__all__ = ["install"]\n',
            '__all__ = ["install", "replace_execution_manifest"]\n',
            "manifest helper export",
        )
        MANIFEST_CACHE.write_text(text, encoding="utf-8")


def patch_orchestrator() -> None:
    text = ORCHESTRATOR.read_text(encoding="utf-8")
    if "reconcile_project_mutation," not in text:
        text = replace_once(
            text,
            "    mark_post_generation,\n    tune_gradle_resources,\n",
            "    mark_post_generation,\n    reconcile_project_mutation,\n    tune_gradle_resources,\n",
            "orchestrator reconciliation import",
        )
    if "from .project_manifest_hash_efficiency_contract import replace_execution_manifest\n" not in text:
        text = replace_once(
            text,
            "from .proposal_store import write_sharded_complete_proposal\n",
            "from .project_manifest_hash_efficiency_contract import replace_execution_manifest\nfrom .proposal_store import write_sharded_complete_proposal\n",
            "orchestrator manifest cache import",
        )
    repair_line = "                repair_result = RepairEngine(router=router, gradle_cache=cache, policy=self.policy).repair(project_root, run_gametest=options.run_gametest, max_attempts=options.max_repair_attempts)\n"
    reconciliation = repair_line + '''                if isinstance(repair_result, dict) and repair_result.get('patch_receipts'):\n                    repaired_manifest = reconcile_project_mutation(project_root, repair_result)\n                    if repaired_manifest is not None:\n                        replace_execution_manifest(project_root, repaired_manifest)\n'''
    if "repaired_manifest = reconcile_project_mutation" not in text:
        text = replace_once(text, repair_line, reconciliation, "repair manifest reconciliation")

    final_manifest = "        final_manifest = self._project_manifest_hash(project_root)\n"
    validation_start = "        def validate_final_source() -> dict[str, Any]:\n"
    build_start = "        reported_jar = _jar_path(build)\n"
    jar_validation = "        jar_validation = run_named_checkpoint(ledger, 'validate-jar'"
    manifest_at = text.index(final_manifest)
    validation_at = text.index(validation_start, manifest_at)
    build_at = text.index(build_start, validation_at)
    jar_at = text.index(jar_validation, build_at)
    build_success_at = text.index("self._succeed_work_node(ledger, 'build-project'", build_at, jar_at)
    if build_success_at >= validation_at:
        validation_section = text[validation_at:build_at]
        build_section = text[build_at:jar_at]
        text = text[:validation_at] + build_section + "\n" + validation_section + text[jar_at:]

    new_validation_at = text.index(validation_start, manifest_at)
    new_build_success_at = text.index("self._succeed_work_node(ledger, 'build-project'", manifest_at, new_validation_at)
    if not (manifest_at < new_build_success_at < new_validation_at):
        raise RuntimeError("build-project was not moved before final validation")
    ORCHESTRATOR.write_text(text, encoding="utf-8")


def patch_tests() -> None:
    text = REGRESSION_TESTS.read_text(encoding="utf-8")
    marker = "def test_build_work_node_is_committed_before_final_validation()"
    if marker not in text:
        text += '''\n\ndef test_build_work_node_is_committed_before_final_validation() -> None:\n    source = Path("minecraft_mod_ai/complete_orchestrator.py").read_text(encoding="utf-8")\n    manifest_at = source.index("final_manifest = self._project_manifest_hash(project_root)")\n    build_done_at = source.index("self._succeed_work_node(ledger, 'build-project'", manifest_at)\n    final_validation_at = source.index("def validate_final_source", manifest_at)\n    final_node_at = source.index("'validate-source-final'", final_validation_at)\n    assert manifest_at < build_done_at < final_validation_at < final_node_at\n'''
    marker2 = "def test_repair_receipt_reconciles_execution_project_index_and_manifest_cache("
    if marker2 not in text:
        text += '''\n\ndef test_repair_receipt_reconciles_execution_project_index_and_manifest_cache(tmp_path: Path) -> None:\n    from minecraft_mod_ai.project_index import ProjectIndex\n    from minecraft_mod_ai import project_index_execution_reuse_contract as reuse\n    from minecraft_mod_ai import project_manifest_hash_efficiency_contract as manifest_cache\n\n    source = tmp_path / "src/main/java/example/Test.java"\n    source.parent.mkdir(parents=True)\n    source.write_text("class Test {}\\n", encoding="utf-8")\n\n    @reuse.execution_scoped\n    def scenario() -> tuple[str, str]:\n        reuse.mark_post_generation()\n        index = reuse.project_index(ProjectIndex, tmp_path)\n        before = str(index.manifest_receipt()["sha256"])\n        key = str(tmp_path.resolve())\n        token = manifest_cache._EXECUTION_CACHE.set({key: before})\n        try:\n            source.write_text("class Test { int repaired = 1; }\\n", encoding="utf-8")\n            receipt = {\n                "status": "PASS",\n                "patch_receipts": [\n                    {\n                        "schema_version": "mmm/source-patch-receipt-v1",\n                        "status": "APPLIED",\n                        "operations": [\n                            {"operation": "replace", "path": "src/main/java/example/Test.java"}\n                        ],\n                    }\n                ],\n            }\n            after = reuse.reconcile_project_mutation(tmp_path, receipt)\n            assert after is not None\n            manifest_cache.replace_execution_manifest(tmp_path, after)\n            assert manifest_cache._EXECUTION_CACHE.get()[key] == after\n            return before, after\n        finally:\n            manifest_cache._EXECUTION_CACHE.reset(token)\n\n    before, after = scenario()\n    assert after != before\n'''
    REGRESSION_TESTS.write_text(text, encoding="utf-8")

    text = MANIFEST_TESTS.read_text(encoding="utf-8")
    old = '''def test_execution_manifest_reuse_stays_before_build_repair_mutation() -> None:\n    source = (ROOT / "minecraft_mod_ai/complete_orchestrator.py").read_text(encoding="utf-8")\n    tree = ast.parse(source)\n    cls = next(\n        node\n        for node in tree.body\n        if isinstance(node, ast.ClassDef) and node.name == "CompleteProductionOrchestrator"\n    )\n    execute = next(\n        node\n        for node in cls.body\n        if isinstance(node, ast.FunctionDef) and node.name == "execute"\n    )\n    build_repair = next(\n        node\n        for node in execute.body\n        if isinstance(node, ast.FunctionDef) and node.name == "build_with_repair"\n    )\n    build_bundle = next(\n        node\n        for node in execute.body\n        if isinstance(node, ast.Assign)\n        and any(isinstance(target, ast.Name) and target.id == "build_bundle" for target in node.targets)\n    )\n\n    def manifest_calls(node: ast.AST) -> list[ast.Call]:\n        return [\n            item\n            for item in ast.walk(node)\n            if isinstance(item, ast.Call)\n            and isinstance(item.func, ast.Attribute)\n            and item.func.attr == "_project_manifest_hash"\n        ]\n\n    calls = manifest_calls(execute)\n    assert len(calls) >= 2\n    assert manifest_calls(build_repair) == []\n    assert all(call.lineno <= build_bundle.end_lineno for call in calls)\n    assert not any(call.lineno > build_bundle.end_lineno for call in calls)\n'''
    new = '''def test_execution_manifest_reuse_refreshes_once_after_build_repair_mutation() -> None:\n    source = (ROOT / "minecraft_mod_ai/complete_orchestrator.py").read_text(encoding="utf-8")\n    tree = ast.parse(source)\n    cls = next(\n        node\n        for node in tree.body\n        if isinstance(node, ast.ClassDef) and node.name == "CompleteProductionOrchestrator"\n    )\n    execute = next(\n        node\n        for node in cls.body\n        if isinstance(node, ast.FunctionDef) and node.name == "execute"\n    )\n    build_repair = next(\n        node\n        for node in execute.body\n        if isinstance(node, ast.FunctionDef) and node.name == "build_with_repair"\n    )\n    build_bundle = next(\n        node\n        for node in execute.body\n        if isinstance(node, ast.Assign)\n        and any(isinstance(target, ast.Name) and target.id == "build_bundle" for target in node.targets)\n    )\n\n    def manifest_calls(node: ast.AST) -> list[ast.Call]:\n        return [\n            item\n            for item in ast.walk(node)\n            if isinstance(item, ast.Call)\n            and isinstance(item.func, ast.Attribute)\n            and item.func.attr == "_project_manifest_hash"\n        ]\n\n    calls = manifest_calls(execute)\n    post_build = [call for call in calls if call.lineno > build_bundle.end_lineno]\n    assert manifest_calls(build_repair) == []\n    assert len(post_build) == 1\n    assert "reconcile_project_mutation(project_root, repair_result)" in source\n    assert "replace_execution_manifest(project_root, repaired_manifest)" in source\n'''
    if old in text:
        text = text.replace(old, new, 1)
    elif "def test_execution_manifest_reuse_refreshes_once_after_build_repair_mutation()" not in text:
        raise RuntimeError("manifest efficiency regression block not found")
    MANIFEST_TESTS.write_text(text, encoding="utf-8")


def main() -> None:
    patch_index_reuse()
    patch_manifest_cache()
    patch_orchestrator()
    patch_tests()


if __name__ == "__main__":
    main()
