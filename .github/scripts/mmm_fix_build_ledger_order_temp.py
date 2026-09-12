from __future__ import annotations

from pathlib import Path


ORCHESTRATOR = Path("minecraft_mod_ai/complete_orchestrator.py")
INDEX_CONTRACT = Path("minecraft_mod_ai/project_index_execution_reuse_contract.py")
TESTS = Path("tests/test_validation_repair_regressions.py")
INDEX_TESTS = Path("tests/test_project_index_execution_reuse_contract.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def patch_index_contract() -> None:
    text = INDEX_CONTRACT.read_text(encoding="utf-8")
    if "def update_from_receipt(" not in text:
        marker = "\n\ndef tune_gradle_resources(project_root: str | Path, *args: Any, **kwargs: Any):\n"
        wrapper = '''\n\ndef update_from_receipt(project_root: str | Path, receipt: Any) -> None:\n    \"\"\"Incrementally reconcile an execution-scoped index from a mutation receipt.\"\"\"\n    _update_from_receipt(project_root, receipt)\n'''
        text = replace_once(text, marker, wrapper + marker, "public receipt update wrapper")
    if '    "update_from_receipt",\n' not in text:
        text = replace_once(
            text,
            '    "tune_gradle_resources",\n',
            '    "tune_gradle_resources",\n    "update_from_receipt",\n',
            "public receipt update export",
        )
    INDEX_CONTRACT.write_text(text, encoding="utf-8")


def patch_orchestrator() -> None:
    text = ORCHESTRATOR.read_text(encoding="utf-8")
    import_old = '''from .project_index_execution_reuse_contract import (\n    execution_scoped,\n    mark_post_generation,\n    tune_gradle_resources,\n)\n'''
    import_new = '''from .project_index_execution_reuse_contract import (\n    execution_scoped,\n    mark_post_generation,\n    tune_gradle_resources,\n    update_from_receipt as update_execution_project_index_from_receipt,\n)\n'''
    if "update_execution_project_index_from_receipt" not in text:
        text = replace_once(text, import_old, import_new, "orchestrator receipt update import")

    build_hash_old = "'project_manifest': self._project_manifest_hash(project_root), 'run_gametest': options.run_gametest"
    build_hash_new = "'project_manifest': validation_manifest, 'run_gametest': options.run_gametest"
    if build_hash_old in text:
        text = replace_once(text, build_hash_old, build_hash_new, "build manifest reuse")

    repair_old = '''        if isinstance(repair, dict):\n            module_receipts.append({'schema_version': 'mmm/repair-receipt-v2', **repair})\n        if build.get('status') != 'PASS':\n'''
    repair_new = '''        if isinstance(repair, dict):\n            module_receipts.append({'schema_version': 'mmm/repair-receipt-v2', **repair})\n            if repair.get('patch_receipts'):\n                update_execution_project_index_from_receipt(project_root, repair)\n        if build.get('status') != 'PASS':\n'''
    if "update_execution_project_index_from_receipt(project_root, repair)" not in text:
        text = replace_once(text, repair_old, repair_new, "repair index reconciliation")

    final_old = "        final_manifest = self._project_manifest_hash(project_root)\n"
    final_new = '''        final_manifest = str(\n            execution_project_index(ProjectIndex, project_root, policy=self.policy)\n            .manifest_receipt()['sha256']\n        )\n'''
    if final_old in text:
        text = replace_once(text, final_old, final_new, "final manifest reuse")

    validation_start = "        def validate_final_source() -> dict[str, Any]:\n"
    build_start = "        reported_jar = _jar_path(build)\n"
    jar_validation = "        jar_validation = run_named_checkpoint(ledger, 'validate-jar'"
    final_marker = "        final_manifest = str(\n"

    manifest_at = text.index(final_marker)
    validation_at = text.index(validation_start, manifest_at)
    build_at = text.index(build_start, validation_at)
    jar_at = text.index(jar_validation, build_at)

    build_success_at = text.index(
        "self._succeed_work_node(ledger, 'build-project'", build_at, jar_at
    )
    if build_success_at > validation_at:
        validation_section = text[validation_at:build_at]
        build_section = text[build_at:jar_at]
        text = (
            text[:validation_at]
            + build_section
            + "\n"
            + validation_section
            + text[jar_at:]
        )

    new_validation_at = text.index(validation_start, manifest_at)
    new_build_success_at = text.index(
        "self._succeed_work_node(ledger, 'build-project'", manifest_at, new_validation_at
    )
    if not (manifest_at < new_build_success_at < new_validation_at):
        raise RuntimeError("build-project was not moved before final validation")
    ORCHESTRATOR.write_text(text, encoding="utf-8")


def patch_tests() -> None:
    text = TESTS.read_text(encoding="utf-8")
    marker = "def test_build_work_node_is_committed_before_final_validation()"
    if marker not in text:
        text += '''\n\ndef test_build_work_node_is_committed_before_final_validation() -> None:\n    source = Path("minecraft_mod_ai/complete_orchestrator.py").read_text(encoding="utf-8")\n    final_manifest_at = source.index("final_manifest = str(")\n    repair_refresh_at = source.index("update_execution_project_index_from_receipt(project_root, repair)")\n    build_done_at = source.index("self._succeed_work_node(ledger, 'build-project'", final_manifest_at)\n    final_validation_at = source.index("def validate_final_source", final_manifest_at)\n    final_node_at = source.index("'validate-source-final'", final_validation_at)\n    assert repair_refresh_at < final_manifest_at < build_done_at < final_validation_at < final_node_at\n    assert source.count("self._project_manifest_hash(project_root)", 0, source.index("build_bundle =")) >= 2\n    assert "self._project_manifest_hash(project_root)" not in source[source.index("build_bundle ="):]\n'''
        TESTS.write_text(text, encoding="utf-8")

    index_tests = INDEX_TESTS.read_text(encoding="utf-8")
    index_tests = index_tests.replace(
        "contract._update_from_receipt(tmp_path, _known_receipt())",
        "contract.update_from_receipt(tmp_path, _known_receipt())",
    )
    index_tests = index_tests.replace(
        'contract._update_from_receipt(tmp_path, {"status": "TUNED"})',
        'contract.update_from_receipt(tmp_path, {"status": "TUNED"})',
    )
    INDEX_TESTS.write_text(index_tests, encoding="utf-8")


def main() -> None:
    patch_index_contract()
    patch_orchestrator()
    patch_tests()


if __name__ == "__main__":
    main()
