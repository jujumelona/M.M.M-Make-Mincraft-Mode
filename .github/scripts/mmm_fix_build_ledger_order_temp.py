from __future__ import annotations

from pathlib import Path


ORCHESTRATOR = Path("minecraft_mod_ai/complete_orchestrator.py")
TESTS = Path("tests/test_validation_repair_regressions.py")


def patch_orchestrator() -> None:
    text = ORCHESTRATOR.read_text(encoding="utf-8")
    final_manifest = "        final_manifest = self._project_manifest_hash(project_root)\n"
    validation_start = "        def validate_final_source() -> dict[str, Any]:\n"
    build_start = "        reported_jar = _jar_path(build)\n"
    jar_validation = "        jar_validation = run_named_checkpoint(ledger, 'validate-jar'"

    manifest_at = text.index(final_manifest)
    validation_at = text.index(validation_start, manifest_at)
    build_at = text.index(build_start, validation_at)
    jar_at = text.index(jar_validation, build_at)

    build_success_at = text.index(
        "self._succeed_work_node(ledger, 'build-project'", build_at, jar_at
    )
    if build_success_at < validation_at:
        return

    validation_section = text[validation_at:build_at]
    build_section = text[build_at:jar_at]
    rewritten = (
        text[:validation_at]
        + build_section
        + "\n"
        + validation_section
        + text[jar_at:]
    )

    new_validation_at = rewritten.index(validation_start, manifest_at)
    new_build_success_at = rewritten.index(
        "self._succeed_work_node(ledger, 'build-project'", manifest_at, new_validation_at
    )
    if not (manifest_at < new_build_success_at < new_validation_at):
        raise RuntimeError("build-project was not moved before final validation")
    ORCHESTRATOR.write_text(rewritten, encoding="utf-8")


def patch_tests() -> None:
    text = TESTS.read_text(encoding="utf-8")
    marker = "def test_build_work_node_is_committed_before_final_validation()"
    if marker in text:
        return
    text += '''\n\ndef test_build_work_node_is_committed_before_final_validation() -> None:\n    source = Path("minecraft_mod_ai/complete_orchestrator.py").read_text(encoding="utf-8")\n    manifest_at = source.index("final_manifest = self._project_manifest_hash(project_root)")\n    build_done_at = source.index("self._succeed_work_node(ledger, 'build-project'", manifest_at)\n    final_validation_at = source.index("def validate_final_source", manifest_at)\n    final_node_at = source.index("'validate-source-final'", final_validation_at)\n    assert manifest_at < build_done_at < final_validation_at < final_node_at\n'''
    TESTS.write_text(text, encoding="utf-8")


def main() -> None:
    patch_orchestrator()
    patch_tests()


if __name__ == "__main__":
    main()
