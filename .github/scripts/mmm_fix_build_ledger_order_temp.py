from __future__ import annotations

from pathlib import Path


TESTS = Path("tests/test_validation_repair_regressions.py")


def main() -> None:
    text = TESTS.read_text(encoding="utf-8")
    marker = "def test_repair_receipt_updates_execution_project_index_manifest("
    if marker in text:
        return
    text += '''\n\ndef test_repair_receipt_updates_execution_project_index_manifest(tmp_path: Path) -> None:\n    from minecraft_mod_ai import project_index_execution_reuse_contract as reuse\n    from minecraft_mod_ai.project_index import ProjectIndex\n\n    source = tmp_path / "src/main/java/example/Test.java"\n    source.parent.mkdir(parents=True)\n    source.write_text("class Test {}\\n", encoding="utf-8")\n\n    @reuse.execution_scoped\n    def scenario() -> tuple[str, str]:\n        reuse.mark_post_generation()\n        index = reuse.project_index(ProjectIndex, tmp_path)\n        before = str(index.manifest_receipt()["sha256"])\n        source.write_text("class Test { int repaired = 1; }\\n", encoding="utf-8")\n        reuse.update_from_receipt(\n            tmp_path,\n            {\n                "status": "PASS",\n                "patch_receipts": [\n                    {\n                        "schema_version": "mmm/source-patch-receipt-v1",\n                        "status": "APPLIED",\n                        "operations": [\n                            {\n                                "operation": "replace",\n                                "path": "src/main/java/example/Test.java",\n                            }\n                        ],\n                    }\n                ],\n            },\n        )\n        assert reuse.project_index(ProjectIndex, tmp_path) is index\n        after = str(index.manifest_receipt()["sha256"])\n        return before, after\n\n    before, after = scenario()\n    assert after != before\n'''
    TESTS.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
