from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


validation_path = Path("minecraft_mod_ai/validation_checkpoint_policy.py")
validation = validation_path.read_text(encoding="utf-8")
validation = replace_once(
    validation,
    '_VALIDATION_CHECKPOINTS = frozenset({"validate-source", "validate-jdt"})\n',
    '''_VALIDATION_CHECKPOINT_FAMILIES = {\n    "validate-source": "validate-source",\n    "validate-source-final": "validate-source",\n    "validate-jdt": "validate-jdt",\n    "validate-jdt-final": "validate-jdt",\n}\n\n\ndef _canonical_validation_checkpoint(checkpoint_id: str) -> str:\n    family = _VALIDATION_CHECKPOINT_FAMILIES.get(checkpoint_id)\n    if family is None:\n        raise ValueError(f"Unsupported validation checkpoint: {checkpoint_id}")\n    return family\n''',
    "validation checkpoint families",
)
validation = replace_once(
    validation,
    '''    if checkpoint_id not in _VALIDATION_CHECKPOINTS:\n        raise ValueError(f"Unsupported validation checkpoint: {checkpoint_id}")\n\n    digest = hashlib.sha256()\n    for module in _validation_modules(checkpoint_id):\n''',
    '''    checkpoint_family = _canonical_validation_checkpoint(checkpoint_id)\n\n    digest = hashlib.sha256()\n    for module in _validation_modules(checkpoint_family):\n''',
    "validation implementation canonicalization",
)
validation = replace_once(
    validation,
    '''def cached_validation_is_reusable(checkpoint_id: str, value: Any) -> bool:\n    if not isinstance(value, dict):\n        return False\n    if checkpoint_id == "validate-source":\n        return _complete_source_receipt(value)\n    if checkpoint_id == "validate-jdt":\n        return _complete_jdt_receipt(value)\n    return False\n''',
    '''def cached_validation_is_reusable(checkpoint_id: str, value: Any) -> bool:\n    if not isinstance(value, dict):\n        return False\n    checkpoint_family = _VALIDATION_CHECKPOINT_FAMILIES.get(checkpoint_id)\n    if checkpoint_family == "validate-source":\n        return _complete_source_receipt(value)\n    if checkpoint_family == "validate-jdt":\n        return _complete_jdt_receipt(value)\n    return False\n''',
    "cached validation canonicalization",
)
validation_path.write_text(validation, encoding="utf-8")

repair_path = Path("minecraft_mod_ai/repair_engine.py")
repair = repair_path.read_text(encoding="utf-8")
repair = replace_once(
    repair,
    "import json\nimport os\nimport re\n",
    "import hashlib\nimport json\nimport os\nimport re\n",
    "repair hashlib import",
)
repair = replace_once(
    repair,
    "_HARD_REPAIR_ATTEMPTS = 2\n",
    "_HARD_REPAIR_ATTEMPTS = 2\n_REPAIR_LOG_SNIPPET_CHARS = 3000\n_REPAIR_LOG_READ_BYTES = 16384\n_REPAIR_BUILD_LOG_LIMIT = 4\n_REPAIR_QUERY_PART_LIMIT = 12\n",
    "repair log bounds",
)
class_marker = "\n\nclass RepairEngine:\n"
helper = '''


def _read_bounded_build_log(log_path: Any) -> str:
    raw_path = str(log_path or "").strip()
    if not raw_path:
        return ""
    path = Path(raw_path).expanduser()
    try:
        if not path.is_file() or path.is_symlink():
            return ""
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - _REPAIR_LOG_READ_BYTES))
            payload = handle.read(_REPAIR_LOG_READ_BYTES)
    except (OSError, ValueError):
        return ""
    text = payload.decode("utf-8", errors="replace")
    normalized = text.replace("\\r\\n", "\\n").replace("\\r", "\\n").strip()
    return normalized[-_REPAIR_LOG_SNIPPET_CHARS:]


def _failed_build_log_diagnostics(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    build = evidence.get("build")
    if not isinstance(build, dict):
        return []
    commands = [item for item in build.get("commands", []) if isinstance(item, dict)]
    failed = [
        item
        for item in commands
        if bool(item.get("timed_out"))
        or (
            isinstance(item.get("exit_code"), int)
            and not isinstance(item.get("exit_code"), bool)
            and item.get("exit_code") != 0
        )
    ]
    if not failed and build.get("status") == "FAIL" and commands:
        failed = [commands[-1]]

    diagnostics: list[dict[str, Any]] = []
    for command in failed[-_REPAIR_BUILD_LOG_LIMIT:]:
        output = _read_bounded_build_log(command.get("log_path"))
        item: dict[str, Any] = {
            "name": command.get("name"),
            "exit_code": command.get("exit_code"),
            "timed_out": bool(command.get("timed_out")),
        }
        if output:
            item["output"] = output
        diagnostics.append(item)
    return diagnostics
'''
repair = replace_once(repair, class_marker, helper + class_marker, "repair build log helpers")
old_signature = '''    @staticmethod
    def _signature(evidence: dict[str, Any]) -> str:
        diagnostics = []
        for item in _diagnostic_items(evidence.get("diagnostics")):
            if not isinstance(item, dict):
                continue
            diagnostics.append(
                {
                    "path": item.get("path") or item.get("uri"),
                    "message": item.get("message"),
                    "code": item.get("code"),
                    "severity": item.get("severity"),
                }
            )
        build = evidence.get("build", {})
        return json.dumps(
            {
                "diagnostics": diagnostics,
                "build_status": build.get("status"),
                "build_error": build.get("error"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
'''
new_signature = '''    @staticmethod
    def _signature(evidence: dict[str, Any]) -> str:
        diagnostics = []
        for item in _diagnostic_items(evidence.get("diagnostics")):
            if not isinstance(item, dict):
                continue
            diagnostics.append(
                {
                    "path": item.get("path") or item.get("uri"),
                    "message": item.get("message"),
                    "code": item.get("code"),
                    "severity": item.get("severity"),
                }
            )
        build = evidence.get("build", {})
        build_logs = []
        for item in _failed_build_log_diagnostics(evidence):
            output = str(item.get("output") or "")
            build_logs.append(
                {
                    "name": item.get("name"),
                    "exit_code": item.get("exit_code"),
                    "timed_out": item.get("timed_out"),
                    "output_sha256": (
                        hashlib.sha256(output.encode("utf-8")).hexdigest()
                        if output
                        else ""
                    ),
                }
            )
        return json.dumps(
            {
                "diagnostics": diagnostics,
                "build_status": build.get("status"),
                "build_error": build.get("error"),
                "build_logs": build_logs,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
'''
repair = replace_once(repair, old_signature, new_signature, "repair signature")
repair = replace_once(
    repair,
    '''        for command in evidence.get("build", {}).get("commands", []):
            if not isinstance(command, dict):
                continue
            output = command.get("output")
            if isinstance(output, str):
                query_parts.append(output)
        from .production_tools import ProjectRAGIndex
''',
    '''        build_logs = _failed_build_log_diagnostics(evidence)
        for command in build_logs:
            output = command.get("output")
            if isinstance(output, str) and output:
                query_parts.append(output)
        from .production_tools import ProjectRAGIndex
''',
    "repair context log loading",
)
repair = replace_once(
    repair,
    '            query = " ".join(query_parts) if query_parts else "Minecraft Fabric mod build repair"\n',
    '            bounded_query_parts = query_parts[-_REPAIR_QUERY_PART_LIMIT:]\n            query = " ".join(bounded_query_parts) if bounded_query_parts else "Minecraft Fabric mod build repair"\n',
    "repair RAG query bound",
)
repair = replace_once(
    repair,
    '''        return {
            "diagnostics_files": tuple(sorted(set(diagnostic_paths))),
            "rag": {"hits": rag_hits},
        }
''',
    '''        return {
            "diagnostics_files": tuple(sorted(set(diagnostic_paths))),
            "build_logs": build_logs,
            "rag": {"hits": rag_hits},
        }
''',
    "repair context build logs",
)
repair_path.write_text(repair, encoding="utf-8")

test_path = Path("tests/test_validation_repair_regressions.py")
test_path.write_text('''from __future__ import annotations

from pathlib import Path

import pytest

import minecraft_mod_ai.production_tools as production_tools
import minecraft_mod_ai.repair_engine as repair_module
from minecraft_mod_ai.repair_engine import RepairEngine
from minecraft_mod_ai.validation_checkpoint_policy import (
    cached_validation_is_reusable,
    validation_checkpoint_input,
    validation_implementation_fingerprint,
)


def _failed_evidence(log_path: Path) -> dict[str, object]:
    return {
        "passed": False,
        "diagnostics": None,
        "build": {
            "status": "FAIL",
            "error": "Gradle build failed.",
            "commands": [
                {
                    "name": "build",
                    "exit_code": 1,
                    "timed_out": False,
                    "log_path": str(log_path),
                }
            ],
        },
    }


def test_final_validation_checkpoint_aliases_share_validator_family() -> None:
    assert validation_implementation_fingerprint("validate-source-final") == validation_implementation_fingerprint("validate-source")
    assert validation_implementation_fingerprint("validate-jdt-final") == validation_implementation_fingerprint("validate-jdt")
    scoped = validation_checkpoint_input("validate-source-final", {"project_manifest": "abc"})
    assert scoped["project_manifest"] == "abc"
    assert scoped["_mmm_validation_implementation"].startswith("sha256:")
    source_receipt = {"status": "PASS", "checks_run": 1, "findings": []}
    assert cached_validation_is_reusable("validate-source-final", source_receipt)
    jdt_receipt = {
        "schema_version": "mmm/java-diagnostics-v2",
        "diagnostics": {},
        "pages": [],
        "files_opened": 0,
        "page_count": 0,
        "error_count": 0,
        "warning_count": 0,
    }
    assert cached_validation_is_reusable("validate-jdt-final", jdt_receipt)


def test_unknown_validation_checkpoint_remains_fail_closed() -> None:
    with pytest.raises(ValueError, match="Unsupported validation checkpoint"):
        validation_implementation_fingerprint("validate-unknown")
    assert not cached_validation_is_reusable("validate-unknown", {"status": "PASS", "checks_run": 1, "findings": []})


def test_repair_context_reads_bounded_failed_gradle_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "gradle-build.log"
    marker = "error: cannot find symbol ExampleRegistry"
    log_path.write_text("x" * 5000 + "\\n" + marker + "\\n", encoding="utf-8")

    class BrokenRag:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("RAG intentionally unavailable in unit test")

    monkeypatch.setattr(production_tools, "ProjectRAGIndex", BrokenRag)
    engine = object.__new__(RepairEngine)
    context = engine._context(tmp_path, _failed_evidence(log_path))
    assert context["build_logs"]
    output = context["build_logs"][0]["output"]
    assert marker in output
    assert len(output) <= repair_module._REPAIR_LOG_SNIPPET_CHARS


def test_repair_signature_tracks_actual_failed_gradle_log(tmp_path: Path) -> None:
    log_path = tmp_path / "gradle-build.log"
    log_path.write_text("first concrete compiler failure", encoding="utf-8")
    evidence = _failed_evidence(log_path)
    first = RepairEngine._signature(evidence)
    assert first == RepairEngine._signature(evidence)
    log_path.write_text("second different resource failure", encoding="utf-8")
    assert RepairEngine._signature(evidence) != first


def test_repair_log_reader_fails_safe_for_missing_file(tmp_path: Path) -> None:
    evidence = _failed_evidence(tmp_path / "missing.log")
    assert RepairEngine._signature(evidence)
    assert repair_module._failed_build_log_diagnostics(evidence) == [
        {"name": "build", "exit_code": 1, "timed_out": False}
    ]
''', encoding="utf-8")
