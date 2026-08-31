from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one old block, found {count}")
    return text.replace(old, new, 1)


path = ROOT / "minecraft_mod_ai" / "agentic_optimization_contract.py"
text = path.read_text(encoding="utf-8")

text = replace_once(
    text,
    "from .preference_training import PreferenceCandidate, PreferenceTraceStore\n",
    "from .preference_training import PreferenceCandidate, PreferenceTraceStore\n"
    "from .validation_diagnostic_contract import (\n"
    "    diagnostic_errors,\n"
    "    diagnostic_items,\n"
    "    run_diagnostics,\n"
    ")\n",
    "canonical diagnostic imports",
)

text = replace_once(
    text,
    "def _compact_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:\n"
    "    try:\n"
    "        from .repair_diagnostics_contract import flatten_diagnostics\n"
    "        diagnostics = flatten_diagnostics(evidence.get('diagnostics'))\n"
    "    except Exception:\n"
    "        raw = evidence.get('diagnostics', {})\n"
    "        diagnostics = raw.get('diagnostics', []) if isinstance(raw, Mapping) else []\n"
    "        diagnostics = diagnostics if isinstance(diagnostics, list) else []\n",
    "def _diagnostic_receipt(value: Any) -> Mapping[str, Any] | None:\n"
    "    if isinstance(value, Mapping):\n"
    "        return value\n"
    "    if isinstance(value, list):\n"
    "        return {'diagnostics': value}\n"
    "    return None\n\n"
    "def _compact_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:\n"
    "    diagnostics = diagnostic_items(_diagnostic_receipt(evidence.get('diagnostics')))\n",
    "compact evidence diagnostic authority",
)

text = replace_once(
    text,
    "    try:\n"
    "        from .repair_diagnostics_contract import diagnostic_errors\n"
    "        errors = diagnostic_errors(evidence.get('diagnostics'))\n"
    "    except Exception:\n"
    "        errors = []\n",
    "    errors = diagnostic_errors(_diagnostic_receipt(evidence.get('diagnostics')))\n",
    "candidate count diagnostic authority",
)

text = replace_once(
    text,
    "def _diagnostic_paths(evidence: Mapping[str, Any]) -> set[str]:\n"
    "    try:\n"
    "        from .repair_diagnostics_contract import flatten_diagnostics\n"
    "        values = flatten_diagnostics(evidence.get('diagnostics'))\n"
    "    except Exception:\n"
    "        values = []\n",
    "def _diagnostic_paths(evidence: Mapping[str, Any]) -> set[str]:\n"
    "    values = diagnostic_items(_diagnostic_receipt(evidence.get('diagnostics')))\n",
    "diagnostic path authority",
)

text = replace_once(
    text,
    "        from .source_patch import TransactionalSourcePatcher\n"
    "        from .validation_diagnostic_contract import diagnostic_errors, run_diagnostics\n",
    "        from .source_patch import TransactionalSourcePatcher\n",
    "remove verifier local diagnostic import",
)

path.write_text(text, encoding="utf-8")


test_path = ROOT / "tests" / "test_worker12_shared_core.py"
tests = test_path.read_text(encoding="utf-8")
marker = "\ndef test_colab_profile_switch_removes_stale_remote_credentials(monkeypatch) -> None:\n"
new_tests = '''\n\ndef test_agentic_diagnostics_use_canonical_authority_without_legacy_adapter() -> None:\n    source = inspect.getsource(agentic_optimization_contract)\n    assert "repair_diagnostics_contract" not in source\n    receipt = {\n        "status": "AVAILABLE",\n        "diagnostics": {\n            "file:///src/main/java/A.java": [\n                {"severity": 1, "message": "compile error", "code": "E1"}\n            ]\n        },\n    }\n    compact = agentic_optimization_contract._compact_evidence(\n        {"diagnostics": receipt, "build": {"status": "PASS"}}\n    )\n    assert compact["diagnostics"] == [\n        {\n            "path": "file:///src/main/java/A.java",\n            "message": "compile error",\n            "code": "E1",\n            "severity": 1,\n        }\n    ]\n    assert agentic_optimization_contract._diagnostic_paths(\n        {"diagnostics": receipt}\n    ) == {"file:///src/main/java/A.java", "A.java"}\n'''
if "test_agentic_diagnostics_use_canonical_authority_without_legacy_adapter" not in tests:
    if marker not in tests:
        raise SystemExit("worker12 test insertion marker missing")
    tests = tests.replace(marker, new_tests + marker, 1)
    test_path.write_text(tests, encoding="utf-8")

print("worker12 native cleanup applied")
