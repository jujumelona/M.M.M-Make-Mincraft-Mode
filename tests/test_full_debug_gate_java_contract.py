from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "full-debug-gate.yml"


def test_full_debug_gate_uses_jdtls_compatible_host_jdk() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert 'java-version: "21"' in text
    assert 'java-version: "17"' not in text
