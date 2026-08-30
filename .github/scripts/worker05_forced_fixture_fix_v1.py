from __future__ import annotations

from pathlib import Path

path = Path(__file__).with_name("worker05_forced_capability_canonical_v1.py")
text = path.read_text(encoding="utf-8")
replacements = {
    '''def test_transient_failure_cools_down_without_poisoning(monkeypatch) -> None:\\n    _reset()\\n''': '''def test_transient_failure_cools_down_without_poisoning(monkeypatch) -> None:\\n    _reset()\\n    monkeypatch.setattr(forced, "_native_probe_request", lambda request: request)\\n''',
    '''def test_protocol_negative_expires(monkeypatch) -> None:\\n    _reset()\\n''': '''def test_protocol_negative_expires(monkeypatch) -> None:\\n    _reset()\\n    monkeypatch.setattr(forced, "_native_probe_request", lambda request: request)\\n''',
    '''def test_concurrent_probe_is_deduplicated_per_endpoint() -> None:\\n    _reset()\\n''': '''def test_concurrent_probe_is_deduplicated_per_endpoint(monkeypatch) -> None:\\n    _reset()\\n    monkeypatch.setattr(forced, "_native_probe_request", lambda request: request)\\n''',
}
for old, new in replacements.items():
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"forced capability staging fixture anchor count={count}: {old[:72]!r}")
    text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
Path(__file__).unlink(missing_ok=True)
