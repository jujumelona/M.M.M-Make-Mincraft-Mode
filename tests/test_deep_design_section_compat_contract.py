from __future__ import annotations

import inspect
from types import SimpleNamespace

from minecraft_mod_ai import deep_design_section_compat_contract as compat


def test_compat_wrapper_preserves_prose_first_section_api(monkeypatch):
    calls = []

    def base(*, prompt, section_id, fields, research):
        calls.append((prompt, section_id, tuple(fields), research))
        return [{"role": "system", "content": "BASE"}]

    def stale(*, prompt, section_id, fields, research, prior_error, prior_candidate):
        raise AssertionError("stale wrapper must be replaced")

    stale.__wrapped__ = base
    deep = SimpleNamespace(
        _deep_section_messages=stale,
        _SECTION_MARKER="__mmm_deep_design_section_prompt__",
    )
    agentic = SimpleNamespace(_section_messages=stale)

    monkeypatch.setattr(compat, "_INSTALLED", False)
    monkeypatch.setattr(compat, "_repair", lambda: compat._repair_for(agentic, deep))
    compat.install()

    result = agentic._section_messages(
        prompt="p",
        section_id="identity_and_loop",
        fields=("title", "pitch", "core_loop"),
        research={"source": "host"},
    )

    assert calls == [("p", "identity_and_loop", ("title", "pitch", "core_loop"), {"source": "host"})]
    assert "PRODUCTION DEPTH" in result[0]["content"]
    signature = inspect.signature(agentic._section_messages)
    assert tuple(signature.parameters) == ("prompt", "section_id", "fields", "research")
