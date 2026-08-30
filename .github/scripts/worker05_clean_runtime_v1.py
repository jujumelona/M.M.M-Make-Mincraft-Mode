from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_exact(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# 1) Finite tool-round policy belongs to its canonical owner, not a runtime wrapper.
replace_exact(
    "minecraft_mod_ai/model_router.py",
    '''def _agent_tool_round_limit() -> int | None:\n    raw = os.environ.get("MMM_AGENT_TOOL_ROUNDS", "").strip()\n    if not raw:\n        return None\n    try:\n        value = int(raw)\n    except ValueError:\n        return None\n    return value if value > 0 else None\n''',
    '''_DEFAULT_AGENT_TOOL_ROUNDS = 128\n_MIN_AGENT_TOOL_ROUNDS = 16\n_MAX_AGENT_TOOL_ROUNDS = 512\n\n\ndef _default_agent_tool_rounds() -> int:\n    raw = os.environ.get("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "").strip()\n    if not raw:\n        return _DEFAULT_AGENT_TOOL_ROUNDS\n    try:\n        value = int(raw)\n    except ValueError:\n        return _DEFAULT_AGENT_TOOL_ROUNDS\n    return max(_MIN_AGENT_TOOL_ROUNDS, min(_MAX_AGENT_TOOL_ROUNDS, value))\n\n\ndef _agent_tool_round_limit() -> int:\n    raw = os.environ.get("MMM_AGENT_TOOL_ROUNDS", "").strip()\n    if not raw:\n        return _default_agent_tool_rounds()\n    try:\n        value = int(raw)\n    except ValueError:\n        return _default_agent_tool_rounds()\n    return value if value > 0 else _default_agent_tool_rounds()\n''',
)

replace_exact(
    "minecraft_mod_ai/runtime_bootstrap.py",
    "        model_router,\n    )\n",
    "        model_router,\n    )\n",
)
# Remove only the explicit shim import/install; the model_router import remains used.
replace_exact(
    "minecraft_mod_ai/runtime_bootstrap.py",
    "    from .llama_tool_round_safety_contract import install as install_tool_round_safety\n",
    "",
)
replace_exact(
    "minecraft_mod_ai/runtime_bootstrap.py",
    "    install_tool_round_safety(model_router)\n",
    "",
)

# 2) The emergency context owner already performs deterministic, protocol-safe shrinking.
# Never fall back to [first, last-2, last-1], and never spend a third decode on that lossy window.
replace_exact(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '''        if not _replace_live_messages(messages, emergency):\n            if len(messages) > 3:\n                forced = [messages[0], messages[-2], messages[-1]]\n                _replace_live_messages(messages, tuple(forced))\n            else:\n                mark_context_recovery_exhausted(exc)\n                raise\n''',
    '''        if not _replace_live_messages(messages, emergency):\n            mark_context_recovery_exhausted(exc)\n            raise\n''',
)
replace_exact(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '''            if retry_kind == CONTEXT_PRESSURE:\n                if len(messages) > 3:\n                    forced = [messages[0], messages[-2], messages[-1]]\n                    if _replace_live_messages(messages, tuple(forced)):\n                        ultra_request = replace(\n                            turn_request,\n                            messages=tuple(messages),\n                            media_paths=media_paths,\n                        )\n                        try:\n                            with router._generation_scope(config):\n                                return adapter.generate_turn(ultra_request)\n                        except BaseException as ultra_exc:\n                            if completion_boundary_kind(ultra_exc) == OUTPUT_EXHAUSTED:\n                                return _retry_atomic_after_output_exhaustion(\n                                    router,\n                                    config=config,\n                                    adapter=adapter,\n                                    request=ultra_request,\n                                    messages=messages,\n                                    media_paths=media_paths,\n                                )\n                mark_context_recovery_exhausted(exc)\n                raise exc from retry_exc\n''',
    '''            if retry_kind == CONTEXT_PRESSURE:\n                mark_context_recovery_exhausted(exc)\n                raise exc from retry_exc\n''',
)
replace_exact(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    "        if round_limit is not None and state.step_index > round_limit:\n",
    "        if state.step_index > round_limit:\n",
)
replace_exact(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '                    "Agent reached the explicit tool-round limit before required "\n',
    '                    "Agent reached the host tool-round limit before required "\n',
)
replace_exact(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '                    "Writable coder reached the explicit tool-round limit before a "\n',
    '                    "Writable coder reached the host tool-round limit before a "\n',
)
replace_exact(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '                    f"The explicitly configured host tool limit was reached after {round_limit} rounds. "\n',
    '                    f"The host tool-round limit was reached after {round_limit} rounds. "\n',
)

# 3) Once the lossy fallback is gone, its defensive replacement wrapper is dead code.
context_path = ROOT / "minecraft_mod_ai/llama_context_safety_contract.py"
context_text = context_path.read_text(encoding="utf-8")
context_text = context_text.replace('from functools import wraps\n', 'from functools import wraps\n')
replace_exact(
    "minecraft_mod_ai/llama_context_safety_contract.py",
    '_REPLACE_MARKER = "_mmm_protocol_safe_live_replace_v1"\n',
    "",
)
start = context_text.find("\ndef _is_unsafe_three_message_fallback(")
end = context_text.find("\ndef install(context_module: Any) -> None:")
if start < 0 or end < 0 or end <= start:
    raise SystemExit("llama_context_safety_contract.py: replacement-guard block not found")
context_text = context_text[:start] + context_text[end:]
context_text = context_text.replace("\n    _install_tool_loop_guards(context_module)\n", "\n")
context_text = context_text.replace('    "_is_unsafe_three_message_fallback",\n', "")
context_path.write_text(context_text, encoding="utf-8")

# 4) Move shim tests to the owner and delete wrapper-only assertions.
old_test = ROOT / "tests/test_llama_tool_round_safety_contract.py"
if not old_test.exists():
    raise SystemExit("missing tests/test_llama_tool_round_safety_contract.py")
old_test.unlink()
(ROOT / "tests/test_agent_tool_round_policy.py").write_text(
    '''from __future__ import annotations\n\nfrom minecraft_mod_ai import model_router\n\n\ndef test_default_tool_round_limit_is_finite(monkeypatch) -> None:\n    monkeypatch.delenv("MMM_AGENT_TOOL_ROUNDS", raising=False)\n    monkeypatch.delenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", raising=False)\n    assert model_router._agent_tool_round_limit() == 128\n\n\ndef test_default_tool_round_limit_is_tunable_with_safe_bounds(monkeypatch) -> None:\n    monkeypatch.delenv("MMM_AGENT_TOOL_ROUNDS", raising=False)\n    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "64")\n    assert model_router._agent_tool_round_limit() == 64\n    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "1")\n    assert model_router._agent_tool_round_limit() == 16\n    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "9999")\n    assert model_router._agent_tool_round_limit() == 512\n\n\ndef test_explicit_positive_tool_round_limit_wins(monkeypatch) -> None:\n    monkeypatch.setenv("MMM_AGENT_TOOL_ROUNDS", "37")\n    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "64")\n    assert model_router._agent_tool_round_limit() == 37\n\n\ndef test_invalid_or_nonpositive_explicit_limit_stays_finite(monkeypatch) -> None:\n    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "64")\n    for raw in ("garbage", "0", "-7"):\n        monkeypatch.setenv("MMM_AGENT_TOOL_ROUNDS", raw)\n        assert model_router._agent_tool_round_limit() == 64\n''',
    encoding="utf-8",
)

# Context tests should assert canonical behavior, not a wrapper marker around a dead fallback.
test_context = ROOT / "tests/test_llama_context_safety_contract.py"
test_text = test_context.read_text(encoding="utf-8")
legacy_start = test_text.find("\ndef test_ultra_context_fallback_never_silently_drops_authored_task()")
legacy_end = test_text.find("\ndef test_runtime_context_safety_wrappers_are_installed()", legacy_start)
if legacy_start < 0 or legacy_end < 0:
    raise SystemExit("context safety legacy test block not found")
test_text = test_text[:legacy_start] + test_text[legacy_end:]
test_text = test_text.replace(
    '''    assert getattr(\n        tool_loop._replace_live_messages,\n        "_mmm_protocol_safe_live_replace_v1",\n        False,\n    )\n''',
    "",
)
test_text = test_text.replace("from minecraft_mod_ai import progress_aware_tool_loop as tool_loop\n", "")
test_context.write_text(test_text, encoding="utf-8")

# Delete the production shim after all references have been migrated.
shim = ROOT / "minecraft_mod_ai/llama_tool_round_safety_contract.py"
if not shim.exists():
    raise SystemExit("missing llama_tool_round_safety_contract.py")
shim.unlink()

# No production or test reference may survive this cleanup.
for base in (ROOT / "minecraft_mod_ai", ROOT / "tests"):
    for path in base.rglob("*.py"):
        if "llama_tool_round_safety_contract" in path.read_text(encoding="utf-8"):
            raise SystemExit(f"stale tool-round shim reference: {path.relative_to(ROOT)}")

# Remove the one-shot scaffolding from the resulting repository state.
(ROOT / ".github/workflows/worker05-clean-runtime-v1.yml").unlink(missing_ok=True)
Path(__file__).unlink(missing_ok=True)
